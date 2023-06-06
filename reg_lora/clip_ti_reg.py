import re
import os
import copy
import random
from pathlib import Path
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))

import PIL
import numpy as np
from PIL import Image
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from transformers.models.clip.modeling_clip import clip_loss
from torch.utils.data import Dataset, IterableDataset
import torch.nn as nn

def _randomset(lis):
    ret = []
    for i in range(len(lis)):
        if random.random() < 0.5:
            ret.append(lis[i])
    return ret

def _shuffle(lis):
    return random.sample(lis, len(lis))
IMAGENET_TEMPLATES_TINY = [
    "a photo of a {}",
]

IMAGENET_TEMPLATES_SMALL = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

IMAGENET_STYLE_TEMPLATES_SMALL = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

CONTRASTIVE_CONCEPT = [
    "teddy bear",
    "dog",
    "cat",
    "chair",
    "bird",
    "chair",
    "glasses",
    "sofa",
    "car",
    "bicycle",]



# class CLIPMaskTextModel(CLIPTextModel):
def get_identifier_masked_causal_attention_mask(bs, seq_len, identifier_indices: torch.Tensor = None, class_token_len: int = -1, dtype = torch.float16):

    if identifier_indices is not None:
        assert class_token_len >= 1
        for i, identifier_indice in zip(identifier_indices[0],identifier_indices[1]):
            mask = torch.empty(bs, seq_len, seq_len, dtype=dtype)
            mask.fill_(torch.tensor(torch.finfo(dtype).min))
            mask.triu_(1)  # zero out the lower diagonal
            mask = mask.unsqueeze(1)  # expand mask
            mask[i,:,identifier_indice, 1:max(identifier_indice,1)] = torch.finfo(dtype).min
            mask[i,:,identifier_indice+class_token_len+1:,identifier_indice] = torch.finfo(dtype).min

    else:
        mask = torch.empty(bs, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask

    return mask


def build_causal_attention_mask( bs, seq_len, dtype):
    mask = torch.empty(bs, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask

class CLIPTiDataset(Dataset):
    def __init__(self,
                 reg_texts_file,
                 placeholder_tokens,
                 class_token,
                 repeat = 2,
                 learnable_property = "style",
                 class_data_root = None,
                 contrastive_training = False,
                 stochastic_attribute = None,
                 initializer_token_as_class = False,
                ) -> None:
        super().__init__()
        self.placeholder_token = placeholder_tokens

        assert os.path.exists(reg_texts_file)
        with open(reg_texts_file) as f:
            self.templates = f.readlines()
        self.templates = [t.strip() for t in self.templates]
        self.num_templates = len(self.templates) 

        placeholder_tokens = ' '.join(placeholder_tokens.split('+'))
        self.class_token = class_token
        self.customized_token = placeholder_tokens + ' ' + class_token if initializer_token_as_class else placeholder_tokens
       
        self.contrastive_training = contrastive_training
        self._length = self.num_templates
        self.repeat = repeat

        self.stochastic_attribute = (
            stochastic_attribute.split(",") if stochastic_attribute else []
        )

        self.sd_templates = IMAGENET_TEMPLATES_TINY
    def __len__(self):
        return self._length if ~self.repeat else self.repeat * self._length

    def __getitem__(self, index) -> dict:
        example = {}

        reg_text  = random.choice(self.templates)
        text = reg_text.replace(self.class_token, self.customized_token)
        example["text"] = text
        example["reg_text"] = reg_text

        return example

class CLIPTiScoreCalculator(nn.Module):
    def __init__(self,text_model, tokenizer, placeholder_tokens, class_token_len, version='openai/clip-vit-large-patch14') -> None:
        super().__init__()
        placeholder_tokens = placeholder_tokens.split('+')
        assert len(placeholder_tokens) == 1
        self.placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_tokens)[0] # TODO multiple placeholder tokens
        self.class_token_len = class_token_len

        self.model = CLIPTextModelWithProjection.from_pretrained(version)
        del self.model.text_model
        self.model.text_model  = text_model
        self.tokenizer =  tokenizer
    
        self.logit_scale = torch.tensor([0.5],requires_grad=False)

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype
        
    def _get_text_features(self, input_ids, mask_identifier_causal_attention = False, output_attentions = False): 

        if mask_identifier_causal_attention:
            identifier_indices= torch.where(input_ids == self.placeholder_token_id)
            causal_attention_mask = get_identifier_masked_causal_attention_mask(input_ids.size(0), 77 ,identifier_indices, class_token_len= self.class_token_len, dtype= self.dtype).to(input_ids.device)
        else:
            causal_attention_mask = build_causal_attention_mask(input_ids.size(0), 77, dtype= self.dtype).to(input_ids.device)

        hidden_states = self.model.text_model.embeddings(input_ids=input_ids)
        outputs = self.model.text_model.encoder(
            hidden_states,
            causal_attention_mask = causal_attention_mask,
            output_attentions = output_attentions,
        )
        last_hidden_states =  outputs.last_hidden_state # this is used for controlling SD
        last_hidden_states = self.model.text_model.final_layer_norm(last_hidden_states)
        # we added new token, so we need to find the last token of the input_ids
        pooled_output = last_hidden_states[ 
                    torch.arange(last_hidden_states.size(0), device=input_ids.device),
                    [(row == 49407).nonzero().min() for row in input_ids]
                    ]
        text_embeds = self.model.text_projection(pooled_output.to(self.dtype))
        output_attentions = None
        if output_attentions:
            output_attentions = outputs.attentions

        return (text_embeds, output_attentions)

    def _check_type(self, inputs):
        if isinstance(inputs, torch.LongTensor):
            inputs = inputs.to(self.device)
        elif isinstance(inputs, torch.FloatTensor):
            inputs = inputs.to(self.device,dtype = self.dtype)
        return inputs
        
    def clip_text_forward(self, texts, reg_texts, mask_identifier_causal_attention = False, output_attentions = False):
        input_ids = self.tokenizer(texts, return_tensors="pt", padding='max_length',truncation=True)['input_ids']
        reg_input_ids = self.tokenizer(reg_texts, return_tensors="pt", padding='max_length',truncation=True)['input_ids']
        
        input_ids = self._check_type(input_ids)
        reg_input_ids = self._check_type(reg_input_ids)
        
        text_embeds, output_text_attentions = self._get_text_features(input_ids,
                                               mask_identifier_causal_attention = mask_identifier_causal_attention,
                                               output_attentions = output_attentions)
        
        reg_text_embeds, output_reg_text_attentions = self._get_text_features(reg_input_ids,
                                                  mask_identifier_causal_attention = mask_identifier_causal_attention,
                                                    output_attentions = output_attentions)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        reg_text_embeds = reg_text_embeds / reg_text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp().to(self.device, self.dtype)
        logits_per_text = torch.matmul(text_embeds, reg_text_embeds.t()) * logit_scale
        logits_per_reg_text = logits_per_text.t()

        return (
            logits_per_reg_text,
            logits_per_text,
            text_embeds,
            reg_text_embeds,
            output_text_attentions,
        )
        
    def forward(self, texts, reg_texts, mask_identifier_causal_attention = False, contrastive_loss = False ):
        """_summary_

        Args:
            text (List(str)): text with placeholder token.
            reg_text (List(str)): text with no placeholder token. 
            mask_identifier_causal_attention (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        (
            logits_per_reg_text,
            logits_per_text,
            text_embeds,
            reg_text_embeds,
            output_text_attentions,
        ) = self.clip_text_forward(texts=texts,
                                reg_texts=reg_texts,
                                mask_identifier_causal_attention = mask_identifier_causal_attention,)
        

        if contrastive_loss: 
            loss = clip_loss(logits_per_text)
            print(logits_per_reg_text)
        else:
            text_embeds = text_embeds
            reg_text_embeds = reg_text_embeds.detach()
            # simple mse loss
            loss = 1 - torch.cosine_similarity(text_embeds, reg_text_embeds).mean()
        return loss

# class CLIPTiScorer(nn.Module):
#     def __init__(self) -> None:
#         super().__init__() 

if __name__ == "__main__":
    dataset = CLIPTiDataset(reg_texts_file = "custom_data/data_reg/dog_reg.txt",
                            placeholder_tokens = "<krk1>",
                            class_token = "dog",
                            repeat = 2,)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)
    model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14') 
    clip = CLIPTiScoreCalculator(model.text_model, tokenizer, placeholder_tokens = "<krk1>", class_token_len = 1)
    for batch in dataloader:
        texts = batch['text']
        reg_texts = batch['reg_text']
        clip(texts, reg_texts, mask_identifier_causal_attention = False, contrastive_loss = True)

        break
import re
import os
import copy
import random
from pathlib import Path
import torch
from typing import Any, Optional, Tuple, Union
import sys

from transformers.configuration_utils import PretrainedConfig

sys.path.append(str(Path(__file__).parent.parent))

import PIL
import numpy as np
from PIL import Image
from transformers import CLIPTextModelWithProjection, CLIPProcessor, CLIPTokenizer, CLIPVisionModel,CLIPModel
from transformers.models.clip.modeling_clip import clip_loss,CLIPTextTransformer, BaseModelOutputWithPooling,CLIPTextConfig, _expand_mask, CLIPTextModel
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


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    
# class CLIPMaskTextModel(CLIPTextModel):
def get_identifier_masked_causal_attention_mask_V2(bs, seq_len, identifier_indices: torch.Tensor = None, class_token_len: int = -1, dtype = torch.float16):

    if identifier_indices is not None and len(identifier_indices[0]): 
        assert class_token_len >= 1
        for i, identifier_indice in zip(identifier_indices[0],identifier_indices[1]):
            mask = torch.empty(bs, seq_len, seq_len, dtype=dtype)
            mask.fill_(torch.tensor(torch.finfo(dtype).min))
            mask.triu_(1)  # zero out the lower diagonal
            mask = mask.unsqueeze(1)  # expand mask
            mask[i,:,identifier_indice, :max(identifier_indice,1)] = torch.finfo(dtype).min
            mask[i,:,identifier_indice+class_token_len+1:,identifier_indice] = torch.finfo(dtype).min
    else:
        mask = torch.empty(bs, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask

    return mask


# class CLIPMaskTextModel(CLIPTextModel):
def get_identifier_masked_causal_attention_mask(bs, seq_len, identifier_indices, class_token_len: int = -1, dtype = torch.float16):

    if identifier_indices is not None and len(identifier_indices[0]): 
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

class CLIPTiTextTransformer(CLIPTextTransformer):
    def __init__(self, config: CLIPTextConfig,
                mask_identifier_causal_attention: Optional[bool] = False,
                class_token_len: Optional[int] = -1,
                placeholder_token_id: Optional[int] = 49408, 
        ):
        super().__init__(config)
        self.mask_identifier_causal_attention = mask_identifier_causal_attention
        self.class_token_len = class_token_len
        self.placeholder_token_id = placeholder_token_id

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        bsz, seq_len = input_ids.size()
        
        if self.mask_identifier_causal_attention:
            identifier_indices= torch.where(input_ids == self.placeholder_token_id)
            causal_attention_mask = get_identifier_masked_causal_attention_mask_V2(input_ids.size(0), 77 ,identifier_indices,\
                            class_token_len=self.class_token_len, dtype= hidden_states.dtype).to(hidden_states.device)
        else:
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[ 
                        torch.arange(last_hidden_state.size(0), device=input_ids.device),
                        [(row == 49407).nonzero().min() for row in input_ids]
                        ]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class CLIPTiTextModel(CLIPTextModel):

    config_class = CLIPTextConfig
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: PretrainedConfig,
                 **kwargs):
        super(CLIPTextModel, self).__init__(config)
        self.text_model = CLIPTiTextTransformer(config, **kwargs)
        self.post_init()

class CLIPTiDataset(Dataset):
    def __init__(self,
                 reg_texts_file, # files that include the reg texts
                 custom_token, # e.g. <krk1>
                 class_token, # e.g. dog
                 repeat = 10,
                 reg_images_root = None, # if not None, the images will be sampled from the class_data_root
                ) -> None:
        super().__init__()

        assert os.path.exists(reg_texts_file)
        with open(reg_texts_file) as f:
            self.templates = f.readlines()

        self.templates = [t.strip() for t in self.templates]
        self.num_templates = len(self.templates) 

        self.class_token = class_token
        self.custom_token = custom_token
       
        self._length = self.num_templates
        self.repeat = repeat

        self.reg_images_root = reg_images_root 
        if self.reg_images_root:
            self.reg_images_path = list(Path(self.reg_images_root).iterdir())
            self.num_reg_images = len(self.reg_images_path)
            
    def __len__(self):
        return self.repeat * self._length

    def __getitem__(self, index) -> dict:
        
        example = {}
        reg_text  = self.templates[index % self.num_templates]
        text = reg_text.replace(self.class_token, self.custom_token)
        example["text"] = text
        example["reg_text"] = reg_text

        if self.reg_images_root is not None:
            reg_instance_image = Image.open(self.reg_images_path[index % self.num_reg_images])
            if not reg_instance_image.mode == "RGB":
                reg_instance_image = reg_instance_image.convert("RGB")
            example["reg_image"] = np.array(reg_instance_image)

        return example

class CLIPTiScoreCalculator(nn.Module):
    
    def __init__(self,
                text_model: CLIPTiTextModel, 
                tokenizer: CLIPTokenizer,
                mode = 'text',
                version='openai/clip-vit-large-patch14',) -> None:
        super().__init__()

        # TODO multiple placeholder tokens

        # self.text_model = CLIPTextModelWithProjection.from_pretrained(version)
        self.model = CLIPModel.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version)
        del self.model.text_model
        del self.processor.tokenizer
        self.processor.tokenizer = tokenizer
        self.model.text_model  = text_model.text_model

        self.logit_scale = self.model.logit_scale

        if mode == 'text':
            self.forward = self.clip_to_text_forward
        elif mode == 'image':
            self.forward = self.clip_to_image_forward

        elif 'text' in mode and 'image' in mode:
            self.forward = self.forward_to_both_forward
        else :
            raise NotImplementedError

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype
        
    def _check_type(self, inputs):
        if isinstance(inputs, torch.LongTensor):
            inputs = inputs.to(self.device)
        elif isinstance(inputs, torch.FloatTensor):
            inputs = inputs.to(self.device,dtype = self.dtype)
        return inputs

    def sim_loss(self, text_embeds, reg_embeds):
        assert text_embeds.shape == reg_embeds.shape
        text_embeds = text_embeds
        reg_embeds = reg_embeds.detach()
        sim_loss = 1 - torch.cosine_similarity(text_embeds, reg_embeds).mean() 

        return sim_loss

    def clip_to_text_forward(self, texts1, reg_texts,  **kwargs):
        input_ids = self.processor.tokenizer(texts1, return_tensors="pt", padding='max_length',truncation=True)['input_ids']
        reg_input_ids = self.processor.tokenizer(reg_texts, return_tensors="pt", padding='max_length',truncation=True)['input_ids']
        input_ids = self._check_type(input_ids)
        reg_input_ids = self._check_type(reg_input_ids)
        reg_input_ids.requires_grad = False
        text_embeds= self.model.get_text_features(input_ids)
        reg_text_embeds = self.model.get_text_features(reg_input_ids)

        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        reg_text_embeds = reg_text_embeds / reg_text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp().to(self.device, self.dtype)
        logits_per_text = torch.matmul(text_embeds, reg_text_embeds.t()) * logit_scale

        to_text_contrastive_loss = clip_loss(logits_per_text)
           
        return to_text_contrastive_loss

    def clip_to_image_forward(self, texts2, reg_images, **kwargs):
        """_summary_

        Args:
            texts (List(str)): text with placeholder token, it should be paired with images or paired with the first image in reg_images when its length is 1.
            reg_images (List(PIL.Image)): list of images # the images should be paired with the texts
        """
        processed = self.processor(texts2, reg_images, return_tensors="pt", padding='max_length',truncation=True) 
        input_ids = self._check_type(processed['input_ids'])
        pixel_values = self._check_type(processed['pixel_values'])
        pixel_values.requires_grad = False
        clip_outputs = self.model(pixel_values = pixel_values, input_ids = input_ids )
        logits_per_text = clip_outputs.logits_per_text
        logits_per_image = clip_outputs.logits_per_image
        
        to_image_contrastive_loss = (contrastive_loss(logits_per_text[:1]) + contrastive_loss(logits_per_image[1:]))/2

        return to_image_contrastive_loss
        
    def forward_to_both_forward(self, texts1, reg_texts, texts2, reg_images,):
        """_summary_

        Args:
            text (List(str)): text with placeholder token.
            reg_text (List(str)): text with no placeholder token. 
            reg_images (List(PIL.Image)): list of images.
            mask_identifier_causal_attention (bool, optional): _description_. Defaults to False.
        """
        to_text_constrastive_loss = self.clip_to_text_forward(texts1, reg_texts)
        to_image_constrastive_loss = self.clip_to_image_forward(texts2, reg_images)
        print(to_text_constrastive_loss.detach(), to_image_constrastive_loss.detach())
        return to_text_constrastive_loss + to_image_constrastive_loss 

    def forward(self, *args, **kwargs):
        return


if __name__ == "__main__":
    dataset = CLIPTiDataset(reg_texts_file = "custom_data/data_reg/dog_reg.txt",
                            placeholder_token = "<krk1>",
                            class_token = "dog",
                            repeat = 2,)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)
    model = CLIPTiTextModel.from_pretrained('openai/clip-vit-large-patch14',
                                             mask_identifier_causal_attention = True, 
                                             output_attentions = True,
                                             class_token_len = 1)
    tokenizer = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14').tokenizer
 
    clip = CLIPTiScoreCalculator(model, tokenizer, placeholder_token = "<krk1>", class_token_len = 1)
    for batch in dataloader:
        texts = batch['text']
        reg_texts = batch['reg_text']
        clip(texts, reg_texts, mask_identifier_causal_attention = False, contrastive_loss = True)
        break
import re
import os
import copy
import random
from pathlib import Path

import PIL
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor, CLIPTextModel
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
                 instance_data_root,
                 placeholder_tokens,
                 class_token = None,
                 repeat = 2,
                 learnable_property = "style",
                 class_data_root = None,
                 contrastive_training = False,
                 stochastic_attribute = None,
                ) -> None:
        super().__init__()
        self.instance_data_root = Path(instance_data_root)
        self.placeholder_token = placeholder_tokens
        self.instance_images_path = list(self.instance_data_root.iterdir())
        self.num_instance_images = len(self.instance_images_path)
        placeholder_tokens = ' '.join(placeholder_tokens.split('+'))
        self.customized_token = placeholder_tokens + ' ' + class_token if class_token else placeholder_tokens
        self.templates = (
            IMAGENET_STYLE_TEMPLATES_SMALL
            if learnable_property == "style"
            else IMAGENET_TEMPLATES_SMALL
        )
        
        self.contrastive_training = contrastive_training
        self.contrastive_concepts = CONTRASTIVE_CONCEPT
        self._length = self.num_instance_images
        self.repeat = repeat

        self.class_data_root = class_data_root #TODO
        self.stochastic_attribute = (
            stochastic_attribute.split(",") if stochastic_attribute else []
        )

    def __len__(self):
        return self._length if self.class_data_root is not None else self.repeat * self._length

    def __getitem__(self, index) -> dict:
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["np_instance_image"] = np.array(instance_image)
        
        text = random.choice(self.templates).format(
            ", ".join(
                [self.customized_token]
                + _shuffle(_randomset(self.stochastic_attribute))
            )
        )
        example["text"] = text
        example["contrastive_text"] = ""

        if self.contrastive_training:
            contrastive_text =  random.choice(self.templates).format(
                ", ".join(
                    [random.choice(self.contrastive_concepts)]
                    + _shuffle(_randomset(self.stochastic_attribute))
                )
            )
            example["contrastive_text"] = contrastive_text
        return example

class CLIPTiScoreCalculator(nn.Module):
    def __init__(self,text_model, tokenizer, placeholder_tokens, class_token_len, version='openai/clip-vit-large-patch14', device='cuda:0', weight_dtype = torch.float32) -> None:
        super().__init__()
        placeholder_tokens = placeholder_tokens.split('+')
        assert len(placeholder_tokens) == 1
        self.placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_tokens)[0] # TODO multiple placeholder tokens
        self.class_token_len = class_token_len

        self.model = CLIPModel.from_pretrained(version).to(device,dtype = weight_dtype)
        self.processor = CLIPProcessor.from_pretrained(version)

        del self.model.text_model
        del self.processor.tokenizer
        
        self.model.text_model  = text_model
        self.processor.tokenizer =  tokenizer
        self.device = device
        self.weight_dtype = weight_dtype
        
    def _get_text_features(self, input_ids, mask_identifier_causal_attention = False, output_attentions = False): 

        if mask_identifier_causal_attention:
            identifier_indices= torch.where(input_ids == self.placeholder_token_id)
            causal_attention_mask = get_identifier_masked_causal_attention_mask(input_ids.size(0), 77 ,identifier_indices, class_token_len= self.class_token_len, dtype= self.weight_dtype).to(input_ids.device)
        else:
            causal_attention_mask = build_causal_attention_mask(input_ids.size(0), 77, dtype= self.weight_dtype).to(input_ids.device)

        hidden_states = self.model.text_model.embeddings(input_ids=input_ids)
        outputs = self.model.text_model.encoder(
            hidden_states,
            causal_attention_mask = causal_attention_mask,
            output_attentions = output_attentions,
        )
        last_hidden_states =  outputs.last_hidden_state # this is used for controlling SD
        last_hidden_states = self.model.text_model.final_layer_norm(last_hidden_states)
        pooled_output = last_hidden_states[
                    torch.arange(last_hidden_states.size(0), device=input_ids.device),
                    [(row == 49407).nonzero().min() for row in input_ids]
                    ]
        text_embeds = self.model.text_projection(pooled_output.to(self.weight_dtype))
        output_attentions = None
        if output_attentions:
            output_attentions = outputs.attentions

        return (text_embeds, output_attentions)
    
    def clip_forward(self, text, images, mask_identifier_causal_attention = False, output_attentions = False):
        
        _inputs = self.processor(text=text, images=images,max_length=77,  return_tensors="pt", padding='max_length',truncation=True)
        for k, v in _inputs.items():
            if isinstance(v, torch.LongTensor):
                _inputs[k] = v.to(self.device)
            elif isinstance(v, torch.FloatTensor):
                _inputs[k] = v.to(self.device,dtype = self.weight_dtype)

        input_ids = _inputs.pop("input_ids")
        pixel_values = _inputs.pop("pixel_values") # not return attention mask

        image_embeds = self.model.get_image_features(pixel_values = pixel_values)
        
        text_embeds, output_text_attentions = self._get_text_features(input_ids,
                                               mask_identifier_causal_attention = mask_identifier_causal_attention,
                                               output_attentions = output_attentions)
        
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        return (
            logits_per_image,
            logits_per_text,
            text_embeds,
            image_embeds,
            output_text_attentions,
        )
        
    def forward(self, text, images, mask_identifier_causal_attention = False, contrastive_loss = False ):
        """_summary_

        Args:
            text (List(str)): text with placeholder token.
            images (List(np.numpy)): images to be cumtomized using text iversion.
            mask_identifier_causal_attention (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        _,logits_per_text, text_embeds, image_embeds, _ = self.clip_forward(text, 
                                    images,
                                    mask_identifier_causal_attention = mask_identifier_causal_attention,
                                    )
        if contrastive_loss: 
            loss = clip_loss(logits_per_text)
        else:
            text_embeds = text_embeds
            image_embeds = image_embeds.detach()
            # simple mse loss
            loss = 1 - torch.cosine_similarity(text_embeds, image_embeds).mean()
        return loss

# class CLIPTiScorer(nn.Module):
#     def __init__(self) -> None:
#         super().__init__() 


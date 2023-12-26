import re
import os
import copy
import random
from pathlib import Path
import torch
from typing import Any, Optional, Tuple, Union
import sys


sys.path.append(str(Path(__file__).parent.parent))

import PIL
import numpy as np
from PIL import Image
from .modules import CLIPTiTextModel
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
def get_identifier_masked_causal_attention_mask_V2(bs, seq_len, identifier_indices: torch.Tensor = None, class_token_len: int = -1, dtype = torch.float16, mask_identifier_ratio=0.5):

    if identifier_indices is not None and len(identifier_indices[0]) and random.random() < mask_identifier_ratio: 
        assert class_token_len >= 1
        for i, identifier_indice in zip(identifier_indices[0],identifier_indices[1]):
            mask = torch.empty(bs, seq_len, seq_len, dtype=dtype)
            mask.fill_(torch.tensor(torch.finfo(dtype).min))
            mask.triu_(1)  # zero out the lower diagonal
            mask = mask.unsqueeze(1)  # expand mask
            # mask[i,:,identifier_indice, :max(identifier_indice,1)] = torch.finfo(dtype).min
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


class CLIPTiScoreCalculator(nn.Module):
    
    def __init__(self,
                text_model: CLIPTiTextModel, 
                tokenizer: CLIPTokenizer,
                mode = 'text',
                version='models/clip-vit-large-patch14',) -> None:
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)
    model = CLIPTiTextModel.from_pretrained('models/clip-vit-large-patch14',
                                             mask_identifier_causal_attention = True, 
                                             output_attentions = True,
                                             class_token_len = 1)
    tokenizer = CLIPProcessor.from_pretrained('models/clip-vit-large-patch14').tokenizer
 
    clip = CLIPTiScoreCalculator(model, tokenizer, placeholder_token = "<krk1>", class_token_len = 1)
    for batch in dataloader:
        texts = batch['text']
        reg_texts = batch['reg_text']
        clip(texts, reg_texts, mask_identifier_causal_attention = False, contrastive_loss = True)
        break
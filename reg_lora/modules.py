import re
import os
import copy
import random
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import PIL
import torch
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



def get_identifier_masked_causal_attention_mask_V2(bs, seq_len, identifier_indices: torch.Tensor = None, class_token_len: int = -1, dtype = torch.float16,mask_identifier_ratio=0.5):

    if identifier_indices is not None and len(identifier_indices[0]) and random.random() < mask_identifier_ratio: 
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


class CLIPTiTextTransformer(CLIPTextTransformer):
    def __init__(self, config: CLIPTextConfig,
                mask_identifier_causal_attention: Optional[bool] = False,
                class_token_len: Optional[int] = -1,
                placeholder_token_id: Optional[int] = 49408,
                mask_identifier_ratio=0.5, 
        ):
        super().__init__(config)
        self.mask_identifier_causal_attention = mask_identifier_causal_attention
        self.class_token_len = class_token_len
        self.placeholder_token_id = placeholder_token_id
        self.mask_identifier_ratio = mask_identifier_ratio

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
            causal_attention_mask = get_identifier_masked_causal_attention_mask_V2(
                bsz,
                seq_len,
                identifier_indices,
                class_token_len=self.class_token_len, dtype= hidden_states.dtype,
                mask_identifier_ratio=self.mask_identifier_ratio,
                            ).to(hidden_states.device)
        else:
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = get_identifier_masked_causal_attention_mask_V2(bsz, seq_len, dtype= hidden_states.dtype).to(
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

    def __init__(self, config,
                 **kwargs):
        super(CLIPTextModel, self).__init__(config)
        self.text_model = CLIPTiTextTransformer(config, **kwargs)
        self.post_init()


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
    def __init__(self,text_model, tokenizer, placeholder_tokens, class_token_len, version='models/clip-vit-large-patch14') -> None:
        super().__init__()
        placeholder_tokens = placeholder_tokens.split('+')
        assert len(placeholder_tokens) == 1
        self.placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_tokens)[0] # TODO multiple placeholder tokens
        self.class_token_len = class_token_len

        self.model = CLIPModel.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version)

        del self.model.text_model
        del self.processor.tokenizer
        
        self.model.text_model  = text_model
        self.processor.tokenizer =  tokenizer

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
            causal_attention_mask = get_identifier_masked_causal_attention_mask(input_ids.size(0), 77, dtype= self.dtype).to(input_ids.device)

        hidden_states = self.model.text_model.embeddings(input_ids=input_ids)
        outputs = self.model.text_model.encoder(
            hidden_states,
            causal_attention_mask = causal_attention_mask,
            output_attentions = output_attentions,
        )
        last_hidden_states =  outputs.last_hidden_state # this is used for controlling SD
        last_hidden_states = self.model.text_model.final_layer_norm(last_hidden_states)
        # we added new token , so we need to find the last token of the input_ids
        pooled_output = last_hidden_states[ 
                    torch.arange(last_hidden_states.size(0), device=input_ids.device),
                    [(row == 49407).nonzero().min() for row in input_ids]
                    ]
        text_embeds = self.model.text_projection(pooled_output.to(self.dtype))
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
                _inputs[k] = v.to(self.device,dtype = self.dtype)

        input_ids = _inputs.pop("input_ids")
        pixel_values = _inputs.pop("pixel_values") # not return attention mask

        image_embeds = self.model.get_image_features(pixel_values = pixel_values)
        
        text_embeds, output_text_attentions = self._get_text_features(input_ids,
                                               mask_identifier_causal_attention = mask_identifier_causal_attention,
                                               output_attentions = output_attentions)
        
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp().to(self.device, self.dtype)
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
        (
            _,
            logits_per_text,
            text_embeds,
            image_embeds,
            _,
         )= self.clip_forward(text, images, mask_identifier_causal_attention = mask_identifier_causal_attention, )

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


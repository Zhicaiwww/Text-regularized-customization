import re
import os
import copy
import random
from pathlib import Path

import PIL
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor
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

class CLIPTiDataset(Dataset):
    def __init__(self,
                 instance_data_root,
                 placeholder_token,
                 class_token = None,
                 repeat = 2,
                 learnable_property = "style",
                 class_data_root = None,
                 contrastive_training = False,
                 stochastic_attribute = None,
                ) -> None:
        super().__init__()
        self.instance_data_root = Path(instance_data_root)
        self.placeholder_token = placeholder_token
        self.instance_images_path = list(self.instance_data_root.iterdir())
        self.num_instance_images = len(self.instance_images_path)
        placeholder_token = ' '.join(placeholder_token.split('+'))
        self.customized_token = placeholder_token + ' ' + class_token if class_token else placeholder_token
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
    def __init__(self,text_model, tokenizer, version='openai/clip-vit-large-patch14', device='cuda:0', weight_dtype = torch.float32) -> None:
        super().__init__()

        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device,dtype = weight_dtype)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        del self.model.text_model
        del self.processor.tokenizer
        self.model.text_model  = text_model
        self.processor.tokenizer =  tokenizer
        self.device = device
        self.weight_dtype = weight_dtype
        
    def forward(self, text, images):
        _inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        for k, v in _inputs.items():
            if isinstance(v, torch.LongTensor):
                _inputs[k] = v.to(self.device)
            elif isinstance(v, torch.FloatTensor):
                _inputs[k] = v.to(self.device,dtype = self.weight_dtype)
        outputs = self.model(**_inputs)
        text_embeds = outputs.text_embeds
        image_embeds = outputs.image_embeds.detach()
        # simple mse loss
        sim_loss = 1 - torch.cosine_similarity(text_embeds, image_embeds).mean()
        return sim_loss

# class CLIPTiScorer(nn.Module):
#     def __init__(self) -> None:
#         super().__init__() 


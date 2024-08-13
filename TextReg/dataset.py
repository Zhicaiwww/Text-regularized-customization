import torch
import os
import math
import random
from custom_datasets.utils import IMAGENET_TEMPLATES_SMALL, IMAGENET_STYLE_TEMPLATES_SMALL, IMAGENET_TEMPLATES_TINY
from torchvision import transforms
from pathlib import Path
from PIL import Image

class ConcatenateDataset(torch.utils.data.Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2

    def __len__(self):
        return math.ceil(len(self.ds1) * len(self.ds2) / math.gcd(len(self.ds1), len(self.ds2)))  # Return the maximum length

    def __getitem__(self, index):
        example_1 = self.ds1[index % len(self.ds1)]
        example_2 = self.ds2[index % len(self.ds2)] 
        instance_images = torch.stack((example_1["instance_images"], example_2["instance_images"]), dim=0)   
        instance_prompt_ids = torch.stack((example_1["instance_prompt_ids"], example_2["instance_prompt_ids"]), dim=0)  

        return  dict(instance_images = instance_images, instance_prompt_ids = instance_prompt_ids)

class DreamBoothTiDataset(torch.utils.data.Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        learnable_property,
        custom_token,
        tokenizer,
        size=512,
        class_token=None,
        prompts_file=None,
        return_reg_text=False,
        center_crop=False,
        reg_prompts= None,
        repeat = 20,
    ):
        if return_reg_text:
            assert class_token is not None
        self.class_token = class_token
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.repeat = repeat
        self.return_reg_text = return_reg_text
        self.custom_token = custom_token
        self.reg_prompts = reg_prompts
        self.prompts = None

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        try:
            try:
                self.instance_images_path = sorted(list(Path(instance_data_root).iterdir()), key = lambda x: int(str(x).split('/')[-1].split('.', 1)[0]))
            except Exception:
                self.instance_images_path = sorted(list(Path(instance_data_root).iterdir()), key = lambda x: int(str(x).split('/')[-1].split('-', 1)[0]))
        except Exception:
            self.instance_images_path = sorted(list(Path(instance_data_root).iterdir()))
        
        self.num_instance_images = len(self.instance_images_path)

        self.templates = (
            IMAGENET_STYLE_TEMPLATES_SMALL
            if learnable_property == "style"
            else IMAGENET_TEMPLATES_SMALL
        )

        if prompts_file is not None and os.path.exists(prompts_file):
            with open(prompts_file, 'r', encoding='utf-8') as f:
                self.prompts = [line.strip() for line in f.readlines()]
            assert self.num_instance_images == len(self.prompts)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(0.2, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.repeat * self.num_instance_images

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images]).convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)

        if self.prompts is None:
            text = random.choice(self.templates).format(self.custom_token)
        else:
            text = self.prompts[index % self.num_instance_images]

        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.return_reg_text:
            if self.reg_prompts is None:
                reg_text = random.choice(IMAGENET_TEMPLATES_TINY).format(self.class_token)
            else:
                reg_text = random.choice(self.reg_prompts)

            example["reg_cpp_prompt_ids"] = self.tokenizer(
                reg_text,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        return example

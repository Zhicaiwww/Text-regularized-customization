
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import os
import json
import math
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/zhicai/poseVideo/Text-regularized-customization/')
from lora_diffusion import LoraInjectedConv2d, LoraInjectedLinear, patch_pipe, tune_lora_scale
from lora_diffusion.lora import _find_modules, UNET_CROSSATTN_TARGET_REPLACE
from reg_lora.visual import visualize_images
from reg_lora.clip_reg import CLIPTiDataset, CLIPTiScoreCalculator
from lora_diffusion.lora import patch_pipe

# os.environ["DISABLE_TELEMETRY"] = 'YES'
model_id = "runwayml/stable-diffusion-v1-5"
device = torch.device("cuda:1")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,local_files_only=True,revision='39593d5650112b4cc580433f6b0435385882d819').to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

prompt = "style of <s1><s2>, baby lion"
torch.manual_seed(0)
# image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]

# image  # nice. diffusers are cool.

lora_ckpt = 'lora_output/checkpoints/output_dog_Ti-clip_Nonorm_3e-5/lora_weight_e6_s1000.safetensors'
torch.manual_seed(0)

patch_pipe(
    pipe,
    lora_ckpt,
    patch_text=False,
    patch_ti=True,
    patch_unet=True,
    filter_crossattn_str = 'cross+self'
)

data  = CLIPTiDataset(instance_data_root="custom_data/data/dog",
                      placeholder_tokens="<krk1>",
                      class_token='dog'
                      )
clip = CLIPTiScoreCalculator(text_model = pipe.text_encoder.text_model,
                             tokenizer = pipe.tokenizer,
                             placeholder_tokens = "<krk1>",
                             class_token_len = 1,
                             device= device,
                             weight_dtype= torch.float16,)
# TODO batch_size >= 1
data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=1)

clip_batch = next(iter(data_loader))
instance_texts = clip_batch['text']
instance_images = [image for image in clip_batch['np_instance_image']]

henutputs = clip.forward(instance_texts, instance_images, mask_identifier_causal_attention = True)


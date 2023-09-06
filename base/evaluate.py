from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import os
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys add pwd path
sys.path.append('/data/zhicai/code/Text-regularized-customization')
from lora_diffusion import patch_pipe, tune_lora_scale
from evaluation_pipe import  evaluate_pipe, prepare_clip_model_sets
import time
import json
from collections import defaultdict
import os
import argparse

os.environ["DISABLE_TELEMETRY"] = 'YES'
os.environ["HTTP_PROXY"] = "http://localhost:8890"
os.environ["HTTPS_PROXY"] = "http://localhost:8890"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora-ckpt', type=str, required=True)
    parser.add_argument('--instance-data-dir', type=str, required=True)
    parser.add_argument('--class-tokens', type=str, default='dog')
    parser.add_argument('--custom-token', type=str, default='<krk1> dog')
    parser.add_argument('--n-test', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=6)
    args = parser.parse_args()
    n_test = args.n_test

    model_id = "runwayml/stable-diffusion-v1-5"
    device = torch.device(f"cuda:{args.gpu}")
    time.sleep(0.01)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,local_ckpt_files_only=True,revision='39593d5650112b4cc580433f6b0435385882d819').to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    torch.manual_seed(0)


    preped_clip = prepare_clip_model_sets()
    instance_data_dir = args.instance_data_dir
    class_tokens = args.class_tokens
    custom_token = args.custom_token
    lora_ckpt = args.lora_ckpt
    # root_dir = '/data/zhicai/code/Text-regularized-customization/outputs/checkpoints/dog/ewc_reg_shareTI/baseline_lora_rank_10'
    # lora_ckpts = sorted(glob.glob(os.path.join(root_dir, "lora_weight_s*.safetensors")))
    logs = defaultdict()
    # for lora_ckpt in lora_ckpts:
    patch_pipe(
        pipe,
        lora_ckpt,
        patch_text=False,
        patch_ti=True,
        patch_unet=True,
        filter_crossattn_str = 'cross+self'
    )
    tune_lora_scale(pipe.unet, 1)

    test_image_path = instance_data_dir
    images = []
    for file in os.listdir(test_image_path):
        if (
            file.lower().endswith(".png")
            or file.lower().endswith(".jpg")
            or file.lower().endswith(".jpeg")
        ):
            images.append(
                Image.open(os.path.join(test_image_path, file))
            )

    log = evaluate_pipe(
        pipe,
        target_images=images,
        class_token=class_tokens,
        learnt_token=custom_token,
        n_test=n_test,
        n_step=50,
        clip_model_sets=preped_clip,
    ),
    
    
    logs['text_alignment_avg'] = log[0]["text_alignment_avg"]
    logs['image_alignment_avg'] = log[0]["image_alignment_avg"]

    print(logs)

    with open(os.path.join(os.path.dirname(os.path.abspath(lora_ckpt)),'evaluate_logs.json'), 'w') as f:
        json.dump(logs, f)
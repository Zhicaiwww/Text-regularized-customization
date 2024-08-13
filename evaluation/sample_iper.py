import os, sys, time, json, datetime, math, pdb
sys.path.append('/data/zhicai/code/Text-regularized-customization')
import argparse
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from collections import defaultdict
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline
from lora_diffusion import patch_pipe, tune_lora_scale

os.environ["DISABLE_TELEMETRY"] = 'YES'
os.environ["HTTP_PROXY"] = "http://localhost:8890"
os.environ["HTTPS_PROXY"] = "http://localhost:8890"

'''
CUDA_VISIBLE_DEVICES=1 python evaluation/sample_iper.py \
    --lora_ckpt 'logs/log_iper/person1/2023-12-28T22-55-39_person_baseline/lora_weight.safetensors' \
    --from_file prompts/TEMP.txt \
    --batch_size 1 \
    --n_img 1
'''

def image_grid(_imgs, rows=None, cols=None):

    if rows is None and cols is None:
        rows = cols = math.ceil(len(_imgs) ** 0.5)

    if rows is None:
        rows = math.ceil(len(_imgs) / cols)
    if cols is None:
        cols = math.ceil(len(_imgs) / rows)

    w, h = _imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(_imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


if __name__  == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--filter_crossattn_str', type=str, default="cross")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--n_img', type=int, default=10)
    parser.add_argument('--n_step', type=int, default=50)
    parser.add_argument('--prompts', type=str, default=None)
    parser.add_argument('--from_file', type=str, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    batch_size = args.batch_size

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    device = torch.device("cuda")
    time.sleep(0.01)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("models/stable-diffusion-v1-5", torch_dtype=torch.float16, revision='39593d5650112b4cc580433f6b0435385882d819', safety_checker=None, local_files_only=True).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, )
    pipe.set_progress_bar_config(disable=True)

    if args.lora_ckpt is not None:
        lora_ckpt = os.path.join(args.lora_ckpt, "lora_weight.safetensors") if ".safetensors" not in args.lora_ckpt else args.lora_ckpt
        patch_pipe(
            pipe,
            lora_ckpt,
            patch_text=False,
            patch_ti=True,
            patch_unet=True,
            filter_crossattn_str=args.filter_crossattn_str,
        )
    else:
        lora_ckpt = "lora_weight_pt.safetensors"
        patch_pipe(
            pipe,
            lora_ckpt,
            patch_text=False,
            patch_ti=False,
            patch_unet=False,
            filter_crossattn_str=args.filter_crossattn_str,
        )
    tune_lora_scale(pipe.unet, 1)

    if args.outdir is not None:
        save_path = os.path.join(args.outdir, f"samples_{now}") 
    else:
        save_path = os.path.join(os.path.dirname(lora_ckpt), f"samples_{now}") 
    os.makedirs(os.path.join(save_path, "samples"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "grids"), exist_ok=True)

    assert (args.prompts is None) ^ (args.from_file is None)
    if args.prompts is not None:
        prompt_list = [prompt for prompt in args.prompts.split(';') for _ in range(args.n_img)]
    elif args.from_file is not None:
        with open(args.from_file, "r") as f:
            data = f.read().splitlines()
            prompt_list = [prompt for prompt in data for _ in range(args.n_img)]

    seed_everything(args.seed)

    images, prompt_list_maybe_shuffled = [], []
    prompt_dataset = DataLoader(prompt_list, batch_size=batch_size)
    ref_dir = "custom_datasets/iper_subset/person2"
    for ref in sorted(os.listdir(ref_dir)):
        for prompts in tqdm(prompt_dataset, desc="Sampling"):
            with torch.autocast("cuda"):
                imgs_ = pipe(image=Image.open(os.path.join(ref_dir, ref)), prompt=prompts, strength=0.5, num_inference_steps=args.n_step, guidance_scale=5.0).images
                # imgs__ = [img.resize((512, 512), Image.BILINEAR) for img in imgs_]
            images.extend(imgs_)
            prompt_list_maybe_shuffled.extend(prompts)
    del pipe

    for idx, img in enumerate(images):
        img.save(os.path.join(save_path, "samples", f"{idx:05}_{prompt_list_maybe_shuffled[idx].replace(' ', '_')}.png"))
    for i in range(0, len(images), batch_size**2):
        slice = images[i:i+batch_size**2]
        grid = image_grid(slice, cols=batch_size)
        grid.save(os.path.join(save_path, "grids", f"{(i):05}-{(i+len(slice)-1):05}.png"))
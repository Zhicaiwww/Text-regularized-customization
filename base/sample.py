import torch
import os
import re
import copy
import argparse
import pathlib
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
os.environ["HTTP_PROXY"]="http://localhost:8890"
os.environ["HTTPS_PROXY"]="http://localhost:8890"
os.environ["WANDB_DISABLED"] = "true"
sys.path.append(os.getcwd())
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from lora_diffusion import LoraInjectedConv2d, LoraInjectedLinear, patch_pipe, tune_lora_scale, parse_safeloras
from utils import parse_templates_class_name





def sample_diffuser(pipe, templates, placehodler_name,bs, n_per_prompt, outpath, prompt=None, clear_cache=True):
    prompts = [prompt.replace('<placeholder>', placehodler_name) for prompt in templates] if prompt is None else [prompt]
    print(f'prompts: \n {prompts}')
    prompts = prompts * n_per_prompt
    # count png files in outpath
    cnt = len([name for name in os.listdir(outpath) if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg')])
    print(f'found {cnt} images in {outpath}')
    if clear_cache and cnt > 0:
        import shutil
        # remove files in outpath but keep the dir
        shutil.rmtree(outpath)
        os.mkdir(outpath)
        print(f'remove {outpath}')
        cnt = 0 
    # pipe.unet

    # split prompts into batches
    for i in tqdm.tqdm(range(0, len(prompts), bs), desc='sampling'):
        prompt = prompts[i:i+bs]
        img = pipe(prompt = prompt, num_inference_steps=50, guidance_scale=6).images
        for j in range(len(img)):
            img[j].save(f'{outpath}/{cnt}.png') 
            cnt += 1
        
        
        # _dir, _name = lora_ckpt.rsplit('/', 1) 
        # _dir = _dir.replace('checkpoints', 'figures')
        # _name = _name.replace('.safetensors', '')
        # outpath = os.path.join(_dir, _name)
        # os.makedirs(_dir, exist_ok=True)
        # print(f'saving to {outpath}')

OUTPUT_DIR = '/data/zhicai/code/Text-regularized-customization/dataset/sample'
if __name__  == '__main__':
    #  python scripts_my/sample.py  --lora-path outputs/checkpoints/cat_decay+clip_maskId/lora_weight_e10_s1100.safetensors --concept-name '<krk1> cat' --template concept --gpu 2 --seed 0 --n-per-prompt 4 --n-row 4 
    # python scripts_my/sample.py  --lora-path outputs/checkpoints/decay_maskId/lora_weight_e6_s1000.safetensors --concept-name 'dog' --template concept --gpu 2 --seed 0 --n-per-prompt 4 --n-row 4
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora-ckpt', type=str, default=None, help = 'path to lora ckpt or dir')
    parser.add_argument('--concept-path-name', type=str, default='dog', help = 'concept path name')
    parser.add_argument('--placeholder', type=str, default='<krk1>')
    parser.add_argument('--class-only', action='store_true', help = 'only sample class')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-per-prompt', type =int, default= 50)
    parser.add_argument('--bs', type =int, default= 4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    weight_dtype = torch.float16
    os.environ["DISABLE_TELEMETRY"] = 'YES'
    
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=weight_dtype,
                                                   local_ckpt_files_only=True,revision='39593d5650112b4cc580433f6b0435385882d819').to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    templates, class_name  = parse_templates_class_name(args.concept_path_name)

    pipe_copy = copy.deepcopy(pipe).to(device)
    
    if args.lora_ckpt is not None:
        print(f'loading lora from {args.lora_ckpt}')
        placehodler_name = ' '.join([args.placeholder,class_name])  if not args.class_only else class_name
        patch_pipe(
            pipe_copy,
            args.lora_ckpt,
            patch_text=False,
            patch_ti=True,
            patch_unet=True,
            filter_crossattn_str = 'cross+self'
        )
        tune_lora_scale(pipe.unet, 1)
        tune_lora_scale(pipe.text_encoder, 1)
        _dir, _name = args.lora_ckpt.rsplit('/', 1) 
        _dir = _dir.replace('checkpoints', 'sample')
        _name = _name.replace('.safetensors', '')
        outpath = os.path.join(_dir, _name, 'concept' if not args.class_only else 'class')
        os.makedirs(outpath, exist_ok=True)
    else:
        placehodler_name = class_name
        outpath = os.path.join(OUTPUT_DIR,class_name)
        os.makedirs(outpath, exist_ok=True) 

    print(f'placeholder name: {placehodler_name}')
    print(f'saving to {outpath}')
    print(f'n-per-prompt: {args.n_per_prompt}')
    print(f'batch size: {args.bs}')
    pipe.text_encoder.text_model.to(device, dtype=weight_dtype)
    
    sample_diffuser(
        pipe = pipe_copy,
        templates = templates,
        placehodler_name = placehodler_name,
        bs = args.bs,
        n_per_prompt=args.n_per_prompt,
        outpath = outpath,
        prompt=args.prompt,)
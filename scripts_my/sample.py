import torch
import os
import re
import copy
import argparse
import pathlib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/zhicai/poseVideo/Text-regularized-customization/')

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from lora_diffusion import LoraInjectedConv2d, LoraInjectedLinear, patch_pipe, tune_lora_scale, parse_safeloras
from lora_diffusion.lora import _find_modules, UNET_CROSSATTN_TARGET_REPLACE, DEFAULT_TARGET_REPLACE
from reg_lora.visual import visualize_images

prompt_templates = ['photo of a placeholder',
                    'photo of a placeholder in a swimming pool',
                    'photo of a placeholder in grand canyon,']

prompt_stype_templates = ['photo of a placeholder']

if __name__  == '__main__':
    #  python scripts_my/sample.py  --lora-path lora_output/checkpoints/NT-Decay --concept-name '<krk1> dog' --template concept --gpu 4 --seed 0 --n-per-prompt 4 --n-row 4
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora-path', type=str, required=True, help = 'path to lora ckpt or dir')
    parser.add_argument('--concept-name', type=str, required=True)
    parser.add_argument('--template', type=str, default="concept", help="concept or style")
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-per-prompt', type =int, default= 4)
    parser.add_argument('--n-row', type =int, default= 4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    weight_dtype = torch.float16
    os.environ["DISABLE_TELEMETRY"] = 'YES'
    
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=weight_dtype,
                                                   local_ckpt_files_only=True,revision='39593d5650112b4cc580433f6b0435385882d819').to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    templates = prompt_templates if args.template == 'concept' else prompt_stype_templates
    prompts = [prompt.replace('placeholder', args.concept_name) for prompt in templates] if args.prompt is None else [args.prompt]

    if pathlib.Path(args.lora_path).is_dir():
        _ckpt_files = os.listdir(args.lora_path)
        ckpt_pattern = r'lora_weight_e\d+_s\d{3,4}.safetensors'
        ckpt_files = [f for f in _ckpt_files if re.match(ckpt_pattern, f)]
        ckpt_files = sorted(ckpt_files, key = lambda x: int(re.match(r'.*e([0-9]+).*', x.split('/')[-1])[1]), reverse=False)
        lora_ckpts = [os.path.join(args.lora_path, f) for f in ckpt_files]
    else:
        lora_ckpts = [args.lora_path]

    bs = args.n_per_prompt
    pipe_copy = copy.deepcopy(pipe).to(device)

    for lora_ckpt in lora_ckpts:    
        _dir, _name = lora_ckpt.rsplit('/', 1)  
        print(f'visualizing {lora_ckpt}')
        patch_pipe(
            pipe_copy,
            lora_ckpt,
            patch_text=False,
            patch_ti=True,
            patch_unet=True,
            filter_crossattn_str = 'cross+self'
        )
        # pipe.unet
        tune_lora_scale(pipe_copy.unet, 1)
        tune_lora_scale(pipe_copy.text_encoder, 1)

        pipe_copy.text_encoder.text_model.to(device, dtype=weight_dtype)
        images = []
        for prompt in prompts:
            prompt = [prompt]*bs
            img = pipe_copy(prompt = prompt, num_inference_steps=50, guidance_scale=6).images
            images.extend(img) 
        
        _dir, _name = lora_ckpt.rsplit('/', 1) 
        _dir = _dir.replace('checkpoints', 'figures')
        _name = _name.replace('.safetensors', '')
        outpath = os.path.join(_dir, _name)
        os.makedirs(_dir, exist_ok=True)
        print(f'saving to {outpath}')

        visualize_images(images, outpath=outpath, nrow=args.n_row, show=False, save=True)
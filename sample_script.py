#!/bin/bash
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']= "7"
prompt_file = 'sample_prompt.txt'
root_dir = ['logs/2023-03-28T04-05-49_dog/checkpoints']
for dir in root_dir:
    for file in os.listdir(dir):
        if file.endswith(".ckpt"):
            delta_ckpt = os.path.join(dir, file)
            os.system(f'python sample.py  --from-file "{prompt_file}"  --delta_ckpt "{delta_ckpt}" --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2')


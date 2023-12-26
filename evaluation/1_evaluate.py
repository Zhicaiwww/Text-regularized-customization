import os, sys, time, json, datetime, pdb
sys.path.append('/data/zhicai/code/Text-regularized-customization')
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from collections import defaultdict
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from lora_diffusion import patch_pipe, tune_lora_scale
from evaluation_pipe import  *

os.environ["DISABLE_TELEMETRY"] = 'YES'
os.environ["HTTP_PROXY"] = "http://localhost:8890"
os.environ["HTTPS_PROXY"] = "http://localhost:8890"

'''
CUDA_VISIBLE_DEVICES=3 python evaluation/1_evaluate.py \
    --superclass bear_plushie \
    --data_dir custom_datasets/data/bear_plushie \
    --enable_saving
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--superclass', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--filter_crossattn_str', type=str, default=None)
    parser.add_argument('--placeholder', type=str, default="<krk1>")
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--n_step', type=int, default=50)
    parser.add_argument('--enable_saving', action="store_true")
    parser.add_argument('--ti_step', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    device = torch.device("cuda")
    seed_everything(args.seed)

    superclass = args.superclass
    learnt_token = f"{args.placeholder} {superclass}" if args.lora_ckpt is not None else superclass

    if args.lora_ckpt is not None:
        if args.ti_step is None:
            lora_ckpt = os.path.join(args.lora_ckpt, "lora_weight.safetensors")
            save_path = os.path.join(args.lora_ckpt, "eval", f"samples_{now}") if args.enable_saving else None
        else:
            lora_ckpt = os.path.join(args.lora_ckpt, f"lora_weight_s{args.ti_step}.safetensors")
            save_path = os.path.join(args.lora_ckpt, "eval", f"samples_{now}_{args.ti_step}") if args.enable_saving else None
    else:
        save_path = os.path.join("samples", superclass, f"samples_{now}") if args.enable_saving else None

    time.sleep(0.01)
    pipe = StableDiffusionPipeline.from_pretrained("models/stable-diffusion-v1-5", torch_dtype=torch.float16, revision='39593d5650112b4cc580433f6b0435385882d819', safety_checker=None, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    if args.lora_ckpt is not None:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, )
        patch_pipe(
            pipe,
            lora_ckpt,
            patch_text=False,
            patch_ti=True,
            patch_unet=True,
            filter_crossattn_str = args.filter_crossattn_str,
        )
        tune_lora_scale(pipe.unet, 1)

    logs = defaultdict()

    log_OD = evaluate_pipe1(
        pipe=pipe,
        save_path=save_path,
        superclass=superclass,
        target_dir=args.data_dir,
        learnt_token=learnt_token,
        batch_size=args.batch_size,
        n_test=args.n_test,
        n_step=args.n_step,
        device=device,
    )
    
    log_ID = evaluate_pipe2(
        pipe=pipe,
        save_path=save_path,
        superclass=superclass,
        target_dir=args.data_dir,
        learnt_token=learnt_token,
        batch_size=args.batch_size,
        n_test=args.n_test,
        n_step=args.n_step,
        device=device,
    )

    logs['TA_OD'] = log_OD["text_alignment_avg"]
    logs['IA_OD'] = log_OD["image_alignment_avg"]
    logs['IA_ID'] = log_ID["image_alignment_avg"]
    logs['KID_ID'] = f"{log_ID['KID_score']:.2f}"
    print(logs)

    json_path = os.path.join(save_path, 'evaluate_logs.json')
    with open(json_path, 'w') as f:
        json.dump(logs, f)
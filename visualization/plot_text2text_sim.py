
import copy
import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append('../')
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

from lora_diffusion import patch_pipe
from utils import LIVE_SUBJECT_PROMPT_LIST

os.environ["DISABLE_TELEMETRY"] = 'YES'
model_id = "runwayml/stable-diffusion-v1-5"
color_palette = ["#304F6D", "#899481", "#E07D54", "#FFE1A0", "#E2F3FD", "#E6E1DD"]

class MovAvg(object):
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.data_queue = []

    def update(self, data):
        if len(self.data_queue) == self.window_size:
            del self.data_queue[0]
        self.data_queue.append(data)
        return sum(self.data_queue)/len(self.data_queue)

def plot_with_fill(x, y, yerr, color, label):
    plt.plot(x, y, c=color, marker='s', label=label)
    plt.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

font = {'family': 'serif',
        'weight': 'normal',
        'size': 12,
        }


def compute_text2text_sim(
    pipe,
    target_dir,
    custom_tokens='<krk1> dog',
    class_token='dog',
    runs=None,
    device='cuda'):

    # ckpt_pattern = r'lora_weight_s\d{3,4}.safetensors'
    # captions = ["photo of a <krk1> dog swimming in a pool", "photo of a <krk1> dog ",  "photo of a dog swimming in a pool"]
    weight_dtype = torch.float32
    caption_lists = [
        [template.format(custom_tokens), 
        f"photo of a {custom_tokens}", 
        template.format(class_token)] 
        for template in LIVE_SUBJECT_PROMPT_LIST
    ]
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device, dtype = weight_dtype)


    ckpt_files = [file  for file in os.listdir(target_dir) if '_s' in file and 'lora_weight' in file]
    target_steps = [100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900]
    ckpt_files = list(filter(lambda x: int(re.findall(r'.*s(\d+).*',x)[0]) in target_steps, ckpt_files))
    sorted_ckpt_files = sorted(ckpt_files, key = lambda x: int(re.findall(r'.*s(\d+).*',x)[0]))
    c2c_sim_outer_list = []
    c2c_sim_iner_list = []

    for i, _file in enumerate(sorted_ckpt_files):
        if runs is not None and i != runs:
            continue
        pipe_copy = copy.deepcopy(pipe).to(device)
        lora_path = os.path.join(target_dir, _file)
        print(f"loading from: {lora_path}")
        patch_pipe( 
            pipe_copy,
            lora_path,
            patch_text=False,
            patch_ti=True,
            patch_unet=True,
            filter_crossattn_str = 'cross'
        )

        model.text_model = copy.deepcopy(pipe_copy.text_encoder.text_model.to(device, dtype=weight_dtype))
        # token_embedding = model.text_model.embeddings.token_embedding
        # token_embedding.weight.data[49408] = token_embedding.weight.data[42170] 
        c2c_sim_iner_list = []
        for captions in caption_lists:   
            input_ids = pipe_copy.tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(device)
            outputs = pipe_copy.text_encoder.text_model(input_ids)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[torch.arange(last_hidden_state.size(0), device=input_ids.device),
                                            [(row == 49407).nonzero().min() for row in input_ids]]
            text_embeds = model.text_projection(pooled_output)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            c2c_sim = torch.matmul(text_embeds,text_embeds.t())
            diag_idx = torch.arange(min(c2c_sim.size(0), c2c_sim.size(1)))
            c2c_sim = c2c_sim[0,1:]  
            c2c_sim = torch.softmax(c2c_sim, dim = -1)
            c2c_sim_iner_list.append(c2c_sim.detach().cpu().numpy()[0])
        c2c_sim_outer_list.append(np.array(c2c_sim_iner_list))


            
    del pipe_copy
    data = np.stack(c2c_sim_outer_list)
    return data

def plot(pipe, target_dir, target_dir_mask):

    start_data = compute_text2text_sim(pipe, target_dir, custom_tokens='ktn dog', class_token='dog', runs = 0)
    start_data_mask = compute_text2text_sim(pipe, target_dir_mask, custom_tokens='ktn dog', class_token='dog', runs = 0)
    
    data = compute_text2text_sim(pipe, target_dir, custom_tokens='<krk1> dog', class_token='dog')
    data_mask = compute_text2text_sim(pipe, target_dir_mask, custom_tokens='<krk1> dog', class_token='dog')

    # data_1 = np.concatenate([start_data_1, data_1], axis=0)
    data = np.concatenate([start_data, data], axis=0)
    data_mask = np.concatenate([start_data_mask, data_mask], axis=0)


    # '[V] class'
    mean_values = np.mean(data, axis=1)
    smooth_fn = MovAvg(2)
    smoothed_mean_values = [smooth_fn.update(value) for value in mean_values]
    std_values = np.std(data, axis=1)
    x_labels = list(range(0, 100 * len(data),100))

    ax = plt.errorbar(x=x_labels, y=smoothed_mean_values, yerr=0.3 * std_values, c='black',capsize=0.4,marker='s',
            ecolor='#79AC78', mec='#79AC78', mfc='#79AC78', ms=3, mew=4, alpha = 0.7, label = 'Vanilla TI')

    # '[V] class' (masked)
    mean_values = np.mean(data_mask, axis=1)
    smooth_fn = MovAvg(2)
    smoothed_mean_values = [smooth_fn.update(value) for value in mean_values]
    std_values = np.std(data_mask, axis=1)
    x_labels = list(range(0, 100 * len(data_mask),100))

    ax2 = plt.errorbar(x=x_labels, y=smoothed_mean_values, yerr=0.3 * std_values, c='gray',capsize=0.4,marker='s',
            ecolor='#e76f51', mec='#e76f51', mfc='#e76f51', ms=3, mew=4, alpha = 0.7, label='MaskTI')
    # plt.title()
    plt.xticks(list(range(0, 100 * len(data) + 100,200)))
    plt.xlabel('TI Steps',fontdict=font)
    plt.ylabel('Prompts Similarity',fontdict=font)
    plt.grid(axis='y')
    plt.legend()
    plt.savefig('text2text_sim.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    target_dir = '/data/zhicai/code/Text-regularized-customization/outputs/checkpoints/dog/TiReg/TiReg_norm0.37_weight_0_rank_10_seed_0'
    target_dir_mask = '/data/zhicai/code/Text-regularized-customization/outputs/checkpoints/dog/TiReg/TiReg_norm0.37_MaskPlaceholder_weight_0_rank_10_seed_0'

    device = torch.device("cuda:1")
    time.sleep(0.01)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                                   torch_dtype=torch.float16,
                                                   local_ckpt_files_only=True,
                                                   revision='39593d5650112b4cc580433f6b0435385882d819',
                                                   local_files_only=True).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    plot(pipe, target_dir, target_dir_mask)

import copy
import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionPipeline
from PIL import Image

from lora_diffusion import patch_pipe

sys.path.append('../')

os.environ["DISABLE_TELEMETRY"] = 'YES'
model_id = "/data/zhicai/code/Text-regularized-customization/models/stable-diffusion-v1-5"
device = torch.device("cuda:3")
time.sleep(0.01)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,revision='39593d5650112b4cc580433f6b0435385882d819',local_files_only=True).to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# from custom_datasets import LIVE_SUBJECT_PROMPT_LIST

target_dir = '/data/zhicai/code/Text-regularized-customization/outputs/dog/TiReg/TiReg_InitToken_weight_0_rank_10_seed_0'
ckpt_pattern = r'lora_weight_s\d{3,4}.safetensors'
root_img_path = '../custom_datasets/data/dog'
captions = ["photo of a <krk1> dog swimming in a pool",
            "photo of a <krk1> dog ",  "photo of a dog swimming in a pool"]
weight_dtype = torch.float32

# model = CLIPModel.from_pretrained(
#     "models/clip-vit-large-patch14").to(device, dtype=weight_dtype)
# processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14")

images = [Image.open(os.path.join(root_img_path, img_path))
          for img_path in os.listdir(root_img_path)]


group = [ckpt_file for ckpt_file in os.listdir(
    target_dir) if '_s' in ckpt_file and 'lora_weight' in ckpt_file]
print(group)
sorted_group = sorted(group, key=lambda x: int(
    re.findall(r'.*s(\d+).*', x)[0]))[:20]

all_c2c_sim_list = []
all_c2i_sim_list = []
c2i_sim_list = []
c2c_sim_list = []
for _file in sorted_group:
    pipe_copy = copy.deepcopy(pipe).to(device)
    lora_path = os.path.join(target_dir, _file)
    print(f"loading from: {lora_path}")
    patch_pipe(
        pipe_copy,
        lora_path,
        patch_text=False,
        patch_ti=True,
        patch_unet=True,
        filter_crossattn_str='cross'
    )


all_c2c_sim_list.append(c2c_sim_list)
all_c2i_sim_list.append(c2i_sim_list)

c2c_sim_outer_list = []
c2c_sim_iner_list = []

for _file in sorted_group:
    pipe_copy = copy.deepcopy(pipe).to(device)
    lora_path = os.path.join(target_dir, _file)
    print(f"loading from: {lora_path}")
    patch_pipe(
        pipe_copy,
        lora_path,
        patch_text=False,
        patch_ti=True,
        patch_unet=True,
        filter_crossattn_str='cross'
    )

    model.text_model = copy.deepcopy(
        pipe_copy.text_encoder.text_model.to(device, dtype=weight_dtype))
    # token_embedding = model.text_model.embeddings.token_embedding
    # token_embedding.weight.data[49408] = token_embedding.weight.data[42170]
    c2c_sim_iner_list = []
    for captions in caption_lists:
        input_ids = pipe_copy.tokenizer(
            captions, return_tensors="pt", padding=True).input_ids.to(device)
        outputs = pipe_copy.text_encoder.text_model(input_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.size(0), device=input_ids.device),
                                          [(row == 49407).nonzero().min() for row in input_ids]]
        text_embeds = model.text_projection(pooled_output)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        c2c_sim = torch.matmul(text_embeds, text_embeds.t())
        diag_idx = torch.arange(min(c2c_sim.size(0), c2c_sim.size(1)))
        c2c_sim = c2c_sim[0, 1:]
        c2c_sim = torch.softmax(c2c_sim, dim=-1)
        c2c_sim_iner_list.append(c2c_sim.detach().cpu().numpy()[0])
    c2c_sim_outer_list.append(np.array(c2c_sim_iner_list))
    



class MovAvg(object):
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.data_queue = []

    def update(self, data):
        if len(self.data_queue) == self.window_size:
            del self.data_queue[0]
        self.data_queue.append(data)
        return sum(self.data_queue)/len(self.data_queue)


font = {'family': 'serif',
        'weight': 'normal',
        'size': 12,
        }

data = np.stack(c2c_sim_outer_list)

# 计算每列数据的均值和标准差
mean_values = np.mean(data, axis=1)
smooth_fn = MovAvg(2)
smoothed_mean_values = [smooth_fn.update(value) for value in mean_values]
std_values = np.std(data, axis=1)

x_labels = list(range(0, 100 * len(data), 100))

ax = plt.errorbar(x=x_labels, y=smoothed_mean_values, yerr=0.2 * std_values, c='black', capsize=0.4, marker='s',
                  ecolor='#79AC78', mec='#79AC78', mfc='#79AC78', ms=3, mew=4, alpha=0.7)

# plt.title()
plt.xticks(list(range(0, 100 * len(data) + 100, 200)))
plt.xlabel('steps', fontdict=font)
plt.ylabel('similarity', fontdict=font)
plt.grid(axis='x')
plt.show()
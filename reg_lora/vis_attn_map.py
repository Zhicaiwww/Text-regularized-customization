from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from lora_diffusion import LoraInjectedConv2d, LoraInjectedLinear, patch_pipe, tune_lora_scale, parse_safeloras
from lora_diffusion.lora import _find_modules, UNET_CROSSATTN_TARGET_REPLACE, DEFAULT_TARGET_REPLACE
from reg_lora.visual import visualize_images
from einops import rearrange
from torch import einsum
from diffusers.models.attention import CrossAttention 
import matplotlib.pyplot as plt
import copy

activation = {}
hooks = []

def get_attn_softmax(name):
    def hook(model, input, kwargs, output):
        if kwargs is not None and 'encoder_hidden_states' in kwargs.keys() and kwargs['encoder_hidden_states'] is not None:
            with torch.no_grad():
                x= input[0]
                h = model.heads
                q = model.to_q(x)
                context =  kwargs['encoder_hidden_states']
                k = model.to_k(context)
                v = model.to_v(context)   
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
                sim = einsum('b i d, b j d -> b i j', q, k) * model.scale
                attn = sim.softmax(dim=-1)
                if name not in activation:
                    activation[name] = [attn]
                else:
                    activation[name].append(attn)
        else:
            pass
    return hook
def add_attn_vis_hook(model, hook_name: str = None):
    hooks = []
    for name, module in model.named_modules():
        if  hook_name is not None:
            if hook_name == name:
                print("added hook to", name)
                hooks.append(module.register_forward_hook(get_attn_softmax(name),with_kwargs=True))
        elif type(module) ==  CrossAttention:
            if 'attn2' in name:
                print("added hook to", name)
                hooks.append(module.register_forward_hook(get_attn_softmax(name),with_kwargs=True))





# os.environ["DISABLE_TELEMETRY"] = 'YES'
if __name__ == '__main__':
    # python reg_lora/vis_attn_map.py --lora_path outputs/checkpoints/output_dog_Ti-clip_norm/lora_weight_e49_s8000.safetensors --prompt "a <krk1> dog in grand canyon" --gpu 0 --name up_blocks.2.attentions.1.transformer_blocks.0.attn2 --save_tag 'decay'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--name', type=str, default='up_blocks.2.attentions.1.transformer_blocks.0.attn2')
    parser.add_argument('--save_tag', type=str, default='')

    args = parser.parse_args()
    torch.manual_seed(0)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    model_id = "models/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,local_files_only=True,revision='39593d5650112b4cc580433f6b0435385882d819').to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe_copy = copy.deepcopy(pipe)
    patch_pipe(
        pipe_copy,
        args.lora_path,
        patch_text=False,
        patch_ti=True,
        patch_unet=True,
        filter_crossattn_str = 'cross+self'
    )
    name = args.name
    add_attn_vis_hook(pipe_copy.unet, name)

    images = pipe_copy([args.prompt], num_inference_steps=50, guidance_scale=6).images
    del pipe_copy
    basename = '_'.join(args.lora_path.split("/")[-2:]).split('.')[0]
    prompt_name = args.prompt.replace(' ','_')
    path = f'figures/attn_map/{prompt_name}_{basename}_{args.save_tag}'
    visualize_images(images,  nrow=1, outpath = path+'_ori_img', show=False, save=True,)


    attn_map = activation[name]
    for i in range(len(attn_map)):
        #  attn_map is of shape [8+8 , 4096, 77]
        if i == 40:
            fig_shape = int(math.sqrt(attn_map[i].size(1)))
            # reshape to (shape,shape) and sum over heads
            vis_map = attn_map[i].reshape(attn_map[i].size(0),fig_shape,fig_shape,77)
            uncond_attn_map , cond_attn_map = torch.chunk(vis_map, 2, dim=0)
            
            # mean over head [h, w, 77]
            cond_attn_map = cond_attn_map.mean(0)
            uncond_attn_map = uncond_attn_map.mean(0)
            # vis_map_token = vis_map[:,:,:,2].unsqueeze(1)
            # grid = make_grid(vis_map_token, nrow=4)
            # # to image
            # grid = 2550. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            # img = Image.fromarray(grid.astype(np.uint8))
            fig, ax = plt.subplots(1, 10, figsize=(20, 2))
            for j in range(10):
                map = cond_attn_map[:,:,j].unsqueeze(-1).cpu().numpy()
                ax[j].imshow(map) 
                # no axis for subplot
                ax[j].axis('off')
            plt.savefig(f'{path}_attn_map.jpg')
            # plt.imshow(vis_map[:,:,2].cpu().numpy())
                # plt.close(fig)
            print("saved attn map at ", f"{path}_attn_map.jpg")
            break
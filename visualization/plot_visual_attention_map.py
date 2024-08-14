import os
import math
import torch
import sys
import time
import copy

sys.path.append("../")
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
from torch import einsum
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)
from diffusers.models.attention import Attention
from lora_diffusion import patch_pipe


CKPT_MAPPING = {
    "v": "/data/zhicai/code/Text-regularized-customization/outputs/checkpoints/dog/TiReg/TiReg_norm0.37_weight_0_Init_rank_10_seed_0/lora_weight_s1000.safetensors",
    "vclass": "/data/zhicai/code/Text-regularized-customization/outputs/checkpoints/dog/TiReg/TiReg_norm0.37_weight_0_rank_10_seed_0/lora_weight_s1000.safetensors",
    "vclassMask": "/data/zhicai/code/Text-regularized-customization/outputs/checkpoints/dog/TiReg/TiReg_norm0.37_MaskPlaceholder_weight_0_rank_10_seed_0/lora_weight_s1000.safetensors",
}


def visualize_attn_map(
    pipe,
    lora_path,
    filter_crossattn_str: str = "cross",
    prompt: str = "a <krk1> dog swimming in a pool",
    hook_target_name: str = "up_blocks.2.attentions.1.transformer_blocks.0.attn2",
):
    print(f"loading from {lora_path}")

    activation = {}
    hooks = []

    pipe_copy = copy.deepcopy(pipe)
    model = pipe_copy.unet
    patch_pipe(
        pipe_copy,
        lora_path,
        patch_text=False,
        patch_ti=True,
        patch_unet=True,
        filter_crossattn_str=filter_crossattn_str,
    )

    def get_attn_softmax(name):
        def hook(model, input, kwargs, output):
            if (
                kwargs is not None
                and "encoder_hidden_states" in kwargs.keys()
                and kwargs["encoder_hidden_states"] is not None
            ):
                with torch.no_grad():
                    x = input[0]
                    h = model.heads
                    q = model.to_q(x)
                    context = kwargs["encoder_hidden_states"]
                    k = model.to_k(context)
                    v = model.to_v(context)
                    q, k, v = map(
                        lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
                    )
                    sim = einsum("b i d, b j d -> b i j", q, k) * model.scale
                    attn = sim.softmax(dim=-1)
                    if name not in activation:
                        activation[name] = [attn]
                    else:
                        activation[name].append(attn)
            else:
                pass

        return hook

    for name, module in model.named_modules():
        if hook_target_name is not None:
            if hook_target_name == name:
                print("added hook to", name)
                hooks.append(
                    module.register_forward_hook(
                        get_attn_softmax(name), with_kwargs=True
                    )
                )
        elif type(module) == Attention:
            if "attn2" in name:
                print("added hook to", name)
                hooks.append(
                    module.register_forward_hook(
                        get_attn_softmax(name), with_kwargs=True
                    )
                )

    image = pipe_copy([prompt], num_inference_steps=50, guidance_scale=6).images[0]
    attn_map = activation[hook_target_name]
    del pipe_copy
    return image, attn_map


if __name__ == "__main__":

    os.environ["DISABLE_TELEMETRY"] = "YES"
    model_id = "runwayml/stable-diffusion-v1-5"
    device = torch.device("cuda:1")
    torch.manual_seed(0)
    time.sleep(0.01)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_ckpt_files_only=True,
        revision="39593d5650112b4cc580433f6b0435385882d819",
        local_files_only=True,
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    save_path = "figures/vis_attn_map/"
    prompt = "a <krk1> dog swimming in a pool"
    ti_type = "vclassMask"

    image, attn_map = visualize_attn_map(
        pipe, lora_path=CKPT_MAPPING[ti_type], prompt=prompt
    )

    for time_step in range(len(attn_map)):
        #  attn_map is of shape [8+8 , 4096, 77]
        if time_step == 40:
            fig_shape = int(math.sqrt(attn_map[time_step].size(1)))
            # reshape to (shape,shape) and sum over heads
            vis_map = attn_map[time_step].reshape(
                attn_map[time_step].size(0), fig_shape, fig_shape, 77
            )
            uncond_attn_map, cond_attn_map = torch.chunk(vis_map, 2, dim=0)
            # mean over head [h, w, 77]
            cond_attn_map = cond_attn_map.mean(0)
            uncond_attn_map = uncond_attn_map.mean(0)
            fig, ax = plt.subplots(1, 10, figsize=(20, 2))
            for j in range(10):
                map = cond_attn_map[:, :, j].unsqueeze(-1).cpu().numpy()
                ax[j].imshow(map)
                ax[j].axis("off")
            save_name = "_".join(
                [f"{ti_type}", prompt.replace(" ", "_"), f"{time_step}"]
            )
            plt.savefig(f"{save_path}/{save_name}.pdf")
            image.save(f"{save_path}/{save_name}_image.pdf")
            print("saved attn map at ", f"{save_path}/{save_name}.pdf")
            break

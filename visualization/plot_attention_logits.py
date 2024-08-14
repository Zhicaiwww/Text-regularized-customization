import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionPipeline

sys.path.append("../")
import datetime
import time

from lora_diffusion import patch_pipe

color_palette = ["#304F6D", "#899481", "#E07D54", "#FFE1A0", "#E2F3FD", "#E6E1DD"]

ROOT_DIR = "figures"


def plot_attention_logits(
    lora_path, prompt, prefix, vis_token_id, vis_token_len, device
):
    pipe_copy = copy.deepcopy(pipe).to(device)
    patch_pipe(
        pipe_copy,
        lora_path,
        patch_text=False,
        patch_ti=True,
        patch_unet=True,
        filter_crossattn_str="cross",
    )

    text_inputs = pipe_copy.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    device = pipe_copy.text_encoder.device
    text_input_ids = text_inputs.input_ids.to(device)
    # global_eot_ids = [(row == 49407).nonzero().min() for row in text_input_ids]
    bs = len(prompt)
    outputs = pipe_copy.text_encoder(
        text_input_ids.to(device),
        attention_mask=None,
        output_attentions=True,
    )
    attentions = []

    for attention in outputs.attentions:
        attentions.append(
            attention.mean(dim=1)[
                torch.arange(bs, device=device), vis_token_id, :vis_token_len
            ]
            .detach()
            .cpu()
            .numpy()
        )
    attentions = np.stack(attentions, axis=1)

    for idx, data in enumerate(attentions):
        data =data.round(2)
        xticklabels = ["<sot>"] + prompt[idx].split(" ") + ["<eot>"]
        # padding if not enough
        if len(xticklabels) < len(data):
            xticklabels += ["<eot>"] * (12 - len(xticklabels))
        # cut if too much
        if len(xticklabels) > len(data):
            xticklabels = xticklabels[: len(data)]

        yticklabels = np.arange(12) + 1
        plt.figure()
        ax = sns.heatmap(
            data,
            annot=True,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            linewidths=0.5,
            cmap="YlGnBu",
        )
        ax.set_xlabel("Prompt")
        ax.set_ylabel("Text-transformer Layer")
        # timetag
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d-%H-%M")
        plt.savefig(
            "{}/attention_heatmap_{}_{}_{}.png".format(
                ROOT_DIR, prefix, idx, formatted_time
            )
        )
        plt.close()


if __name__ == "__main__":
    os.environ["DISABLE_TELEMETRY"] = "YES"
    model_id = "runwayml/stable-diffusion-v1-5"
    device = torch.device("cuda:1")
    time.sleep(0.01)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_ckpt_files_only=True,
        revision="39593d5650112b4cc580433f6b0435385882d819",
        local_files_only=True,
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    prefix = "V"
    mapping = {
        "Vmask": "/data/zhicai/code/Text-regularized-customization/outputs/checkpoints/dog/TiReg/TiReg_norm0.37_MaskPlaceholder_weight_0_rank_10_seed_0/lora_weight_s2000.safetensors",
        "V": "/data/zhicai/code/Text-regularized-customization/outputs/checkpoints/dog/TiReg/TiReg_norm0.37_weight_0_Init_rank_10_seed_0/lora_weight_s2000.safetensors",
        "VC": "/data/zhicai/code/Text-regularized-customization/outputs/checkpoints/dog/TiReg/TiReg_norm0.37_weight_0_rank_10_seed_0/lora_weight_s2000.safetensors",
    }
    lora_path = mapping[prefix]
    prompt = [
        "a photo of a ktn dog swimming in a pool",
        "a photo of a <krk1> dog swimming in a pool",
    ]
    vis_token_id = 11
    vis_token_len = 12
    plot_attention_logits(
        lora_path, prompt, prefix, vis_token_id, vis_token_len, device
    )

import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange

# import rearrange
from PIL import Image
from torchvision.utils import make_grid

from lora_diffusion import patch_pipe, tune_lora_scale
from lora_diffusion.lora import (
    DEFAULT_TARGET_REPLACE,
    LoraInjectedLinear,
    _find_modules,
)


def visual_unet_scales(
    pipe,
    prompt="a <krk> dog in grand canyon",
    type: str = "cross",
    scales=None,
    seed=0,
    batch_size=1,
):
    if scales is None:
        scales = [1, 0.5, 0.1, 0]
    images = []
    prompts = [prompt] * batch_size
    torch.manual_seed(seed)
    if type == "cross":
        # in_feature = 768
        print("visualize tuned lora scale in cross attention")
    elif type == "self":
        # in_feature = 320
        print("visualize tuned lora scale in self attention")
    elif type == "all":
        print("visualize tuned lora scale in all module")
    else:
        raise ValueError("type must be cross or self")
    tune_lora_scale(pipe.unet, 1)
    tune_lora_scale(pipe.text_encoder, 1)
    for scale in scales:
        cnt = 0
        for target, name, module in _find_modules(
            pipe.unet, DEFAULT_TARGET_REPLACE, search_class=[LoraInjectedLinear]
        ):
            # print(name)
            if type == "all":
                module.scale = scale
            elif type == "cross+self":
                if name == "to_k" or name == "to_v":
                    module.scale = scale
            elif type == "cross":
                if (
                    name == "to_k" or name == "to_v"
                ) and module.linear.in_features == 768:
                    module.scale = scale
                else:
                    module.scale = 0
            elif type == "self":
                if (
                    name == "to_k" or name == "to_v"
                ) and module.linear.in_features != 768:
                    module.scale = scale
                else:
                    module.scale = 0
            cnt += 1

        image = pipe(prompts, num_inference_steps=50, guidance_scale=6).images
        images.extend(image)
    print(f"finished tuned lora scales in unet {type} attention within {scales}")
    return images


def visualize_images(
    images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
    nrow: int = 4,
    show=False,
    save=True,
    outpath=None,
):

    if isinstance(images[0], Image.Image):
        transform = transforms.ToTensor()
        images_ts = torch.stack([transform(image) for image in images])
    elif isinstance(images[0], torch.Tensor):
        images_ts = torch.stack(images)
    elif isinstance(images[0], np.ndarray):
        images_ts = torch.stack([torch.from_numpy(image) for image in images])
    # save images to a grid
    grid = make_grid(images_ts, nrow=nrow, normalize=True, scale_each=True)
    # set plt figure size to (4,16)

    if show:
        plt.figure(
            figsize=(4 * nrow, 4 * (len(images) // nrow + (len(images) % nrow > 0)))
        )
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.show()
        # remove the axis
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
    img = Image.fromarray(grid.astype(np.uint8))
    if save:
        assert outpath is not None
        if os.path.dirname(outpath) and not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
        img.save(f"{outpath}")
    return img

# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

import argparse, pdb
import hashlib
import itertools
import math
import os
import random
import inspect
from pathlib import Path
from typing import Optional
import sys
sys.path.append('/data/liox/Text-regularized-customization')

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms
from functools import partial

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from collections import defaultdict


from lora_diffusion import *
# from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers
from custom_datasets.utils import IMAGENET_TEMPLATES_SMALL, IMAGENET_STYLE_TEMPLATES_SMALL, IMAGENET_TEMPLATES_TINY
from reg_lora.clip_ti_reg import  CLIPTiScoreCalculator, CLIPTiTextModel


os.environ['DISABLE_TELEMETRY'] = 'YES'
os.environ['HTTP_PROXY'] = 'http://localhost:8890'
os.environ['HTTPS_PROXY'] = 'http://localhost:8890'

INITIALIZER_TOKEN = 'ktn'
INITIALIZER_ID = 42170

class ConcatenateDataset(torch.utils.data.Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2

    def __len__(self):
        return math.ceil(len(self.ds1) * len(self.ds2) / math.gcd(len(self.ds1), len(self.ds2)))  # Return the maximum length

    def __getitem__(self, index):
        example_1 = self.ds1[index % len(self.ds1)] 
        example_2 = self.ds2[index % len(self.ds2)]  
        instance_images = torch.stack((example_1["instance_images"], example_2["instance_images"]), dim=0)   
        instance_prompt_ids = torch.stack((example_1["instance_prompt_ids"], example_2["instance_prompt_ids"]), dim=0)  

        return  dict(instance_images = instance_images, instance_prompt_ids = instance_prompt_ids)

class CLIPTiDataset(Dataset):
    def __init__(self,
                 custom_token, # e.g. <krk1> (dog)
                 class_token, # e.g. dog
                 tokenizer,
                 ti_reg_texts_file= None , # files that include the reg texts
                 init_placeholder_as_class = False,
                 repeat = 10,
                 learnable_property = 'object',
                ) -> None:
        super().__init__()

        if ti_reg_texts_file is not None and os.path.exists(ti_reg_texts_file):
            with open(ti_reg_texts_file) as f:
                self.templates = f.readlines()
            self.templates = [t.strip() for t in self.templates]
        else:
            self.templates = IMAGENET_STYLE_TEMPLATES_SMALL if learnable_property == "style" else IMAGENET_TEMPLATES_SMALL

        self.tokenizer = tokenizer
        self.num_templates = len(self.templates) 
        self.init_placeholder_as_class = init_placeholder_as_class

        self.class_token = INITIALIZER_TOKEN + ' ' + class_token  if not self.init_placeholder_as_class else class_token
        self.custom_token = custom_token
       
        self._length = self.num_templates
        self.repeat = repeat
            
    def __len__(self):
        return self.repeat * self._length

    def __getitem__(self, index) -> dict:
        
        example = {}
        template = self.templates[index % self.num_templates]
        reg_text = template.format(self.class_token)
        text = template.format(self.custom_token)
        print(text , " ------- ", reg_text)
        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        example["reg_instance_prompt_ids"] = self.tokenizer(
            reg_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example

class DreamBoothTiDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        learnable_property,
        custom_token,
        tokenizer,
        size=512,
        class_token=None,
        prompts_file=None,
        return_reg_text=False,
        center_crop=False,
        reg_prompts= None,
        repeat = 20,
    ):
        if return_reg_text:
            assert class_token is not None
        self.class_token = class_token
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.repeat = repeat
        self.return_reg_text = return_reg_text 

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        try:
            try:
                self.instance_images_path = sorted(list(Path(instance_data_root).iterdir()), key = lambda x: int(str(x).split('/')[-1].split('.', 1)[0]))
            except:
                self.instance_images_path = sorted(list(Path(instance_data_root).iterdir()), key = lambda x: int(str(x).split('/')[-1].split('-', 1)[0]))
        except:
            self.instance_images_path = sorted(list(Path(instance_data_root).iterdir()))
        
        self.num_instance_images = len(self.instance_images_path)
        self.custom_token = custom_token

        self.templates = (
            IMAGENET_STYLE_TEMPLATES_SMALL
            if learnable_property == "style"
            else IMAGENET_TEMPLATES_SMALL
        )

        self._length = self.num_instance_images

        self.reg_prompts = reg_prompts
        self.prompts = None
        if prompts_file is not None and os.path.exists(prompts_file):
            with open(prompts_file, 'r') as f:
                self.prompts = [line.strip() for line in f.readlines()]
            assert self.num_instance_images == len(self.prompts)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(0.2, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.repeat * self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.prompts is None:
            _template = random.choice(self.templates)
            text = _template.format(self.custom_token) 
        else:
            text = self.prompts[index % self.num_instance_images]

        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.return_reg_text:
            if self.reg_prompts is None:
                reg_text = random.choice(IMAGENET_TEMPLATES_TINY).format(self.class_token)
            else:
                reg_text = random.choice(self.reg_prompts)
            example["reg_prompt_ids"] = self.tokenizer(
                reg_text,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        return example


logger = get_logger(__name__)


def save_progress(text_encoder, placeholder_token, placeholder_token_id, accelerator, save_path):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[placeholder_token_id]
    )
    print("Current Learned Embeddings: ", learned_embeds[:4])
    print("saved to ", save_path)
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


def loss_step(batch, 
              unet, 
              vae,
              text_encoder,
              noise_scheduler,
              weight_dtype,
              cached_latents=False,
              with_prior_preservation=False,
              prior_loss_weight: float = 1.0,
              return_verbose=False):
    
    if not cached_latents:
        latents = vae.encode(
            batch["pixel_values"].to(dtype=weight_dtype)
        ).latent_dist.sample()
        latents = latents * 0.18215
    else:
        latents = batch["pixel_values"]

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (bsz,),
        device=latents.device,
    )
    timesteps = timesteps.long()


    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # if args.mask_identifier_causal_attention and not args.init_placeholder_as_class:
    #     encoder_hidden_states = get_masked_identifier_latents(text_encoder, batch["input_ids"], batch["identifier_indices"],class_len,dtype = latents.dtype)
    # # Get the text embedding for conditioning
    # else:
    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

    # Predict the noise residual
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        )

    if with_prior_preservation:
        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
        target, target_prior = torch.chunk(target, 2, dim=0)

        # Compute instance loss
        loss = (
            F.mse_loss(model_pred.float(), target.float(), reduction="none")
            .mean([1, 2, 3])
            .mean()
        )

        # Compute prior loss
        prior_loss = F.mse_loss(
            model_pred_prior.float(), target_prior.float(), reduction="mean"
        )

        # Add the prior loss to the instance loss.
        loss = loss + prior_loss_weight * prior_loss
    else:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    if not return_verbose:
        return loss
    else:
        return loss, timesteps,


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--stochastic_attribute",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance",
    )

    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be \
            sampled with prompts_file."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pt", "safe", "both"],
        default="both",
        help="The output format of the model predicitions and checkpoints.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Whether to apply color jitter to images",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=1e-5,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="object",
        help="Conecpt to learn : style or object?",
    )
    parser.add_argument(
        "--class_tokens",
        type=str,
        default=None,
        help="A token to use as initializer word.",
    )

    parser.add_argument(
        "--just_ti",
        action="store_true",
        help="Debug to see just ti",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Should images be resized to --resolution before training?",
    )
    parser.add_argument(
        "--use_xformers", action="store_true", help="Whether or not to use xformers"
    )
    parser.add_argument(
        "--cached_latents", action="store_true", help="cached_latents"
    )

    parser.add_argument("--filter_crossattn_str",type = str, default='full', help='full | self | cross | cross+self')

    # Hyper for Self-NormReg and Cross-TextReg 
    parser.add_argument("--enable_norm_reg", action="store_true", help="enable_norm_reg")
    parser.add_argument("--enable_text_reg", action="store_true", help="enable_text_reg")
    parser.add_argument("--norm_reg_alpha_weight", type=float, default=None, help="norm_reg_alpha_weight")
    parser.add_argument("--text_reg_alpha_weight", type=float, default=0.01, help="text_reg_alpha_weight")
    parser.add_argument("--text_reg_beta_weight", type=float, default=0.01, help="text_reg_beta_weight")
    parser.add_argument("--reg_prompts", type = str, default=None, nargs='*', help="if set as None, default templates will be used")

    # Hyper for Textual inversion 
    parser.add_argument("--resume_ti_embedding_path", type =str, default=None, help="resume_ti_embedding_path")
    parser.add_argument("--ti_reg_type", type = str, default=None, help="text or image or text+image")
    parser.add_argument("--ti_text_reg_alpha_weight", type=float, default=0.01, help="ti_text_reg_alpha_weight")
    parser.add_argument("--ti_reg_texts_file", type = str, default=None, help="ti_reg_texts_file")
    parser.add_argument("--ti_train_step", type=int, default=500, help="Save checkpoint every X updates steps.",) 
    parser.add_argument("--learning_rate_ti", type=float, default=5e-4, help="Initial learning rate for embedding of textual inversion (after the potential warmup period) to use.",)

    # Other important Hypers
    parser.add_argument("--init_placeholder_as_class", action= "store_true", help="Whether to use the initializer token as a class token.",) 
    parser.add_argument("--mask_identifier_causal_attention", action="store_true", help="cached_latents")
    parser.add_argument("--mask_identifier_ratio", type=float, default=0.5, help="text_reg_beta_weight")

    parser.add_argument("--enable_ewc_reg", action="store_true", help="enable_ewc_reg")
    parser.add_argument("--log_evaluation", action="store_true", help="log_evaluation")
    parser.add_argument("--log_evaluation_step", type=int, default=50, help="log_evaluation_step")
    parser.add_argument("--ewc_epochs", type=int,default=1, help="ewc_epochs")
    parser.add_argument("--local_files_only", action="store_true", help="local_files_only")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")

    if args.enable_norm_reg or args.enable_text_reg or args.enable_ewc_reg:
        assert args.filter_crossattn_str in ['self', 'cross', 'cross+self']
        if args.enable_text_reg and args.filter_crossattn_str == 'cross':
            print(f"Uses text regularization: text_reg_alpha_weight={args.text_reg_alpha_weight}")
        elif args.enable_text_reg and args.filter_crossattn_str == 'cross+self':
            print(f"Uses both text regularization and norm regularization: \
                  norm_reg_alpha_weight={args.norm_reg_alpha_weight}, text_reg_alpha_weight={args.text_reg_alpha_weight}")
        else:
            print(f"Uses norm regularization: norm_reg_alpha_weight={args.norm_reg_alpha_weight}")

    if not safetensors_available:
        if args.output_format == "both":
            print(
                "Safetensors is not available - changing output format to just output PyTorch files"
            )
            args.output_format = "pt"
        elif args.output_format == "safe":
            raise ValueError(
                "Safetensors is not available - either install it, or change output_format."
            )
    return args


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


def collate_fn(examples):
    input_ids = torch.stack([example["instance_prompt_ids"] for example in examples]) # bs, 1, 77
    pixel_values = torch.stack([example["instance_images"] for example in examples])

    if len(pixel_values.size()) == 5: # with class images
        pixel_values = torch.concat([pixel_values[:,0], pixel_values[:,1]], dim=0)
        input_ids = torch.concat([input_ids[:,0], input_ids[:,1]], dim=0)
    
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    if "reg_prompt_ids" in examples[0].keys():
        reg_ids = torch.cat([example["reg_prompt_ids"] for example in examples])
        batch["reg_ids"] = reg_ids 
    return batch


def clip_collate_fn(examples):
    input_ids = torch.stack([example["instance_prompt_ids"] for example in examples])
    reg_input_ids = torch.stack([example["reg_instance_prompt_ids"] for example in examples])
    batch = {
         "input_ids": input_ids,
         "reg_input_ids": reg_input_ids
    }
    # return list batch
    return batch


def main(args):
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
    placeholder_token = args.placeholder_token.split('+')[-1]
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token.split('+'))
    placeholder_token_id = tokenizer(placeholder_token, truncation=False, add_special_tokens=False)["input_ids"]
    class_token_ids = tokenizer(args.class_tokens, truncation=False, add_special_tokens=False)["input_ids"]

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTiTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        mask_identifier_causal_attention=args.mask_identifier_causal_attention,
        class_token_len=len(class_token_ids),
        placeholder_token_id=placeholder_token_id[0], # We only support on placeholder token for now
        local_files_only=args.local_files_only,
        mask_identifier_ratio=args.mask_identifier_ratio
    )

    if args.resume_ti_embedding_path is not None:
        from lora_diffusion.lora import apply_learned_embed_in_clip, parse_safeloras_embeds
        from safetensors.torch import safe_open

        print("Loading learned embeddings from: ", args.resume_ti_embedding_path)
        safeloras = safe_open(args.resume_ti_embedding_path, framework="pt", device="cpu")

        tok_dict = parse_safeloras_embeds(safeloras)
        apply_learned_embed_in_clip(
            tok_dict,
            text_encoder,
            tokenizer,
            token=args.placeholder_token.split('+')[:-1],  # the last placeholder needs to be learned
            idempotent=True,
        )
    else:
        tok_dict = None
        text_encoder.resize_token_embeddings(len(tokenizer))

    text_encoder.get_input_embeddings().weight.data[-1] = torch.normal(0, 0.014, torch.Size([768]))  # Initialize the last placeholder    
    initializer_norm = text_encoder.get_input_embeddings().weight.data[-1].norm()
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
        local_files_only=args.local_files_only,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    if args.resume_ti_embedding_path is not None:
        # Add loras.unet.prev to pre-trained SD
        safeloras = safe_open(args.resume_ti_embedding_path, framework="pt", device="cpu")
        (lora, ranks, target) = parse_safeloras(safeloras)["unet"]
        add_lora_params_to_model_by_lox(unet, lora, target, ranks, filter_crossattn_str=args.filter_crossattn_str)
        # Get loras.unet.prev.weights
        loras_unet_prev = [i.data for i in parse_safeloras(safeloras)['unet'][0]]
        lora_prev_matrix = []
        for idx in range(int(len(loras_unet_prev)/2)):
            lora_prev_matrix.append(torch.matmul(loras_unet_prev[2*idx].float(), loras_unet_prev[2*idx+1].float()).to("cuda"))
    else:
        loras_unet_prev = None

    vae.requires_grad_(False)
    unet.requires_grad_(False)

    if args.init_placeholder_as_class:
        custom_token = placeholder_token
    else: 
        custom_token = ' '.join([placeholder_token, args.class_tokens])  # '<krk1> dog'
        
    if args.ti_reg_type is not None:
        use_text_reg_with_ti = 'text' in args.ti_reg_type
        use_decay_reg_with_ti = 'decay' in args.ti_reg_type
    else:
        use_text_reg_with_ti = False
        use_decay_reg_with_ti = False

    if use_text_reg_with_ti : 

        clip_dataset = CLIPTiDataset(
            custom_token=custom_token,
            class_token = args.class_tokens,
            tokenizer = tokenizer,
            ti_reg_texts_file = args.ti_reg_texts_file,
            init_placeholder_as_class = args.init_placeholder_as_class,
            repeat = 10,
        )
        
        clip_dataloader = torch.utils.data.DataLoader(clip_dataset, collate_fn = clip_collate_fn, batch_size=2, shuffle=True)
        clip_data_iterator = iter(clip_dataloader)
        clip_data_iterator = itertools.cycle(clip_data_iterator)

    if args.filter_crossattn_str == 'full':
        # TODO
        raise ValueError("filter_crossattn_str cannot be full, not supported yet")
        target_module = UNET_DEFAULT_TARGET_REPLACE
    else:
        target_module = UNET_CROSSATTN_TARGET_REPLACE
    unet_lora_params, _ = inject_trainable_lora(unet,
                                                target_replace_module=target_module, 
                                                r=args.lora_rank,
                                                freeze_down_lora=args.enable_ewc_reg, # freeze down lora if ewc is enabled
                                                filter_crossattn_str = args.filter_crossattn_str
                                                )
    to_reg_params = filter_unet_to_norm_weights(unet, target_replace_module=target_module)

    for _up, _down in extract_lora_ups_down(unet):
        print(f"\n{'*' * 25}{'[Before training] UNet':^30}{'*' * 25}")
        print(f"Unet First Layer lora up ↑: {_up.weight.shape} \n{_up.weight.data}")
        print(f"Unet First Layer lora down ↓: {_down.weight.shape} \n{_down.weight.data} \n{'*' * 80}")
        break

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    text_encoder_lora_params, _ = inject_trainable_lora(
        text_encoder,
        target_replace_module=["CLIPAttention"],
        r=args.lora_rank,
        freeze_down_lora=args.enable_ewc_reg, 
    )
    for _up, _down in extract_lora_ups_down(
        text_encoder, target_replace_module=["CLIPAttention"]
    ):
        print(f"\n{'*' * 22}{'[Before training] text encoder':^36}{'*' * 22}")
        print(f"text encoder First Layer lora up ↑: {_up.weight.shape} \n{_up.weight.data}")
        print(f"text encoder First Layer lora down ↓: {_down.weight.shape} \n{_down.weight.data} \n{'*' * 80}")
        break

    # if args.use_xformers:
    #     set_use_memory_efficient_attention_xformers(unet, True)
    #     set_use_memory_efficient_attention_xformers(vae, True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:   
        use_sgd=False
        if use_sgd:
            optimizer_class = partial(torch.optim.SGD, momentum=0.9)
        else:
            optimizer_class = partial(torch.optim.AdamW,
                                betas=(args.adam_beta1, args.adam_beta2),
                                weight_decay=args.adam_weight_decay,
                                eps=args.adam_epsilon,)
    
    params_ti_to_optimize = [ 
        {"params": text_encoder.get_input_embeddings().parameters(),
             "lr": args.learning_rate_ti,
        },
    ]
    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params) if not args.just_ti else [], 
             "lr": args.learning_rate
        },
        {"params": itertools.chain(*text_encoder_lora_params) if not args.just_ti and args.train_text_encoder else [],
             "lr": args.learning_rate_text,
        },
    ]

    optimizer_ti = optimizer_class(
        params_ti_to_optimize,
    )
    optimizer = optimizer_class(
        params_to_optimize,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=args.local_files_only,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)

    ################################################ Prepare Dataset ################################################ 

    # region [Prepare Dataset]
    train_dataset = DreamBoothTiDataset(
        instance_data_root=args.instance_data_dir,
        custom_token=custom_token,
        learnable_property=args.learnable_property,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        return_reg_text=args.enable_text_reg,
        class_token=args.class_tokens,
        reg_prompts=args.reg_prompts,
    )

    if args.with_prior_preservation or args.enable_ewc_reg:
        class_dataset = DreamBoothTiDataset(
            instance_data_root=args.class_data_dir,
            custom_token=args.class_tokens,
            learnable_property=args.learnable_property,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
            return_reg_text=False,
            prompts_file=args.prompts_file,
            repeat=1,
        )

    if args.with_prior_preservation:
        train_dataset = ConcatenateDataset(train_dataset, class_dataset)

    # Might be too long for concatenated dataset
    if args.cached_latents:
        cached_latents_dataset = []

        for idx in tqdm(range(len(train_dataset))):
            batch = train_dataset[idx]
            # rint(batch)
            latents = vae.encode(
                batch["instance_images"].unsqueeze(0).to(dtype=vae.dtype).to(vae.device)
            ).latent_dist.sample()
            latents = latents * 0.18215
            batch["instance_images"] = latents.squeeze(0)
            cached_latents_dataset.append(batch)
        train_dataloader = torch.utils.data.DataLoader(
            cached_latents_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        accelerator.log("Using cached latent.")
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
    # endregion

    ############################################# Prepare Training Hyper ############################################ 

    # region [Prepare Training Hyper]
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler_ti = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_ti,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.ti_train_step * args.gradient_accumulation_steps,
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(args.max_train_steps - args.ti_train_step) * args.gradient_accumulation_steps,
    )

    (unet, text_encoder, train_dataloader) = accelerator.prepare(unet, text_encoder, train_dataloader)

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    if args.enable_ewc_reg and args.norm_reg_alpha_weight != 0:
        if accelerator.is_local_main_process:
            class_dataloader = torch.utils.data.DataLoader(
                class_dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            class_dataloader = accelerator.prepare(class_dataloader)
            ewc_step = 0
            for name, param in unet.named_parameters():
                param.requires_grad = True
            for _ in range(args.ewc_epochs):
                for _, batch in enumerate(tqdm(class_dataloader)):
                    unet.zero_grad()
                    loss = loss_step(batch=batch, 
                                    unet=unet,
                                    vae=vae,
                                    text_encoder=text_encoder,
                                    noise_scheduler=noise_scheduler,
                                    weight_dtype=weight_dtype,
                                    cached_latents=args.cached_latents,
                                    with_prior_preservation=False,
                                    prior_loss_weight=args.prior_loss_weight,
                                    )
                    accelerator.backward(loss)
                    ewc_step += 1
                    logs = defaultdict(list)
                    for idx, module in enumerate(to_reg_params["other_loras"]):
                        module.update_fisher()
                        logs[f"EWC-{idx}_lora_up_grad"] = module.lora_up.weight.grad.norm().detach().item()
                        # logs[f"EWC-{idx}_linear_grad"] = module.linear.weight.grad.norm().detach().item()
                    accelerator.log(logs, step=ewc_step)
        # scale fisher matrix
        global_scaler = torch.stack([module.fisher.mean() for module in to_reg_params["other_loras"]]).mean()
        ewc_statics = defaultdict(list)
        for _, module in enumerate(to_reg_params["other_loras"]):
            ewc_statics['fisher_list'].append(module.fisher)
            ewc_statics['lora_up_list'].append(module.lora_up.weight.data)
            ewc_statics['lora_down_list'].append(module.lora_down.weight.data)
            # local_scaler = module.fisher.mean()
            module.fisher = module.fisher / global_scaler 
        torch.save(ewc_statics, os.path.join(args.output_dir, "fisher.pt")) 
    else:
        pass
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps / args.gradient_accumulation_steps}")
    # Only show the progress bar once on each machine.

    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    # Let's make sure we don't update any embedding weights besides the newly added token
    index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool)
    for _placeholder_token_id in placeholder_token_id:
        index_no_updates *= torch.arange(len(tokenizer)) != _placeholder_token_id 
    index_updates = ~index_no_updates

    if args.cached_latents:
        del vae
        vae = None
        
    if args.log_evaluation:
        preped_clip = prepare_clip_model_sets()
    # endregion

    ################################################ Start Training ################################################# 

    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # region [Textual inversion training: freeze unet and text encoder during ti training]
            if global_step < args.ti_train_step:
                text_encoder.eval()
                unet.eval()
                loss = loss_step(
                    batch=batch, 
                    unet=unet,
                    vae=vae,
                    text_encoder=text_encoder,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    cached_latents=args.cached_latents,
                    with_prior_preservation=False,
                ) 
                if use_text_reg_with_ti:
                    clip_batch = next(clip_data_iterator)
                    _ti_reg_loss = []
                    input_conditions = text_encoder(clip_batch["input_ids"].to(text_encoder.device))[0]
                    reg_conditions = text_encoder(clip_batch["reg_input_ids"].to(text_encoder.device))[0]
                    for module in to_reg_params["cross_project_k_loras"]:
                        _ti_reg_loss.append(torch.norm(module.linear(input_conditions) - module.linear(reg_conditions), dim =[1,2], p = 2).mean())
                    if len(_ti_reg_loss):
                        ti_reg_loss = torch.mean(sum(_ti_reg_loss)/len(_ti_reg_loss))
                        loss += args.ti_text_reg_alpha_weight * ti_reg_loss

                    # print(f"sim_loss is {sim_loss.detach().item()}")
                    # if global_step > 10:
                    #     loss+= 0.01 * sim_loss

                accelerator.backward(loss)
                optimizer_ti.step()
                lr_scheduler_ti.step()
                progress_bar.update(1)
                optimizer_ti.zero_grad()
                global_step += 1
                ti_logs = {"ti_loss": loss.detach().item()}
                if use_text_reg_with_ti: 
                    ti_logs = {"ti_reg_loss": ti_reg_loss.detach().item()}

                ########################################### Norm Embedding ###########################################

                with torch.no_grad():
                    if use_decay_reg_with_ti:
                        # normalize embeddings
                        pre_norm = (
                            text_encoder.get_input_embeddings()
                            .weight[index_updates, :]
                            .norm(dim=-1, keepdim=True)
                        )
                        lambda_ = min(1.0, 100 * lr_scheduler_ti.get_last_lr()[0])
                        
                        text_encoder.get_input_embeddings().weight[
                            index_updates
                        ] = F.normalize(
                            text_encoder.get_input_embeddings().weight[
                                index_updates, :
                            ],
                            dim=-1,
                        ) * (
                            pre_norm + lambda_ * (initializer_norm - pre_norm)
                        )
                        # print(f"Pre Norm: {pre_norm}")

                    current_norm = (
                        text_encoder.get_input_embeddings()
                        .weight[index_updates, :]
                        .norm(dim=-1)
                    )

                    text_encoder.get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]
                    ti_logs['current_norm'] = current_norm.detach().item()
            
                accelerator.log(ti_logs, step=global_step)
                # endregion
                
            # region [Unet and text encoder training: begin learning with unet and text encoder]
            else:
                text_encoder.train()  
                unet.train() 
                loss, _ = loss_step(
                    batch=batch, 
                    unet=unet,
                    vae=vae,
                    text_encoder=text_encoder,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    cached_latents=args.cached_latents,
                    with_prior_preservation=args.with_prior_preservation,
                    prior_loss_weight=args.prior_loss_weight,
                    return_verbose=True,
                ) 

                if args.enable_norm_reg or args.enable_ewc_reg:
                    # TODO
                    _norm_reg_loss = []
                    for module in to_reg_params["other_loras"]:
                        _norm_reg_loss.append(module.get_reg_loss())
                    if len(_norm_reg_loss):
                        norm_reg_loss = torch.mean(sum(_norm_reg_loss)/len(_norm_reg_loss))
                        loss += args.norm_reg_alpha_weight * norm_reg_loss

                if args.enable_text_reg:
                    reg_conditions = text_encoder(batch["reg_ids"])[0]
                    input_conditions = text_encoder(batch["input_ids"])[0]

                    _text_reg_loss = []
                    for module in to_reg_params["cross_project_loras"]:
                        _text_reg_loss.append(module.get_reg_loss(reg_conditions))
                    if len(_text_reg_loss):
                        text_reg_loss = torch.mean(sum(_text_reg_loss)/len(_text_reg_loss))
                        loss += args.text_reg_alpha_weight * text_reg_loss

                    _text_reg_k_loss = []
                    for module in to_reg_params["cross_project_k_loras"]:
                        _text_reg_k_loss.append(module.get_reg_loss(input_conditions))
                    if len(_text_reg_loss):
                        text_reg_k_loss = torch.mean(sum(_text_reg_k_loss)/len(_text_reg_k_loss))
                        loss += args.text_reg_beta_weight * text_reg_k_loss

                if args.resume_ti_embedding_path is not None:
                    _c_lora_loss = []
                    for idx, module in enumerate(to_reg_params["lox_loras"]):
                        lora_now = torch.matmul(module.lora_up.weight, module.lora_down.weight)
                        _c_lora_loss.append(
                            torch.norm(torch.mul(lora_prev_matrix[idx], lora_now), p=2)
                        )
                    if len(_c_lora_loss):
                        c_lora_loss = torch.mean(sum(_c_lora_loss)/len(_c_lora_loss))
                        loss += 1e8 * c_lora_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                optimizer.zero_grad()
                global_step += 1
            # endregion

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.save_steps and global_step - last_save >= args.save_steps:
                    if accelerator.is_main_process:
                        # newer versions of accelerate allow the 'keep_fp32_wrapper' arg. without passing
                        # it, the models will be unwrapped, and when they are then used for further training,
                        # we will crash. pass this, but only to newer versions of accelerate. fixes
                        # https://github.com/huggingface/diffusers/issues/1566
                        # accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                        #     inspect.signature(
                        #         accelerator.unwrap_model
                        #     ).parameters.keys()
                        # )
                        # extra_args = (
                        #     {"keep_fp32_wrapper": True}
                        #     if accepts_keep_fp32_wrapper
                        #     else {}
                        # )
                        # unwarp_text_encoder = accelerator.unwrap_model(text_encoder, **extra_args)
                        # unwarp_unet = accelerator.unwrap_model(unet, **extra_args)
                        unwarp_text_encoder = text_encoder
                        unwarp_unet = unet

                        
                        if args.output_format == "pt" or args.output_format == "both":
                            filename_unet = (
                                f"{args.output_dir}/lora_weight_s{global_step}.pt"
                            )
                            filename_text_encoder = f"{args.output_dir}/lora_weight_s{global_step}.text_encoder.pt"
                            print(f"save weights {filename_unet}, {filename_text_encoder}")
                            save_lora_weight(unwarp_unet, filename_unet,target_replace_module = target_module)

                            save_lora_weight(
                                unwarp_text_encoder,
                                filename_text_encoder,
                                target_replace_module=["CLIPAttention"],
                            )
                            filename_ti = f"{args.output_dir}/lora_weight_s{global_step}.ti.pt"

                            save_progress(
                                unwarp_text_encoder,
                                placeholder_token,
                                placeholder_token_id,
                                accelerator,
                                filename_ti,
                            )

                        if args.output_format == "safe" or args.output_format == "both":
                            loras = {}
                            loras["unet"] = (unwarp_unet, target_module)
                            loras["text_encoder"] = (unwarp_text_encoder, {"CLIPAttention"})
                            embeds = {}
                            learned_embed = (
                                unwarp_text_encoder
                                .get_input_embeddings()
                                .weight[placeholder_token_id]
                            )

                            embeds[placeholder_token] = learned_embed.detach().cpu()
                            save_safeloras_with_embeds(
                                loras, embeds, args.output_dir + f"/lora_weight_s{global_step}.safetensors",
                                tok_dict=tok_dict, loras_unet_prev=loras_unet_prev
                            ) 

                        last_save = global_step

                    probe_prompts = ["a photo of a dog swimming in a pool","a photo of a <krk1> dog swimming in a pool"]
                    input_ids = tokenizer(probe_prompts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
                    global_eot_ids = [(row == 49407).nonzero().min() for row in input_ids]
                    attentions = text_encoder(input_ids.to(text_encoder.device), output_attentions = True).attentions
                    for attention in attentions:
                        print(attention.mean(dim=1)[torch.arange(input_ids.size(0), device = text_encoder.device), global_eot_ids, :12].detach().cpu().numpy().round(2))
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                # "placeholder_norm": current_norm.detach().item() if current_norm else 0.0,
            }
            
            if global_step > args.ti_train_step:
                if args.enable_text_reg:
                    logs["text_reg_loss"] = text_reg_loss.detach().item()
                    logs["text_reg_k_loss"] = text_reg_k_loss.detach().item()
                if args.enable_norm_reg or args.enable_ewc_reg:
                    logs["norm_reg_loss"] = norm_reg_loss.detach().item()

                for idx, module in enumerate(to_reg_params["other_loras"]):
                    logs[f"{idx}_lora_up_weight_norm"] = module.lora_up.weight.norm().detach().item()
                    logs[f"{idx}_lora_down_weight_norm"] = module.lora_down.weight.norm().detach().item()
                    # logs[f"{idx}_lora_up_weight_grad_norm"] =module.lora_up.grad.norm().detach().item()
                    if args.norm_reg_alpha_weight > 0 and args.enable_ewc_reg:
                        logs[f"{idx}_fisher"] = module.fisher.detach().mean().item()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if args.log_evaluation and global_step % args.log_evaluation_step == 1 and accelerator.is_main_process:
                with torch.no_grad():
                    pipe = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            scheduler=noise_scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                        )

                    test_image_path = args.instance_data_dir
                    images = []
                    for file in os.listdir(test_image_path):
                        if (
                            file.lower().endswith(".png")
                            or file.lower().endswith(".jpg")
                            or file.lower().endswith(".jpeg")
                        ):
                            images.append(
                                Image.open(os.path.join(test_image_path, file))
                            )
                    
                    accelerator.log(
                        evaluate_pipe(
                            pipe,
                            target_images=images,
                            class_token=args.class_tokens,
                            learnt_token=custom_token,
                            n_test=10,
                            n_step=50,
                            clip_model_sets=preped_clip,
                        ),
                        step=global_step
                    )

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
    accelerator.wait_for_everyone()

    # Create the pipeline using the trained modules and save it.
    if accelerator.is_main_process:
        # pipeline = StableDiffusionPipeline.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     unet=accelerator.unwrap_model(unet),
        #     text_encoder=accelerator.unwrap_model(text_encoder),
        #     revision=args.revision,
        #     local_files_only=args.local_files_only,
        # )

        print("\n\nLora TRAINING DONE!\n\n")

        if args.output_format == "pt" or args.output_format == "both":
            save_lora_weight(unet, args.output_dir + "/lora_weight.pt")

            save_lora_weight(
                text_encoder,
                args.output_dir + "/lora_weight.text_encoder.pt",
                target_replace_module=["CLIPAttention"],
            )

        if args.output_format == "safe" or args.output_format == "both":
            loras = {}
            loras["unet"] = (unet, target_module)
            loras["text_encoder"] = (text_encoder, {"CLIPAttention"})

            learned_embeds = (
                text_encoder
                .get_input_embeddings()
                .weight[placeholder_token_id]
            )

            embeds = {placeholder_token: learned_embeds.detach().cpu()}

            save_safeloras_with_embeds(
                loras, embeds, args.output_dir + "/lora_weight.safetensors",
                tok_dict=tok_dict, loras_unet_prev=loras_unet_prev
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

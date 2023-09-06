#!/bin/bash
#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="dataset/data/wooden_pot"
export CLASS_PRIOR_DIR="dataset/prior_real/samples_wooden_pot/wooden_pot" # prior preserving dataset dir
export CLASS_CAPTION_DIR="dataset/prior_real/samples_wooden_pot/caption.txt" # caption of prior preserving dataset dir, e.g.: 
# init_placeholder_as_class:  use the class token as the initializer, e.g., "wooden_pot" -> "<krk> wooden_pot"
# train_text_encoder: train the text encoder
# with_prior_preservation: use prior preserving dataset
# class_data_dir: CLASS_PRIOR_DIR
# class_prompt_or_file: CLASS_CAPTION_DIR
# filter_crossattn_str: self, cross, full,cross+self
# enbale_norm_reg 
# enable_text_reg
# norm_reg_loss_weight
# text_reg_loss_weight
# reg_prompts: photo of a wooden_pot
# scale_norm_reg: scale the norm reg loss accpording to the 1/SNR
# mask_identifier_causal_attention: mask the identifier in the causal attention

# accelerate launch
lora_rank=(1 10 100)
for _lora_rank in "${lora_rank[@]}"; do
  export lora_rank=$_lora_rank
  export OUTPUT_DIR="outputs/checkpoints/wooden_pot/ewc_reg/prior-loss_lora_rank_${lora_rank}"
  CUDA_VISIBLE_DEVICES=7 python -m pdb training_scripts/train_lora_w_ti.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --class_data_dir=$CLASS_PRIOR_DIR \
    --prompts_file=$CLASS_CAPTION_DIR \
    --resolution=512 \
    --train_batch_size=1 \
    --learning_rate=1e-4 \
    --learning_rate_text=1e-5 \
    --learning_rate_ti=5e-4 \
    --color_jitter \
    --lr_scheduler="constant" \
    --lr_warmup_steps=100 \
    --ti_train_step=1000 \
    --max_train_steps=2000 \
    --placeholder_token="<krk1>" \
    --learnable_property="object" \
    --class_tokens="wooden pot" \
    --save_steps=100 \
    --resize=True \
    --center_crop \
    --scale_lr \
    --lora_rank=$lora_rank \
    --gradient_accumulation_steps=4 \
    --output_format=safe \
    --mixed_precision=bf16 \
    --filter_crossattn_str=cross+self \
    --reg_prompts="photo of a wooden pot" \
    --with_prior_preservation \
    --text_reg_loss_weight=0.001 \
    --ti_reg_type="decay" \
    --prior_loss_weight=1 \
    --resume_ti_embedding_path="outputs/checkpoints/dog/ewc_reg_shareTI/baseline_lora_rank_1/lora_weight_s1100.safetensors"
done

  # --ti_reg_type="decay" \
  # --mask_identifier_causal_attention \
  # --reg_texts_file="dataset/data_reg/wooden_pot_reg.txt" \
  # --reg_images_root="dataset/prior_real/samples_wooden_pot/wooden_pot" \
  # --stochastic_attribute="game character,3d render,4k,highres" # these attributes will be randomly appended to the prompts
  
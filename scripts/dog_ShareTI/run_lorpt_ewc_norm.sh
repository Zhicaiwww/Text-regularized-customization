#!/bin/bash
#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="dataset/data/dog"
export CLASS_PRIOR_DIR="dataset/prior_gen/samples_dog/dog" # prior preserving dataset dir
export CLASS_CAPTION_DIR="dataset/prior_gen/samples_dog/caption.txt" # caption of prior preserving dataset dir, e.g.: 

# init_placeholder_as_class:  use the class token as the initializer, e.g., "dog" -> "<krk> dog"
# train_text_encoder: train the text encoder
# with_prior_preservation: use prior preserving dataset
# class_data_dir: CLASS_PRIOR_DIR
# class_prompt_or_file: CLASS_CAPTION_DIR
# filter_crossattn_str: self, cross, full,cross+self
# enbale_norm_reg 
# enable_text_reg
# norm_reg_loss_weight
# text_reg_loss_weight
# reg_prompts: photo of a dog
# scale_norm_reg: scale the norm reg loss accpording to the 1/SNR
# mask_identifier_causal_attention: mask the identifier in the causal attention

# accelerate launch
norm_reg_loss_weights=(0.01 1 100)
lora_rank=(10)
seeds=(0)
for seed in "${seeds[@]}"; do 
  for _norm_reg_loss_weight in "${norm_reg_loss_weights[@]}"; do
    for _lora_rank in "${lora_rank[@]}"; do
      export seed=$seed
      export lora_rank=$_lora_rank
      export norm_reg_loss_weight=$_norm_reg_loss_weight
      export OUTPUT_DIR="outputs/checkpoints/dog/ewc_reg_shareTI/ewc_gen_weight_${norm_reg_loss_weight}_rank_${lora_rank}_seed_${seed}"
      CUDA_VISIBLE_DEVICES=7 python training_scripts/train_lora_w_ti.py \
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
        --class_tokens="dog" \
        --save_steps=100 \
        --resize=True \
        --center_crop \
        --scale_lr \
        --lora_rank=$lora_rank \
        --gradient_accumulation_steps=4 \
        --output_format=safe \
        --mixed_precision=bf16 \
        --filter_crossattn_str=cross+self \
        --reg_prompts="photo of a dog" \
        --enable_norm_reg \
        --enable_text_reg \
        --text_reg_loss_weight=0.001 \
        --norm_reg_loss_weight=$norm_reg_loss_weight \
        --ti_reg_type="decay" \
        --resume_ti_embedding_path="outputs/checkpoints/dog/ewc_reg/baseline_lora_rank_10/lora_weight_e124_s1000.safetensors"
    done
  done
done
  # --ti_reg_type="decay" \
  # --mask_identifier_causal_attention \

  # --reg_texts_file="dataset/data_reg/dog_reg.txt" \
  # --reg_images_root="dataset/prior_real/samples_dog/dog" \

  
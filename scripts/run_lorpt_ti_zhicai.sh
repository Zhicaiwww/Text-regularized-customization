#!/bin/bash
#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth

# init_placeholder_as_class:  use the class token as the initializer, e.g., "cat" -> "<krk> cat"
# train_text_encoder: train the text encoder
# with_prior_preservation: use prior preserving dataset
# filter_crossattn_str: self, cross, full,cross+self
# enable_text_reg
# text_reg_alpha_weight
# reg_prompts: photo of a cat
# scale_norm_reg: scale the norm reg loss accpording to the 1/SNR
# mask_identifier_causal_attention: mask the identifier in the causal attention
# text_reg_beta_weight: K-projection 额外正则项 photo of a <new> cat


# accelerate launch
seeds=(0)
lora_rank=(10)
text_reg_beta_weights=(0.01)

for seed in "${seeds[@]}"; do 
  for _lora_rank in "${lora_rank[@]}"; do
      for text_reg_beta_weight in "${text_reg_beta_weights[@]}"; do
      export seed=$seed
      export lora_rank=$_lora_rank
      export text_reg_beta_weight=$text_reg_beta_weight
      export OUTPUT_DIR="logs_zhicai/cat/$(date "+%Y-%m-%dT%H-%M-%S")_cat_textReg"
      CUDA_VISIBLE_DEVICES=5 python training_scripts/train_lora_w_ti.py \
        --pretrained_model_name_or_path="models/stable-diffusion-v1-5" \
        --instance_data_dir="custom_datasets/data/cat" \
        --output_dir=$OUTPUT_DIR \
        --resolution=512 \
        --train_batch_size=1 \
        --learning_rate=1e-4 \
        --learning_rate_text=1e-5 \
        --learning_rate_ti=5e-4 \
        --color_jitter \
        --lr_scheduler="constant" \
        --lr_warmup_steps=10 \
        --ti_train_step=1000 \
        --max_train_steps=1000 \
        --placeholder_token="<krk1>" \
        --learnable_property="object" \
        --class_tokens="cat" \
        --save_steps=100 \
        --resize \
        --center_crop \
        --scale_lr \
        --lora_rank=$lora_rank \
        --gradient_accumulation_steps=4 \
        --output_format=safe \
        --mixed_precision=bf16 \
        --filter_crossattn_str="cross" \
        --enable_text_reg \
        --text_reg_alpha_weight=0.01 \
        --text_reg_beta_weight=$text_reg_beta_weight\
        --ti_reg_type="decay" \
        --local_files_only \
        --mask_identifier_causal_attention
    done
  done
done

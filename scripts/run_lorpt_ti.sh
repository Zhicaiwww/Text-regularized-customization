export target_name="cat"
export superclass="cat"
export datapath="data"
# export OUTPUT_DIR="logs/log_ablation/TI_ablation/${target_name}/$(date "+%Y-%m-%dT%H-%M-%S")_${superclass}_textReg"
export OUTPUT_DIR="logs/results/${target_name}/$(date "+%Y-%m-%dT%H-%M-%S")_${superclass}_textReg"

# ============================= Baseline =============================
# CUDA_VISIBLE_DEVICES=3 python training_scripts/train_lora_w_ti.py \
#   --pretrained_model_name_or_path="models/stable-diffusion-v1-5" \
#   --instance_data_dir="custom_datasets/${datapath}/${target_name}" \
#   --output_dir=$OUTPUT_DIR \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --learning_rate=1e-4 \
#   --learning_rate_ti=5e-4 \
#   --color_jitter \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=10 \
#   --ti_train_step=500 \
#   --max_train_steps=2000 \
#   --placeholder_token="<krk1>" \
#   --learnable_property="object" \
#   --class_tokens="${superclass}" \
#   --save_steps=100 \
#   --resize \
#   --center_crop \
#   --scale_lr \
#   --lora_rank=10 \
#   --gradient_accumulation_steps=4 \
#   --output_format=safe \
#   --mixed_precision=bf16 \
#   --filter_crossattn_str="cross" \
#   --mask_identifier_causal_attention \
#   --mask_identifier_ratio 0.75 \
#   --local_files_only


# ============================= TextReg =============================
CUDA_VISIBLE_DEVICES=1 python train_lora_w_ti.py \
  --pretrained_model_name_or_path="models/stable-diffusion-v1-5" \
  --instance_data_dir="custom_datasets/${datapath}/${target_name}" \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=10 \
  --ti_train_step=500 \
  --max_train_steps=2000 \
  --placeholder_token="<krk1>" \
  --learnable_property="object" \
  --class_tokens="${superclass}" \
  --save_steps=100 \
  --resize \
  --center_crop \
  --scale_lr \
  --lora_rank=10 \
  --gradient_accumulation_steps=4 \
  --output_format=safe \
  --mixed_precision=bf16 \
  --filter_crossattn_str="cross" \
  --enable_text_reg \
  --text_reg_alpha_weight=0.01 \
  --text_reg_beta_weight=0.01 \
  --local_files_only

#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="custom_data/data/dog"
export OUTPUT_DIR="lora_output/checkpoints/debug_decay_0.001"
export CLASS_PRIOR_DIR="dataset/real_reg/samples_dog/dog" # prior preserving dataset dir
export CLASS_CAPTION_DIR="dataset/real_reg/samples_dog/caption.txt" # caption of prior preserving dataset dir, e.g.: 
# initializer_token_as_class:  use the class token as the initializer, e.g., "dog" -> "<krk> dog"
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

# accelerate launch
CUDA_VISIBLE_DEVICES=0 python training_scripts/train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_PRIOR_DIR \
  --class_prompt_or_file='photo of a dog' \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --ti_train_step=10000 \
  --max_train_steps=10000 \
  --placeholder_token="<krk1>" \
  --learnable_property="object" \
  --initializer_token="dog" \
  --save_steps=200 \
  --resize=True \
  --center_crop \
  --scale_lr \
  --lora_rank=4 \
  --gradient_accumulation_steps=1 \
  --output_format=safe \
  --mixed_precision=fp16 \
  --filter_crossattn_str=cross+self \
  --norm_reg_loss_weight=0.001 \
  --text_reg_loss_weight=0.001 \
  --reg_prompts="photo of a dog" \
  --ti_reg_type="decay" \

  # --stochastic_attribute="game character,3d render,4k,highres" # these attributes will be randomly appended to the prompts
  
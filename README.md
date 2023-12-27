# Prior Preserved Concept Learning Without Image Regularization
## Introduction
Concept learning aims to enable T2I models to generate specific concepts and is widely used in the community. Nevertheless, the process of concept learning often involves model fine-tuning, which in turn brings the potential risk of language drift. Language drift typically manifests itself as degraded knowledge within the T2I model and a diminished editability for the target concept.
Our observation indicates that the commonly employed regularization method, which involves image regularization through the introduction of an image-caption subset, has certain limitations in addressing language drift.
Consequently, we introduce \textbf{TextReg}, a method designed to explicitly regulate model fine-tuning through the exclusive use of text prompts. Furthermore, we draw attention to the challenge of semantic drift that arises when integrating visual concepts into textual tokens and propose use identifier-mask to alleviate this issue. 

## Datasets
We provide Concept Set, which is the combination of concept datasets from [Dreambooth](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion) and [Custom Diffusion](https://github.com/adobe-research/custom-diffusion) and contains a total of 39 unique objects and living animals. Please download the dataset from [Here](https://drive.google.com/file/d/10QQMQsOfiLozUneESdoSRC7CibLRmdf_/view?usp=sharing), unzip the file, and place it wihtin the `custom_datasets` folder. The structure of the dataset folder should be organized as follows:

```
custom_datasets/data/
├── backpack
│   ├── 00.jpg
│   ├── 01.jpg
│   ├── 02.jpg
│   ├── 03.jpg
│   ├── 04.jpg
│   └── 05.jpg
├── backpack_dog
│   ├── 00.jpg
│   ├── 01.jpg
│   ├── 02.jpg
│   ├── 03.jpg
│   └── 04.jpg
├── barn
```
## Code Description 

Text regularization does not require the preparation of an image regularization subset. Fine-tuning can be conducted seamlessly, as long as the pre-trained Stable Diffusion checkpoint is pre-cached. 

### 1. Fine-tuning Stable Diffusion using Text-Regularization.

To fine-tune on a given instance, specify the foler `INSTANCE_DATA_DIR` that cotains user-specified images and give the class descriptor in `CLASS_TOKENS`, and run following command
```bash
LORA_RANK=10
INSTANCE_DATA_DIR="custom_datasets/data/cat"
LEARNABLE_PROPERTY="object"
CLASS_TOKENS="cat"
OUTPUT_DIR="logs/cat/2023-11-16T23-07-45_cat_textReg"

CUDA_VISIBLE_DEVICES=5 python training_scripts/train_lora_w_ti.py \
--pretrained_model_name_or_path="models/stable-diffusion-v1-5" \
--instance_data_dir=$INSTANCE_DATA_DIR \
--output_dir=$OUTPUT_DIR \
--resolution=512 \
--train_batch_size=1 \
--learning_rate=1e-4 \
--learning_rate_text=1e-5 \
--learning_rate_ti=5e-4 \
--resize \
--center_crop \
--scale_lr \
--color_jitter \
--lr_scheduler="constant" \
--lr_warmup_steps=10 \
--ti_train_step=500 \
--max_train_steps=2000 \
--placeholder_token="<krk1>" \
--learnable_property=$LEARNABLE_PROPERTY\
--class_tokens=$CLASS_TOKENS \
--save_steps=100 \
--lora_rank=$LORA_RANK \
--gradient_accumulation_steps=4 \
--output_format=safe \
--mixed_precision=bf16 \
--filter_crossattn_str="cross" \
--enable_text_reg \
--text_reg_alpha_weight=0.01 \
--text_reg_beta_weight=0.01\
--ti_reg_type="decay" \
--mask_identifier_causal_attention
```
The total fine-tuing steps are 2000, which conprises 500 steps for the TI stage and the other 1500 for the UT stage.
The identifier-masked textual inversion is set as the default strategy during the first Textual Inversion stage (If one does not want to use identifier mask, remove the command line parameter `--mask_identifier_causal_attention`). The identifier to anchor the given instance in the semantic space is `<krk1>`, which is a new injected token and treated as an adjective word for the class descriptor (i.e., `<krk1> cat`). Other TextReg-related hyper-parameters are fixed based on our evaluation results on extensive concepts. 

### 2. Sampling with the personalized checkpoint.

As the fine-tuning process finished, the sampling python script is 

```bash
CUDA_VISIBLE_DEVICES=1 python evaluation/sample.py \
--lora_ckpt 'logs/cat/2023-11-16T23-07-45_cat_textReg/lora_weight_s2000.safetensors' \
--prompts 'photo of a <krk1> cat swimming in a pool' \
--n_img 10
``` 

please specify the desired checkpoint path in `lora_ckpt` and the prompt containing `<krk> cat` in `prompts`. Enjoy it !

## Acknowledgements

This project is built upon the repository [lora](https://github.com/cloneofsimo/lora) and [diffusers](https://github.com/huggingface/diffusers). Special thanks to the contributors.

## Requirements


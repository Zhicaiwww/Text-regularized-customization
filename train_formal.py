import os, sys, argparse, datetime, pdb
import shlex
import subprocess
from custom_datasets.utils import *

'''
python train_formal.py \
    --log_dir logs/log_test \
    --dataset D \
    --specific_classes 'dog' \
    --gpu_ids 6

python train_formal.py \
    --log_dir logs/Pick_Images/transferable_identifier/ratio=0.9 \
    --mask_identifier_ratio 0.9 \
    --dataset D \
    --specific_classes 'cat' \
    --required_mode 0 \
    --gpu_ids 2

rsync -av --exclude="eval" --exclude="logs" --exclude="*lora_weight_s*" logs/log_formal_D /data/zhicai/code/Text-regularized-customization
!@#$%^+lds@ustc123456
'''

mode_map = {0: "baseline", 1: "priorReal", 2: "priorGen", 3: "textReg"}
dataset_dict = {
    "D": {**OBJECT_CLASSES, **LIVE_SUBJECT_CLASSES}, 
    "I": {**IMAGENET_LIVE_CLASSES, **IMAGENET_OBJECT_CLASSES},
    "A": {**ABLATION_CLASSES},
    "O": {**OXFORD_PET_CLASSES},
}

parser = argparse.ArgumentParser()
# Script Settings
parser.add_argument('--log_dir', type=str, default="logs")
parser.add_argument('--dataset', type=str, default="A")
parser.add_argument('--gpu_ids', type=str, default="0")
parser.add_argument('--required_mode', type=str, default='0,1,2,3')
parser.add_argument('--specific_classes', type=str, default="")
parser.add_argument('--enable_eval', action='store_true')
# Hyperparameters
parser.add_argument('--text_reg_alpha_weight', type=float, default=0.01)
parser.add_argument('--text_reg_beta_weight', type=float, default=0.1)
parser.add_argument('--mask_identifier_ratio', type=float, default=0.75)
parser.add_argument('--ti_train_step', type=int, default=500)
parser.add_argument('--unet_train_steps', type=int, default=500)
parser.add_argument('--textreg_extra_train_steps', type=int, default=1000)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

ti_train_step = args.ti_train_step
max_train_steps = args.ti_train_step + args.unet_train_steps

def train(args, target_name: str, superclass: str, mode: int, tag: str = "_"):

    data_path = which_target_dataset(target_name)
    log_dir = os.path.join(args.log_dir, target_name)

    try:
        print_box(f"Customization on '{target_name} : {superclass}' | mode {mode}")
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        command = f"python training_scripts/train_lora_w_ti.py \
                        --pretrained_model_name_or_path 'models/stable-diffusion-v1-5' \
                        --instance_data_dir '{data_path}' \
                        --train_batch_size 1 \
                        --lr_warmup_steps 10 \
                        --ti_train_step {ti_train_step} \
                        --placeholder_token '<krk1>' \
                        --class_tokens '{superclass}' \
                        --resize \
                        --center_crop \
                        --color_jitter \
                        --scale_lr \
                        --output_format safe \
                        --prior_loss_weight 1 \
                        --mixed_precision bf16 \
                        --gradient_accumulation_steps 4 \
                        --lora_rank 10 \
                        --filter_crossattn_str 'cross' \
                        --ti_reg_type 'decay' \
                        --mask_identifier_causal_attention \
                        --mask_identifier_ratio {args.mask_identifier_ratio} \
                        --local_files_only "

        if mode == 0:
            command += f" --max_train_steps {max_train_steps} \
                          --output_dir '{log_dir}/{now}_{superclass}_baseline' " 
        elif mode == 1:
            command += f" --max_train_steps {max_train_steps} \
                          --with_prior_preservation \
                          --class_data_dir custom_datasets/prior_real/{superclass}/{superclass} \
                          --prompts_file custom_datasets/prior_real/{superclass}/caption.txt \
                          --output_dir '{log_dir}/{now}_{superclass}_priorReal' " 
        elif mode == 2:
            command += f" --max_train_steps {max_train_steps} \
                          --with_prior_preservation \
                          --class_data_dir custom_datasets/prior_gen/{superclass}/{superclass} \
                          --prompts_file custom_datasets/prior_gen/{superclass}/caption.txt \
                          --output_dir '{log_dir}/{now}_{superclass}_priorGen' " 
        elif mode == 3:
            command += f" --max_train_steps {max_train_steps + args.textreg_extra_train_steps} \
                          --enable_text_reg \
                          --text_reg_alpha_weight {args.text_reg_alpha_weight} \
                          --text_reg_beta_weight {args.text_reg_beta_weight} \
                          --output_dir '{log_dir}/{now}_{superclass}_textReg' " 
            
        if (mode in [1, 2, 3]) and (0 in modes):
            for root, dirs, files in sorted(os.walk(os.path.join(args.log_dir, target_name))):
                for dir in sorted(dirs):
                    if 'baseline' in dir:
                        command += f" --resume_ti_embedding_path {os.path.join(root, dir, f'lora_weight_s{ti_train_step}.safetensors')}"

        # pdb.set_trace()
        p = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
        output, err = p.communicate()
        print(output.decode("utf-8"))

    except subprocess.CalledProcessError as e:
        print(e.output)


if __name__ == '__main__':

    modes = [int(i) for i in args.required_mode.replace(' ', '').split(',')]
    data_dict = dataset_dict[args.dataset]

    for target_name, superclass in data_dict.items():
        if (target_name in args.specific_classes.replace(' ', '').split(',')) or (len(args.specific_classes) == 0):
            for mode in modes:
                if (
                    (not os.path.exists(os.path.join(args.log_dir, target_name))) or 
                    (all(mode_map[mode] not in i for i in os.listdir(os.path.join(args.log_dir, target_name))))
                ):
                    train(args, target_name, superclass, mode)

    if args.enable_eval:
        os.system(
            f"python evaluation/evaluate_script.py --log_dir {args.log_dir} --gpu_ids {args.gpu_ids}"
        )

    print(f"\nFinish Training.")
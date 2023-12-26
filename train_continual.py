import os, sys, argparse, datetime, pdb
import shutil
import shlex
import subprocess
from custom_datasets.utils import parse_templates_class_name, which_target_dataset, print_box

'''
scp -P 2125 -r liox@112.29.111.160:/data/zhicai/code/Text-regularized-customization/custom_datasets/Oxford-IIIT-Pet /data/liox/Text-regularized-customization/custom_datasets
!@#$%^+lds@ustc123456

python train_continual.py \
    --target_names 'Abyssinian, Bengal, Birman, Bombay, British_Shorthair, Egyptian_Mau, Maine_Coon, Persian, Ragdoll, Russian_Blue' \
    --enable_eval \
    --gpu_ids 7

python train_continual.py \
    --target_names 'backpack, barn, bear_plushie, berry_bowl, can, candle, chair, clock' \
    --enable_eval \
    --gpu_ids 4

python train_continual.py \
    --target_names 'american_bulldog, american_pit_bull_terrier, basset_hound, beagle, boxer, chihuahua, english_cocker_spaniel, english_setter, german_shorthaired, great_pyrenees' \
    --enable_eval \
    --gpu_ids 5

python train_continual.py \
    --target_names 'dog, dog1, cat, cat1, n01530575, n01531178, n02483708, n02484975' \
    --enable_eval \
    --gpu_ids 6

python train_continual.py \
    --target_names 'dog1, dog, n02483708, n02484975, cat, cat1, dog3, dog7, n01530575, n01531178' \
    --required_mode 1,3 \
    --gpu_ids 4

rsync -av --exclude="eval" --exclude="logs" --exclude="*lora_weight_s*" -e "ssh -p 2125" \
liox@112.29.111.160:/data/zhicai/code/Text-regularized-customization/logs/log_formal_D/dog1/2023-11-09T04-44-29_dog_textReg \
logs/logs_continual_test_2/dog1-dog-n02483708-n02484975-cat-cat1-dog3-dog7-n01530575-n01531178
!@#$%^+lds@ustc123456
'''

mode_map = {0: "baseline", 1: "priorReal", 2: "priorGen", 3: "textReg"}

parser = argparse.ArgumentParser()
# Script Settings
parser.add_argument('--log_dir', type=str, default="logs/logs_continual_test")
parser.add_argument('--target_names', type=str, default='')
parser.add_argument('--gpu_ids', type=str, default="0")
parser.add_argument('--required_mode', type=str, default='0,1,2,3')
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

def train(target_name, superclass, placeholder_all, sub_task_dir, mode, prev_target_dir=None, tag="_"):

    data_path = which_target_dataset(target_name)

    try:
        print_box(f"Continual learning on '{target_name}' | superclass '{superclass}' | mode '{mode}'")
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        command = f"python training_scripts/train_lora_continual.py \
                        --pretrained_model_name_or_path 'models/stable-diffusion-v1-5' \
                        --instance_data_dir '{data_path}' \
                        --train_batch_size 1 \
                        --lr_warmup_steps 10 \
                        --ti_train_step {ti_train_step} \
                        --placeholder_token '{placeholder_all}' \
                        --class_tokens '{superclass}' \
                        --resize \
                        --center_crop \
                        --color_jitter \
                        --scale_lr \
                        --output_format safe \
                        --mixed_precision bf16 \
                        --gradient_accumulation_steps 4 \
                        --lora_rank 10 \
                        --filter_crossattn_str 'cross' \
                        --ti_reg_type 'decay' \
                        --mask_identifier_causal_attention \
                        --mask_identifier_ratio {args.mask_identifier_ratio} \
                        --local_files_only "

        if prev_target_dir is not None:
            resume_ti_embedding_path = os.path.join(prev_target_dir, "lora_weight.safetensors")
            command += f" --resume_ti_embedding_path '{resume_ti_embedding_path}'"
            
        if mode == 0:
            command += f" --max_train_steps {max_train_steps} \
                          --output_dir '{sub_task_dir}/{now}_{superclass}_baseline' " 
        elif mode == 1:
            command += f" --max_train_steps {max_train_steps} \
                          --with_prior_preservation \
                          --class_data_dir custom_datasets/prior_real/{superclass}/{superclass} \
                          --prompts_file custom_datasets/prior_real/{superclass}/caption.txt \
                          --output_dir '{sub_task_dir}/{now}_{superclass}_priorReal' " 
        elif mode == 2:
            command += f" --max_train_steps {max_train_steps} \
                          --with_prior_preservation \
                          --class_data_dir custom_datasets/prior_gen/{superclass}/{superclass} \
                          --prompts_file custom_datasets/prior_gen/{superclass}/caption.txt \
                          --output_dir '{sub_task_dir}/{now}_{superclass}_priorGen' " 
        elif mode == 3:
            command += f" --max_train_steps {max_train_steps + args.textreg_extra_train_steps} \
                          --enable_text_reg \
                          --text_reg_alpha_weight {args.text_reg_alpha_weight} \
                          --text_reg_beta_weight {args.text_reg_beta_weight} \
                          --output_dir '{sub_task_dir}/{now}_{superclass}_textReg' " 
        
        # pdb.set_trace()
        p = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
        output, err = p.communicate()
        print(output.decode("utf-8"))

    except subprocess.CalledProcessError as e:
        print(e.output)


if __name__ == '__main__':

    target_names = args.target_names.replace(' ', '').split(',')
    modes = [int(i) for i in args.required_mode.replace(' ', '').split(',')]
    task_dir = os.path.join(args.log_dir, f"{'-'.join(target_names)}")
    os.makedirs(task_dir, exist_ok=True)

    for idx, target_name in enumerate(target_names):

        placeholder = f"<krk{idx+1}>"
        placeholder_all = placeholder if idx == 0 else f"{placeholder_all}+{placeholder}"
        _, superclass = parse_templates_class_name(target_name)

        sub_task_dir = os.path.join(task_dir, f"{placeholder}_{target_name}")
        prev_sub_task_dir = os.path.join(task_dir, f"<krk{idx}>_{target_names[idx-1]}") if idx != 0 else None

        for mode in modes:
            if (not os.path.exists(sub_task_dir)) or (all(mode_map[mode] not in i for i in os.listdir(sub_task_dir))):
                if prev_sub_task_dir is None:
                    train(target_name, superclass, placeholder, sub_task_dir, mode, prev_target_dir=None)
                else:
                    for target_dir in sorted(os.listdir(prev_sub_task_dir)):
                        if mode_map[mode] in target_dir:
                            train(target_name, superclass, placeholder_all, sub_task_dir, mode, prev_target_dir=os.path.join(prev_sub_task_dir, target_dir))

    if args.enable_eval:
        os.system(
            f"python evaluation/evaluate_continual_script.py --log_dir {task_dir} --gpu_ids {args.gpu_ids}"
        )
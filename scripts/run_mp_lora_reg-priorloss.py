import os 
import sys
sys.path.append(os.getcwd())
import subprocess
import argparse
import multiprocessing
from multiprocessing import Queue
from base.utils import (OBJECT_CLASSES, LIVE_SUBJECT_CLASSES, DATA_ROOT, PLACE_HODLER, DEBUG_OUTPUT_ROOT, OUTPUT_ROOT, DEBUG_DIRS)
# merge two dicts
ALL_CLASSES = {**OBJECT_CLASSES, **LIVE_SUBJECT_CLASSES}
def run_command(queue, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    while not queue.empty():
        args = queue.get()
        if args is None:
            break
        subprocess.run(args, check=True)

def main():
    parser = argparse.ArgumentParser(description="Parallelize and queue commands")
    args = parser.parse_args()

    debug = True

    args.data_root = DATA_ROOT
    args.init_placeholder_as_class=False
    args.lora_rank=10

    # 需要运行的命令列表，每个元素都是一个命令
    
    basic_command = [
       "python",
       "training_scripts/train_lora_w_ti.py",
        "--resolution=512",
        "--train_batch_size=1",
        "--learning_rate=1e-4",
        "--learning_rate_text=1e-5",
        "--learning_rate_ti=5e-4",
        "--color_jitter",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=100",
        "--ti_train_step=1000",
        "--max_train_steps=2000",
        "--resize=True",
        "--center_crop",
        "--scale_lr",
        "--gradient_accumulation_steps=4",
        "--output_format=safe",
        "--mixed_precision=bf16",
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
    ]
    
    command_queue = Queue()

    data_dir_name_list = DEBUG_DIRS if debug else os.listdir(args.data_root)
    for data_dir_name in data_dir_name_list:
        try:
            class_name = ALL_CLASSES[data_dir_name]
        except KeyError:
            print(f'key {data_dir_name} not found in ALL_CLASSES')
            continue
        instance_dir = f'dataset/data/{data_dir_name}'
        text_reg_prompt = f'photo of a {class_name}'

        extra_command = [
            f"--instance_data_dir={instance_dir}",
            f"--class_tokens={class_name}",
            f"--lora_rank={args.lora_rank}",
            f"--reg_prompts={text_reg_prompt}",
            "--learnable_property=object",
            f"--placeholder_token={PLACE_HODLER}",
            "--save_steps=100",
            "--filter_crossattn_str=cross+self",
            "--ti_reg_type='decay'",
            f"--resume_ti_embedding_path=outputs/checkpoints/{data_dir_name}/shareTI/decay_lr_5e-4_gas_4/lora_weight_s1000.safetensors"
        ]

        extra_command.extend([f"--class_data_dir=dataset/prior_gen/{class_name}/{class_name}",
                              f"--prompts_file=dataset/prior_gen/{class_name}/caption.txt",
                              "--with_prior_preservation",])

        output_dir = f"{DEBUG_OUTPUT_ROOT if debug else OUTPUT_ROOT}/checkpoints/{data_dir_name}/shareTI_text_0.01_L-priorloss/decay_lr_5e-4_gas_4"  
        extra_command.extend([f"--output_dir={output_dir}"])
        command = [*basic_command, *extra_command]

        command_queue.put(command)

    processes = []

    gpus =[0,1,3,4,5,6,7]
    for gpu_id in gpus:
        p = multiprocessing.Process(target=run_command, args=(command_queue,gpu_id))
        processes.append(p)
        p.start()

    for process in processes:
        command_queue.put(None)

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()

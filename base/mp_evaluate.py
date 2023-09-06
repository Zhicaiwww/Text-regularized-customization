import subprocess
import os
import sys
sys.path.append(os.getcwd())
import multiprocessing
from multiprocessing import Queue
from utils import (
    parse_templates_class_name,
    DATA_ROOT,
    OUTPUT_ROOT,
    DEBUG_OUTPUT_ROOT,
    PLACE_HODLER,
    LORA_CKPT_TEMPLATE,
    DEBUG_DIRS)

def run_sample(queue, args):
    while not queue.empty():
        task = queue.get()
        if task is None:
            break
        cmd = f"python base/evaluate.py --lora-ckpt {task['lora_ckpt']} --instance-data-dir {task['data_dir_name']} --gpu {args.gpu_id} --class-tokens {task['class_name']} --custom-token='{task['custom_token']}' --n-test 200"
        subprocess.call(cmd, shell=True)

class SampleArgs:
    def __init__(self, gpu_id=0, n_per_prompt = 50, bs = 2):
        self.n_per_prompt = n_per_prompt
        self.bs = bs
        self.gpu_id = gpu_id



def main():
    debug = True
    experiment_types =['shareTI_text_0.01_norm_0.01', 
                    'shareTI_text_0.01_G-ewc_0.01',
                    'shareTI_text_0.01_L-ewc_0.01',
                    'shareTI_text_0.01_L-priorloss']
    gpus = [4,5,6,7]

    data_dir_names = DEBUG_DIRS if debug else os.listdir(DATA_ROOT)

    task_queue = Queue()
    args = SampleArgs()
    for data_dir_name in data_dir_names:
        _ , class_name = parse_templates_class_name(data_dir_name)
        custom_name = PLACE_HODLER + ' ' + class_name
        for experiment_type in experiment_types:
            lora_ckpt = os.path.join(OUTPUT_ROOT if not debug else DEBUG_OUTPUT_ROOT ,LORA_CKPT_TEMPLATE.format(data_dir_name, experiment_type))
            kwargs = {"data_dir_name": os.path.join(DATA_ROOT,data_dir_name),
                        "class_name": class_name,
                        "custom_token": custom_name,
                        "lora_ckpt": lora_ckpt,
                        }
            print(kwargs)
            task_queue.put(kwargs)

    processes = []

    for gpu_id in gpus:
        args.gpu_id = gpu_id
        p = multiprocessing.Process(target=run_sample, args=(task_queue, args))
        processes.append(p)
        p.start()

    for process in processes:
        task_queue.put(None)

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()

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
        cmd = f"python base/sample.py --concept-path-name {task['data_dir_name']} --lora-ckpt {task['lora_ckpt']} --n-per-prompt {args.n_per_prompt} --gpu {args.gpu_id} --bs {args.bs}  {'--class-only' if args.class_only else ''}"
        subprocess.call(cmd, shell=True)

class SampleArgs:
    def __init__(self, gpu_id=0, n_per_prompt = 20, bs = 2,class_only=False):
        self.n_per_prompt = n_per_prompt
        self.bs = bs
        self.gpu_id = gpu_id
        self.class_only= class_only


    
def main():
    # sample image with class names, using lora checkpoint, test overfitting in gloabl class domains 
    debug = True
    gpus = [1,2,3,4,5,6]
    class_only = False # only sample with class names
    experiment_type='shareTI_text_0.01_norm_0.01'
    data_dir_names = DEBUG_DIRS if debug else os.listdir(DATA_ROOT)
    # data_dir_name_l = ["berry_bowl","backpack"]
    task_queue = Queue()
    args = SampleArgs()
    args.class_only = class_only

    for data_dir_name in data_dir_names:
        lora_ckpt = os.path.join(OUTPUT_ROOT if not debug else DEBUG_OUTPUT_ROOT ,LORA_CKPT_TEMPLATE.format(data_dir_name, experiment_type))
        if os.path.exists(lora_ckpt):
            task_queue.put({"data_dir_name": data_dir_name, "lora_ckpt":lora_ckpt})
        else:
            print(f"Skip {data_dir_name}")

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

import os, sys, json, pdb
sys.path.append('/data/liox/Text-regularized-customization')
import argparse
import torch
from multiprocessing import Process, Queue
from custom_datasets.utils import *

'''
'''

mode_map = {0: "baseline", 1: "priorReal", 2: "priorGen", 3: "textReg"}

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default="")
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=6)
args = parser.parse_args()


def worker(queue, gpu_id):

    while not queue.empty():

        lora_ckpt, target_name, placeholder, superclass, mode = queue.get()
        outdir = os.path.join(lora_ckpt, "images", f"{placeholder}_{target_name}")

        print_box(f"A: Evaluation on '{placeholder} {target_name}' | superclass '{superclass}' | mode '{mode}'\nSaving to {os.path.join(lora_ckpt, 'images')}")
        if not os.path.exists(outdir):
            os.system(
                f"CUDA_VISIBLE_DEVICES={gpu_id} python evaluation/sample.py \
                    --lora_ckpt '{lora_ckpt}' \
                    --filter_crossattn_str cross \
                    --batch_size {args.batch_size} \
                    --n_img 200 \
                    --prompts 'photo of {placeholder} {superclass}' \
                    --outdir '{outdir}' "
            )
            
        if not os.path.exists(os.path.join(outdir, os.listdir(outdir)[0], "evaluate_logs.json")):
            dir1 = which_target_dataset(target_name)
            dir2 = os.path.join(outdir, os.listdir(outdir)[0], "samples")
            os.system(
                f"CUDA_VISIBLE_DEVICES={gpu_id} python evaluation/2_evaluate.py \
                    --dir1 '{dir1}' \
                    --dir2 '{dir2}'"
            )
        

def find_key_by_value(input_dict, search_value):
    for key, value in input_dict.items():
        if value in search_value:
            return key
    return None


def custom_sort(item):
    return int(item.split('_', 1)[0][4:-1])


if __name__ == '__main__':

    queue, task_list = Queue(), sorted(os.listdir(args.log_dir), key=custom_sort)
    mode_and_files = {find_key_by_value(mode_map, l): l for l in sorted(os.listdir(os.path.join(args.log_dir, task_list[-1])))}
    N, placeholder_list, target_name_list = len(task_list), [i.split('_', 1)[0] for i in task_list], [j.split('_', 1)[1] for j in task_list]
    superclass_list = [parse_templates_class_name(k)[1] for k in target_name_list]

    for mode in mode_and_files.keys():
        lora_ckpt = os.path.join(args.log_dir, task_list[-1], mode_and_files[mode])
        for idx, target_name in enumerate(target_name_list):
            queue.put((lora_ckpt, target_name, placeholder_list[idx], superclass_list[idx], mode))

    processes = []
    for i in [int(x) for x in args.gpu_ids.split(',')]:
        p = Process(target=worker, args=(queue, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
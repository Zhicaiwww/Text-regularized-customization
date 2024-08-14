import json
import os
import pdb
import sys

sys.path.append('/data/zhicai/code/Text-regularized-customization')
import argparse
from multiprocessing import Process, Queue

from utils import (
    parse_templates_class_name,
    parse_templates_from_superclass,
    print_box,
    which_target_dataset,
)

'''
python evaluation/evaluate_script.py --log_dir logs/log_ablation/image_reg/logs/log_type --gpu_ids 1,2,3

python evaluation/evaluate_script.py --log_dir logs/Visualize_Images/vis-GenReg_type/dog1/1_default --disable_phase_1 --enable_phase_2 --gpu_ids 6
'''

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default="")
parser.add_argument('--disable_phase_1', default=False, action='store_true')
parser.add_argument('--enable_phase_2', default=False, action='store_true')
parser.add_argument('--enable_phase_3', default=False, action='store_true')
parser.add_argument('--filter_crossattn_str', type=str, default="cross")
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=5)
args = parser.parse_args()

batch_size = args.batch_size


def worker(queue, gpu_id):

    while not queue.empty():

        eval_dir, target_name, superclass, placeholder = queue.get()
        os.makedirs(eval_dir, exist_ok=True)
        target_dataset = which_target_dataset(target_name)

        # =============================================================== Phase 1 ===============================================================

        if not args.disable_phase_1:
            print_box(f"Phase 1: Strat evaluating models on '{os.path.dirname(eval_dir)}' | target_name '{target_name}' | superclass '{superclass}'")
            os.system(
                f"CUDA_VISIBLE_DEVICES={gpu_id} python evaluation/1_evaluate.py \
                    --superclass '{superclass}' \
                    --lora_ckpt '{os.path.dirname(eval_dir)}' \
                    --filter_crossattn_str '{args.filter_crossattn_str}' \
                    --data_dir '{target_dataset}' \
                    --placeholder '{placeholder}' \
                    --batch_size {batch_size} \
                    --enable_saving"
            )
        
        # =============================================================== Phase 2 ===============================================================

        if args.enable_phase_2:
            print_box(f"Phase 2: Strat evaluating models on '{os.path.dirname(eval_dir)}' | target_name '{target_name}' | superclass '{superclass}'")
            caption_dir = os.path.join(eval_dir, "caption.txt")
            templates = parse_templates_from_superclass(superclass)
            with open(caption_dir, "w") as file:
                for prompt in templates:
                    file.write(prompt + "\n")
            os.system(
                f"CUDA_VISIBLE_DEVICES={gpu_id} python evaluation/sample.py \
                    --lora_ckpt '{os.path.dirname(eval_dir)}' \
                    --filter_crossattn_str '{args.filter_crossattn_str}' \
                    --batch_size {batch_size} \
                    --n_img 10 \
                    --from_file '{caption_dir}' \
                    --outdir '{eval_dir}' "
            )
            os.remove(caption_dir)

            dir1 = os.path.join("custom_datasets/samples_from_pretrained", superclass, "samples")
            dir2 = os.path.join(eval_dir, sorted(os.listdir(eval_dir))[-1], "samples")
            os.system(
                f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation/2_evaluate.py \
                    --dir1 {dir1} \
                    --dir2 {dir2}'
            )

        # =============================================================== Phase 3 ===============================================================

        if args.enable_phase_3:
            print_box(f"Phase 3: Strat evaluating models on '{os.path.dirname(eval_dir)}' | target_name '{target_name}' | superclass '{superclass}'")
            os.system(
                f"CUDA_VISIBLE_DEVICES={gpu_id} python evaluation/sample.py \
                    --lora_ckpt '{os.path.dirname(eval_dir)}' \
                    --filter_crossattn_str '{args.filter_crossattn_str}' \
                    --batch_size {batch_size} \
                    --n_img 1 \
                    --from_file 'custom_datasets/samples_for_general/general_prompts.txt' \
                    --outdir '{eval_dir}' "
            )

            dir1 = "custom_datasets/samples_for_general/samples"
            dir2 = os.path.join(eval_dir, sorted(os.listdir(eval_dir))[-1], "samples")
            os.system(
                f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation/2_evaluate.py \
                    --dir1 {dir1} \
                    --dir2 {dir2}'
            )


if __name__ == '__main__':

    queue = Queue()
    for root, dirs, files in sorted(os.walk(args.log_dir)):
        for dir in sorted(dirs):
            if any(k in dir for k in ['baseline', 'priorReal', 'priorGen', 'textReg']):
                eval_dir = os.path.join(root, dir, "eval")
                placeholder, target_name  = "<krk1>", root.split("/")[-1]
                _, superclass = parse_templates_class_name(target_name)
                if (not os.path.exists(eval_dir)) or (len(os.listdir(eval_dir)) == 1):
                    queue.put((eval_dir, target_name, superclass, placeholder))
                else:
                    pass

    processes = []
    for i in [int(x) for x in args.gpu_ids.split(',')]:
        p = Process(target=worker, args=(queue, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    os.system(
        f'python evaluation/json_to_csv.py {args.log_dir}'
        )
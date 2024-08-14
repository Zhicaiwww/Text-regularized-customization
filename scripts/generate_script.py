import os
import pdb
import sys

sys.path.append('/data/zhicai/code/custom-diffusion')
import argparse
import shutil
from multiprocessing import Process, Queue

from utils import (
    IMAGENET_LIVE_CLASSES,
    IMAGENET_OBJECT_CLASSES,
    LIVE_SUBJECT_CLASSES,
    OBJECT_CLASSES,
)

'''
python custom_datasets/prior_gen/generate_script.py --gpu_ids 0
'''

parser = argparse.ArgumentParser()
parser.add_argument('--prior_gen_dir', type=str, default="/data/zhicai/code/custom-diffusion/custom_datasets/prior_gen")
parser.add_argument('--num_class_images', type=int, default=200)
parser.add_argument('--sample_batch_size', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_ids', type=str, default='0')
args = parser.parse_args()


def worker(queue, gpu_id):

    while not queue.empty():

        superclass, image_dir, caption_dir = queue.get()

        print_box(f"Strat generating images for '{superclass}'")
        os.makedirs(image_dir)
        shutil.copy2(os.path.join(args.prior_gen_dir, 'caption_template.txt'), caption_dir)
        with open(caption_dir, 'r') as file:
            content = file.read()
        content = content.replace("{concept}", superclass)
        with open(caption_dir, 'w') as file:
            file.write(content)

        os.system(
            f'python {args.prior_gen_dir}/generate_data.py  \
                --class_tokens {superclass} \
                --class_data_dir {image_dir} \
                --prompts_file {caption_dir} \
                --num_class_images {args.num_class_images} \
                --sample_batch_size {args.sample_batch_size} \
                --seed {args.seed} \
                --gpu_ids {gpu_id}'
            )


def print_box(text):
    lines = text.split('\n')
    max_length = max(len(line) for line in lines)
    print("+" + "-" * (max_length + 2) + "+")
    for line in lines:
        print(f"| {line.ljust(max_length)} |")
    print("+" + "-" * (max_length + 2) + "+")


if __name__ == '__main__':

    queue = Queue()
    superclass_list = sorted(list(set([value for dictionary in [OBJECT_CLASSES, LIVE_SUBJECT_CLASSES, IMAGENET_LIVE_CLASSES, IMAGENET_OBJECT_CLASSES] for value in dictionary.values()])))

    for idx, superclass in enumerate(superclass_list):
        class_dir = os.path.join(args.prior_gen_dir, superclass)
        image_dir = os.path.join(class_dir, superclass)
        caption_dir = os.path.join(class_dir, 'caption.txt')
        if not os.path.exists(image_dir):
            queue.put((superclass, image_dir, caption_dir))
        else:
            pass

    processes = []
    for i in [int(x) for x in args.gpu_ids.split(',')]:
        p = Process(target=worker, args=(queue, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

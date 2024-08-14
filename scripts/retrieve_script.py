import os
import pdb
import sys

sys.path.append('/data/zhicai/code/custom-diffusion')
import argparse
from multiprocessing import Process, Queue

from utils import (
    IMAGENET_LIVE_CLASSES,
    IMAGENET_OBJECT_CLASSES,
    LIVE_SUBJECT_CLASSES,
    OBJECT_CLASSES,
)

'''
python custom_datasets/prior_real/retrieve_script.py
'''

parser = argparse.ArgumentParser()
parser.add_argument('--prior_real_dir', type=str, default="/data/zhicai/code/custom-diffusion/custom_datasets/prior_real")
parser.add_argument('--num_class_images', type=int, default=200)
args = parser.parse_args()


def worker(queue, gpu_id):

    while not queue.empty():
        superclass = queue.get()
        print_box(f"Strat retrieving images for '{superclass}'")
        os.system(
            f'python {args.prior_real_dir}/retrieve.py \
                --target_name {superclass} \
                --outpath {args.prior_real_dir} \
                --num_class_images {args.num_class_images}'
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
        class_dir = os.path.join(args.prior_real_dir, superclass)
        if not os.path.exists(class_dir):
            queue.put((superclass))
            print((superclass))
        else:
            pass

    processes = []
    for i in [int(x) for x in args.gpu_ids.split(',')]:
        p = Process(target=worker, args=(queue, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

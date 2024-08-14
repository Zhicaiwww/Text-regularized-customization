import os
import pdb
import sys

sys.path.append('/data/zhicai/code/Text-regularized-customization/custom_datasets')
import argparse
import shutil
from multiprocessing import Process, Queue

from utils import *

'''
python custom_datasets/samples_from_pretrained/sample_from_pretrained_script.py --gpu_ids 0,1,4,5
'''

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default="/data/zhicai/code/Text-regularized-customization/custom_datasets/samples_from_pretrained")
parser.add_argument('--n_img', type=int, default=20)
parser.add_argument('--gpu_ids', type=str, default='0')
args = parser.parse_args()


def worker(queue, gpu_id):

    while not queue.empty():

        superclass, class_dir, templates = queue.get()

        # Pre-process
        os.makedirs(class_dir)
        caption_dir = os.path.join(class_dir, "caption.txt")
        with open(caption_dir, "w") as file:
            for prompt in templates:
                file.write(prompt + "\n")

        print_box(f"Strat generating (pretrained) images for '{superclass}'")
        os.system(
            f"CUDA_VISIBLE_DEVICES={gpu_id} python evaluation/sample.py \
                --from_file {caption_dir} \
                --n_img {args.n_samples} \
                --outdir {class_dir} "
            )
        
        # Post-process
        try:
            os.remove(caption_dir)
            for root, dirs, files in os.walk(class_dir):
                for keyword in ["grids", "samples"]:
                    if keyword in dirs:
                        shutil.move(os.path.join(root, keyword), class_dir)
            shutil.rmtree(os.path.join(class_dir, "images"))
        except:
            pass


if __name__ == '__main__':

    queue = Queue()
    superclass_list = sorted(list(set([value for dictionary in [OBJECT_CLASSES, LIVE_SUBJECT_CLASSES, IMAGENET_LIVE_CLASSES, IMAGENET_OBJECT_CLASSES] for value in dictionary.values()])))

    for idx, superclass in enumerate(superclass_list):
        class_dir = os.path.join(args.save_dir, superclass)
        if not os.path.exists(class_dir):
            templates = parse_templates_from_superclass(superclass)
            queue.put((superclass, class_dir, templates))
        else:
            pass

    processes = []
    for i in [int(x) for x in args.gpu_ids.split(',')]:
        p = Process(target=worker, args=(queue, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

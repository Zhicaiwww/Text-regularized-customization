import subprocess
import os
import sys
sys.path.append(os.getcwd())
import multiprocessing
from utils import DEBUG_DIRS, DATA_ROOT
from multiprocessing import Queue

def run_sample(queue, args):
    while not queue.empty():
        task = queue.get()
        if task is None:
            break
        cmd = f"python base/sample.py --concept-path-name {task['data_dir_name']} --n-per-prompt {args.n_per_prompt} --gpu {args.gpu_id} --bs {args.bs}"
        subprocess.call(cmd, shell=True)

class SampleArgs:
    def __init__(self, gpu_id=0, n_per_prompt = 50, bs = 2):
        self.n_per_prompt = n_per_prompt
        self.bs = bs
        self.gpu_id = gpu_id

def main():
    gpus = [0,1,3,4,5,6,7]
    data_dir_names = list(set(os.listdir(DATA_ROOT)))
    task_queue = Queue()
    args = SampleArgs()
    for data_dir_name in data_dir_names:
        task_queue.put({"data_dir_name": data_dir_name})

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

import os
import re
from multiprocessing import Process, Queue

def worker(queue,gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    while not queue.empty():
        path,prompt_file = queue.get()
        os.system(f'python sample.py  --from-file "{prompt_file}"  --delta_ckpt "{path}" --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 1')

if __name__ == '__main__':
    gpu_ids = [2,3,6,7]
    regex = '2023'
    dir_list = os.listdir('logs/logs_toTest')
    log_dir = [os.path.join('logs/logs_toTest',x) for x in dir_list if regex in x ]

    queue = Queue()
    for dir in log_dir:
        matchObj = re.match(r'.*(tortoise_plushy|teddybear|cat|dog).*', dir, re.M|re.I)
        prompt_file = os.path.join('data',f'{matchObj.group(1)}.txt')
        # prompt_file = 'prompts/dog_test.txt'
        # for file in os.listdir(os.path.join( dir, 'checkpoints')):
        #     if file.endswith(".ckpt"):
        #         matchObj2 = re.match(r'.*epoch=0+(\d+).*.ckpt',file)
        #         if  matchObj2 is not None and matchObj2[1] is not None and int(matchObj2[1]) >= 14 :
        #             delta_ckpt = os.path.join( dir, 'checkpoints', file)
        #             queue.put((delta_ckpt,prompt_file))
        file = sorted(os.listdir(os.path.join( dir, 'checkpoints')))[-1]
        print(file)
        if file.endswith(".ckpt"):
            delta_ckpt = os.path.join( dir, 'checkpoints', file)
            queue.put((delta_ckpt,prompt_file))

    processes = []
    for i in gpu_ids:
        p = Process(target=worker, args=(queue,i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


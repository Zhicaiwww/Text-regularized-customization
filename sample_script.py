import os
import re
from multiprocessing import Process, Queue




def worker(queue,gpu_id):
    n_samples = 8
    n_iter = 2
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    while not queue.empty():
        path,prompt_file = queue.get()
        if path == 'Stable-diffusion/sd-v1-4-full-ema.ckpt':
            os.system(f'python sample.py  --from-file "{prompt_file}"  --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples {n_samples} --n_iter {n_iter}')
        else:
            os.system(f'python sample.py  --from-file "{prompt_file}"  --delta_ckpt "{path}" --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples {n_samples} --n_iter {n_iter}')

if __name__ == '__main__':
    original_sd = True
    demand_prompt_file = 'prompts/dog_test.txt'

    gpu_ids = [2,3,5]
    regex = '2023'
    no_regex = 'realReg'
    dir_list = os.listdir('logs/logs_lds11')
    log_dir = [os.path.join('logs/logs_lds11',x) for x in dir_list if regex in x and no_regex not in x ]
    log_dir = ['logs/2023-04-13T17-29-57_dog_classBias_reg0.1-0.1_scale0-0']
    queue = Queue()
    for dir in log_dir:
        matchObj = re.match(r'.*(tortoise_plushy|teddybear|cat|dog).*', dir, re.M|re.I)
        prompt_file = os.path.join('prompts',f'{matchObj.group(1)}.txt') if demand_prompt_file is None else demand_prompt_file
        # prompt_file = 'prompts/dog_test.txt'
        for file in os.listdir(os.path.join( dir, 'checkpoints')):
            if file.endswith(".ckpt"):
                matchObj2 = re.match(r'.*epoch=0+(\d+).*.ckpt',file)
                if  matchObj2 is not None and matchObj2[1] is not None and int(matchObj2[1]) >= 8 :
                    delta_ckpt = os.path.join( dir, 'checkpoints', file)
                    queue.put((delta_ckpt,prompt_file))
        # file = sorted(os.listdir(os.path.join( dir, 'checkpoints')))[-1]
        # print(file)
        # if file.endswith(".ckpt"):
        #     delta_ckpt = os.path.join( dir, 'checkpoints', file)
        #     queue.put((delta_ckpt,prompt_file))
    if original_sd:
        queue.put(('Stable-diffusion/sd-v1-4-full-ema.ckpt',prompt_file))
            
    processes = []
    for i in gpu_ids:
        p = Process(target=worker, args=(queue,i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


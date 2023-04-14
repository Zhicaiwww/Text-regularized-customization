import shlex
import os
import subprocess
import sys
def train(concept: str, use_class_bias):
    use_BLIP = False
    use_real_reg = False
    reg_k_list = [0.1]
    reg_v_list = [0.1]
    norm_k_list = [0]
    norm_v_list = [0]
    reg_prompt_file = f'data_reg/{concept}_reg.txt'  # reg prompts path for training 
    caption = f"'<new1> {concept.replace('_',' ')}'" if not use_BLIP else f'data/{concept}.txt'
    data_path = f'data/{concept}' # data path for training
    prompt_path = f'prompts/{concept}.txt' # <new1> prompts path for sampling

    new_prompt = f"'a <new1> {concept}'" # unused

    tag = "_"
    if use_BLIP:
        tag += '_BLIP'
    origin_log_dir = os.listdir('logs')
    for reg_k_scale in reg_k_list:
        for reg_v_scale in reg_v_list:
            if reg_k_scale != reg_v_scale:
                continue
            for norm_k_scale in norm_k_list:
                for norm_v_scale in norm_v_list:
                    try:
                        print("caption: ", caption)
                        print("reg_prompt_file: ", reg_prompt_file)
                        command = f"python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 4,5 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption {caption} --datapath {data_path} --modifier_token <new1> --name {concept} --no-test --reg_k_scale {reg_k_scale} --norm_k_scale {norm_k_scale} --reg_v_scale {reg_v_scale} --norm_v_scale {norm_v_scale}  --new_prompt {new_prompt} --repeat 50  --reg_prompt_file {reg_prompt_file}"
                        
                        if use_real_reg:
                            command +=f" --reg_datapath real_reg/samples_{concept}/images.txt --reg_caption real_reg/samples_{concept}/caption.txt" 
                            tag +="realReg_"
                        if use_class_bias:
                            command += f" --concept_classes {caption}"
                            tag +="classBias_"
                        command += f" --postfix {tag}reg{reg_k_scale}-{reg_v_scale}_scale{norm_k_scale}-{norm_v_scale}"
                        p = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
                        output, err = p.communicate()
                        print(output.decode("utf-8"))
                        cur_log_dir = os.listdir('logs')
                        ckpt_log_dir = [x for x in cur_log_dir if x not in origin_log_dir]
                        # ckpt_path = f'logs/{ckpt_log_dir[0]}/checkpoints/last.ckpt'
                        origin_log_dir = cur_log_dir

                    except subprocess.CalledProcessError as e:
                        print(e.output)
                        
def train_joint(concept1: str, concept2: str):
    use_BLIP = False
    reg_k_list = [0.1,1]
    reg_v_list = [0]
    norm_k_list = [0]
    norm_v_list = [0]
    reg_prompt_file = f'data_reg/{concept1}_{concept2}_reg.txt'
    if not os.path.exists(reg_prompt_file):
        reg_prompt_file1 = f'data_reg/{concept1}_reg.txt'  # reg prompts path for training 
        reg_prompt_file2 = f'data_reg/{concept2}_reg.txt'  # reg prompts path for training 
        with open(reg_prompt_file,'w') as file, open(reg_prompt_file1, 'r') as file1, open(reg_prompt_file2, 'r') as file2:
            file.write(file1.read())
            file.write('\n')
            file.write(file2.read())
    caption1 = f"'<new1> {concept1.replace('_',' ')}'" if not use_BLIP else f'data/{concept1}.txt'
    caption2 = f"'<new2> {concept2.replace('_',' ')}'" if not use_BLIP else f'data/{concept2}.txt'
    
    data_path1 = f'data/{concept1}' # data path for training
    data_path2 = f'data/{concept2}' # data path for training
    

    new_prompt = f"'a <new1> {concept1}'" # unused

    tag = f"a_{concept1}_{concept2}"
    if use_BLIP:
        tag += '_BLIP'
    origin_log_dir = os.listdir('logs')
    for reg_k_scale in reg_k_list:
        for reg_v_scale in reg_v_list:
            for norm_k_scale in norm_k_list:
                for norm_v_scale in norm_v_list:
                    try:
                        print("caption: ", caption1,' ',caption2)
                        print("reg_prompt_file: ", reg_prompt_file)
                        subprocess.check_output(shlex.split(f"python train.py --base configs/custom-diffusion/finetune_joint2.yaml -t --gpus 0,1 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption {caption1} --datapath {data_path1} --caption2 {caption2} --datapath2 {data_path2} --modifier_token <new1>+<new2> --name {concept1}_{concept2} --no-test --reg_k_scale {reg_k_scale} --norm_k_scale {norm_k_scale} --reg_v_scale {reg_v_scale} --norm_v_scale {norm_v_scale} --postfix reg_{reg_k_scale}-{reg_v_scale}_scale{norm_k_scale}-{norm_v_scale}_{tag} --reg_prompt_file {reg_prompt_file} --new_prompt {new_prompt} --repeat 50 "))
                        cur_log_dir = os.listdir('logs')
                        ckpt_log_dir = [x for x in cur_log_dir if x not in origin_log_dir]
                        ckpt_path = f'logs/{ckpt_log_dir[0]}/checkpoints/last.ckpt'
                        origin_log_dir = cur_log_dir

                    except subprocess.CalledProcessError as e:
                        print(e.output)
#// --reg_datapathgen_reg/edn_subject2_train20_realistic6
#// --reg_captionwoman

if __name__ == '__main__':
    joint = False
    if joint:
        train_joint('dog', 'cat')
    else:
        concepts = ['cat']
        for concept in concepts:
            train(concept,use_class_bias=True)
            train(concept,use_class_bias=False)

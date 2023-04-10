import os

def traverse_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.ckpt') and 'compressed' not in file:
                print(root)
                file_path = os.path.join(root, file)
                command = f"python src2/compress.py --delta_ckpt {file_path} --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt"
                os.system(command)

traverse_folder('logs')
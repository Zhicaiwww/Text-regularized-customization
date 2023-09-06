import time
import subprocess

# 定义要运行的四个 Python 脚本的文件路径
script_paths = [
    "scripts/run_mp_lora_reg-text-ewc.py",
    "scripts/run_mp_lora_reg-text-freeze.py",
    "scripts/run_mp_lora_reg-text-norm.py",
    "scripts/run_mp_lora_reg-text.py",
    "scripts/run_mp_lora_reg-priorloss.py",
]

# 定义每过多少秒运行一次
interval_hours = 2
interval_seconds = interval_hours * 60 * 60

while True:
    for script_path in script_paths:
        print(f"Running {script_path}...")
        subprocess.run(["python", script_path])  # 运行脚本
        print(f"{script_path} executed.")

        print(f"Waiting for the next {interval_hours} hours...")
        if script_path == 'scripts/run_mp_lora_reg-text-ewc.py':
            time.sleep(interval_seconds + 60 * 60)  # 等待一段时间后再继续循环
        else:
            time.sleep(interval_seconds)  # 等待一段时间后再继续循环

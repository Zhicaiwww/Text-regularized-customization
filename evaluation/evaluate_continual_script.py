import os, sys, json, argparse, json, pdb
sys.path.append('/data/zhicai/code/Text-regularized-customization')

'''
python evaluation/evaluate_continual_script.py \
    --log_dir logs/logs_continual/dog-dog1-cat-cat1-n01530575-n01531178-n02483708-n02484975 \
    --gpu_ids 4
'''

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default="logs/logs_continual/dog-dog1-dog2-dog3-dog5-dog6-dog7-dog8")
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=5)
args = parser.parse_args()
log_dir = args.log_dir


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0
    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n
    def item(self):
        return f"{self.v:.2f}"
    def len(self):
        return self.n


def custom_sort(item):
    return int(item.split('_', 1)[0][4:-1])


if __name__ == '__main__':

    A = {"mode": "A_KID A_FID"}
    F = {"mode": "F_KID F_FID"}
    modes = ["baseline", "priorReal", "priorGen", "textReg"]
    N = len(os.listdir(log_dir))

    os.system(
        f"python evaluation/evaluate_continual_A.py --log_dir {log_dir} --gpu_ids {args.gpu_ids} --batch_size {args.batch_size}"
    )

    os.system(
        f"python evaluation/evaluate_continual_F.py --log_dir {log_dir} --gpu_ids {args.gpu_ids} --batch_size {args.batch_size}"
    )

    # For A-metric
    for mode in modes:
        KID, FID = Averager(), Averager()
        task_list = sorted(os.listdir(log_dir), key=custom_sort)
        for (root, dirs, filenames) in sorted(os.walk(os.path.join(log_dir, task_list[-1]))):
            for filename in sorted(filenames):
                if (filename == "evaluate_logs.json") and mode in root:
                    with open(os.path.join(root, filename), 'r') as file:
                        data = json.load(file)
                        KID.add(float(data['clean_KID']))
                        FID.add(float(data['clean_FID']))
        assert (KID.len() == N) and (FID.len() == N)
        A[mode] = f"{KID.item()}, {FID.item()}"

    # For F-metric
    for mode in modes:
        KID, FID = Averager(), Averager()
        for task in task_list[:-1]:
            for (root, dirs, filenames) in sorted(os.walk(os.path.join(log_dir, task))):
                for filename in sorted(filenames):
                    if (filename == "evaluate_logs.json") and (mode in root):
                        with open(os.path.join(root, filename), 'r') as file:
                            data = json.load(file)
                            KID.add(float(data['clean_KID']))
                            FID.add(float(data['clean_FID']))
        assert (KID.len() == N-1) and (FID.len() == N-1)
        F[mode] = f"{KID.item()}, {FID.item()}"

    # Output
    json_path = os.path.join(log_dir, 'evaluate_logs.json')
    with open(json_path, 'w') as f:
        json.dump({"A": A, "F": F}, f, indent=2)
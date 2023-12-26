import os, sys, pdb
import pandas as pd
import csv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('log_dir', type=str)
args = parser.parse_args()

csv_file = os.path.join(args.log_dir, f"{args.log_dir.split('/')[-1]}.csv")

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    for root, dirs, files in sorted(os.walk(args.log_dir)):
        for dir in sorted(dirs):
            if any(k in dir for k in ['baseline', 'priorReal', 'priorGen', 'textReg']):
                eval_dir = os.path.join(root, dir, "eval")
                target_name, mode = root.split("/")[-1], dir.split("_")[-1]
                assert mode in ['baseline', 'priorReal', 'priorGen', 'textReg']
                row = [target_name, mode]
                try:
                    for eval_result in  sorted(os.listdir(eval_dir)):
                        json_path = os.path.join(eval_dir, eval_result, "evaluate_logs.json")
                        with open(json_path, 'r') as file:
                            row.extend(json.load(file).values())
                    writer.writerow(row)
                except:
                    pass

df = pd.read_csv(csv_file, header=None)
num_cols = df.shape[1]
column_names = ['Target_name', 'Mode']
column_names.extend(['Metric{}'.format(i) for i in range(1, num_cols - 1)])
df.columns = column_names

result = df.groupby('Mode').mean()
result.to_csv(csv_file, mode='a', header=False)

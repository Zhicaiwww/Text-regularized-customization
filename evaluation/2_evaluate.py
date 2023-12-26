import os, sys, argparse, json
sys.path.append("/data/zhicai/code/Text-regularized-customization")
from cleanfid import fid
from evaluation_pipe import *

'''
CUDA_VISIBLE_DEVICES=1 python evaluation/2_evaluate.py \
    --dir1 eval/samples_231030-110006/samples1 \
    --dir2 eval/samples_231030-110006/samples2
'''

parser = argparse.ArgumentParser()
parser.add_argument('--dir1', type=str, default="")
parser.add_argument('--dir2', type=str, default="")
args = parser.parse_args()
device = torch.device("cuda")

dir1, dir2 = args.dir1, args.dir2
clean_FID = fid.compute_fid(dir1, dir2, batch_size=128, use_dataparallel=False)
# region [KID Evaluation]
feat_model = build_feature_extractor(mode="clean", use_dataparallel=False, device=device)
source_feats = get_folder_features(dir1, feat_model, verbose=False)
gen_feats = get_folder_features(dir2, feat_model, verbose=False)
clean_KID = 1e3 * kernel_distance(source_feats, gen_feats)
# endregion
logs = {
    'clean_KID': f"{clean_KID:.2f}",
    'clean_FID': f"{clean_FID:.2f}",
}
print(logs)
json_path = os.path.join(os.path.dirname(dir2), 'evaluate_logs.json')
with open(json_path, 'w') as f:
    json.dump(logs, f)
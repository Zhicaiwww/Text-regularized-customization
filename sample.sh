#!/bin/bash
ARRAY=(logs/2023-03-27T08-55-01_dog/checkpoints/epoch=000009.ckpt logs/2023-03-27T08-55-01_dog/checkpoints/epoch=000014.ckpt)

for i in "${ARRAY[@]}"
do
CUDA_VISIBLE_DEVICES=7 python sample_script.py --root_dir $i
done
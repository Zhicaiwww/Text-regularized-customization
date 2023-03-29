#!/bin/bash
ARRAY=(logs/2023-03-27T08-55-01_dog/checkpoints/epoch=000009.ckpt logs/2023-03-27T08-55-01_dog/checkpoints/epoch=000014.ckpt)

for i in "${ARRAY[@]}"
do
for file in "$dir"/*
do
    # Check if the file is a regular file
    if [ -f "$file" ]
    then
        # Add the file to the array
        files+=("$file")
    fi
done
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt 'photo of a <new1> dog on the beach' --delta_ckpt $i --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
done
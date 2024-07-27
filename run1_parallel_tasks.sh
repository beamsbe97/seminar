#!/bin/bash

# Ensure GNU Parallel is installed
command -v parallel >/dev/null 2>&1 || { echo >&2 "GNU Parallel is required but it's not installed. Aborting."; exit 1; }

# Define the commands to run in parallel
cmd1="python train_vp_detection.py --mode spimg_spmask --output_dir output_samples --device cuda:2 --base_dir ./pascal-5i --batch-size 1 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --simidx 1 --dropout 0.25"
cmd2="python train_vp_detection.py --mode spimg_spmask --output_dir output_samples --device cuda:2 --base_dir ./pascal-5i --batch-size 1 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --simidx 2 --dropout 0.25"
cmd3="python train_vp_detection.py --mode spimg_spmask --output_dir output_samples --device cuda:3 --base_dir ./pascal-5i --batch-size 1 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --simidx 4 --dropout 0.25"
cmd4="python train_vp_detection.py --mode spimg_spmask --output_dir output_samples --device cuda:3 --base_dir ./pascal-5i --batch-size 1 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --simidx 8 --dropout 0.25"

# Run the commands in parallel
parallel ::: "$cmd1" "$cmd2" "$cmd3" "$cmd4"

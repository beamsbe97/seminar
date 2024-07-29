#!/bin/bash

# data_root=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL
data_root=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL

# Define the commands to run in parallel
cmd1="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:0 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 0.9"
cmd2="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:1 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 1.0"
cmd3="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:2 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 1.1"
cmd4="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:3 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 1.2"
cmd5="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:4 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 1.3"
cmd6="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:5 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 1.4"
cmd7="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:6 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 1.5"
cmd8="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:7 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 1.6"

# Function to kill all background processes
terminate() {
    echo "Terminating background processes..."
    kill $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8
    exit 1
}

# Trap SIGINT (Ctrl+C) signal to terminate background processes
trap terminate SIGINT

# Run the commands in parallel
$cmd1 &
pid1=$!
$cmd2 &
pid2=$!
$cmd3 &
pid3=$!
$cmd4 &
pid4=$!
$cmd5 &
pid5=$!
$cmd6 &
pid6=$!
$cmd7 &
pid7=$!
$cmd8 &
pid8=$!

# Wait for all background tasks to complete
wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5
wait $pid6
wait $pid7
wait $pid8

echo "All tasks completed."
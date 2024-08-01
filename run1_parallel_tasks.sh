#!/bin/bash

# data_root=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL
data_root=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL

# Define the commands to run in parallel
cmd1="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:0 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 150 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 0.1"
cmd2="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:1 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 150 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 2.0"
cmd3="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:2 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 150 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 3.0"
cmd4="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:3 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 150 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 0.4"
cmd5="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:4 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 150 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 0.5"
cmd6="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:5 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 150 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 0.6"
cmd7="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:6 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 150 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 0.7"
cmd8="python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:7 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 150 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 0.8"

# Function to kill all background processes
terminate() {
    echo "Terminating background processes..."
    kill $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8
    exit 1
}

# Trap SIGINT (Ctrl+C) signal to terminate background processes
trap terminate SIGINT

pids=()

for cmd in "${cmds[@]}"; do
    echo "Running: $cmd"
    $cmd &
    pids+=($!)
    sleep 600  # wait for 10 minutes before starting the next command
done

# Wait for all background tasks to complete
for pid in "${pids[@]}"; do
    wait $pid
done


echo "All tasks completed."
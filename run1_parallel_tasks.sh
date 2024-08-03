#!/bin/bash

# data_root=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL
data_root=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL

cmds=(
    "python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:0 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.03 --epoch 150 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 0.7"
    "python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:1 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.03 --epoch 150 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 0.8"
    "python3 train_vp_detection.py --mode spimg_spmask --output_dir ${data_root}/output/logs/ --device cuda:2 --base_dir ${data_root}/pascal-5i/ --batch-size 16 --lr 0.03 --epoch 150 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt ${data_root}/weights/checkpoint-1000.pth --vq_ckpt_dir ${data_root}/weights/vqgan --save_base_dir ${data_root}/  --simidx 20  --dropout 0.25 --sigma 0.9"
)

# Function to kill all background processes
terminate() {
    echo "Terminating background processes..."
    for pid in "${pids[@]}"; do
        kill $pid
    done
    exit 1
}

# Trap SIGINT (Ctrl+C) signal to terminate background processes
trap terminate SIGINT

pids=()

# echo "begin  "
# Initial delay for the first command (720 seconds = 12 minutes)
delay=1

for cmd in "${cmds[@]}"; do
    echo "Running: $cmd"
    $cmd &
    pids+=($!)
    sleep $delay  # wait for the specified delay before starting the next command
    delay=$((delay + 1))  # Increment delay by 1 minute (60 seconds) for the next command
done

# echo "ltt"

# Wait for all background tasks to complete
for pid in "${pids[@]}"; do
    wait $pid
done


echo "All tasks completed."
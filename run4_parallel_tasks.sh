#!/bin/bash

# Define the commands to run in parallel
cmd1="python Codes/train_vp_detection.py --mode spimg_spmask --output_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/output/logs/ --device cuda:0 --base_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/weights/checkpoint-1000.pth --save_base_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/  --simidx 20  --dropout 0.25 --sigma 1.3"
cmd2="python Codes/train_vp_detection.py --mode spimg_spmask --output_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/output/logs/ --device cuda:1 --base_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/weights/checkpoint-1000.pth --save_base_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/  --simidx 20  --dropout 0.25 --sigma 1.4"
cmd3="python Codes/train_vp_detection.py --mode spimg_spmask --output_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/output/logs/ --device cuda:2 --base_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/weights/checkpoint-1000.pth --save_base_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/  --simidx 20  --dropout 0.25 --sigma 1.5"
cmd4="python Codes/train_vp_detection.py --mode spimg_spmask --output_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/output/logs/ --device cuda:3 --base_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/pascal-5i/ --batch-size 16 --lr 0.02 --epoch 400 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model Prompt --p-eps 1 --ckpt /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/weights/checkpoint-1000.pth --save_base_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/wangjinpeng08/tianci/VisualICL/  --simidx 20  --dropout 0.25 --sigma 1.6"

# Function to kill all background processes
terminate() {
    echo "Terminating background processes..."
    kill $pid1 $pid2 $pid3 $pid4
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

# Wait for all background tasks to complete
wait $pid1
wait $pid2
wait $pid3
wait $pid4

echo "All tasks completed."
#!/bin/bash

python3 Codes/val_vp_coloring.py \
 --mode spimg_spmask \
 --output_dir Data/output/logs\
 --device cuda:0\
 --base_dir Data/imagenet/ \
 --batch-size 16 \
 --lr 0.03\
 --epoch 7\
 --arr a1\
 --vp-model Prompt\
 --p-eps 1\
 --ckpt Data/weights/checkpoint-1000.pth\
 --vq_ckpt_dir Data/weights/vqgan\
 --save_base_dir Data/\
 --simidx 16\
 --dropout 0.25\
 --align_q 0 \
 --save_model_path Data/ckpt/Col_K_16.pth
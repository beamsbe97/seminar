#!/bin/bash

python3 Codes/val_vp_detection.py \
 --mode spimg_spmask \
 --output_dir VisualICL/output/logs\
 --device cuda:4\
 --base_dir VisualICL/pascal-5i/ \
 --batch-size 8 \
 --lr 0.03\
 --epoch 150\
 --arr a1\
 --vp-model Prompt\
 --p-eps 1\
 --ckpt VisualICL/weights/checkpoint-1000.pth\
 --vq_ckpt_dir VisualICL/weights/vqgan\
 --save_base_dir VisualICL/\
 --simidx 16\
 --dropout 0.25\
 --align_q 0 \
 --save_model_path VisualICL/ckpt/Det_K_16.pth
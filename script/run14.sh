#!/bin/bash

python3 Codes/val_vp_segmentation_for_coco.py \
 --fold 0 \
 --mode spimg_spmask \
 --output_dir Data/output/logs\
 --device cuda:0\
 --base_dir Data/coco/ \
 --batch-size 8 \
 --lr 0.03\
 --epoch 150\
 --arr a1\
 --vp-model Prompt\
 --p-eps 1\
 --ckpt Data/weights/checkpoint-1000.pth\
 --vq_ckpt_dir Data/weights/vqgan\
 --save_base_dir Data/\
 --simidx 16\
 --dropout 0.25\
 --align_q 0 \
 --save_model_path Data/ckpt/Coco_K_16_Folder_0.pth
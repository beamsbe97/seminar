#!/bin/bash -l

#SBATCH --job-name=test_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=70G
#SBATCH --time=7:00:00
#SBATCH --partition=gpu-a100-80g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1

module purge

module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load GCC/11.3.0
module load Miniforge3

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo $CUDA_VISIBLE_DEVICES
nvidia-smi

conda init
conda activate condenser

export CWD=$(pwd)
echo "[$SHELL] CWD: "$CWD

echo "running calculate_pre_feature_for_support.py val"

/zfsstore/user/s4561341/.conda/envs/condenser/bin/python -m Codes.val_vp_segmentation \
 --fold 1\
 --mode spimg_spmask\
 --output_dir Data/output/logs/\ 
 --device cuda:0\ 
 --base_dir Data/pascal-5i/\ 
 --batch-size 8\
 --lr 0.03\
 --epoch 150\
 --arr a1\
 --vp-model Prompt\
 --p-eps 1\
 --ckpt Data/weights/checkpoint-1000.pth\
 --vq_ckpt_dir Data/weights/vqgan\
 --save_base_dir Data/\
 --simidx 1\
 --dropout 0.25\
 --align_q 0 \
 --save_model_path SAVE_MODEL_PATH
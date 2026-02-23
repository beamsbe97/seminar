#!/bin/bash -l

#SBATCH --job-name=test_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

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

echo "running feature_extractor_segmentation.py val"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python $HOME/seminar/Codes/tools/feature_extractor_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level val

echo "running feature_extractor_segmentation.py trn"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python $HOME/seminar/Codes/tools/feature_extractor_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level trn

echo "running calculate_similariity.py val"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python $HOME/seminar/Codes/tools/calculate_similariity.py features_vit-laion2b_pixel-level val trn
echo "running calculate_similariity.py trn"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python $HOME/seminar/Codes/tools/calculate_similariity.py features_vit-laion2b_pixel-level trn trn

echo "running calculate_pre_feature_for_query.py"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python $HOME/seminar/Codes/tools/calculate_pre_feature_for_query.py

echo "running calculate_pre_feature_for_support.py"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python $HOME/seminar/Codes/tools/calculate_pre_feature_for_support.py
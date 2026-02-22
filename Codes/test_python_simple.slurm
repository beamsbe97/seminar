#!/bin/bash -l

#SBATCH --job-name=test_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=5G
#SBATCH --time=00:02:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module purge

module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load GCC/11.3.0
module load Miniforge3

nvidia-smi

conda init

conda activate condenser
echo $CONDA_DEFAULT_ENV

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES

# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python tools/feature_extractor_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level val
echo "[$SHELL] ## Script finished"

echo "Active env: $CONDA_DEFAULT_ENV"
which python
python -V
python -c "import sys; print(sys.executable)"
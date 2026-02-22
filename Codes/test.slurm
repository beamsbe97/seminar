#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --mail-user="<your-email-address>"
#SBATCH --mail-type="ALL"
#SBATCH --time=00:01:00
#SBATCH --partition=testing
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:tesla_t4:1

# making sure we start with a clean module environment
module purge

echo "## Starting GPU test on $HOSTNAME"

echo "## Loading module"
module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load GCC/11.3.0

TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"

echo "## Available CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

echo "## Running test"
$HOME/cuda-samples/Samples/6_Performance/transpose/transpose

echo "## Test finished. Goodbye"
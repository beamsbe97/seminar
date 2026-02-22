#!/bin/bash -l
#SBATCH --job-name=test_python_simple
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# load modules (assuming you start from the default environment)

# in case the default settings change
module load Miniforge3
conda init

conda activate condenser
echo $CONDA_DEFAULT_ENV

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"

# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python tools/feature_extractor_segmentation.py
echo "[$SHELL] ## Script finished"

echo "Active env: $CONDA_DEFAULT_ENV"
which python
python -V
python -c "import sys; print(sys.executable)"
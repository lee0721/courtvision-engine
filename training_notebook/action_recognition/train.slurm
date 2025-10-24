#!/bin/bash
#SBATCH --job-name=action_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00

# Correctly load Conda functionality
module load anaconda3/2022.10
source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-zen2/gcc-13.2.0/anaconda3-2022.10*/etc/profile.d/conda.sh
conda activate yolov8  # Ensure the environment name matches the one where the packages are installed

# Change to the working directory and ensure logs exist
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Run the training
python train.py
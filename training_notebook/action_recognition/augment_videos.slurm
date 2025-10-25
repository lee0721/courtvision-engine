#!/bin/bash
#SBATCH --job-name=augment_videos
#SBATCH --output=logs/augment_%j.out
#SBATCH --error=logs/augment_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00

# Load the anaconda module provided by the HPC system
module load anaconda3/2022.10-gcc-13.2.0

# Activate the conda virtual environment (e.g., named 'bball')
source activate bball

# Change to the working directory
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Run data augmentation
python augment_videos.py
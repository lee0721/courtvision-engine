#!/bin/bash
#SBATCH --job-name=action_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00

# Load the anaconda module provided by the HPC system
module load anaconda3/2022.10-gcc-13.2.0

# Activate the conda virtual environment (e.g., named 'bball')
source activate bball

# Change to the working directory and ensure logs exist
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Run the training
python train.py
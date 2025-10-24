#!/bin/bash
#SBATCH --job-name=train_player_yolov8
#SBATCH --output=logs/train_player_yolov8_%j.out
#SBATCH --error=logs/train_player_yolov8_%j.err
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00

# Correctly load Conda functionality
module load anaconda3/2022.10
source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-zen2/gcc-13.2.0/anaconda3-2022.10*/etc/profile.d/conda.sh
conda activate yolov8  # Ensure the environment name matches the one where the packages are installed

# Change to the working directory and ensure logs exist
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Train the model
echo "Start training player model with YOLOv8..."
python train_player_yolov8.py  # Use relative path

# After training, zip the results, logs, and checkpoints
echo "Zipping training results, logs, and checkpoints..."

# Zip the runs, logs, and model_checkpoints folders
zip -r player_training_results.zip runs logs model_checkpoints

echo "Training complete and zipped results are saved as player_training_results.zip."
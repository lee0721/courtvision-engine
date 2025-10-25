#!/bin/bash
#SBATCH --job-name=train_ball_yolov8
#SBATCH --output=logs/train_ball_yolov8_%j.out
#SBATCH --error=logs/train_ball_yolov8_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00

# Load the anaconda module provided by the HPC system
module load anaconda3/2022.10-gcc-13.2.0

# Activate the conda virtual environment (e.g., named 'bball')
source activate bball

# Change to the working directory and ensure logs exist
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Train the model
echo "Start training ball model with YOLOv8..."
python train_ball_yolov8.py  # Use relative path

# After training, zip the results, logs, and checkpoints
echo "Zipping training results, logs, and checkpoints..."

# Zip the runs, logs, and model_checkpoints folders
zip -r ball_training_results.zip runs logs model_checkpoints

echo "Training complete and zipped results are saved as ball_training_results.zip."
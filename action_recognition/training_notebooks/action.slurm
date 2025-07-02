#!/bin/bash
#SBATCH --job-name=action_recognition
#SBATCH --output=logs/action_%j.out
#SBATCH --error=logs/action_%j.err
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00

# 載入 Conda 模組並啟用環境
module load anaconda3/2022.10
source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-zen2/gcc-13.2.0/anaconda3-2022.10*/etc/profile.d/conda.sh
conda activate yolov8

# 切換到執行目錄
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# 執行動作辨識腳本
python action.py
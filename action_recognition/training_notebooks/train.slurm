#!/bin/bash
#SBATCH --job-name=action_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00

# ✅ 正確載入 Conda 功能
module load anaconda3/2022.10
source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-zen2/gcc-13.2.0/anaconda3-2022.10*/etc/profile.d/conda.sh
conda activate yolov8  # 🔁 確保是你安裝套件的環境名稱

# ✅ 進入工作目錄並確保 logs 存在
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# ✅ 執行訓練 
python train.py
#!/bin/bash
#SBATCH --job-name=basketball_analysis
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1              # 要 1 張 GPU
#SBATCH --mem=32G                 # 記憶體
#SBATCH --cpus-per-task=4         # 分配 CPU core 數
#SBATCH --time=02:00:00           # 最長執行時間
#SBATCH --output=job_output.log   # 輸出 log 檔案

# 進入工作目錄
cd ~/basketball_analysis

# 載入 HPC 提供的 anaconda 模組
module load anaconda3/2022.10-gcc-13.2.0

# 啟用你用 conda 建立的虛擬環境（例如叫 bball）
source activate bball

# 執行你的 Python 程式
python main.py input_videos/video_1.mp4 --output_video output_videos/video_result_1.mp4
#python main.py input_videos/test_1.mp4 --output_video output_videos/test_result_1.mp4
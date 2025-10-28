#!/bin/bash
#SBATCH --job-name=courtvision_engine
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=32G                 
#SBATCH --cpus-per-task=4         
#SBATCH --time=02:00:00           
#SBATCH --output=job_output.log   
#SBATCH --error=job_error.log
# Change to the working directory
cd ~/courtvision-engine

# Load the anaconda module provided by the HPC system
module load anaconda3/2022.10-gcc-13.2.0

# Activate the conda virtual environment (e.g., named 'bball')
source activate bball

# Run the Python program
#python main.py input_videos/video_1.mp4 --output_video output_videos/video_result_1.mp4
#python main.py input_videos/video_2.mp4 --output_video output_videos/video_result_2.mp4
#python main.py input_videos/test_1.mp4 --output_video output_videos/test_result_1.mp4
python main.py input_videos/test_2.mp4 --output_video output_videos/test_result_2.mp4
#python main.py input_videos/test_3.mp4 --output_video output_videos/test_result_3.mp4

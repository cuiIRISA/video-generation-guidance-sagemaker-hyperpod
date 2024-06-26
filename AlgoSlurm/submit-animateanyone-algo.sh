#!/bin/bash
#SBATCH --job-name=video-gen
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -o video-gen-stage-1.out

export OMP_NUM_THREADS=1

# Activate the conda environment
source ~/miniconda3/bin/activate
conda activate videogen

srun accelerate launch train_stage_1.py --config configs/train/stage1.yaml
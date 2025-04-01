#!/bin/bash
#SBATCH --job-name=ML-training
#SBATCH --mail-user=yuanmis1@uci.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=dtobias_lab_gpu
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A30:2
#SBATCH --output=out/ML-training_KL_improved_2.out

./trainit.sh

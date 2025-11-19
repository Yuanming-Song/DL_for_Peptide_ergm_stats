#!/bin/bash
#SBATCH --job-name=gen-rl-lr1e5-bs32-ep100
#SBATCH --mail-type=FAIL
#SBATCH --account=dtobias_lab_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:A30:1
#SBATCH --out=logs/%x.out
#SBATCH --err=logs/%x.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the RL training script
./train_generative_model_rl.sh


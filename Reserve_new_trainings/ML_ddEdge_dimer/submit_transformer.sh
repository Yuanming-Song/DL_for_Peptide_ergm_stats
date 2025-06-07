#!/bin/bash
#SBATCH --job-name=transformer-regression
#SBATCH --mail-type=FAIL
#SBATCH --account=dtobias_lab_gpu
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A30:2
#SBATCH --output=out/transformer_regression_%j.out

./train_transformer.sh 
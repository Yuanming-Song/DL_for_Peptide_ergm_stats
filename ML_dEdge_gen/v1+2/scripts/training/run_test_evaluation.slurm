#!/bin/bash
#SBATCH --job-name=gen-test-eval
#SBATCH --mail-type=FAIL
#SBATCH --account=dtobias_lab_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:A30:1
#SBATCH --out=logs/%x.out
#SBATCH --err=logs/%x.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module load anaconda/2024.06
module load gcc/11.2.0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dl_py309

# Run test evaluation
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training

python -u run_test_evaluation.py

echo "Test evaluation completed!"




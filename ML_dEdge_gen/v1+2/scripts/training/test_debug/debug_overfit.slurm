#!/bin/bash
#SBATCH --job-name=debug_overfit
#SBATCH --mail-type=FAIL
#SBATCH --account=dtobias_lab_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:A30:1
#SBATCH --out=test_debug/debug_overfit_%j.out
#SBATCH --err=test_debug/debug_overfit_%j.err

# Load modules
module load anaconda/2024.06
module load gcc/11.2.0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dl_py309

# Run debug script
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training

echo "Starting overfitting test with all bugs fixed..."
echo "Results will be written to: test_debug/overfit_results.txt"
echo "="

python -u test_debug/debug_overfit.py

echo "="
echo "Test complete. Check test_debug/overfit_results.txt for results."

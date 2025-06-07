#!/bin/bash

# Default iteration number if not provided
ITERATION=${1:-1}
echo "Submitting job for iteration ${ITERATION}"

# Create job name with iteration number
JOB_NAME="transformer-regression-iter${ITERATION}"

#SBATCH --mail-type=FAIL
#SBATCH --account=dtobias_lab_gpu
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A30:2

# Create output directory if it doesn't exist
mkdir -p out

# Submit the job
sbatch \
    --job-name=${JOB_NAME} \
    --output=out/${JOB_NAME}_%j.out \
    --wrap="cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_ddEdge/scripts/training && ./train_transformer.sh ${ITERATION}"

echo "Job submitted for iteration ${ITERATION}"
echo "Check job status with: squeue -u $USER"

#now perdict
./predict_seq.sh
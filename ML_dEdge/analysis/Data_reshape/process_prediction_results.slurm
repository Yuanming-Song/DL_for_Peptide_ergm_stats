#!/bin/bash
#SBATCH --job-name=process-pred
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yuanmis1@uci.edu
#SBATCH --account=dtobias_lab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --out=logs/%x.out
#SBATCH --err=logs/%x.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules
module load R
# Run the R script
Rscript process_prediction_results.R 
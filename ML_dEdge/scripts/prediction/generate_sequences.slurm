#!/bin/bash
#SBATCH --job-name=generate-seqs
#SBATCH --mail-type=FAIL
#SBATCH --account=dtobias_lab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --out=logs/%x.out
#SBATCH --err=logs/%x.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules
module load anaconda/2024.06
module load gcc/11.2.0

# Change to the project directory
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/scripts/prediction
source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh

# Activate the conda environment
conda activate dl_py309

# Create output directory
PREDICTION_DIR="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/data/iteration2/Prediction"
mkdir -p ${PREDICTION_DIR}

# Generate sequences for lengths 5-10
for LENGTH in {5..10}; do
    echo "Processing ${LENGTH}-mer sequences..."
    
    # For each position in the sequence
    for POS in $(seq 0 $((LENGTH-1))); do
        echo "  Generating sequences with C at position ${POS}..."
        
        # Define output file
        OUTPUT_FILE="${PREDICTION_DIR}/dEdge_predict_seqs_${LENGTH}mer_pos${POS}.csv"
        
        # Generate sequences
        python generate_cysteine_sequences.py \
            --length ${LENGTH} \
            --include_positions ${POS} \
            --batch_size 10000 \
            --output_file "${OUTPUT_FILE}"
            
        echo "  Completed position ${POS}"
    done
    
    echo "Completed ${LENGTH}-mer sequences"
done 
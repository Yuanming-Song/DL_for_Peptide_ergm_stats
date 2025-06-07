#!/bin/bash
#SBATCH --job-name=transformer-iter2
#SBATCH --mail-type=FAIL
#SBATCH --account=dtobias_lab_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A30:2
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

# Base directories
PREDICTION_DIR="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/data/iteration2/Prediction"
SOURCE_DIR="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/scripts/prediction/Sequential_Peptides_edges"

# Create prediction directory if it doesn't exist
mkdir -p ${PREDICTION_DIR}

# Move only files for lengths 5-7
for length in {5..7}; do
    # Move sequence files
    mv ${SOURCE_DIR}/dEdge_predict_seqs_${length}mer_pos*.csv ${PREDICTION_DIR}/
    
    # Remove existing result files for this length
    rm -f ${PREDICTION_DIR}/dEdge_predict_seqs_${length}mer_pos*_results.csv
done

# Model parameters
SRC_VOCAB_SIZE=21
SRC_LEN=10
DROPOUT=0.1
MAX=3.0
MIN=-1.0

# Loop through sequence files for lengths 5-7
for length in {5..7}; do
    for SEQ_FILE in ${PREDICTION_DIR}/dEdge_predict_seqs_${length}mer_pos*.csv; do
        # Skip if not a sequence file
        if [[ ! $SEQ_FILE =~ dEdge_predict_seqs_[0-9]+mer_pos[0-9]+\.csv$ ]]; then
            continue
        fi
        
        # Get corresponding result file name
        RESULT_FILE="${SEQ_FILE%.csv}_results.csv"
        
        echo "Processing ${SEQ_FILE}..."
        
        # Run prediction
        python predict_seq.py \
            --task_type "Regression" \
            --src_vocab_size ${SRC_VOCAB_SIZE} \
            --src_len ${SRC_LEN} \
            --model "Transformer" \
            --dropout ${DROPOUT} \
            --max ${MAX} \
            --min ${MIN} \
            --batch_size 50 \
            --model_path "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/models/iteration2/Transformer_curriculum_lr_0.2_bs_1024.pt" \
            --input_file "${SEQ_FILE}" \
            --output_file "${RESULT_FILE}"
            
        echo "Completed prediction for ${SEQ_FILE}"
    done
done 
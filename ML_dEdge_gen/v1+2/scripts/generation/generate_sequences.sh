#!/bin/bash

# Generation script for creating sequences with target dEdge values
# Usage: ./generate_sequences.sh <model_path> <dEdge_min> <dEdge_max> <seq_length_min> <seq_length_max> <num_sequences>

# Load required modules
module load anaconda/2024.06
module load gcc/11.2.0

# Activate conda environment
source activate dl_py309

# Parse arguments
MODEL_PATH=$1
DEDGE_MIN=$2
DEDGE_MAX=$3
SEQ_LENGTH_MIN=$4
SEQ_LENGTH_MAX=$5
NUM_SEQUENCES=$6
TEMPERATURE=${7:-1.0}  # Default to 1.0 if not provided

# Define base path once at the beginning
PROJECT_ROOT="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide"
BASE_DIR="${PROJECT_ROOT}/ML_dEdge_gen/v1+2"
SCRIPT_DIR="${BASE_DIR}/scripts/generation"

# Change to script directory
cd "${SCRIPT_DIR}"

# Run the generation script
python generate_sequences.py \
    --model_path "${MODEL_PATH}" \
    --dEdge_min ${DEDGE_MIN} \
    --dEdge_max ${DEDGE_MAX} \
    --seq_length_min ${SEQ_LENGTH_MIN} \
    --seq_length_max ${SEQ_LENGTH_MAX} \
    --num_sequences ${NUM_SEQUENCES} \
    --temperature ${TEMPERATURE} \
    --src_len 10 \
    --src_vocab_size 21 \
    --d_model 512 \
    --d_ff 2048 \
    --d_k 64 \
    --d_v 64 \
    --n_layers 6 \
    --n_heads 8 \
    --dropout 0.1

echo "Generation completed!"

#!/bin/bash

# Training script for conditional generative model
# This script trains a generative model that can generate sequences based on dEdge values

# Load required modules
module load anaconda/2024.06
module load gcc/11.2.0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dl_py309

# Model architecture parameters
D_MODEL=768
D_FF=3072
N_LAYERS=8
N_HEADS=12
SRC_VOCAB_SIZE=21
SRC_LEN=10

# Training parameters
EPOCHS=400
BATCH_SIZE=512
LEARNING_RATE=0.003
SEED=42

# Define base path once at the beginning
PROJECT_ROOT="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide"
BASE_DIR="${PROJECT_ROOT}/ML_dEdge_gen/v1+2"
SCRIPT_DIR="${BASE_DIR}/scripts/training"

# Change to script directory
cd "${SCRIPT_DIR}"

# Run the training script
python train_generative_model.py \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --src_vocab_size ${SRC_VOCAB_SIZE} \
    --src_len ${SRC_LEN} \
    --d_model ${D_MODEL} \
    --d_ff ${D_FF} \
    --d_k 64 \
    --d_v 64 \
    --n_layers ${N_LAYERS} \
    --n_heads ${N_HEADS} \
    --dropout 0.1

# Print summary
echo "Training completed with parameters:"
echo "Model: ConditionalGenerator"
echo "Architecture: d_model=${D_MODEL}, d_ff=${D_FF}, n_layers=${N_LAYERS}, n_heads=${N_HEADS}"
echo "Training: epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, lr=${LEARNING_RATE}"
echo "Data: Combined iteration1 (v1) and iteration2 (v2)"


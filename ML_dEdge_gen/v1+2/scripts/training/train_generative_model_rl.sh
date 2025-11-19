#!/bin/bash

# RL fine-tuning script for conditional generative model
# This script fine-tunes a pre-trained MLE generator using RL with ML_dEdge iteration 2 as critic

# Load required modules
module load anaconda/2024.06
module load gcc/11.2.0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dl_py309

# RL training parameters
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-5
REWARD_WEIGHT=1.0
ENTROPY_WEIGHT=0.01
BASELINE_DECAY=0.99
TEMPERATURE=1.0

# Define base path
PROJECT_ROOT="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide"
BASE_DIR="${PROJECT_ROOT}/ML_dEdge_gen/v1+2"
SCRIPT_DIR="${BASE_DIR}/scripts/training"
MODEL_DIR="${BASE_DIR}/models"

# MLE model path (update this to point to your trained MLE model)
# Example: ConditionalGenerator_v1v2_minmax_lr_0.003_bs_512.pt
MLE_MODEL_PATH="${MODEL_DIR}/ConditionalGenerator_v1v2_minmax_lr_0.003_bs_512.pt"

# Critic model path (ML_dEdge iteration 2)
CRITIC_MODEL_PATH="${PROJECT_ROOT}/ML_dEdge/models/iteration2/Transformer_curriculum_lr_0.2_bs_1024.pt"

# Change to script directory
cd "${SCRIPT_DIR}"

# Check if MLE model exists
if [ ! -f "${MLE_MODEL_PATH}" ]; then
    echo "ERROR: MLE model not found at ${MLE_MODEL_PATH}"
    echo "Please update MLE_MODEL_PATH in this script to point to your trained MLE model"
    exit 1
fi

# Check if critic model exists
if [ ! -f "${CRITIC_MODEL_PATH}" ]; then
    echo "ERROR: Critic model not found at ${CRITIC_MODEL_PATH}"
    exit 1
fi

# Run the RL training script
python -u train_generative_model_rl.py \
    --model_path "${MLE_MODEL_PATH}" \
    --ml_dedge_model_path "${CRITIC_MODEL_PATH}" \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --reward_weight ${REWARD_WEIGHT} \
    --entropy_weight ${ENTROPY_WEIGHT} \
    --baseline_decay ${BASELINE_DECAY} \
    --temperature ${TEMPERATURE} \
    --experiment_name "generative_rl"

# Print summary
echo "RL fine-tuning completed with parameters:"
echo "MLE Model: ${MLE_MODEL_PATH}"
echo "Critic Model: ${CRITIC_MODEL_PATH}"
echo "RL Training: epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, lr=${LEARNING_RATE}"
echo "RL Parameters: reward_weight=${REWARD_WEIGHT}, entropy_weight=${ENTROPY_WEIGHT}, baseline_decay=${BASELINE_DECAY}"


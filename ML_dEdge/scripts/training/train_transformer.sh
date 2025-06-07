#!/bin/bash

# Default iteration number if not provided
ITERATION=${1:-1}
echo "Running iteration ${ITERATION}"

# Load necessary modules
module load anaconda/2024.06
module load gcc/11.2.0

# Change to the project directory
source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh

# Activate the conda environment
conda activate dl_py309

# Base directories for this iteration
BASE_DIR="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_ddEdge"
ITER_DIR="${BASE_DIR}/data/iteration${ITERATION}"
MODEL_DIR="${BASE_DIR}/models/iteration${ITERATION}"
RESULTS_DIR="${ITER_DIR}/results_transformer"

# Create directories for this iteration
mkdir -p ${ITER_DIR}/training
mkdir -p ${MODEL_DIR}
mkdir -p ${RESULTS_DIR}
mkdir -p out

# Model architecture parameters
D_MODEL=512
D_FF=2048
N_LAYERS=4
N_HEADS=4
SRC_VOCAB_SIZE=21
SRC_LEN=10
D_K=32
D_V=32
DROPOUT=0.1

# Training parameters
EPOCHS=200
BATCH_SIZE=64
LEARNING_RATE=0.22
SEED=42

# Run the training script with updated parameters
python main_seq_clean.py \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --src_vocab_size $SRC_VOCAB_SIZE \
    --src_len $SRC_LEN \
    --d_k $D_K \
    --d_v $D_V \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --seed $SEED \
    --iteration $ITERATION \
    --model_dir $MODEL_DIR \
    --results_dir $RESULTS_DIR \
    --data_dir ${ITER_DIR}/training

# Print the parameters used
echo "Training completed with the following parameters:"
echo "----------------------------------------"
echo "Iteration: ${ITERATION}"
echo "Directories:"
echo "  Model Directory: ${MODEL_DIR}"
echo "  Results Directory: ${RESULTS_DIR}"
echo "  Data Directory: ${ITER_DIR}/training"
echo ""
echo "Model Architecture:"
echo "  Embedding Dimension: $D_MODEL"
echo "  Feed Forward Dimension: $D_FF"
echo "  Number of Layers: $N_LAYERS"
echo "  Number of Heads: $N_HEADS"
echo "  Key Dimension: $D_K"
echo "  Value Dimension: $D_V"
echo "  Dropout Rate: $DROPOUT"
echo "  Source Vocabulary Size: $SRC_VOCAB_SIZE"
echo "  Sequence Length: $SRC_LEN"
echo ""
echo "Training Parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Random Seed: $SEED"
echo "----------------------------------------" 
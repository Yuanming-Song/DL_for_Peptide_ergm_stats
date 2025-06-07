#!/bin/bash

# Default iteration number if not provided
ITERATION=${1:-2}
echo "Running curriculum learning for iteration ${ITERATION}"

# Load necessary modules
module load anaconda/2024.06
module load gcc/11.2.0

# Change to the project directory
source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh

# Activate the conda environment
conda activate dl_py309

# Base directories
BASE_DIR="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_ddEdge"
PREV_ITER=$((ITERATION - 1))
OLD_DATA_DIR="${BASE_DIR}/data/iteration${PREV_ITER}/training/Sequential_Peptides_edges"
NEW_DATA_DIR="${BASE_DIR}/data/iteration${ITERATION}/training/Sequential_Peptides_edges"
PREV_MODEL="${BASE_DIR}/models/iteration${PREV_ITER}/Transformer_lr_0.2_bs_1024.pt"
MODEL_DIR="${BASE_DIR}/models/iteration${ITERATION}"

# Create directories for this iteration
mkdir -p ${MODEL_DIR}

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
WARMUP_EPOCHS=20
CURRICULUM_STEPS=3

# Run the curriculum training script
python main_seq_curriculum.py \
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
    --prev_model $PREV_MODEL \
    --old_data_dir $OLD_DATA_DIR \
    --new_data_dir $NEW_DATA_DIR \
    --warmup_epochs $WARMUP_EPOCHS \
    --curriculum_steps $CURRICULUM_STEPS

# Print the parameters used
echo "Training completed with the following parameters:"
echo "----------------------------------------"
echo "Iteration: ${ITERATION}"
echo "Directories:"
echo "  Previous Model: ${PREV_MODEL}"
echo "  Old Data: ${OLD_DATA_DIR}"
echo "  New Data: ${NEW_DATA_DIR}"
echo ""
echo "Curriculum Parameters:"
echo "  Warmup Epochs: $WARMUP_EPOCHS"
echo "  Curriculum Steps: $CURRICULUM_STEPS"
echo "  Total Epochs: $EPOCHS"
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
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Random Seed: $SEED"
echo "----------------------------------------" 
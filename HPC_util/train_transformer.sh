#!/bin/bash

# Load necessary modules
module load anaconda/2024.06
module load gcc/11.2.0

# Change to the project directory
source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh

# Activate the conda environment
conda activate dl_py309

# Model architecture parameters
D_MODEL=512
D_FF=2048
N_LAYERS=6
N_HEADS=8
SRC_VOCAB_SIZE=21
SRC_LEN=10

# Training parameters
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001
SEED=42

# Create output directory if it doesn't exist
mkdir -p out
mkdir -p results_transformer

# Run the training script with all parameters
python main_seq_clean.py \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --src_vocab_size $SRC_VOCAB_SIZE \
    --src_len $SRC_LEN \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --seed $SEED

# Print the parameters used
echo "Training completed with the following parameters:"
echo "----------------------------------------"
echo "Model Architecture:"
echo "  Embedding Dimension: $D_MODEL"
echo "  Feed Forward Dimension: $D_FF"
echo "  Number of Layers: $N_LAYERS"
echo "  Number of Heads: $N_HEADS"
echo "  Source Vocabulary Size: $SRC_VOCAB_SIZE"
echo "  Sequence Length: $SRC_LEN"
echo ""
echo "Training Parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Random Seed: $SEED"
echo "----------------------------------------" 
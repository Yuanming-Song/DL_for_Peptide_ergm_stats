#!/bin/bash

# Load necessary modules
module load anaconda/2024.06
module load gcc/11.2.0

# Change to the project directory
source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh

#conda init
#Activate the conda environment located in your writable directory
conda activate dl_py309

# Model architecture parameters
MODEL="Transformer"
D_MODEL=256
D_FF=4
N_LAYERS=2
N_HEADS=4
DROPOUT=0.2

# Training parameters
EPOCHS=300
BATCH_SIZE=128
LEARNING_RATE=0.000005
WEIGHT_DECAY=0.005
GRAD_CLIP=0.5
WARMUP_STEPS=1000
PATIENCE=300

# Data parameters
DIST_DIM=6
SRC_LEN=10
SRC_VOCAB_SIZE=21

# Run the training script with all parameters
python main_seq_kl_improved.py \
    --model $MODEL \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --grad_clip $GRAD_CLIP \
    --warmup_steps $WARMUP_STEPS \
    --patience $PATIENCE \
    --dist_dim $DIST_DIM \
    --src_len $SRC_LEN \
    --src_vocab_size $SRC_VOCAB_SIZE

# Print the parameters used
echo "Training completed with the following parameters:"
echo "----------------------------------------"
echo "Model Architecture:"
echo "  Model Type: $MODEL"
echo "  Embedding Dim: $D_MODEL"
echo "  Feed Forward Dim: $D_FF"
echo "  Number of Layers: $N_LAYERS"
echo "  Number of Heads: $N_HEADS"
echo "  Dropout Rate: $DROPOUT"
echo ""
echo "Training Parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Gradient Clipping: $GRAD_CLIP"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "  Early Stopping Patience: $PATIENCE"
echo ""
echo "Data Parameters:"
echo "  Distribution Dimension: $DIST_DIM"
echo "  Source Length: $SRC_LEN"
echo "  Vocabulary Size: $SRC_VOCAB_SIZE"
echo "----------------------------------------"

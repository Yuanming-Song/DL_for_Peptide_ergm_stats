#!/bin/bash

# Load required modules
module load anaconda/2024.06
module load gcc/11.2.0

# Activate conda environment
source activate dl_py309

# Model architecture parameters
D_MODEL=512
D_FF=2048
N_LAYERS=6
N_HEADS=8
SRC_VOCAB_SIZE=21
SRC_LEN=10

# Training parameters
EPOCHS=100
BATCH_SIZE=1024
LEARNING_RATE=0.2
SEED=42

# Directory paths
BASE_DIR="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge"
ITER1_DATA_DIR="${BASE_DIR}/data/iteration1/training/Sequential_Peptides_edges"
ITER2_DATA_DIR="${BASE_DIR}/data/iteration2/training/Sequential_Peptides_edges"
ITER1_MODEL="${BASE_DIR}/models/iteration1/Transformer_lr_0.2_bs_1024.pt"

# Create output directories
mkdir -p "${BASE_DIR}/models/iteration2"
mkdir -p "${BASE_DIR}/results/iteration2"

# Run the curriculum learning script
python main_seq_curriculum.py \
    --task_type Regression \
    --seed ${SEED} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --src_vocab_size ${SRC_VOCAB_SIZE} \
    --src_len ${SRC_LEN} \
    --batch_size ${BATCH_SIZE} \
    --model Transformer \
    --prev_model "${ITER1_MODEL}" \
    --old_data_dir "${ITER1_DATA_DIR}" \
    --new_data_dir "${ITER2_DATA_DIR}" \
    --curriculum_steps 3 \
    --warmup_epochs 10

# Print summary
echo "Training completed with parameters:"
echo "Model: Transformer"
echo "Architecture: d_model=${D_MODEL}, d_ff=${D_FF}, n_layers=${N_LAYERS}, n_heads=${N_HEADS}"
echo "Training: epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, lr=${LEARNING_RATE}"
echo "Curriculum: steps=3, warmup_epochs=10"
echo "Data: Combined iteration 1 and 2" 
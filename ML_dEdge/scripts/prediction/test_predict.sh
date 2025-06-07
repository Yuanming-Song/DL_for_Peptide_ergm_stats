#!/bin/bash

# Load necessary modules
module load anaconda/2024.06
module load gcc/11.2.0

# Activate conda environment
source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh
conda activate dl_py309

# Base directories
BASE_DIR="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge"
MODEL_DIR="${BASE_DIR}/models/iteration2"

# Run prediction
python predict_seq.py \
    --task_type "Regression" \
    --src_vocab_size 21 \
    --src_len 10 \
    --model "Transformer" \
    --dropout 0.1 \
    --max 3.0 \
    --min -1.0 \
    --batch_size 2 \
    --model_path "${MODEL_DIR}/Transformer_curriculum_lr_0.2_bs_1024.pt" \
    --input_file "test_sequences.csv" \
    --output_file "test_predictions.csv" 
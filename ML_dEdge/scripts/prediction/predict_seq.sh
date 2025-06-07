#!/bin/bash

# Load necessary modules
module load anaconda/2024.06
module load gcc/11.2.0

# Change to the project directory
source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh

# Activate the conda environment
conda activate dl_py309

# First generate the sequences
python generate_C_sequences.py

# Model parameters (must match training parameters)
SRC_VOCAB_SIZE=21
SRC_LEN=10  # Must match training value (10 for decapeptides)
DROPOUT=0.1
MAX=3.0
MIN=-1.0

# Run the prediction script
python predict_seq.py \
    --task_type "Regression" \
    --src_vocab_size $SRC_VOCAB_SIZE \
    --src_len $SRC_LEN \
    --model "Transformer" \
    --dropout $DROPOUT \
    --max $MAX \
    --min $MIN \
    --model_path "Transformer_lr_0.2_bs_1024.pt" \
    --input_file "Sequential_Peptides_edges/ddedge_predict_seqs_raw.csv" \
    --output_file "Sequential_Peptides_edges/ddedge_predict_seqs.csv" 
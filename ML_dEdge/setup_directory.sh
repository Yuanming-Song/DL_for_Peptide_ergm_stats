#!/bin/bash

# Base directory
BASE_DIR="/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_ddEdge"

# Create main directory structure (only if they don't exist)
echo "Creating directory structure..."
mkdir -p "${BASE_DIR}/models/iteration1"
mkdir -p "${BASE_DIR}/data/iteration1/training"
mkdir -p "${BASE_DIR}/data/iteration1/predictions"
mkdir -p "${BASE_DIR}/scripts/sequence_generation"
mkdir -p "${BASE_DIR}/scripts/training"
mkdir -p "${BASE_DIR}/scripts/prediction"
mkdir -p "${BASE_DIR}/scripts/submission"
mkdir -p "${BASE_DIR}/analysis/plotting"

# Show move commands for review
echo "Review these move commands and execute them manually after verification:"
echo

echo "# Moving model files"
echo "mv ${BASE_DIR}/Transformer_lr_0.2_bs_1024.pt ${BASE_DIR}/models/iteration1/"
echo

echo "# Moving data directories"
echo "mv ${BASE_DIR}/Sequential_Peptides_edges ${BASE_DIR}/data/iteration1/training/"
echo "mv ${BASE_DIR}/results_transformer ${BASE_DIR}/data/iteration1/predictions/"
echo "mv ${BASE_DIR}/selected_sequences_tetrapeptide.txt ${BASE_DIR}/data/iteration1/"
echo

echo "# Moving training scripts"
echo "mv ${BASE_DIR}/main_seq_clean.py ${BASE_DIR}/scripts/training/"
echo "mv ${BASE_DIR}/train_transformer.sh ${BASE_DIR}/scripts/training/"
echo

echo "# Moving submission scripts"
echo "mv ${BASE_DIR}/selective_submit_ML_mon_tetrapeptide.sh ${BASE_DIR}/scripts/submission/"
echo "mv ${BASE_DIR}/selective_submit_ML_tetrapeptide.sh ${BASE_DIR}/scripts/submission/"
echo "mv ${BASE_DIR}/submit_transformer.sh ${BASE_DIR}/scripts/submission/"
echo

echo "# Moving analysis scripts"
echo "mv ${BASE_DIR}/*plot*.R ${BASE_DIR}/analysis/plotting/"
echo

echo "# Cleanup"
echo "rm ${BASE_DIR}/setup_directory.sh"

echo
echo "Directory structure is ready. Execute the above commands after verifying them."
echo "Make sure to backup your files before moving them!" 
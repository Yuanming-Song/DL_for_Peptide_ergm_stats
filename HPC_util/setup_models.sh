#!/bin/bash

# Base directories for models
MONOMER_DIR="ML_ddEdge_monomer"
DIMER_DIR="ML_ddEdge_dimer"

# Files to copy
FILES_TO_COPY=(
    "main_seq_clean.py"
    "train_transformer.sh"
    "submit_transformer.sh"
)

# Function to setup a model directory
setup_model_dir() {
    local model_dir=$1
    
    # Create main directory if it doesn't exist
    mkdir -p "$model_dir"
    
    # Create required subdirectories
    mkdir -p "$model_dir/out"
    mkdir -p "$model_dir/results_transformer"
    
    # Copy necessary files
    for file in "${FILES_TO_COPY[@]}"; do
        cp "$file" "$model_dir/"
    done
    
    echo "Setup completed for $model_dir"
}

# Setup both model directories
setup_model_dir "$MONOMER_DIR"
setup_model_dir "$DIMER_DIR"

echo "All directories have been set up successfully!" 
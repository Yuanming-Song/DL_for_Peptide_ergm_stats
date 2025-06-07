#!/bin/bash

# Base directories for models
MODELS=("ML_dEdge_monomer" "ML_dEdge_dimer")

for MODEL in "${MODELS[@]}"; do
    # Create main directory structure
    mkdir -p "${MODEL}/models/iteration1"
    mkdir -p "${MODEL}/data/iteration1/training"
    mkdir -p "${MODEL}/data/iteration1/predictions/results_transformer"
    mkdir -p "${MODEL}/scripts/training"
    mkdir -p "${MODEL}/scripts/prediction"
    mkdir -p "${MODEL}/scripts/submission"
    mkdir -p "${MODEL}/analysis/plotting"
    
    # Copy necessary scripts from ML_ddEdge with appropriate modifications
    if [ -f "ML_ddEdge/scripts/training/train_transformer.sh" ]; then
        cp "ML_ddEdge/scripts/training/train_transformer.sh" "${MODEL}/scripts/training/"
    fi
    
    if [ -f "ML_ddEdge/scripts/submission_ML_train/submit_transformer.sh" ]; then
        cp "ML_ddEdge/scripts/submission_ML_train/submit_transformer.sh" "${MODEL}/scripts/submission/"
    fi
done

echo "Directory structure created for ML_dEdge_monomer and ML_dEdge_dimer"
echo "Next steps:"
echo "1. Copy and modify training data into data/iteration1/training"
echo "2. Adapt training scripts for edge property prediction"
echo "3. Update model architecture in scripts/training"
echo "4. Set up integration with main generative pipeline" 
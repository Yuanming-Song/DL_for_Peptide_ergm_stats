# Transformer for Peptide Property Regression

A Transformer-based model for predicting peptide edge statistics differences between dimer and monomer simulations.

## Project Structure

```
DL_for_Peptide/
├── Data_prepare_R/          # R scripts for data reshaping and CSV generation
├── OG_util_py/             # Original base scripts from DL_for_peptide repo
├── Reserve_new_training/   # Additional models in progress (different ERGM stats)
├── Old_models/            # Archive of past training attempts
├── HPC_util/             # Utility scripts for directory structure and job submission
├── ML_dEdge/             # Current working model for edge statistics
├── training_logs/        # Training logs (gitignored)
└── out/                 # Output directory (gitignored)
```

## Model Architecture

The model is a Transformer neural network with the following specifications:
- 6-layer Transformer with 8-head self-attention
- 512-dimensional embeddings
- 2048-dimensional feed-forward networks
- Trained on edge statistics differences between dimer and monomer simulations

## Training Details

- Uses curriculum learning approach
- Optimized with SGD optimizer
- Supports multiple sequence lengths (5-7 mers)
- Implements early stopping and model checkpointing

## Key Arguments

```
--epochs        Number of training epochs (default: 100)
--batch_size    Batch size (default: 32)
--lr           Learning rate (default: 0.001)
--seed         Random seed (default: 42)

# Model architecture
--d_model      Transformer embedding dimension (default: 512)
--d_ff         Feedforward dimension (default: 2048)
--n_heads      Number of attention heads (default: 8)
--n_layers     Number of transformer layers (default: 6)
```

## Output Format

Test results are saved in CSV format with:
- Feature: Original peptide sequence
- Prediction: Model's predicted value
- True_Value: Actual label
- Absolute_Error: |Prediction - True_Value|

For detailed implementation and usage instructions, please refer to ML_dEdge/README.md. 
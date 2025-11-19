# Transformer for Peptide Property Regression

A Transformer-based model for predicting peptide edge statistics differences between dimer and monomer simulations.

This README documents the Transformer architecture used in the ML_dEdge model. For the complete project structure, see the main [DL_for_Peptide README](../README.md).

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
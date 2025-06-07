# ML_dEdge_monomer: Edge Prediction Model for Monomer States

This directory contains a machine learning model specifically designed to predict edge properties for peptide sequences in their monomer state. This model is part of a larger pipeline that includes both monomer and dimer predictions for generative sequence design.

## Project Structure

```
ML_dEdge_monomer/
├── models/
│   └── iteration1/
│       └── Transformer_monomer_edges.pt  # Best trained model
├── data/
│   └── iteration1/
│       ├── training/
│       │   └── Monomer_Edges/           # Training data for monomer edge properties
│       └── predictions/
│           └── results_transformer/    # Prediction results
├── scripts/
│   ├── training/
│   │   ├── main_monomer_edge.py      # Training script
│   │   └── train_transformer.sh      # Training shell script
│   ├── prediction/
│   │   └── predict_edges.sh          # Prediction script
│   └── submission/
│       └── submit_transformer.sh     # Model submission script
└── analysis/
    └── plotting/
        └── plot_monomer_results.R    # Analysis of monomer predictions
```

## Model Description

- Input: Peptide sequence
- Output: Edge properties prediction for monomer state
- Architecture: Transformer-based model optimized for edge property prediction
- Training data: Edge properties from monomer molecular dynamics simulations

## Integration with Main Pipeline

This model serves as one component of a larger generative pipeline:
1. Takes sequence input from main model
2. Predicts monomer edge properties
3. Feeds predictions back to generative model
4. Helps optimize sequence generation for desired monomer properties

## Usage

### Training
```bash
cd scripts/training
./train_transformer.sh
```

### Prediction
```bash
cd scripts/prediction
./predict_edges.sh <input_sequence_file>
```

### Model Training Submission
```bash
cd scripts/submission
./submit_transformer.sh <iteration_number>
```

## Notes
- Model specifically optimized for monomer state predictions
- Works in conjunction with ML_dEdge_dimer model
- Outputs standardized for integration with main generative pipeline 
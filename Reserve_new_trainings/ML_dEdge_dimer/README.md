# ML_dEdge_dimer: Edge Prediction Model for Dimer States

This directory contains a machine learning model specifically designed to predict edge properties for peptide sequences in their dimer state. This model is part of a larger pipeline that includes both monomer and dimer predictions for generative sequence design.

## Project Structure

```
ML_dEdge_dimer/
├── models/
│   └── iteration1/
│       └── Transformer_dimer_edges.pt  # Best trained model
├── data/
│   └── iteration1/
│       ├── training/
│       │   └── Dimer_Edges/           # Training data for dimer edge properties
│       └── predictions/
│           └── results_transformer/    # Prediction results
├── scripts/
│   ├── training/
│   │   ├── main_dimer_edge.py        # Training script
│   │   └── train_transformer.sh      # Training shell script
│   ├── prediction/
│   │   └── predict_edges.sh          # Prediction script
│   └── submission/
│       └── submit_transformer.sh     # Model submission script
└── analysis/
    └── plotting/
        └── plot_dimer_results.R      # Analysis of dimer predictions
```

## Model Description

- Input: Peptide sequence
- Output: Edge properties prediction for dimer state
- Architecture: Transformer-based model optimized for edge property prediction
- Training data: Edge properties from dimer molecular dynamics simulations

## Integration with Main Pipeline

This model serves as one component of a larger generative pipeline:
1. Takes sequence input from main model
2. Predicts dimer edge properties
3. Feeds predictions back to generative model
4. Helps optimize sequence generation for desired dimer properties

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
- Model specifically optimized for dimer state predictions
- Works in conjunction with ML_dEdge_monomer model
- Outputs standardized for integration with main generative pipeline 
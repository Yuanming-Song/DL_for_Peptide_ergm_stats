# ML_Edge_Integration: Integration Layer for Edge Property Models

This directory contains the integration layer that coordinates between the monomer/dimer edge prediction models and the generative model pipeline.

## Project Structure

```
ML_Edge_Integration/
├── models/
│   └── iteration1/
│       └── generative_model.pt       # Combined generative model
├── scripts/
│   ├── integration/
│   │   ├── edge_predictor.py        # Handles predictions from both models
│   │   └── sequence_generator.py     # Generative model implementation
│   ├── training/
│   │   └── train_generator.py       # Training script for generative model
│   └── utils/
│       ├── data_processing.py       # Data processing utilities
│       └── model_utils.py           # Model utility functions
└── configs/
    └── model_config.yaml            # Configuration for model integration
```

## Integration Flow

1. **Input Processing**
   - Takes sequence input from generative model
   - Prepares input format for edge models

2. **Edge Property Prediction**
   - Loads trained models from:
     - `/ML_dEdge_monomer/models/iteration1/Transformer_monomer_edges.pt`
     - `/ML_dEdge_dimer/models/iteration1/Transformer_dimer_edges.pt`
   - Runs predictions in parallel
   - Combines predictions into unified format

3. **Sequence Generation**
   - Uses edge property predictions to guide sequence generation
   - Optimizes for desired properties in both states
   - Generates new candidate sequences

## Usage

### Prediction Pipeline
```python
from integration.edge_predictor import EdgePredictor
from integration.sequence_generator import SequenceGenerator

# Initialize predictors
predictor = EdgePredictor(
    monomer_model_path='../ML_dEdge_monomer/models/iteration1/Transformer_monomer_edges.pt',
    dimer_model_path='../ML_dEdge_dimer/models/iteration1/Transformer_dimer_edges.pt'
)

# Initialize generator
generator = SequenceGenerator(
    predictor=predictor,
    config_path='configs/model_config.yaml'
)

# Generate sequences
sequences = generator.generate(
    n_sequences=10,
    target_properties={
        'monomer': {...},
        'dimer': {...}
    }
)
```

## Configuration

The `model_config.yaml` file specifies:
- Model paths and parameters
- Prediction thresholds
- Generation parameters
- Integration settings 
# ML_dEdge: Deep Learning for Peptide Edge Statistics

This directory contains the current working model trained on the difference in edge statistics between dimer and monomer simulations.

## Directory Structure

```
ML_dEdge/
├── analysis/         # Analysis scripts and results
│   ├── Data_reshape/    # Scripts for processing prediction results
│   │   └── process_prediction_results.R  # Processes and normalizes prediction results
│   └── plotting/       # R scripts for visualization
│       ├── plot_iteration1_results.R     # Analysis of first iteration
│       └── plot_iteration2_results.R     # Analysis of curriculum learning results
├── scripts/          # Training and prediction scripts
│   ├── training/       # Model training scripts
│   │   ├── main_seq_clean.py            # Current training implementation
│   │   ├── train_transformer.sh         # Training script for iteration 1
│   │   └── iteration2/                  # Iteration 2 training scripts
│   │       └── train_transformer_iter2.sh  # Curriculum learning training script
│   ├── prediction/     # Sequence prediction scripts
│   │   ├── predict_seq.py               # Prediction script
│   │   └── predict_multiple_lengths.slurm  # Multi-length prediction job
│   └── sequence_generation/  # Sequence generation utilities
│       └── generate_cysteine_sequences.py  # Sequence generation script
└── data/            # Data directory
    ├── iteration1/     # First iteration data and results
    │   ├── training/   # Training data and results
    │   └── predictions/  # Prediction results
    └── iteration2/     # Second iteration with curriculum learning
        ├── training/   # Training data and results
        └── predictions/  # Prediction results
```

## Training Methodology

### Initial Model Development (Iteration 1)
The first iteration of model development focused on establishing a baseline performance using a comprehensive dataset of cysteine-containing peptides. The training dataset comprised:
- All possible dipeptides containing cysteine
- All possible tripeptides containing cysteine
- Tetrapeptides with cysteine at the first position

The model was trained using a Transformer architecture with the following specifications:

#### Model Architecture
- Embedding dimension: 512
- Feed-forward dimension: 2048
- Number of layers: 4
- Number of attention heads: 4
- Key/Value dimensions: 32
- Dropout rate: 0.1
- Source vocabulary size: 21
- Maximum sequence length: 10

#### Training Parameters
- Learning rate: 0.22
- Batch size: 64
- Training epochs: 200
- Random seed: 42

#### Hardware Configuration
- Environment: Anaconda 2024.06
- Compiler: GCC 11.2.0
- Python environment: dl_py309
- GPU: NVIDIA A100 (80GB)
- Memory: 32GB RAM
- Storage: /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_ddEdge

This initial training achieved a mean absolute error (MAE) of 0.164 on the test set, establishing a strong baseline for further improvements.

### Enhanced Model with Curriculum Learning (Iteration 2)
To improve the model's performance and generalization capabilities, we implemented a curriculum learning approach in the second iteration. This approach was designed to address the challenges of learning across different sequence lengths and structural variations.

#### Model Architecture
- Embedding dimension: 512
- Feed-forward dimension: 2048
- Number of layers: 6
- Number of attention heads: 8
- Source vocabulary size: 21
- Maximum sequence length: 10

#### Curriculum Learning Stages
1. **Initial Training Phase**
   - Duration: 50 epochs
   - Learning rate: 0.2
   - Batch size: 1024
   - Focus: Establishing a strong foundation using the iteration 1 dataset
   - Objective: Ensure model stability and basic feature learning

2. **New Data Integration Phase**
   - Duration: 30 epochs
   - Learning rate: 0.1
   - Batch size: 1024
   - Focus: Gradual introduction of new sequence variations
   - Objective: Expand model's understanding of sequence patterns

3. **Fine-tuning Phase**
   - Duration: 20 epochs
   - Learning rate: 0.05
   - Batch size: 1024
   - Focus: Optimization of model performance on the combined dataset
   - Objective: Refine predictions and improve accuracy

#### Hardware Configuration
- Environment: Anaconda 2024.06
- Compiler: GCC 11.2.0
- Python environment: dl_py309
- GPU: NVIDIA A100 (80GB)
- Memory: 32GB RAM
- Storage: /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge

#### Training Improvements
- Increased model complexity (6 layers, 8 heads)
- Larger batch size (1024 vs 64)
- Curriculum learning with warmup epochs
- Three-stage training process
- Combined dataset from both iterations

This curriculum learning approach resulted in:
- Improved handling of sequence length variations
- Enhanced prediction accuracy for longer sequences
- Better generalization across different peptide structures
- More robust feature learning through staged training

## Key Scripts

### Analysis Scripts
- `process_prediction_results.R`: Processes prediction results for visualization
- `plot_iteration1_results.R`: Analyzes and visualizes first iteration results
- `plot_iteration2_results.R`: Analyzes and visualizes curriculum learning results

### Training Scripts
- `main_seq_clean.py`: Current training implementation
- `train_transformer.sh`: SLURM script for iteration 1 training
- `train_transformer_iter2.sh`: SLURM script for curriculum learning

### Prediction Scripts
- `predict_seq.py`: Makes predictions using trained model
- `predict_multiple_lengths.slurm`: SLURM script for multi-length predictions

### Sequence Generation
- `generate_cysteine_sequences.py`: Generates sequences for prediction

## Data Organization

### Iteration 1
- Training data in `data/iteration1/training/`
- Prediction results in `data/iteration1/predictions/`
- Model checkpoints in `data/iteration1/training/`

### Iteration 2
- Training data in `data/iteration2/training/`
- Prediction results in `data/iteration2/predictions/`
- Model checkpoints in `data/iteration2/training/`

## Usage

1. **Training**
   ```bash
   cd scripts/training
   sbatch train_transformer.sh
   ```
   ```

3. **Prediction**
   ```bash
   cd scripts/prediction
   sbatch predict_multiple_lengths.slurm
   ```

4. **Analysis**
   ```bash
   cd analysis/plotting
   Rscript plot_iteration2_results.R
   ```

For detailed model architecture and training parameters, please refer to README_transformer.md. 
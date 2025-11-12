# Deep Learning for Peptide Self-Assembly Prediction

This repository is a cleaned-up version of [Yuanming-Song/DL_for_Peptide](https://github.com/Yuanming-Song/DL_for_Peptide), which was forked from [maozhudemi/DL_for_Peptide](https://github.com/maozhudemi/DL_for_Peptide), which was originally forked from [Zihan-Liu-00/BiB_ADV-SCI--DL_for_Peptide](https://github.com/Zihan-Liu-00/BiB_ADV-SCI--DL_for_Peptide).

## Overview

This project focuses on predicting peptide self-assembly properties using deep learning approaches, specifically focusing on edge statistics differences between dimer and monomer simulations. The implementation uses a Transformer-based architecture to process peptide sequences and predict their self-assembly behavior.

## Key Features

- **Transformer-based Architecture**: Implements a state-of-the-art Transformer model for sequence processing
- **Curriculum Learning**: Uses a structured three-stage training approach
- **Multiple Sequence Lengths**: Supports peptides of varying lengths (5-7 mers)
- **Comprehensive Analysis**: Includes tools for data processing, model training, and result visualization
- **Generative Model**: ML_dEdge_gen provides sequence generation capabilities with target dEdge value ranges

## Project Structure

```
DL_for_Peptide/
├── Data_prepare_R/          # R scripts for data reshaping and CSV generation
├── HPC_util/             # Utility scripts for directory structure and job submission
├── ML_dEdge/             # Current working model for edge statistics
└── ML_dEdge_gen/         # Generative model for sequence generation with target dEdge values
```

## Model Architecture

The current implementation uses a Transformer neural network with:
- 6-layer Transformer with 8-head self-attention
- 512-dimensional embeddings
- 2048-dimensional feed-forward networks
- Trained on edge statistics differences between dimer and monomer simulations

## Training Methodology

The model is trained using a curriculum learning approach with three distinct phases:
1. Initial training phase
2. New data integration phase
3. Fine-tuning phase

Each phase uses different learning rates and batch sizes to optimize the learning process.

## Getting Started

1. Clone the repository
2. Install dependencies (see environment.yml or requirements.txt)
3. Prepare your data using the scripts in Data_prepare_R/
4. Train the model using the scripts in ML_dEdge/
5. For generative sequence design, use ML_dEdge_gen to generate sequences with target dEdge values

### Quick Start

**Training ML_dEdge (Regression Model):**
```bash
cd ML_dEdge/scripts/training
sbatch train_transformer.sh  # For iteration 1
# or
sbatch iteration2/train_transformer_iter2.slurm  # For iteration 2 with curriculum learning
```

**Training ML_dEdge_gen (Generative Model):**
```bash
cd ML_dEdge_gen/v1+2/scripts/training
sbatch train_generative_model.slurm
```

**Generating Sequences:**
```bash
cd ML_dEdge_gen/v1+2/scripts/generation
./generate_sequences.sh <model_path> <dEdge_min> <dEdge_max> <seq_length_min> <seq_length_max> <num_sequences>
```

## Models

### ML_dEdge
The standard regression model that predicts dEdge values for given peptide sequences. This model:
- Uses a Transformer encoder architecture
- Trained on edge statistics differences between dimer and monomer simulations
- Supports separate training on iteration1 (v1) or iteration2 (v2) data
- Can also use curriculum learning to combine both iterations

See `ML_dEdge/README.md` for detailed training and usage instructions. For Transformer architecture details, see `ML_dEdge/README_transformer.md`.

### ML_dEdge_gen
A conditional generative model that can directly generate peptide sequences with target dEdge value ranges. This model:
- Uses a conditional Transformer decoder architecture
- Combines data from both iteration1 (v1) and iteration2 (v2) for training
- Generates sequences autoregressively conditioned on dEdge values and sequence lengths
- Takes as input: dEdge value range, sequence length range, and number of sequences
- Outputs: Generated sequences with their target dEdge values and sequence lengths

See `ML_dEdge_gen/README.md` for detailed usage instructions, including training and generation commands.

## References

This work builds upon:
- Liu et al. (2023) "Efficient prediction of peptide self-assembly through sequential and graphical encoding." Briefings in Bioinformatics
- Wang et al. (2023) "Deep Learning Empowers the Discovery of Self‐Assembling Peptides with Over 10 Trillion Sequences." Advanced Science

## Contact

For questions or collaborations, please contact the maintainers of this repository.

## License

[Add appropriate license information] 
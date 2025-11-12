# ML_dEdge_gen: Generative Model for Peptide Sequence Generation

This directory contains a conditional generative model that can directly generate peptide sequences based on target dEdge values and sequence length specifications. Unlike the standard ML_dEdge model which predicts dEdge values for given sequences, this generative model takes dEdge value ranges and sequence length ranges as input and generates sequences accordingly.

## Directory Structure

```
ML_dEdge_gen/
├── v1+2/                    # Combined v1 and v2 generative model
│   ├── scripts/
│   │   ├── training/         # Training scripts for generative model
│   │   │   ├── train_generative_model.py    # Python training script
│   │   │   ├── train_generative_model.sh    # Shell script for training
│   │   │   ├── train_generative_model.slurm # SLURM submission script
│   │   │   └── models_gen.py                # Generative model architecture
│   │   └── generation/      # Sequence generation scripts
│   │       ├── generate_sequences.py     # Main generation script
│   │       └── generate_sequences.sh     # Shell script for generation
│   ├── data/                # Data directory (links to ML_dEdge data)
│   ├── models/              # Trained model checkpoints
│   └── out/                 # Generated sequences output
└── README.md                # This file
```

## Key Features

### Generative Capabilities

The generative model can:
- **Input**: 
  - dEdge value range (min, max)
  - Sequence length range (min, max)
  - Number of sequences to generate
- **Output**: Generated peptide sequences with their target dEdge values and sequence lengths

### Model Architecture

The model uses a **conditional Transformer decoder** architecture:
- Embedding dimension: 512
- Feed-forward dimension: 2048
- Number of decoder layers: 6
- Number of attention heads: 8
- Source vocabulary size: 21 (20 amino acids + empty token)
- Maximum sequence length: 10
- **Condition embedding**: Takes dEdge value and sequence length as conditional inputs

The model is trained to generate sequences autoregressively, conditioned on the target dEdge value and sequence length.

### Training Data

The v1+2 model is trained on **combined data** from:
- **v1 (iteration1)**: Data from `ML_dEdge/data/iteration1/training/Sequential_Peptides_edges/`
  - Files use `ddedge_` prefix
- **v2 (iteration2)**: Data from `ML_dEdge/data/iteration2/training/Sequential_Peptides_edges/`
  - Files use `dedge_` prefix

The training script automatically combines both datasets, providing a more comprehensive training set that includes all data from both iterations.

## Usage

### Step 1: Training the Model

First, train the conditional generative model. You have two options:

#### Option A: Using SLURM (Recommended for HPC)

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training
sbatch train_generative_model.slurm
```

This will submit the training job to the SLURM queue. The script will:
- Load required modules (anaconda, gcc)
- Activate the conda environment (dl_py309)
- Train the model on combined v1+v2 data
- Save the trained model to `ML_dEdge_gen/v1+2/models/ConditionalGenerator_v1v2_lr_0.001_bs_512.pt`

#### Option B: Direct Execution

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training
./train_generative_model.sh
```

**Note**: Make sure you have the required modules loaded and conda environment activated before running directly.

### Step 2: Generating Sequences

Once the model is trained, you can generate sequences using the command line. Navigate to the generation directory and run:

#### Using Shell Script (Recommended)

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/generation
./generate_sequences.sh <model_path> <dEdge_min> <dEdge_max> <seq_length_min> <seq_length_max> <num_sequences> [temperature]
```

**Required Arguments:**
- `model_path`: Path to the trained model checkpoint (e.g., `../models/ConditionalGenerator_v1v2_lr_0.001_bs_512.pt`)
- `dEdge_min`: Minimum dEdge value (float, e.g., `0.5`)
- `dEdge_max`: Maximum dEdge value (float, e.g., `1.0`)
- `seq_length_min`: Minimum sequence length (integer, e.g., `6`)
- `seq_length_max`: Maximum sequence length (integer, e.g., `8`)
- `num_sequences`: Number of sequences to generate (integer, e.g., `100`)

**Optional Arguments:**
- `temperature`: Sampling temperature (float, default: `1.0`)
  - Lower values (e.g., 0.5) = more deterministic, less diverse
  - Higher values (e.g., 1.5) = more random, more diverse

**Example 1: Basic usage**

```bash
./generate_sequences.sh \
    ../models/ConditionalGenerator_v1v2_lr_0.001_bs_512.pt \
    0.5 1.0 6 8 100
```

This generates 100 sequences with:
- dEdge values between 0.5 and 1.0
- Sequence lengths between 6 and 8 amino acids
- Default temperature of 1.0

**Example 2: With custom temperature**

```bash
./generate_sequences.sh \
    ../models/ConditionalGenerator_v1v2_lr_0.001_bs_512.pt \
    0.3 0.7 5 7 50 0.8
```

This generates 50 sequences with:
- dEdge values between 0.3 and 0.7
- Sequence lengths between 5 and 7 amino acids
- Temperature of 0.8 (more deterministic)

#### Using Python Directly

You can also run the Python script directly:

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/generation
python generate_sequences.py \
    --model_path ../models/ConditionalGenerator_v1v2_lr_0.001_bs_512.pt \
    --dEdge_min 0.5 \
    --dEdge_max 1.0 \
    --seq_length_min 6 \
    --seq_length_max 8 \
    --num_sequences 100 \
    --temperature 1.0
```

**All Python Arguments:**
- `--model_path`: Path to trained model (required)
- `--dEdge_min`: Minimum dEdge value (required)
- `--dEdge_max`: Maximum dEdge value (required)
- `--seq_length_min`: Minimum sequence length (required)
- `--seq_length_max`: Maximum sequence length (required)
- `--num_sequences`: Number of sequences to generate (required)
- `--temperature`: Sampling temperature (optional, default: 1.0)
- `--output_file`: Custom output file path (optional, auto-generated if not specified)
- `--seed`: Random seed for reproducibility (optional, default: 42)

## Generation Methodology

The generative model uses a **conditional autoregressive generation** approach:

1. **Condition Encoding**: The target dEdge value and sequence length are encoded as conditional inputs
2. **Autoregressive Generation**: The model generates sequences token-by-token, conditioned on the target properties
3. **Sampling**: Sequences are generated using temperature-based sampling from the model's output distribution

This approach directly generates sequences that match the specified conditions, rather than filtering from random candidates.

## Output Format

The generated sequences are saved as a CSV file in the `ML_dEdge_gen/v1+2/out/` directory with the following columns:
- `Sequence`: The generated peptide sequence (amino acid string)
- `dEdge_target`: The target dEdge value used for generation
- `SeqLength_target`: The target sequence length used for generation
- `SeqLength_actual`: The actual length of the generated sequence

The file is automatically named based on the parameters:
```
generated_sequences_dEdge_{min}_{max}_len{min}_{max}_n{num_sequences}.csv
```

**Example output file:**
```
generated_sequences_dEdge_0.5_1.0_len6_8_n100.csv
```

You can also specify a custom output path using the `--output_file` argument when using Python directly.

## Data Access

The model accesses training data from the original `ML_dEdge` directory:
- v1 data: `/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/data/iteration1/training/Sequential_Peptides_edges/`
- v2 data: `/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/data/iteration2/training/Sequential_Peptides_edges/`

The training script automatically combines both datasets during training.

## Differences from ML_dEdge

| Feature | ML_dEdge | ML_dEdge_gen |
|---------|----------|--------------|
| **Purpose** | Predict dEdge for given sequences | Generate sequences with target dEdge values |
| **Input** | Peptide sequences | dEdge range, sequence length range, number of sequences |
| **Output** | Predicted dEdge values | Generated sequences with target properties |
| **Training Data** | Separate v1 or v2 | Combined v1+v2 |
| **Model Type** | Regression (Transformer encoder + classifier) | Conditional generative (Transformer decoder) |
| **Architecture** | Encoder-based | Decoder-based with condition embedding |

## Model Details

### Conditional Decoder Architecture

The model uses a conditional Transformer decoder that:
- Takes dEdge value and sequence length as conditional inputs
- Generates sequences autoregressively using self-attention and cross-attention
- Uses teacher forcing during training
- Samples from the output distribution during inference

### Training Process

The model is trained to:
- Minimize cross-entropy loss between generated and target sequences
- Learn the mapping from (dEdge, sequence_length) conditions to peptide sequences
- Generalize across the combined v1+v2 dataset

## Notes

- The generative model requires a trained model checkpoint from the conditional generative training
- Sequence length range cannot exceed the model's maximum sequence length (default: 10)
- Temperature parameter controls the randomness of generation (lower = more deterministic, higher = more diverse)
- The model generates sequences directly based on conditions, ensuring better control over output properties

## Future Improvements

Potential enhancements to the generative model:
- Improved condition encoding with learned embeddings
- Multi-objective optimization for multiple properties
- Sequence diversity constraints
- Fine-tuning on specific property ranges
- Integration with reinforcement learning for property optimization

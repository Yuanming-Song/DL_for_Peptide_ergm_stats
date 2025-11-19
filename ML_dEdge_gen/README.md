# ML_dEdge_gen: Generative Model for Peptide Sequence Generation

This directory contains a conditional generative model that can directly generate peptide sequences based on target dEdge values and sequence length specifications. Unlike the standard ML_dEdge model which predicts dEdge values for given sequences, this generative model takes dEdge value ranges and sequence length ranges as input and generates sequences accordingly.

## Directory Structure

```
ML_dEdge_gen/
├── v1+2/                    # Named "v1+2" because it uses ALL data from iteration 1 and iteration 2 from ML_dEdge directory
│   ├── scripts/
│   │   ├── training/         # Training scripts for generative model
│   │   │   ├── train_generative_model.py    # MLE training script
│   │   │   ├── train_generative_model.sh    # Shell script for MLE training
│   │   │   ├── train_generative_model.slurm # SLURM submission script for MLE
│   │   │   ├── train_generative_model_rl.py # RL fine-tuning script (REINFORCE)
│   │   │   ├── train_generative_model_rl.sh  # Shell script for RL training
│   │   │   ├── train_generative_model_rl.slurm # SLURM submission script for RL
│   │   │   ├── run_test_evaluation.py       # Test set evaluation script
│   │   │   ├── run_test_evaluation.slurm    # SLURM script for test evaluation
│   │   │   ├── test_rl_workflow.py          # Debug script for RL workflow
│   │   │   ├── debug_rl_sequences.py        # Debug script for sequence generation
│   │   │   ├── debug_rl_zero_loss.py        # Debug script for zero loss investigation
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

### Training Approach

The model is trained in two stages:

1. **Maximum Likelihood Estimation (MLE) Training** (Initial Training):
   - Pure MLE training: optimizes cross-entropy loss on ground truth sequences
   - Teacher forcing: always uses ground truth tokens during training (no scheduled sampling)
   - Min-max normalization: dEdge values are normalized using min-max scaling (matching the pre-trained ML_dEdge critic model)
   - Validation evaluation: uses pre-trained ML_dEdge model as a critic to evaluate generated sequences (evaluation only, not used during training)
   - See [Training Process](#training-process) section for details

2. **Reinforcement Learning (RL) Fine-tuning** (Optional):
   - Fine-tunes the MLE-trained model using REINFORCE (policy gradient) algorithm
   - Uses frozen pre-trained ML_dEdge model as critic to compute rewards
   - Reward: negative MSE between predicted and target dEdge values
   - Includes entropy bonus to encourage exploration and prevent mode collapse
   - Baseline (exponential moving average) to reduce variance in policy gradient updates
   - See [RL Fine-tuning](#rl-fine-tuning) section for details

### Model Architecture

The model uses a **conditional Transformer decoder** architecture (decoder-only, similar to GPT-style models):

#### Architecture Summary

```
Input: [dEdge, seq_length] + sequence tokens
  ↓
Condition Embedding (Linear: 2 → 768)
  ↓
Token Embedding (Embedding: vocab_size=21 → 768)
  ↓
Positional Encoding (Sinusoidal)
  ↓
8x Decoder Layers:
  ├─ Multi-Head Self-Attention (12 heads, 64 dim each)
  │  └─ Causal mask (prevents attending to future positions)
  ├─ Multi-Head Encoder-Decoder Attention (12 heads, 64 dim each)
  │  └─ Attends to condition embedding (acts as encoder output)
  └─ Feed-Forward Network (768 → 3072 → 768)
     └─ ReLU activation
  ↓
Projection Layer (Linear: 768 → vocab_size=21)
  ↓
Output: Token probabilities for next amino acid
```

#### Architecture Details

- **Type**: Transformer Decoder (decoder-only, not RNN/LSTM)
- **Embedding dimension (d_model)**: 768
- **Feed-forward dimension (d_ff)**: 3072
- **Number of decoder layers**: 8
- **Number of attention heads**: 12
- **Key/Value dimensions**: 64 each
- **Source vocabulary size**: 21 (20 amino acids + padding token)
- **Maximum sequence length**: 10
- **Condition embedding**: Linear layer mapping [dEdge, seq_length] → 768D vector
- **Positional encoding**: Sinusoidal positional encodings

#### Key Components

1. **Multi-Head Attention**: Scaled dot-product attention with 12 heads
2. **Positional Encoding**: Adds position information to token embeddings
3. **Condition Embedding**: Embeds dEdge value and sequence length as conditional inputs
4. **Causal Masking**: Prevents attending to future positions during generation
5. **Teacher Forcing**: Always uses ground truth tokens during training (pure MLE)
6. **Min-Max Normalization**: dEdge values are normalized using min-max scaling to match the pre-trained ML_dEdge critic model

The model is trained to generate sequences autoregressively, conditioned on the target dEdge value and sequence length.

#### Validation Evaluation with Critic Model

During validation, the model uses a pre-trained ML_dEdge model as a critic for evaluation:

**Critic Model (Frozen, Evaluation Only)**:
- Pre-trained ML_dEdge Transformer model (from `ML_dEdge/models/iteration2/Transformer_curriculum_lr_0.2_bs_1024.pt`)
- Used as a frozen critic to evaluate generated sequences during validation
- Predicts dEdge values for generated sequences
- No gradients flow through the critic (frozen)
- **Note**: The critic is only used for evaluation, not during training

**Validation Process**:
1. When a best model is found (based on validation reconstruction loss), the model generates sequences for each unique (dEdge, length) combination in the validation set
2. Generated sequences are evaluated using the critic model to predict their dEdge values
3. Validation metrics include:
   - Unique sequence fraction (diversity measure)
   - Average dEdge prediction error (MSE between predicted and target dEdge)
   - Average predicted dEdge value
4. These validation results are saved to MLflow output directory for analysis

This evaluation process helps assess whether generated sequences actually match target dEdge values, but does not affect the training process.

### Training Data

**Why "v1+2"?** The directory is named `v1+2` because it uses **ALL data from both iteration 1 and iteration 2** from the `ML_dEdge` directory. This distinguishes it from the standard `ML_dEdge` model which can be trained separately on iteration 1 or iteration 2 data.

The v1+2 model is trained on **combined data** from:
- **v1 (iteration1)**: Data from `ML_dEdge/data/iteration1/training/Sequential_Peptides_edges/`
  - Files use `ddedge_` prefix
- **v2 (iteration2)**: Data from `ML_dEdge/data/iteration2/training/Sequential_Peptides_edges/`
  - Files use `dedge_` prefix

#### Stratified Data Splits

The model uses **stratified train/validation/test splits** to ensure proper evaluation of generalization:

- **Stratification**: Data is split by `(dEdge_bin, sequence_length)` combinations, not individual sequences
- **Purpose**: Ensures validation/test sets contain `(dEdge, length)` combinations **not seen during training**
- **Benefit**: Properly tests whether the model can generate sequences for unseen condition combinations
- **Implementation**: Run `create_stratified_split.py` before training to generate stratified splits

**Creating Stratified Splits**:
```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training
python create_stratified_split.py --dEdge_bin_size 0.01 --train_ratio 0.75 --valid_ratio 0.15 --test_ratio 0.1
```

**Note**: The default `dEdge_bin_size` is 0.01 (finer-grained bins) to ensure better stratification of (dEdge, length) combinations.

This creates stratified splits in `ML_dEdge_gen/v1+2/data/stratified/` that are automatically used during training.

**Note**: The training script uses stratified splits by default (`--use_stratified` flag). You can disable this to use random splits from the original ML_dEdge data, but this is not recommended as it doesn't properly test generalization.

## Usage

### Step 0: Create Stratified Data Splits (Required)

Before training, you must create stratified data splits to ensure proper evaluation:

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training
python create_stratified_split.py --dEdge_bin_size 0.01 --train_ratio 0.75 --valid_ratio 0.15 --test_ratio 0.1
```

This will create stratified splits in `ML_dEdge_gen/v1+2/data/stratified/` that ensure `(dEdge, length)` combinations are held out from training. The default `dEdge_bin_size` is 0.01 (finer-grained bins) for better stratification.

### Step 1: Training the Model

After creating stratified splits, train the conditional generative model. You have two options:

#### Option A: Using SLURM (Recommended for HPC)

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training
sbatch train_generative_model.slurm
```

This will submit the training job to the SLURM queue. The script will:
- Load required modules (anaconda, gcc)
- Activate the conda environment (dl_py309)
- Train the model on combined v1+v2 data
- Save the trained model to `ML_dEdge_gen/v1+2/models/ConditionalGenerator_v1v2_minmax_lr_{lr}_bs_{batch_size}.pt`

#### Option B: Direct Execution

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training
./train_generative_model.sh
```

**Note**: Make sure you have the required modules loaded and conda environment activated before running directly.

#### Option C: Custom Training Parameters

You can customize training parameters:

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training
python train_generative_model.py \
    --epochs 400 \
    --lr 0.003 \
    --batch_size 512 \
    --d_model 768 \
    --n_layers 8 \
    --n_heads 12
```

**Training Arguments**:
- `--epochs`: Number of training epochs (default: 400)
- `--lr`: Initial learning rate (default: 0.003)
- `--batch_size`: Batch size (default: 512)
- `--ml_dedge_model_path`: Path to ML_dEdge critic model for validation evaluation (default: auto-detected)
- `--use_stratified`: Use stratified data splits (default: True)

**Example with custom parameters**:
```bash
python train_generative_model.py \
    --epochs 200 \
    --lr 0.001 \
    --batch_size 256
```

### Step 1b: RL Fine-tuning (Optional)

After MLE training, you can optionally fine-tune the model using Reinforcement Learning (REINFORCE) to better match target dEdge values:

#### Using SLURM (Recommended)

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training
sbatch train_generative_model_rl.slurm
```

This will:
- Load the MLE-trained model checkpoint
- Fine-tune using REINFORCE algorithm with frozen ML_dEdge critic
- Use reward = -MSE(predicted_dEdge, target_dEdge)
- Include entropy bonus to prevent mode collapse
- Save checkpoints to `ML_dEdge_gen/v1+2/models/ConditionalGenerator_v1v2_rl_epoch_{epoch}.pt`

#### Direct Execution

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training
./train_generative_model_rl.sh
```

**RL Training Arguments**:
- `--mle_model_path`: Path to MLE-trained model checkpoint (required)
- `--critic_model_path`: Path to ML_dEdge critic model (default: auto-detected)
- `--rl_epochs`: Number of RL training epochs (default: 100)
- `--rl_lr`: Learning rate for RL fine-tuning (default: 1e-5)
- `--rl_batch_size`: Batch size for RL training (default: 32)
- `--temperature`: Sampling temperature for sequence generation (default: 1.0)
- `--entropy_weight`: Weight for entropy bonus (default: 0.01)
- `--baseline_decay`: Decay rate for reward baseline (default: 0.99)

**Note**: RL training is slower than MLE training because it generates sequences one at a time and requires forward passes through both the generator and critic models.

### Step 2: Test Set Evaluation

After training (MLE or RL), evaluate the model on the test set:

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/training
sbatch run_test_evaluation.slurm
```

This will:
- Generate sequences for each unique (dEdge, length) combination in the test set
- Evaluate using the ML_dEdge critic model
- Save test results to `out/{experiment_id}/{run_id}/test_results/test_results_epoch_{epoch}.csv`

### Step 3: Generating Sequences

Once the model is trained, you can generate sequences using the command line. Navigate to the generation directory and run:

#### Using Shell Script (Recommended)

```bash
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/scripts/generation
./generate_sequences.sh <model_path> <dEdge_min> <dEdge_max> <seq_length_min> <seq_length_max> <num_sequences> [temperature]
```

**Required Arguments:**
- `model_path`: Path to the trained model checkpoint (e.g., `../models/ConditionalGenerator_v1v2_minmax_lr_0.003_bs_512.pt`)
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
    ../models/ConditionalGenerator_v1v2_minmax_lr_0.003_bs_512.pt \
    0.5 1.0 6 8 100
```

This generates 100 sequences with:
- dEdge values between 0.5 and 1.0
- Sequence lengths between 6 and 8 amino acids
- Default temperature of 1.0

**Example 2: With custom temperature**

```bash
./generate_sequences.sh \
    ../models/ConditionalGenerator_v1v2_minmax_lr_0.003_bs_512.pt \
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
    --model_path ../models/ConditionalGenerator_v1v2_minmax_lr_0.003_bs_512.pt \
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

The model is trained using **Maximum Likelihood Estimation (MLE)** with the following approach:

#### Training Loss

**Reconstruction Loss (Cross-Entropy)**:
- Uses standard cross-entropy loss between predicted token distributions and ground truth sequences
- Loss: `L_recon = CrossEntropyLoss(predicted_tokens, target_tokens)`
- This is the only loss used during training (pure MLE)

#### Training Techniques

**Teacher Forcing**:
- Always uses ground truth tokens during training (100% teacher forcing)
- No scheduled sampling or mixed sampling
- Ensures stable training and faster convergence

**Min-Max Normalization**:
- dEdge values are normalized using min-max scaling: `dEdge_norm = (dEdge - dEdge_min) / (dEdge_max - dEdge_min)`
- Normalization statistics are computed from all data (train + valid + test) to ensure consistency
- Matches the normalization used by the pre-trained ML_dEdge critic model

**Learning Rate Scheduling**:
- Learning rate warmup: 10 epochs of linear warmup from 0 to initial learning rate
- ReduceLROnPlateau: Reduces learning rate by factor of 0.5 when validation loss plateaus
- Minimum learning rate: 1e-5
- Patience: 10 epochs

**Gradient Clipping**:
- Maximum gradient norm: 1.0
- Prevents exploding gradients during training

#### Validation Evaluation

During validation, the model uses a pre-trained ML_dEdge model as a critic for evaluation (not training):

- **Critic Model**: Pre-trained ML_dEdge Transformer model (frozen, no gradients)
  - Default path: `ML_dEdge/models/iteration2/Transformer_curriculum_lr_0.2_bs_1024.pt`
  - Used only for evaluation, not during training
- **Validation Process**:
  1. When a best model is found (based on validation reconstruction loss), sequences are generated for each unique (dEdge, length) combination
  2. Number of sequences generated per condition: `seq_length * (19/3)^(seq_length-1)`
  3. Generated sequences are evaluated using the critic model to predict their dEdge values
  4. Validation metrics calculated:
     - Unique sequence fraction (diversity measure)
     - Average dEdge prediction error (MSE between predicted and target dEdge)
     - Average predicted dEdge value
  5. Validation results are saved to MLflow output directory: `out/{experiment_id}/{run_id}/validation_results/validation_results_epoch_{epoch}.csv`

**Note**: The critic model is only used for evaluation during validation. It does not affect the training process or gradients.

### RL Fine-tuning Process

After MLE training, the model can be fine-tuned using **Reinforcement Learning (REINFORCE)** to better match target dEdge values:

#### RL Training Loss

**Policy Gradient Loss (REINFORCE)**:
- Loss: `L_RL = -mean((log_probs * advantages) + entropy_bonus * entropy)`
- `log_probs`: Log probabilities of generated sequences (normalized by sequence length)
- `advantages`: `reward - baseline` (detached to prevent gradients through critic)
- `reward`: `-MSE(predicted_dEdge_normalized, target_dEdge_normalized)`
- `baseline`: Exponential moving average of rewards (reduces variance)
- `entropy_bonus`: Weighted entropy term to encourage exploration

#### RL Training Techniques

**Frozen Critic Model**:
- Uses pre-trained ML_dEdge model as frozen critic (no gradients)
- Critic predicts dEdge values for generated sequences
- Rewards are computed as negative MSE between predicted and target dEdge
- Advantages are detached before multiplying with log_probs

**Sequence Generation**:
- Generates sequences one at a time (not batched during generation)
- Uses temperature-based sampling from model's output distribution
- Log probabilities are recomputed by forward passes through the model
- Sequences are truncated to target length after generation

**Gradient Flow**:
- Only generator model parameters receive gradients
- Critic model is frozen (no gradients)
- Advantages are detached to prevent gradients flowing through reward signal
- Log probabilities retain gradients for policy gradient updates

**Debugging Tools**:
- `test_rl_workflow.py`: Tests RL workflow components without backpropagation
- `debug_rl_sequences.py`: Inspects generated sequences and log probability computation
- `debug_rl_zero_loss.py`: Investigates scenarios leading to zero loss/entropy

**Note**: RL training is slower than MLE training because it generates sequences individually and requires forward passes through both generator and critic models.

## Training Metrics and Analysis

### Metrics Logged During Training

Training metrics are automatically logged to MLflow and saved to CSV files with dynamic naming:
- Format: `training_metrics_lr{lr}_bs{bs}_ep{epochs}_{number}.csv`
- Files are numbered sequentially (1, 2, 3, ...) for multiple training runs
- Metrics logged to MLflow: `learning_rate`, `train_loss`, `valid_loss`
- Metrics saved to CSV: epoch, train_loss, valid_loss, learning_rate, is_best

### Analysis and Plotting

An analysis directory is available at `ML_dEdge_gen/v1+2/analysis/plotting/` with R scripts for visualizing training results:

**Plot Training Metrics:**
```r
# In R or RStudio - plots all training runs by default
source("plot_training_metrics.R")

# To plot specific files, set plot_all = FALSE and specify files:
# plot_all <- FALSE
# specific_files <- c("training_metrics_lr0.001_bs512_lam0.5_ep200_1.csv", "training_metrics_lr0.001_bs512_lam0.5_ep200_2.csv")
```

Generates plots for:
- Training and validation loss curves
- Learning rate schedule
- Validation results (when available):
  - Predicted dEdge vs True dEdge (with identity line)
  - MSE dEdge vs True dEdge
  - Generation Loss vs True dEdge

## Notes

- The generative model requires a trained model checkpoint from the conditional generative training
- Sequence length range cannot exceed the model's maximum sequence length (default: 10)
- Temperature parameter controls the randomness of generation (lower = more deterministic, higher = more diverse)
- The model generates sequences directly based on conditions, ensuring better control over output properties
- Training metrics are automatically logged to CSV for analysis and plotting

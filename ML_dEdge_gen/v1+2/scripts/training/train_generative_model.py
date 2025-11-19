"""
Training script for conditional generative model for peptide sequence generation.

This script trains a conditional Transformer decoder that can generate peptide sequences
based on target dEdge values and sequence lengths. The model learns to generate sequences
autoregressively, conditioned on:
- dEdge value (difference in edge statistics between dimer and monomer)
- Sequence length

The model is trained on combined data from both iteration1 (v1) and iteration2 (v2),
which are automatically loaded and merged during training.

The model is trained using Maximum Likelihood Estimation (MLE) with teacher forcing.
During validation, a pre-trained ML_dEdge model is used as a critic to evaluate
whether generated sequences match target dEdge values, but this is for evaluation
only and does not affect training.

Usage:
    python train_generative_model.py [--epochs EPOCHS] [--lr LR] [--batch_size BATCH_SIZE] ...
    
    Or use the shell script:
    ./train_generative_model.sh
    
    Or submit via SLURM:
    sbatch train_generative_model.slurm

Output:
    Trained model saved to: ML_dEdge_gen/v1+2/models/ConditionalGenerator_v1v2_minmax_lr_{lr}_bs_{batch_size}.pt
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
import sys
import argparse
import os
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch

# Define base path once at the beginning
PROJECT_ROOT = '/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide'
BASE_DIR = os.path.join(PROJECT_ROOT, 'ML_dEdge_gen', 'v1+2')
TRAINING_DIR = os.path.join(BASE_DIR, 'scripts', 'training')
DATA_DIR_ITER1 = os.path.join(PROJECT_ROOT, 'ML_dEdge', 'data', 'iteration1', 'training', 'Sequential_Peptides_edges')
DATA_DIR_ITER2 = os.path.join(PROJECT_ROOT, 'ML_dEdge', 'data', 'iteration2', 'training', 'Sequential_Peptides_edges')
STRATIFIED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'stratified')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_SAVE_DIR = os.path.join(BASE_DIR, 'data')

# Add paths to sys.path
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'OG_util_py'))
from utils_seq import *
from models_seq_OG import get_attn_subsequence_mask
sys.path.append(TRAINING_DIR)
from models_gen import *

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

class GenerativeDataset(Dataset):
    """
    Dataset class for conditional generative model training.
    
    This dataset prepares sequences for autoregressive generation with conditional inputs.
    Each sample contains:
    - enc_input: Target sequence (ground truth) encoded as token indices
    - dec_input: Decoder input (shifted by one position for teacher forcing)
    - condition: Conditional inputs [dEdge_value, sequence_length]
    
    Args:
        sequences: Array of peptide sequences (amino acid strings)
        dEdge_values: Array of corresponding dEdge values (labels)
        src_len: Maximum sequence length for padding/truncation
        dEdge_min: Minimum for dEdge min-max normalization (optional)
        dEdge_range: Range (max - min) for dEdge min-max normalization (optional)
    """
    def __init__(self, sequences, dEdge_values, src_len, dEdge_min=None, dEdge_range=None):
        self.sequences = sequences
        self.dEdge_values = dEdge_values
        self.src_len = src_len
        self.dEdge_min = dEdge_min
        self.dEdge_range = dEdge_range if dEdge_range is not None and dEdge_range > 0 else 1.0
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Returns:
            enc_input: Target sequence as token indices [src_len]
            dec_input: Decoder input (shifted target) [src_len]
            condition: Conditional inputs [dEdge, seq_length] [2]
            seq_mask: Mask indicating valid positions (True for valid, False for PAD) [src_len]
        """
        seq = self.sequences[idx]
        dEdge = self.dEdge_values[idx]
        seq_length = len(seq)
        
        # Convert sequence to token indices using vocabulary
        enc_input = [src_vocab[n] for n in list(seq)]
        
        # Create mask for valid positions (before padding)
        seq_mask = [True] * len(enc_input)
        
        # Pad to src_len with padding token (0)
        while len(enc_input) < self.src_len:
            enc_input.append(0)
            seq_mask.append(False)
        
        enc_input = torch.LongTensor(enc_input[:self.src_len])
        seq_mask = torch.BoolTensor(seq_mask[:self.src_len])
        
        # Create decoder input: shift target sequence by one position for teacher forcing
        # Start token = 1, then the full sequence
        # dec_input should be: [START, token1, token2, ..., tokenN] where N = seq_length
        # Then pad the rest
        dec_input_list = [1] + [src_vocab[n] for n in list(seq)]  # Start + full sequence
        
        # Pad decoder input to src_len
        while len(dec_input_list) < self.src_len:
            dec_input_list.append(0)
        
        dec_input = torch.LongTensor(dec_input_list[:self.src_len])
        
        # Normalize dEdge using min-max normalization if parameters are provided
        if self.dEdge_min is not None:
            dEdge_normalized = (dEdge - self.dEdge_min) / self.dEdge_range
        else:
            dEdge_normalized = dEdge
        
        # Condition tensor: [dEdge_value (normalized), sequence_length]
        condition = torch.FloatTensor([dEdge_normalized, seq_length])
        
        return enc_input, dec_input, condition, seq_mask

def create_combined_mask(pad_mask, causal_mask):
    """
    Combine padding mask and causal mask for decoder self-attention.
    
    Args:
        pad_mask: [batch_size, seq_len] - True for PAD tokens, False for valid tokens
        causal_mask: [batch_size, seq_len, seq_len] - Causal mask for autoregressive generation
    
    Returns:
        combined_mask: [batch_size, seq_len, seq_len] - Combined mask (True = masked position)
    """
    # pad_mask is True for PAD, we need to expand it to [batch, seq_len, seq_len]
    # where each row is the same (can't attend TO pad positions)
    pad_mask_expanded = pad_mask.unsqueeze(1).expand(-1, pad_mask.size(1), -1)  # [batch, seq_len, seq_len]
    
    # Combine: position is masked if EITHER causal OR padding says so
    combined_mask = causal_mask | pad_mask_expanded
    
    return combined_mask

def parse_args():
    """
    Parse command line arguments for training.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all training and model parameters
    """
    parser = argparse.ArgumentParser(
        description='Train conditional generative model for peptide sequence generation'
    )
    
    # Training hyperparameters (must be provided by shell script)
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for reproducibility')
    
    # Model architecture parameters (must be provided by shell script)
    parser.add_argument('--src_vocab_size', type=int, required=True,
                        help='Vocabulary size (20 amino acids + padding token)')
    parser.add_argument('--src_len', type=int, required=True,
                        help='Maximum sequence length (sequences will be padded/truncated to this)')
    parser.add_argument('--d_model', type=int, required=True,
                        help='Transformer embedding dimension')
    parser.add_argument('--d_ff', type=int, required=True,
                        help='Feed-forward network dimension')
    parser.add_argument('--d_k', type=int, required=True,
                        help='Key dimension for attention mechanism')
    parser.add_argument('--d_v', type=int, required=True,
                        help='Value dimension for attention mechanism')
    parser.add_argument('--n_layers', type=int, required=True,
                        help='Number of decoder layers')
    parser.add_argument('--n_heads', type=int, required=True,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, required=True,
                        help='Dropout rate')
    
    # Model path for critic (ML_dEdge model)
    parser.add_argument('--ml_dedge_model_path', type=str, 
                        default=os.path.join(PROJECT_ROOT, 'ML_dEdge', 'models', 'iteration2', 'Transformer_curriculum_lr_0.2_bs_1024.pt'),
                        help='Path to pre-trained ML_dEdge model (critic)')
    
    # Data loading parameters
    parser.add_argument('--no_stratified', action='store_true', default=False,
                        help='Disable stratified splits and use random splits instead (stratified is default)')
    
    args = parser.parse_args()
    # Set use_stratified based on no_stratified flag (default is True, unless --no_stratified is set)
    args.use_stratified = not args.no_stratified
    
    return args

def load_combined_data(use_stratified=True):
    """
    Load training data for generative model.
    
    If use_stratified=True, loads stratified splits that ensure (dEdge, length)
    combinations are held out from training for proper generalization evaluation.
    
    If use_stratified=False, loads random splits from original ML_dEdge data.
    
    Args:
        use_stratified: If True, use stratified splits; if False, use random splits
    
    Returns:
        tuple: (df_train, df_valid, df_test) - DataFrames with columns:
            - Feature: Peptide sequences (amino acid strings)
            - Label: dEdge values (float)
    """
    if use_stratified:
        # Load stratified splits (ensures (dEdge, length) combinations are held out)
        stratified_train_path = os.path.join(STRATIFIED_DATA_DIR, 'stratified_train_seqs.csv')
        stratified_valid_path = os.path.join(STRATIFIED_DATA_DIR, 'stratified_valid_seqs.csv')
        stratified_test_path = os.path.join(STRATIFIED_DATA_DIR, 'stratified_test_seqs.csv')
        
        if not all(os.path.exists(p) for p in [stratified_train_path, stratified_valid_path, stratified_test_path]):
            raise FileNotFoundError(
                f"Stratified data files not found. Please run create_stratified_split.py first.\n"
                f"Expected files:\n"
                f"  {stratified_train_path}\n"
                f"  {stratified_valid_path}\n"
                f"  {stratified_test_path}"
            )
        
        print("Loading stratified splits (ensures (dEdge, length) combinations are held out)...")
        df_train = pd.read_csv(stratified_train_path)
        df_valid = pd.read_csv(stratified_valid_path)
        df_test = pd.read_csv(stratified_test_path)
        
        print(f"Stratified training set size: {len(df_train)}")
        print(f"Stratified validation set size: {len(df_valid)}")
        print(f"Stratified test set size: {len(df_test)}")
        print("Note: Validation/test sets contain (dEdge, length) combinations not seen in training")
        
    else:
        # Load random splits from original ML_dEdge data
        print("Loading data from iteration1 (v1)...")
        df_train_v1 = pd.read_csv(f'{DATA_DIR_ITER1}/ddedge_train_seqs.csv')
        df_valid_v1 = pd.read_csv(f'{DATA_DIR_ITER1}/ddedge_valid_seqs.csv')
        df_test_v1 = pd.read_csv(f'{DATA_DIR_ITER1}/ddedge_test_seqs.csv')
        
        print("Loading data from iteration2 (v2)...")
        df_train_v2 = pd.read_csv(f'{DATA_DIR_ITER2}/dedge_train_seqs.csv')
        df_valid_v2 = pd.read_csv(f'{DATA_DIR_ITER2}/dedge_valid_seqs.csv')
        df_test_v2 = pd.read_csv(f'{DATA_DIR_ITER2}/dedge_test_seqs.csv')
        
        print("Combining datasets...")
        # Concatenate datasets from both iterations
        df_train = pd.concat([df_train_v1, df_train_v2], ignore_index=True)
        df_valid = pd.concat([df_valid_v1, df_valid_v2], ignore_index=True)
        df_test = pd.concat([df_test_v1, df_test_v2], ignore_index=True)
        
        print(f"Combined training set size: {len(df_train)}")
        print(f"Combined validation set size: {len(df_valid)}")
        print(f"Combined test set size: {len(df_test)}")
        print("Warning: Using random splits - validation may contain (dEdge, length) combinations seen in training")
    
    return df_train, df_valid, df_test

def load_ml_dedge_critic(model_path, device, src_len=10, dEdge_min=None, dEdge_max=None):
    """
    Load pre-trained ML_dEdge model as a frozen critic for validation evaluation.
    
    Args:
        model_path: Path to saved ML_dEdge model checkpoint
        device: Device to load model on
        src_len: Maximum sequence length (must match training)
    
    Returns:
        Frozen ML_dEdge model ready to use as critic
    """
    # Import models_seq_OG here to avoid import issues when adversarial training is not used
    from models_seq_OG import Transformer
    
    # Create args for ML_dEdge model (must match training configuration)
    # Note: min/max will be set from data when called
    class Args:
        def __init__(self, dEdge_min=None, dEdge_max=None):
            self.task_type = 'Regression'
            self.src_vocab_size = 21
            self.src_len = src_len
            self.model = 'Transformer'
            self.dropout = 0.1
            self.d_model = 512
            self.d_ff = 2048
            self.d_k = 64
            self.d_v = 64
            self.n_layers = 6
            self.n_heads = 8
            # Use provided min/max or defaults
            self.min = dEdge_min if dEdge_min is not None else -2.0
            self.max = dEdge_max if dEdge_max is not None else 3.5
    
    model_args = Args(dEdge_min=dEdge_min, dEdge_max=dEdge_max)
    critic_model = Transformer(model_args).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    # Check if checkpoint is a dict with 'model_state_dict' or just the state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        critic_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Checkpoint is the state dict directly
        critic_model.load_state_dict(checkpoint)
    
    # Freeze model (no gradients)
    for param in critic_model.parameters():
        param.requires_grad = False
    critic_model.eval()
    
    return critic_model

def decode_sequence_from_tokens(token_ids, vocab_reverse):
    """Decode token IDs back to amino acid sequence string."""
    sequence = []
    for token_id in token_ids:
        if token_id == 0:  # PAD token
            break
        token_id_int = int(token_id)
        if token_id_int in vocab_reverse:
            aa = vocab_reverse[token_id_int]
            if aa != 'Empty':
                sequence.append(aa)
    return ''.join(sequence)

def main():
    """
    Main training function for conditional generative model with MLE training.
    """
    print("="*60)
    print("Starting training script...")
    print("="*60)
    sys.stdout.flush()
    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    sys.stdout.flush()
    
    # Set up MLflow tracking
    mlflow_dir = os.path.join(BASE_DIR, 'out')
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment("ML_dEdge_gen_v1v2")
    run_name = f"d{args.d_model}_l{args.n_layers}_h{args.n_heads}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"
    mlflow.start_run(run_name=run_name)
    print(f"MLflow run started: {run_name}")
    sys.stdout.flush()
    
    # Log hyperparameters
    mlflow.log_params({
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'src_vocab_size': args.src_vocab_size,
        'src_len': args.src_len,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'd_k': args.d_k,
        'd_v': args.d_v,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'dropout': args.dropout,
        'use_stratified': args.use_stratified,
    })
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load training data
    print("Loading data...")
    sys.stdout.flush()
    df_train, df_valid, df_test = load_combined_data(use_stratified=args.use_stratified)
    
    # Compute dEdge min-max normalization statistics from ALL data (train + valid + test)
    all_dEdge = np.concatenate([
        df_train["Label"].values,
        df_valid["Label"].values,
        df_test["Label"].values
    ])
    dEdge_min = float(np.min(all_dEdge))
    dEdge_max = float(np.max(all_dEdge))
    dEdge_range = dEdge_max - dEdge_min
    
    print(f"\n{'='*60}")
    print(f"dEdge Min-Max Normalization Statistics (from all data):")
    print(f"  Min: {dEdge_min:.6f}")
    print(f"  Max: {dEdge_max:.6f}")
    print(f"  Range: {dEdge_range:.6f}")
    print(f"  Mean: {np.mean(all_dEdge):.6f}")
    print(f"  Std: {np.std(all_dEdge):.6f}")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    # Load ML_dEdge critic model for validation evaluation (use actual dEdge min/max from data)
    print(f"Loading ML_dEdge critic model from {args.ml_dedge_model_path}")
    sys.stdout.flush()
    critic_model = load_ml_dedge_critic(args.ml_dedge_model_path, device, args.src_len, 
                                       dEdge_min=dEdge_min, dEdge_max=dEdge_max)
    print("Critic model frozen (no gradients)")
    print("Critic model (ML_dEdge) loaded for dEdge prediction")
    sys.stdout.flush()
    
    # Create datasets
    train_dataset = GenerativeDataset(
        sequences=df_train['Feature'].values,
        dEdge_values=df_train['Label'].values,
        src_len=args.src_len,
        dEdge_min=dEdge_min,
        dEdge_range=dEdge_range
    )
    
    valid_dataset = GenerativeDataset(
        sequences=df_valid['Feature'].values,
        dEdge_values=df_valid['Label'].values,
        src_len=args.src_len,
        dEdge_min=dEdge_min,
        dEdge_range=dEdge_range
    )
    
    test_dataset = GenerativeDataset(
        sequences=df_test['Feature'].values,
        dEdge_values=df_test['Label'].values,
        src_len=args.src_len,
        dEdge_min=dEdge_min,
        dEdge_range=dEdge_range
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("Creating model...")
    sys.stdout.flush()
    model = ConditionalGenerator(args).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    sys.stdout.flush()
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD tokens
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5, verbose=True
    )
    
    # Create training metrics log file
    metrics_base_name = f'training_metrics_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}'
    metrics_dir = os.path.join(BASE_DIR, 'data')
    os.makedirs(metrics_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(metrics_dir) if f.startswith(metrics_base_name) and f.endswith('.csv')]
    
    if existing_files:
        numbers = []
        for f in existing_files:
            try:
                num = int(f.replace(metrics_base_name + '_', '').replace('.csv', ''))
                numbers.append(num)
            except ValueError:
                continue
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1
    
    metrics_log_path = os.path.join(metrics_dir, f'{metrics_base_name}_{next_num}.csv')
    metrics_log = []
    print(f"Training metrics will be saved to: {metrics_log_path}")
    sys.stdout.flush()
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"MLE Training Phase: {args.epochs} epochs")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    best_valid_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        
        # Training phase
        for batch_idx, (enc_inputs, dec_inputs, conditions, seq_masks) in enumerate(train_loader):
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            conditions = conditions.to(device)
            seq_masks = seq_masks.to(device)
            
            # Create masks
            pad_mask = ~seq_masks  # True for PAD, False for valid
            causal_mask = get_attn_subsequence_mask(dec_inputs).to(device)
            dec_self_attn_mask = create_combined_mask(pad_mask, causal_mask)
            dec_enc_attn_mask = torch.zeros(dec_inputs.size(0), dec_inputs.size(1), 1).bool().to(device)
            
            # Forward pass
            dec_logits = model(dec_inputs, conditions, dec_self_attn_mask, dec_enc_attn_mask)
            
            # Loss computation: logits[i] predicts enc_inputs[i]
            # Flatten and mask out PAD positions
            logits_flat = dec_logits.view(-1, dec_logits.size(-1))
            targets_flat = enc_inputs.view(-1)
            mask_flat = seq_masks.view(-1)
            
            # Keep only valid positions
            logits_flat = logits_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]
            
            loss = ce_criterion(logits_flat, targets_flat)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        valid_losses = []
        
        with torch.no_grad():
            for enc_inputs, dec_inputs, conditions, seq_masks in valid_loader:
                enc_inputs = enc_inputs.to(device)
                dec_inputs = dec_inputs.to(device)
                conditions = conditions.to(device)
                seq_masks = seq_masks.to(device)
                
                pad_mask = ~seq_masks
                causal_mask = get_attn_subsequence_mask(dec_inputs).to(device)
                dec_self_attn_mask = create_combined_mask(pad_mask, causal_mask)
                dec_enc_attn_mask = torch.zeros(dec_inputs.size(0), dec_inputs.size(1), 1).bool().to(device)
                
                dec_logits = model(dec_inputs, conditions, dec_self_attn_mask, dec_enc_attn_mask)
                
                logits_flat = dec_logits.view(-1, dec_logits.size(-1))
                targets_flat = enc_inputs.view(-1)
                mask_flat = seq_masks.view(-1)
                
                logits_flat = logits_flat[mask_flat]
                targets_flat = targets_flat[mask_flat]
                
                loss = ce_criterion(logits_flat, targets_flat)
                valid_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step(valid_loss)
        
        # Check if best model
        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_loss
            best_epoch = epoch
            
            # Save best model
            model_path = os.path.join(MODEL_SAVE_DIR, 
                                     f'ConditionalGenerator_v1v2_minmax_lr_{args.lr}_bs_{args.batch_size}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'args': args,
                'dedge_min': dEdge_min,
                'dedge_max': dEdge_max,
                'dedge_range': dEdge_range
            }, model_path)
            print(f'Epoch {epoch+1}: New best model saved! (Valid Loss: {valid_loss:.4f})')
            sys.stdout.flush()
        
        # Log metrics
        mlflow.log_metrics({
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'learning_rate': current_lr
        }, step=epoch + 1)
        
        metrics_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'learning_rate': current_lr,
            'is_best': is_best
        })
        
        metrics_df = pd.DataFrame(metrics_log)
        metrics_df.to_csv(metrics_log_path, index=False)
        
        print(f'Epoch {epoch+1}/{args.epochs}: '
              f'Train Loss: {train_loss:.4f} | '
              f'Valid Loss: {valid_loss:.4f} | '
              f'LR: {current_lr:.2e}')
        sys.stdout.flush()
    
    # Training summary
    print(f'\n{"="*60}')
    print(f'MLE Training completed!')
    print(f'Best model from epoch {best_epoch + 1}/{args.epochs}')
    print(f'Best validation loss: {best_valid_loss:.4f}')
    print(f'Model saved to: {model_path}')
    print(f'Training metrics saved to: {metrics_log_path}')
    print(f'{"="*60}')
    sys.stdout.flush()
    
    # Generate test results with best model
    print(f'\n{"="*60}')
    print(f'Generating test results with best model...')
    print(f'{"="*60}')
    sys.stdout.flush()
    
    # Create reverse vocabulary for decoding
    src_vocab_reverse = {v: k for k, v in src_vocab.items()}
    
    # Load best model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_results = []
    
    with torch.no_grad():
        # Get unique conditions from test set
        unique_conditions = {}
        for enc_inputs, dec_inputs, conditions, seq_masks in test_loader:
            conditions_np = conditions.cpu().numpy()
            for cond in conditions_np:
                dEdge_norm = cond[0]
                seq_len = int(cond[1])
                key = (dEdge_norm, seq_len)
                if key not in unique_conditions:
                    unique_conditions[key] = {
                        'dEdge_norm': dEdge_norm,
                        'dEdge_orig': dEdge_norm * dEdge_range + dEdge_min,
                        'seq_len': seq_len
                    }
        
        # Generate sequences for each unique condition
        for (dEdge_norm, seq_len), cond_info in unique_conditions.items():
            dEdge_orig = cond_info['dEdge_orig']
            n_generate = int(seq_len * (19/3) ** (seq_len - 1))
            
            condition_tensor = torch.FloatTensor([[dEdge_norm, seq_len]]).to(device)
            
            generated_seqs = []
            generated_dEdges = []
            
            for _ in range(n_generate):
                generated_tokens = model.generate(
                    condition_tensor,
                    max_len=args.src_len,
                    start_token=1,
                    temperature=1.0
                )
                seq_tokens = generated_tokens[0].cpu().numpy()
                seq_str = decode_sequence_from_tokens(seq_tokens, src_vocab_reverse)
                if len(seq_str) > 0:
                    generated_seqs.append(seq_str)
                    
                    # Predict dEdge using critic
                    gen_enc_inputs = make_data(np.array([seq_str]), args.src_len).to(device)
                    pred_dEdge_orig = critic_model(gen_enc_inputs).squeeze().item()
                    generated_dEdges.append(pred_dEdge_orig)
            
            if len(generated_seqs) > 0:
                unique_seqs = set(generated_seqs)
                unique_fraction = len(unique_seqs) / len(generated_seqs)
                
                dEdge_errors = [(dEdge_orig - pred) ** 2 for pred in generated_dEdges]
                avg_dEdge_error = np.mean(dEdge_errors)
                
                test_results.append({
                    'epoch': best_epoch + 1,
                    'true_dEdge': dEdge_orig,
                    'seq_length': seq_len,
                    'n_generated': len(generated_seqs),
                    'n_unique': len(unique_seqs),
                    'unique_fraction': unique_fraction,
                    'avg_dEdge_error': avg_dEdge_error,
                    'avg_predicted_dEdge': np.mean(generated_dEdges)
                })
    
    # Save test results
    if test_results:
        active_run = mlflow.active_run()
        if active_run:
            experiment_id = active_run.info.experiment_id
            run_id = active_run.info.run_id
            test_results_dir = os.path.join(BASE_DIR, 'out', str(experiment_id), run_id, "test_results")
        else:
            test_results_dir = os.path.join(BASE_DIR, 'out', 'default', f'epoch_{best_epoch+1}', "test_results")
        
        os.makedirs(test_results_dir, exist_ok=True)
        
        test_results_path = os.path.join(test_results_dir, f'test_results_epoch_{best_epoch+1}.csv')
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(test_results_path, index=False)
        
        print(f'Test results saved: {test_results_path}')
        print(f'Average unique fraction: {np.mean([r["unique_fraction"] for r in test_results]):.4f}')
        print(f'Average dEdge error: {np.mean([r["avg_dEdge_error"] for r in test_results]):.4f}')
        print(f'{"="*60}')
        sys.stdout.flush()
    
    # End MLflow run
    mlflow.end_run()
    print("Training completed!")

if __name__ == '__main__':
    main()

"""
Training script for conditional generative model for peptide sequence generation.

This script trains a conditional Transformer decoder that can generate peptide sequences
based on target dEdge values and sequence lengths. The model learns to generate sequences
autoregressively, conditioned on:
- dEdge value (difference in edge statistics between dimer and monomer)
- Sequence length

The model is trained on combined data from both iteration1 (v1) and iteration2 (v2),
which are automatically loaded and merged during training.

Usage:
    python train_generative_model.py [--epochs EPOCHS] [--lr LR] [--batch_size BATCH_SIZE] ...
    
    Or use the shell script:
    ./train_generative_model.sh
    
    Or submit via SLURM:
    sbatch train_generative_model.slurm

Output:
    Trained model saved to: ML_dEdge_gen/v1+2/models/ConditionalGenerator_v1v2_lr_{lr}_bs_{batch_size}.pt
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

# Define base path once at the beginning
PROJECT_ROOT = '/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide'
BASE_DIR = os.path.join(PROJECT_ROOT, 'ML_dEdge_gen', 'v1+2')
TRAINING_DIR = os.path.join(BASE_DIR, 'scripts', 'training')
DATA_DIR_ITER1 = os.path.join(PROJECT_ROOT, 'ML_dEdge', 'data', 'iteration1', 'training', 'Sequential_Peptides_edges')
DATA_DIR_ITER2 = os.path.join(PROJECT_ROOT, 'ML_dEdge', 'data', 'iteration2', 'training', 'Sequential_Peptides_edges')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_SAVE_DIR = os.path.join(BASE_DIR, 'data')

# Add paths to sys.path
# utils_seq.py is in OG_util_py directory, same as other ML_dEdge scripts
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'OG_util_py'))
from utils_seq import *
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
    """
    def __init__(self, sequences, dEdge_values, src_len):
        self.sequences = sequences
        self.dEdge_values = dEdge_values
        self.src_len = src_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Returns:
            enc_input: Target sequence as token indices [src_len]
            dec_input: Decoder input (shifted target) [src_len]
            condition: Conditional inputs [dEdge, seq_length] [2]
        """
        seq = self.sequences[idx]
        dEdge = self.dEdge_values[idx]
        seq_length = len(seq)
        
        # Convert sequence to token indices using vocabulary
        enc_input = [src_vocab[n] for n in list(seq)]
        # Pad to src_len with padding token (0)
        while len(enc_input) < self.src_len:
            enc_input.append(0)
        enc_input = torch.LongTensor(enc_input[:self.src_len])
        
        # Create decoder input: shift target sequence by one position for teacher forcing
        # Start token = 1, then the sequence shifted by one
        dec_input = torch.cat([torch.LongTensor([1]), enc_input[:-1]])  # Start token = 1
        
        # Condition tensor: [dEdge_value, sequence_length]
        # These are the conditional inputs that guide sequence generation
        condition = torch.FloatTensor([dEdge, seq_length])
        
        return enc_input, dec_input, condition

def parse_args():
    """
    Parse command line arguments for training.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all training and model parameters
    """
    parser = argparse.ArgumentParser(
        description='Train conditional generative model for peptide sequence generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model architecture parameters
    parser.add_argument('--src_vocab_size', type=int, default=21,
                        help='Vocabulary size (20 amino acids + padding token)')
    parser.add_argument('--src_len', type=int, default=10,
                        help='Maximum sequence length (sequences will be padded/truncated to this)')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Transformer embedding dimension')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='Feed-forward network dimension')
    parser.add_argument('--d_k', type=int, default=64,
                        help='Key dimension for attention mechanism')
    parser.add_argument('--d_v', type=int, default=64,
                        help='Value dimension for attention mechanism')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Number of decoder layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    return parser.parse_args()

def load_combined_data():
    """
    Load and combine training data from both iteration1 (v1) and iteration2 (v2).
    
    This function:
    1. Loads data files from both iterations (different file prefixes: ddedge_ vs dedge_)
    2. Combines training, validation, and test sets from both iterations
    3. Returns the combined datasets for training
    
    Data sources:
    - iteration1: ML_dEdge/data/iteration1/training/Sequential_Peptides_edges/
      Files use 'ddedge_' prefix
    - iteration2: ML_dEdge/data/iteration2/training/Sequential_Peptides_edges/
      Files use 'dedge_' prefix
    
    Returns:
        tuple: (df_train, df_valid, df_test) - Combined DataFrames with columns:
            - Feature: Peptide sequences (amino acid strings)
            - Label: dEdge values (float)
    """
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
    
    return df_train, df_valid, df_test

def main():
    """
    Main training function for conditional generative model.
    
    Training process:
    1. Set random seeds for reproducibility
    2. Load and combine data from both iterations
    3. Create PyTorch datasets and data loaders
    4. Initialize model, optimizer, and loss function
    5. Train for specified number of epochs with:
       - Teacher forcing during training
       - Gradient clipping for stability
       - Best model checkpointing based on validation loss
    6. Save final model checkpoint
    """
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load and combine data from both iterations (v1 and v2)
    df_train, df_valid, df_test = load_combined_data()
    
    # Prepare PyTorch datasets
    # Each dataset contains sequences, dEdge values, and prepares conditional inputs
    train_dataset = GenerativeDataset(
        np.array(df_train["Feature"]),
        np.array(df_train["Label"]),
        args.src_len
    )
    valid_dataset = GenerativeDataset(
        np.array(df_valid["Feature"]),
        np.array(df_valid["Label"]),
        args.src_len
    )
    
    # Create data loaders with specified batch size
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model, optimizer, and loss function
    model = ConditionalGenerator(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Cross-entropy loss ignoring padding token (index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Model save path includes hyperparameters for easy identification
    model_path = os.path.join(MODEL_SAVE_DIR, f'ConditionalGenerator_v1v2_lr_{args.lr}_bs_{args.batch_size}.pt')
    
    print(f"Training on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    best_valid_loss = float('inf')
    best_epoch = 0
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        
        # Training phase: iterate through training batches
        for enc_inputs, dec_inputs, conditions in train_loader:
            # Move data to device (GPU if available)
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            conditions = conditions.to(device)
            
            # Create attention masks for decoder
            # Self-attention mask: prevents attending to future positions (causal mask)
            dec_self_attn_mask = get_attn_subsequence_mask(dec_inputs).to(device)
            # Encoder-decoder attention mask: no masking needed (using condition as encoder)
            dec_enc_attn_mask = torch.zeros(dec_inputs.size(0), dec_inputs.size(1), 1).bool().to(device)
            
            # Forward pass: generate logits for next token prediction
            dec_logits = model(dec_inputs, conditions, dec_self_attn_mask, dec_enc_attn_mask)
            
            # Reshape for loss calculation: [batch*seq_len, vocab_size] vs [batch*seq_len]
            dec_logits = dec_logits.view(-1, dec_logits.size(-1))
            targets = enc_inputs.view(-1)
            
            # Calculate loss
            loss = criterion(dec_logits, targets)
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase: evaluate on validation set
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for enc_inputs, dec_inputs, conditions in valid_loader:
                enc_inputs = enc_inputs.to(device)
                dec_inputs = dec_inputs.to(device)
                conditions = conditions.to(device)
                
                # Same mask setup as training
                dec_self_attn_mask = get_attn_subsequence_mask(dec_inputs).to(device)
                dec_enc_attn_mask = torch.zeros(dec_inputs.size(0), dec_inputs.size(1), 1).bool().to(device)
                
                # Forward pass (no gradient computation)
                dec_logits = model(dec_inputs, conditions, dec_self_attn_mask, dec_enc_attn_mask)
                dec_logits = dec_logits.view(-1, dec_logits.size(-1))
                targets = enc_inputs.view(-1)
                
                loss = criterion(dec_logits, targets)
                valid_losses.append(loss.item())
        
        # Calculate average losses
        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        
        # Save best model based on validation loss
        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'args': args  # Save arguments for model loading
            }, model_path)
            print(f'Epoch {epoch+1}: New best model saved! (Valid Loss: {valid_loss:.4f})')
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
    
    # Training summary
    print(f'\n{"="*60}')
    print(f'Training completed!')
    print(f'Best model from epoch {best_epoch + 1}/{args.epochs}')
    print(f'Best validation loss: {best_valid_loss:.4f}')
    print(f'Model saved to: {model_path}')
    print(f'{"="*60}')

if __name__ == '__main__':
    main()


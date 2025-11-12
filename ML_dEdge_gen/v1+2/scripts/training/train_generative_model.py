# Training script for conditional generative model
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
sys.path.append(PROJECT_ROOT)
from utils_seq import *
sys.path.append(TRAINING_DIR)
from models_gen import *

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

class GenerativeDataset(Dataset):
    def __init__(self, sequences, dEdge_values, src_len):
        self.sequences = sequences
        self.dEdge_values = dEdge_values
        self.src_len = src_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        dEdge = self.dEdge_values[idx]
        seq_length = len(seq)
        
        # Convert sequence to token indices
        enc_input = [src_vocab[n] for n in list(seq)]
        while len(enc_input) < self.src_len:
            enc_input.append(0)
        enc_input = torch.LongTensor(enc_input[:self.src_len])
        
        # Create decoder input (shifted by one for teacher forcing)
        dec_input = torch.cat([torch.LongTensor([1]), enc_input[:-1]])  # Start token = 1
        
        # Condition: [dEdge, seq_length]
        condition = torch.FloatTensor([dEdge, seq_length])
        
        return enc_input, dec_input, condition

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--src_vocab_size', type=int, default=21)
    parser.add_argument('--src_len', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    return parser.parse_args()

def load_combined_data():
    """Load and combine data from both iterations"""
    print("Loading data from iteration1 (v1)...")
    df_train_v1 = pd.read_csv(f'{DATA_DIR_ITER1}/ddedge_train_seqs.csv')
    df_valid_v1 = pd.read_csv(f'{DATA_DIR_ITER1}/ddedge_valid_seqs.csv')
    df_test_v1 = pd.read_csv(f'{DATA_DIR_ITER1}/ddedge_test_seqs.csv')
    
    print("Loading data from iteration2 (v2)...")
    df_train_v2 = pd.read_csv(f'{DATA_DIR_ITER2}/dedge_train_seqs.csv')
    df_valid_v2 = pd.read_csv(f'{DATA_DIR_ITER2}/dedge_valid_seqs.csv')
    df_test_v2 = pd.read_csv(f'{DATA_DIR_ITER2}/dedge_test_seqs.csv')
    
    print("Combining datasets...")
    df_train = pd.concat([df_train_v1, df_train_v2], ignore_index=True)
    df_valid = pd.concat([df_valid_v1, df_valid_v2], ignore_index=True)
    df_test = pd.concat([df_test_v1, df_test_v2], ignore_index=True)
    
    print(f"Combined training set size: {len(df_train)}")
    print(f"Combined validation set size: {len(df_valid)}")
    print(f"Combined test set size: {len(df_test)}")
    
    return df_train, df_valid, df_test

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load data
    df_train, df_valid, df_test = load_combined_data()
    
    # Prepare datasets
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = ConditionalGenerator(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    
    model_path = os.path.join(MODEL_SAVE_DIR, f'ConditionalGenerator_v1v2_lr_{args.lr}_bs_{args.batch_size}.pt')
    
    print(f"Training on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    best_valid_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        
        for enc_inputs, dec_inputs, conditions in train_loader:
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            conditions = conditions.to(device)
            
            # Create masks
            dec_self_attn_mask = get_attn_subsequence_mask(dec_inputs).to(device)
            dec_enc_attn_mask = torch.zeros(dec_inputs.size(0), dec_inputs.size(1), 1).bool().to(device)
            
            # Forward pass
            dec_logits = model(dec_inputs, conditions, dec_self_attn_mask, dec_enc_attn_mask)
            
            # Reshape for loss calculation
            dec_logits = dec_logits.view(-1, dec_logits.size(-1))
            targets = enc_inputs.view(-1)
            
            loss = criterion(dec_logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for enc_inputs, dec_inputs, conditions in valid_loader:
                enc_inputs = enc_inputs.to(device)
                dec_inputs = dec_inputs.to(device)
                conditions = conditions.to(device)
                
                dec_self_attn_mask = get_attn_subsequence_mask(dec_inputs).to(device)
                dec_enc_attn_mask = torch.zeros(dec_inputs.size(0), dec_inputs.size(1), 1).bool().to(device)
                
                dec_logits = model(dec_inputs, conditions, dec_self_attn_mask, dec_enc_attn_mask)
                dec_logits = dec_logits.view(-1, dec_logits.size(-1))
                targets = enc_inputs.view(-1)
                
                loss = criterion(dec_logits, targets)
                valid_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        
        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'args': args
            }, model_path)
            print(f'Epoch {epoch+1}: New best model saved!')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
    
    print(f'\nTraining completed!')
    print(f'Best model from epoch {best_epoch + 1} with validation loss: {best_valid_loss:.4f}')
    print(f'Model saved to: {model_path}')

if __name__ == '__main__':
    main()


# Import required libraries for numerical operations and deep learning
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
import sys
# Add the project root to Python path for importing custom modules
sys.path.append('/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/')
# Import custom utility functions and models
from utils_seq import *
from models_seq_OG import *
import argparse
import os
from torch.utils.data import ConcatDataset

# Define base paths for saving model and results
BASE_DIR = '/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge'
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models/iteration2')
RESULTS_SAVE_DIR = os.path.join(BASE_DIR, 'data/iteration2/training/Sequential_Peptides_edges')

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

# Define command line arguments for the script
parser = argparse.ArgumentParser()
# Task type argument (Regression or Classification)
parser.add_argument('--task_type', type=str, default='Regression',
                    choices=['Classification','Regression'])
# Random seed for reproducibility
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
# Number of training epochs
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
# Learning rate for optimization
parser.add_argument('--lr', type=float, default=0.2,
                    help='Initial learning rate.')
# Vocabulary size for amino acid sequences
parser.add_argument('--src_vocab_size', type=int, default=21)
# Maximum sequence length
parser.add_argument('--src_len', type=int, default=10)
# Batch size for training
parser.add_argument('--batch_size', type=int, default=1024)
# Model architecture selection
parser.add_argument('--model', type=str, default='Transformer',choices=['RNN','LSTM','Bi-LSTM','Transformer'])
# Path to save the model
parser.add_argument('--model_path', type=str, default='model_val_best.pt')
# Path to previous iteration's model
parser.add_argument('--prev_model', type=str, help='Path to previous iteration model')
# Directory containing previous iteration data
parser.add_argument('--old_data_dir', type=str, help='Directory containing previous iteration data')
# Directory containing new data
parser.add_argument('--new_data_dir', type=str, help='Directory containing new data')
# Number of curriculum learning stages
parser.add_argument('--curriculum_steps', type=int, default=3, help='Number of curriculum stages')
# Number of warmup epochs on old data
parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of epochs to train on old data')

# Parse command line arguments
args = parser.parse_args()

# Set Transformer architecture parameters
if args.model == 'Transformer':
    # Model dimension for Transformer
    args.d_model = 512
    # Feed-forward network dimension
    args.d_ff = 2048
    # Key and value dimensions for attention
    args.d_k = args.d_v = 64
    # Number of Transformer layers
    args.n_layers = 6
    # Number of attention heads
    args.n_heads = 8
else:
    # Default model dimension for other architectures
    args.d_model = 512

# Generate model path with learning rate and batch size
args.model_path = os.path.join(MODEL_SAVE_DIR, '{}_curriculum_lr_{}_bs_{}.pt'.format(args.model, args.lr, args.batch_size))

# Set device to GPU if available, else CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Set random seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

def load_data(data_dir, prefix=''):
    # Determine file prefix based on directory (iteration1 or iteration2)
    if 'iteration1' in data_dir:
        file_prefix = 'ddedge_'
    else:
        file_prefix = 'dedge_'
        
    if args.task_type == 'Regression':
        # Load training data
        df_train = pd.read_csv(f'{data_dir}/{file_prefix}train_seqs.csv')
        # Load validation data
        df_valid = pd.read_csv(f'{data_dir}/{file_prefix}valid_seqs.csv')
        # Load test data
        df_test = pd.read_csv(f'{data_dir}/{file_prefix}test_seqs.csv')
        # Convert labels to tensors and add dimension for regression
        train_label = torch.Tensor(np.array(df_train["Label"])).unsqueeze(1).float()
        valid_label = torch.Tensor(np.array(df_valid["Label"])).unsqueeze(1).float().to(device)
        test_label = torch.Tensor(np.array(df_test["Label"])).unsqueeze(1).float().to(device)
        
        # Set max/min values for label scaling
        args.max = train_label.max().item()
        args.min = train_label.min().item()
    else:
        # Load classification data
        df_train = pd.read_csv(f'{data_dir}/train_seqs_cla.csv')
        df_valid = pd.read_csv(f'{data_dir}/valid_seqs_cla.csv')
        df_test = pd.read_csv(f'{data_dir}/test_seqs_cla.csv')
        # Convert labels to tensors for classification
        train_label = torch.Tensor(np.array(df_train["Label"])).long()
        valid_label = torch.Tensor(np.array(df_valid["Label"])).long().to(device)
        test_label = torch.Tensor(np.array(df_test["Label"])).long().to(device)
    
    # Extract features from dataframes
    train_feat = np.array(df_train["Feature"])
    valid_feat = np.array(df_valid["Feature"])
    test_feat = np.array(df_test["Feature"])

    # Convert features to model input format
    train_enc_inputs = make_data(train_feat, args.src_len)
    valid_enc_inputs = make_data(valid_feat, args.src_len).to(device)
    test_enc_inputs = make_data(test_feat, args.src_len).to(device)
    
    # Return all processed data
    return (train_enc_inputs, train_label, valid_enc_inputs, valid_label, 
            test_enc_inputs, test_label, df_test)

def create_curriculum_loader(old_data, new_data, mix_ratio):
    """Create a DataLoader that mixes old and new data according to mix_ratio"""
    # Unpack old and new data
    old_inputs, old_labels = old_data
    new_inputs, new_labels = new_data
    
    # Calculate number of samples to take from each dataset
    n_old = int(len(old_labels) * mix_ratio)
    n_new = int(len(new_labels) * (1 - mix_ratio))
    
    # Randomly sample indices from each dataset
    old_indices = torch.randperm(len(old_labels))[:n_old]
    new_indices = torch.randperm(len(new_labels))[:n_new]
    
    # Combine the sampled data
    combined_inputs = torch.cat([old_inputs[old_indices], new_inputs[new_indices]])
    combined_labels = torch.cat([old_labels[old_indices], new_labels[new_indices]])
    
    # Create and return DataLoader
    return Data.DataLoader(MyDataSet(combined_inputs, combined_labels), args.batch_size, True)

def main():
    # Print loading message
    print("Loading old and new data...")
    # Load old data
    (old_train_inputs, old_train_label, old_valid_inputs, old_valid_label,
     old_test_inputs, old_test_label, _) = load_data(args.old_data_dir, 'old_')
    
    # Load new data
    (new_train_inputs, new_train_label, new_valid_inputs, new_valid_label,
     new_test_inputs, new_test_label, df_test) = load_data(args.new_data_dir, 'new_')
    
    # Initialize model based on architecture choice
    if args.model == 'Transformer':
        # Create Transformer model
        model = Transformer(args).to(device)
        # Initialize SGD optimizer
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.model == 'LSTM':
        # Create LSTM model
        model = LSTM(args).to(device)
        # Initialize Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.model == 'Bi-LSTM':
        # Create Bidirectional LSTM model
        model = BidirectionalLSTM(args).to(device)
        # Initialize Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.model == 'RNN':
        # Create RNN model
        model = RNN(args).to(device)
        # Initialize Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load previous model if specified
    if args.prev_model:
        print(f"Loading previous model from {args.prev_model}")
        checkpoint = torch.load(args.prev_model)
        model.load_state_dict(checkpoint)

    # Initialize loss function
    loss_mse = torch.nn.MSELoss()
    # Initialize best validation MSE
    valid_mse_saved = float('inf')
    
    # Print start message
    print("Starting curriculum learning...")
    # Stage 1: Warm-up on old data
    print("Stage 1: Warm-up on old data")
    # Create DataLoader for old data
    old_train_loader = Data.DataLoader(MyDataSet(old_train_inputs, old_train_label), args.batch_size, True)
    
    # Warm-up training loop
    for epoch in range(args.warmup_epochs):
        # Set model to training mode
        model.train()
        # Iterate through old data
        for enc_inputs, labels in old_train_loader:
            # Move data to device
            enc_inputs = enc_inputs.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(enc_inputs)
            # Calculate loss
            loss = loss_mse(outputs, labels) if args.task_type == 'Regression' else F.nll_loss(outputs, labels)
            # Zero gradients
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            # Get predictions
            valid_pred = model(old_valid_inputs)
            # Calculate validation MSE
            valid_mse = loss_mse(valid_pred, old_valid_label).item()
            # Print progress
            print(f'Warm-up Epoch {epoch+1}, Valid MSE: {valid_mse:.6f}')
    
    # Stage 2: Gradual introduction of new data
    print("Stage 2: Gradual introduction of new data")
    # Iterate through curriculum stages
    for stage in range(args.curriculum_steps):
        # Calculate mix ratio for this stage
        mix_ratio = 1.0 - (stage + 1) / args.curriculum_steps
        print(f"Curriculum stage {stage+1}, Old:New ratio = {mix_ratio:.2f}:{1-mix_ratio:.2f}")
        
        # Create DataLoader with mixed data
        train_loader = create_curriculum_loader(
            (old_train_inputs, old_train_label),
            (new_train_inputs, new_train_label),
            mix_ratio
        )
        
        # Training loop for this stage
        for epoch in range(args.epochs // args.curriculum_steps):
            # Set model to training mode
            model.train()
            # Iterate through mixed data
            for enc_inputs, labels in train_loader:
                # Move data to device
                enc_inputs = enc_inputs.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(enc_inputs)
                # Calculate loss
                loss = loss_mse(outputs, labels) if args.task_type == 'Regression' else F.nll_loss(outputs, labels)
                # Zero gradients
                optimizer.zero_grad()
                # Backward pass
                loss.backward()
                # Update weights
                optimizer.step()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                # Get predictions for old and new validation data
                old_valid_pred = model(old_valid_inputs)
                new_valid_pred = model(new_valid_inputs)
                # Calculate MSE for both datasets
                old_valid_mse = loss_mse(old_valid_pred, old_valid_label).item()
                new_valid_mse = loss_mse(new_valid_pred, new_valid_label).item()
                # Calculate average MSE
                avg_valid_mse = (old_valid_mse + new_valid_mse) / 2
                
                # Print progress
                print(f'Stage {stage+1}, Epoch {epoch+1}:')
                print(f'  Old Valid MSE: {old_valid_mse:.6f}')
                print(f'  New Valid MSE: {new_valid_mse:.6f}')
                print(f'  Avg Valid MSE: {avg_valid_mse:.6f}')
                
                # Save model if it's the best so far
                if avg_valid_mse < valid_mse_saved:
                    valid_mse_saved = avg_valid_mse
                    torch.save(model.state_dict(), args.model_path)
                    print(f'  Model saved with Avg Valid MSE: {avg_valid_mse:.6f}')
    
    # Final evaluation
    print("\nFinal Evaluation:")
    # Load best model
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Evaluate on test data
    with torch.no_grad():
        # Get predictions for old and new test data
        old_test_pred = model(old_test_inputs)
        new_test_pred = model(new_test_inputs)
        # Calculate MSE for both datasets
        old_test_mse = loss_mse(old_test_pred, old_test_label).item()
        new_test_mse = loss_mse(new_test_pred, new_test_label).item()
        
        # Print final results
        print(f'Old Test MSE: {old_test_mse:.6f}')
        print(f'New Test MSE: {new_test_mse:.6f}')
        print(f'Average Test MSE: {(old_test_mse + new_test_mse) / 2:.6f}')
    
    # Save test results if doing regression
    if args.task_type == 'Regression':
        # Create results dictionary
        results = {
            'Feature': df_test['Feature'],
            'Prediction': new_test_pred.cpu().squeeze().numpy(),
            'Label': new_test_label.cpu().squeeze().numpy(),
            'Old_MSE': old_test_mse,
            'New_MSE': new_test_mse
        }
        # Save results to CSV with full path
        results_path = os.path.join(RESULTS_SAVE_DIR, f'curriculum_test_results_{args.model}_lr_{args.lr}_bs_{args.batch_size}.csv')
        pd.DataFrame(results).to_csv(results_path, index=False)
        print(f'Results saved to: {results_path}')

if __name__ == '__main__':
    main() 
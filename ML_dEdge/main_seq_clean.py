import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
import sys
sys.path.append('/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/')
from utils_seq import *
from models_seq import *
import argparse
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02,
                    help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility.')
    
    # Model parameters
    parser.add_argument('--src_vocab_size', type=int, default=21)
    parser.add_argument('--src_len', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--d_k', type=int, default=32)
    parser.add_argument('--d_v', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dist_dim', type=int, default=1)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load data
    df_train = pd.read_csv('Sequential_Peptides_edges/ddedge_train_seqs.csv')
    df_valid = pd.read_csv('Sequential_Peptides_edges/ddedge_valid_seqs.csv')
    df_test = pd.read_csv('Sequential_Peptides_edges/ddedge_test_seqs.csv')
    
    # Process labels
    train_label = torch.Tensor(np.array(df_train["Label"])).unsqueeze(1).float()
    valid_label = torch.Tensor(np.array(df_valid["Label"])).unsqueeze(1).float().to(device)
    test_label = torch.Tensor(np.array(df_test["Label"])).unsqueeze(1).float().to(device)
    
    # Process features
    train_feat = np.array(df_train["Feature"])
    valid_feat = np.array(df_valid["Feature"])
    test_feat = np.array(df_test["Feature"])

    train_enc_inputs = make_data(train_feat, args.src_len)
    valid_enc_inputs = make_data(valid_feat, args.src_len).to(device)
    test_enc_inputs = make_data(test_feat, args.src_len).to(device)

    train_loader = Data.DataLoader(MyDataSet(train_enc_inputs, train_label), args.batch_size, True)
    
    # Create output directory
    output_directory = 'results_transformer'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Initialize model and optimizer
    model = Transformer(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Initialize metrics tracking
    metrics = {
        'epoch': [],
        'train_mse': [],
        'valid_mse': [],
        'train_mae': [],
        'valid_mae': [],
        'is_best': []
    }
    
    # Training loop
    best_valid_mse = float('inf')
    best_epoch = 0
    model_path = os.path.join(output_directory, 'best_model.pt')
    
    print(f"Training on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        train_mae_list = []
        
        for enc_inputs, labels in train_loader:
            enc_inputs = enc_inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(enc_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_mae_list.append(F.l1_loss(outputs, labels).item())
        
        # Calculate average training metrics
        train_mse = np.mean(train_losses)
        train_mae = np.mean(train_mae_list)
        
        # Validation
        model.eval()
        with torch.no_grad():
            valid_outputs = model(valid_enc_inputs)
            valid_mse = criterion(valid_outputs, valid_label).item()
            valid_mae = F.l1_loss(valid_outputs, valid_label).item()
            
            is_best = valid_mse < best_valid_mse
            if is_best:
                best_valid_mse = valid_mse
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_mse,
                }, model_path)
                
                print(f'Epoch {epoch+1}: New best model saved!')
                print(f'Validation MSE: {valid_mse:.6f}')
                print(f'Validation MAE: {valid_mae:.6f}')
            
            # Store metrics
            metrics['epoch'].append(epoch + 1)
            metrics['train_mse'].append(train_mse)
            metrics['valid_mse'].append(valid_mse)
            metrics['train_mae'].append(train_mae)
            metrics['valid_mae'].append(valid_mae)
            metrics['is_best'].append(is_best)
    
    # Save training metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_directory, 'training_metrics.csv'), index=False)
    
    # Load best model and evaluate
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        test_outputs = model(test_enc_inputs)
        test_mse = criterion(test_outputs, test_label)
        test_mae = F.l1_loss(test_outputs, test_label)
        
        # Save results
        results = pd.DataFrame({
            'Feature': df_test['Feature'],
            'Prediction': test_outputs.cpu().numpy().flatten(),
            'True_Value': test_label.cpu().numpy().flatten(),
            'Absolute_Error': np.abs(test_outputs.cpu().numpy().flatten() - 
                                   test_label.cpu().numpy().flatten())
        })
        
        results.to_csv(os.path.join(output_directory, 'test_results.csv'), index=False)
        
        print('\nTest Results:')
        print(f'Best model from epoch {best_epoch + 1}')
        print(f'Test MSE: {test_mse:.6f}')
        print(f'Test MAE: {test_mae:.6f}')

if __name__ == '__main__':
    main() 
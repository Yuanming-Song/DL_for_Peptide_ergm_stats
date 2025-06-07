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
    parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model parameters
    parser.add_argument('--src_vocab_size', type=int, default=21)
    parser.add_argument('--src_len', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    
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
    
    # Load data
    df_train = pd.read_csv('Sequential_Peptides_edges/train_seqs.csv')
    df_valid = pd.read_csv('Sequential_Peptides_edges/valid_seqs.csv')
    df_test = pd.read_csv('Sequential_Peptides_edges/test_seqs.csv')
    
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
    
    # Training loop
    best_valid_mse = float('inf')
    best_epoch = 0
    model_path = os.path.join(output_directory, 'transformer_best.pt')
    
    print(f"Training on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(args.epochs):
        model.train()
        for enc_inputs, labels in train_loader:
            enc_inputs = enc_inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(enc_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            valid_outputs = model(valid_enc_inputs)
            valid_mse = criterion(valid_outputs, valid_label).item()
            
            if valid_mse < best_valid_mse:
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
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
from models_seq_OG import *
import argparse
import os
from torch.utils.data import ConcatDataset

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='Regression',
                    choices=['Classification','Regression'])
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.2,
                    help='Initial learning rate.')
parser.add_argument('--src_vocab_size', type=int, default=21)
parser.add_argument('--src_len', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--model', type=str, default='Transformer',choices=['RNN','LSTM','Bi-LSTM','Transformer'])
parser.add_argument('--model_path', type=str, default='model_val_best.pt')
parser.add_argument('--prev_model', type=str, help='Path to previous iteration model')
parser.add_argument('--old_data_dir', type=str, help='Directory containing previous iteration data')
parser.add_argument('--new_data_dir', type=str, help='Directory containing new data')
parser.add_argument('--curriculum_steps', type=int, default=3, help='Number of curriculum stages')
parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of epochs to train on old data')

args = parser.parse_args()

# Transformer Parameters
if args.model == 'Transformer':
    args.d_model = 512
    args.d_ff = 2048
    args.d_k = args.d_v = 64
    args.n_layers = 6
    args.n_heads = 8
else:
    args.d_model = 512

args.model_path = '{}_curriculum_lr_{}_bs_{}.pt'.format(args.model, args.lr, args.batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

def load_data(data_dir, prefix=''):
    if args.task_type == 'Regression':
        df_train = pd.read_csv(f'{data_dir}/ddedge_train_seqs.csv')
        df_valid = pd.read_csv(f'{data_dir}/ddedge_valid_seqs.csv')
        df_test = pd.read_csv(f'{data_dir}/ddedge_test_seqs.csv')
        train_label = torch.Tensor(np.array(df_train["Label"])).unsqueeze(1).float()
        valid_label = torch.Tensor(np.array(df_valid["Label"])).unsqueeze(1).float().to(device)
        test_label = torch.Tensor(np.array(df_test["Label"])).unsqueeze(1).float().to(device)
    else:
        df_train = pd.read_csv(f'{data_dir}/train_seqs_cla.csv')
        df_valid = pd.read_csv(f'{data_dir}/valid_seqs_cla.csv')
        df_test = pd.read_csv(f'{data_dir}/test_seqs_cla.csv')
        train_label = torch.Tensor(np.array(df_train["Label"])).long()
        valid_label = torch.Tensor(np.array(df_valid["Label"])).long().to(device)
        test_label = torch.Tensor(np.array(df_test["Label"])).long().to(device)
    
    train_feat = np.array(df_train["Feature"])
    valid_feat = np.array(df_valid["Feature"])
    test_feat = np.array(df_test["Feature"])

    train_enc_inputs = make_data(train_feat, args.src_len)
    valid_enc_inputs = make_data(valid_feat, args.src_len).to(device)
    test_enc_inputs = make_data(test_feat, args.src_len).to(device)
    
    return (train_enc_inputs, train_label, valid_enc_inputs, valid_label, 
            test_enc_inputs, test_label, df_test)

def create_curriculum_loader(old_data, new_data, mix_ratio):
    """Create a DataLoader that mixes old and new data according to mix_ratio"""
    old_inputs, old_labels = old_data
    new_inputs, new_labels = new_data
    
    # Calculate how many samples to take from each dataset
    n_old = int(len(old_labels) * mix_ratio)
    n_new = int(len(new_labels) * (1 - mix_ratio))
    
    # Randomly sample from each dataset
    old_indices = torch.randperm(len(old_labels))[:n_old]
    new_indices = torch.randperm(len(new_labels))[:n_new]
    
    # Create combined dataset
    combined_inputs = torch.cat([old_inputs[old_indices], new_inputs[new_indices]])
    combined_labels = torch.cat([old_labels[old_indices], new_labels[new_indices]])
    
    return Data.DataLoader(MyDataSet(combined_inputs, combined_labels), args.batch_size, True)

def main():
    print("Loading old and new data...")
    # Load old and new data
    (old_train_inputs, old_train_label, old_valid_inputs, old_valid_label,
     old_test_inputs, old_test_label, _) = load_data(args.old_data_dir, 'old_')
    
    (new_train_inputs, new_train_label, new_valid_inputs, new_valid_label,
     new_test_inputs, new_test_label, df_test) = load_data(args.new_data_dir, 'new_')
    
    # Initialize model
    if args.model == 'Transformer':
        model = Transformer(args).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.model == 'LSTM':
        model = LSTM(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.model == 'Bi-LSTM':
        model = BidirectionalLSTM(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.model == 'RNN':
        model = RNN(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load previous model if specified
    if args.prev_model:
        print(f"Loading previous model from {args.prev_model}")
        checkpoint = torch.load(args.prev_model)
        model.load_state_dict(checkpoint)

    loss_mse = torch.nn.MSELoss()
    valid_mse_saved = float('inf')
    
    print("Starting curriculum learning...")
    # Stage 1: Warm-up on old data
    print("Stage 1: Warm-up on old data")
    old_train_loader = Data.DataLoader(MyDataSet(old_train_inputs, old_train_label), args.batch_size, True)
    
    for epoch in range(args.warmup_epochs):
        model.train()
        for enc_inputs, labels in old_train_loader:
            enc_inputs = enc_inputs.to(device)
            labels = labels.to(device)
            outputs = model(enc_inputs)
            loss = loss_mse(outputs, labels) if args.task_type == 'Regression' else F.nll_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            valid_pred = model(old_valid_inputs)
            valid_mse = loss_mse(valid_pred, old_valid_label).item()
            print(f'Warm-up Epoch {epoch+1}, Valid MSE: {valid_mse:.6f}')
    
    # Stage 2: Gradual introduction of new data
    print("Stage 2: Gradual introduction of new data")
    for stage in range(args.curriculum_steps):
        mix_ratio = 1.0 - (stage + 1) / args.curriculum_steps  # Gradually decrease old data ratio
        print(f"Curriculum stage {stage+1}, Old:New ratio = {mix_ratio:.2f}:{1-mix_ratio:.2f}")
        
        train_loader = create_curriculum_loader(
            (old_train_inputs, old_train_label),
            (new_train_inputs, new_train_label),
            mix_ratio
        )
        
        for epoch in range(args.epochs // args.curriculum_steps):
            model.train()
            for enc_inputs, labels in train_loader:
                enc_inputs = enc_inputs.to(device)
                labels = labels.to(device)
                outputs = model(enc_inputs)
                loss = loss_mse(outputs, labels) if args.task_type == 'Regression' else F.nll_loss(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation on both old and new data
            model.eval()
            with torch.no_grad():
                old_valid_pred = model(old_valid_inputs)
                new_valid_pred = model(new_valid_inputs)
                old_valid_mse = loss_mse(old_valid_pred, old_valid_label).item()
                new_valid_mse = loss_mse(new_valid_pred, new_valid_label).item()
                avg_valid_mse = (old_valid_mse + new_valid_mse) / 2
                
                print(f'Stage {stage+1}, Epoch {epoch+1}:')
                print(f'  Old Valid MSE: {old_valid_mse:.6f}')
                print(f'  New Valid MSE: {new_valid_mse:.6f}')
                print(f'  Avg Valid MSE: {avg_valid_mse:.6f}')
                
                if avg_valid_mse < valid_mse_saved:
                    valid_mse_saved = avg_valid_mse
                    torch.save(model.state_dict(), args.model_path)
                    print(f'  Model saved with Avg Valid MSE: {avg_valid_mse:.6f}')
    
    # Final evaluation
    print("\nFinal Evaluation:")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    with torch.no_grad():
        old_test_pred = model(old_test_inputs)
        new_test_pred = model(new_test_inputs)
        old_test_mse = loss_mse(old_test_pred, old_test_label).item()
        new_test_mse = loss_mse(new_test_pred, new_test_label).item()
        
        print(f'Old Test MSE: {old_test_mse:.6f}')
        print(f'New Test MSE: {new_test_mse:.6f}')
        print(f'Average Test MSE: {(old_test_mse + new_test_mse) / 2:.6f}')
    
    # Save test results
    if args.task_type == 'Regression':
        results = {
            'Feature': df_test['Feature'],
            'Prediction': new_test_pred.cpu().squeeze().numpy(),
            'Label': new_test_label.cpu().squeeze().numpy(),
            'Old_MSE': old_test_mse,
            'New_MSE': new_test_mse
        }
        pd.DataFrame(results).to_csv(f'curriculum_test_results_{args.model}_lr_{args.lr}_bs_{args.batch_size}.csv')

if __name__ == '__main__':
    main() 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch.nn.functional as F
import random
import pandas as pd
from utils_seq import *
from models_seq import *
import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='Distribution',
                    choices=['Classification','Regression','Distribution'])
parser.add_argument('--dist_dim', type=int, default=6, help='Dimension of probability distribution.')  # Ensure dist_dim is 6
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')  # Reverted epochs to 100
parser.add_argument('--batch_size', type=int, default=16)  # Reduced batch size
parser.add_argument('--patience', type=int, default=5,
                    help='Number of epochs to wait before early stopping.')  # Reduced patience
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')  # Reduced learning rate
parser.add_argument('--src_vocab_size', type=int, default=21) # number of amino acids + 'Empty'
parser.add_argument('--src_len', type=int, default=10)
parser.add_argument('--model', type=str, default='Transformer',choices=['RNN','LSTM','Bi-LSTM','Transformer'])
parser.add_argument('--model_path', type=str, default='model_val_best.pt')
parser.add_argument('--base_dir', type = str, default='/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for the model.')  # Increased dropout
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer.')

args = parser.parse_args()

# Transformer Parameters
if args.model == 'Transformer':
    args.d_model = 256  # Increased embedding size
    args.d_ff = 1024    # Increased FeedForward dimension
    args.d_k = args.d_v = 32  # Dimension of K(=Q), V
    args.n_layers = 2   # Number of Encoder and Decoder layers
    args.n_heads = 8    # Increased number of heads in Multi-Head Attention

args.model_path=os.path.join(args.base_dir,'{}_lr_{}_bs_{}.pt'.format(args.model,args.lr,args.batch_size))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)  

def normalize_labels(labels):
    return labels / labels.sum(dim=-1, keepdim=True)

def main():
    # Create timestamp for unique experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"dimer_{args.model}_{timestamp}"
    
    # Create directories for results and logs
    output_directory = 'results_seq_dist'
    log_directory = 'training_logs'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    # Initialize logging DataFrame with additional metrics
    log_data = {
        'epoch': [],
        'train_loss': [],
        'valid_ce': [],  # Cross entropy
        'learning_rate': [],
        'best_valid_ce': [],  # Best cross entropy
        'patience_counter': [],
        'max_prob_error': []  # Maximum error in probability prediction
    }
    
    if args.task_type == 'Distribution':
# Ensure the correct file paths for rebinned data
        df_train = pd.read_csv('Sequential_Peptides_Rebin/dimer_train_rebinned.csv')
        df_valid = pd.read_csv('Sequential_Peptides_Rebin/dimer_valid_rebinned.csv')
        df_test = pd.read_csv('Sequential_Peptides_Rebin/dimer_test_rebinned.csv')

        train_label = process_dist_labels(df_train, 'Label').float()
        valid_label = process_dist_labels(df_valid, 'Label').float().to(device)
        test_label = process_dist_labels(df_test, 'Label').float().to(device)
        
        # Normalize the probability distributions
        train_label = normalize_labels(train_label)
        valid_label = normalize_labels(valid_label)
        test_label = normalize_labels(test_label)
        
        assert train_label.shape[-1] == valid_label.shape[-1] == test_label.shape[-1] == args.dist_dim
    
    output_directory = 'results_seq_dist'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    args.max = train_label.max().item()
    args.min = train_label.min().item()
    train_feat = np.array(df_train["Feature"])
    valid_feat = np.array(df_valid["Feature"])
    test_feat = np.array(df_test["Feature"])

    # Convert to long tensors for embedding
    train_enc_inputs = make_data(train_feat,args.src_len).long()  # Ensure long type for embeddings
    valid_enc_inputs = make_data(valid_feat,args.src_len).long().to(device)  # Ensure long type for embeddings
    test_enc_inputs = make_data(test_feat,args.src_len).long().to(device)  # Ensure long type for embeddings

    train_loader = Data.DataLoader(MyDataSet(train_enc_inputs,train_label), args.batch_size, True)

    # Use CrossEntropyLoss with label smoothing
    loss_ce = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing
    
    if args.model == 'Transformer':
        model = Transformer(args).to(device)
        # Use lower initial learning rate with cosine schedule
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)  # Fixed variable name
        # Use cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,  # Reduced first restart cycle length
            T_mult=2,  # Multiply cycle length by 2 after each restart
            eta_min=1e-6  # Minimum learning rate
        )

    # Early stopping setup
    patience_counter = 0
    best_valid_ce = float('inf')

    # Save training configuration
    config = {
        'model': args.model,
        'task_type': args.task_type,
        'dist_dim': args.dist_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'patience': args.patience,
        'learning_rate': args.lr,
        'src_vocab_size': args.src_vocab_size,
        'src_len': args.src_len,
        'seed': args.seed,
        'device': device,
        'data_size': {
            'train': len(df_train),
            'valid': len(df_valid),
            'test': len(df_test)
        },
        'dropout': args.dropout,
        'weight_decay': args.weight_decay
    }
    
    if args.model == 'Transformer':
        config.update({
            'd_model': args.d_model,
            'd_ff': args.d_ff,
            'd_k': args.d_k,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads
        })
    
    # Save configuration to JSON
    import json
    with open(os.path.join(log_directory, f'{experiment_id}_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for enc_inputs, labels in train_loader:
            enc_inputs = enc_inputs.to(device)
            labels = labels.to(device).float()
            
            outputs = model(enc_inputs)
            
            if args.task_type == "Distribution":
                loss = loss_ce(outputs, labels)
                train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Step the scheduler every epoch
        scheduler.step()

        if (epoch+1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                predict = model(valid_enc_inputs)
                predict = F.softmax(predict, dim=-1)
                
                if args.task_type == "Distribution":
                    valid_ce = loss_ce(predict, valid_label)
                    
                    # Calculate maximum probability error
                    max_prob_error = (predict - valid_label).abs().max().item()
                    
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # Log metrics
                    log_data['epoch'].append(epoch + 1)
                    log_data['train_loss'].append(train_loss)
                    log_data['valid_ce'].append(valid_ce.cpu().item())
                    log_data['learning_rate'].append(current_lr)
                    log_data['best_valid_ce'].append(best_valid_ce)
                    log_data['patience_counter'].append(patience_counter)
                    log_data['max_prob_error'].append(max_prob_error)
                    
                    print(f'Epoch {epoch+1}/{args.epochs}')
                    print(f'Training Loss: {train_loss:.6f}')
                    print(f'Validation CE: {valid_ce:.6f}')
                    print(f'Max Prob Error: {max_prob_error:.6f}')
                    print(f'Learning Rate: {current_lr:.6f}')
                    
                    # Early stopping check
                    if valid_ce < best_valid_ce:
                        best_valid_ce = valid_ce.cpu().item()
                        patience_counter = 0
                        print('Saving best model...')
                        args.model_path_best=os.path.join(args.base_dir,f'{experiment_id}_best.pt')
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_valid_ce': best_valid_ce,
                            'config': config
                        }, args.model_path_best)
                        print(f"Model saved at: {args.model_path_best}")
                    else:
                        patience_counter += 1
                        print(f'Early stopping counter: {patience_counter}/{args.patience}')
                    
                    # Early stopping
                    if patience_counter >= args.patience:
                        print(f'Early stopping triggered after {epoch+1} epochs')
                        break

    # Save training logs
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(os.path.join(log_directory, f'{experiment_id}_training_log.csv'), index=False)
    
    print('######## Final Epoch:',epoch+1)
    print('Best validation CE:', best_valid_ce)
    print(f"Model saved at: {args.model_path_best}")
    print(f"Training logs saved at: {os.path.join(log_directory, f'{experiment_id}_training_log.csv')}")

    # Test evaluation
    if args.model == 'Transformer':
        model_load = Transformer(args).to(device)
    
    checkpoint = torch.load(args.model_path_best)
    model_load.load_state_dict(checkpoint['model_state_dict'])
    model_load.eval()
    
    outputs = model_load(test_enc_inputs)

    if args.task_type == 'Distribution':
        # Move tensors to CPU before converting to list
        predict = outputs.detach().cpu().numpy().tolist()
        labels = test_label.cpu().numpy().tolist()

        df_test_save = pd.DataFrame()
        df_test_seq = pd.read_csv('Sequential_Peptides/dimer_test_seqs_dist.csv')
        df_test_save['feature'] = df_test_seq['Feature']
        df_test_save['predict'] = predict
        df_test_save['label'] = labels
        
        # Calculate and save test metrics
        test_metrics = {
            'test_ce': loss_ce(outputs, test_label).item(),
            'test_samples': len(df_test),
            'best_validation_ce': best_valid_ce,
            'final_epoch': epoch + 1
        }
        
        with open(os.path.join(log_directory, f'{experiment_id}_test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)
            
        df_test_save.to_csv(os.path.join(output_directory,f'{experiment_id}_test_results.csv'))

if __name__ == '__main__':
    main()

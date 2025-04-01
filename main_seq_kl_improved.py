import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
from utils_seq import *
from models_seq import *
import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='Distribution', choices=['Distribution'])
parser.add_argument('--dist_dim', type=int, default=6, help='Dimension of probability distribution.')
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--src_vocab_size', type=int, default=21, help='Number of amino acids + "Empty".')
parser.add_argument('--src_len', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model', type=str, default='Transformer', choices=['Transformer'])
parser.add_argument('--model_path', type=str, default='model_val_best.pt')
parser.add_argument('--base_dir', type=str, default='/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide')
parser.add_argument('--d_ff_scaler', type=int, default=4, help='Scaler factor for d_ff computed as d_model * d_ff_scaler')
parser.add_argument('--weight_decay', type=float, default=0.01, help='L2 regularization factor')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping threshold')
parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps for learning rate')
parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait before early stopping')
parser.add_argument('--d_model', type=int, default=256, help='Model embedding dimension')
parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

args = parser.parse_args()

# Transformer Parameters
args.d_k = args.d_v = args.d_model // args.n_heads
args.d_ff = args.d_model * args.d_ff_scaler

args.model_path = os.path.join(args.base_dir, '{}_lr_{}_bs_{}_improved.pt'.format(args.model, args.lr, args.batch_size))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def print_training_config():
    """Print the training configuration in a clear format."""
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print("\nModel Architecture:")
    print(f"  Model Type: {args.model}")
    print(f"  Embedding Dim: {args.d_model}")
    print(f"  Feed Forward Dim: {args.d_ff}")
    print(f"  Number of Layers: {args.n_layers}")
    print(f"  Number of Heads: {args.n_heads}")
    print(f"  Dropout Rate: {args.dropout}")
    
    print("\nTraining Parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Gradient Clipping: {args.grad_clip}")
    print(f"  Warmup Steps: {args.warmup_steps}")
    print(f"  Early Stopping Patience: {args.patience}")
    
    print("\nData Parameters:")
    print(f"  Distribution Dimension: {args.dist_dim}")
    print(f"  Source Length: {args.src_len}")
    print(f"  Vocabulary Size: {args.src_vocab_size}")
    print(f"  Random Seed: {args.seed}")
    print(f"  Device: {device}")
    print("="*50 + "\n")

def main(): 
    # Print training configuration
    print_training_config()
    
    # Load rebinned data
    df_train = pd.read_csv('Sequential_Peptides_Rebin/dimer_train_rebinned.csv')
    df_valid = pd.read_csv('Sequential_Peptides_Rebin/dimer_valid_rebinned.csv')
    df_test = pd.read_csv('Sequential_Peptides_Rebin/dimer_test_rebinned.csv')

    epsilon = 1e-8  # Small value to avoid zeros

    # Debugging: Log raw labels
    print("Raw train labels:", df_train['Label'].head())
    print("Raw valid labels:", df_valid['Label'].head())
    print("Raw test labels:", df_test['Label'].head())

    train_label = process_dist_labels(df_train, 'Label').float() + epsilon
    valid_label = process_dist_labels(df_valid, 'Label').float().to(device) + epsilon
    test_label = process_dist_labels(df_test, 'Label').float().to(device) + epsilon

    # Debugging: Log processed labels
    print("Processed train labels:", train_label[:5])
    print("Processed valid labels:", valid_label[:5])
    print("Processed test labels:", test_label[:5])

    # Data validation checks
    if torch.isnan(train_label).any() or torch.isinf(train_label).any():
        raise AssertionError("Train labels contain invalid values!")

    # Normalize labels
    train_label = train_label / (train_label.sum(dim=-1, keepdim=True) + epsilon)
    valid_label = valid_label / (valid_label.sum(dim=-1, keepdim=True) + epsilon)
    test_label = test_label / (test_label.sum(dim=-1, keepdim=True) + epsilon)

    # Additional validation checks
    assert (train_label >= 0).all(), "Train labels contain negative values!"
    assert torch.allclose(train_label.sum(dim=-1), torch.ones_like(train_label.sum(dim=-1)), atol=1e-6), "Train labels do not sum to 1!"

    train_feat = np.array(df_train["Feature"])
    valid_feat = np.array(df_valid["Feature"])
    test_feat = np.array(df_test["Feature"])

    train_enc_inputs = make_data(train_feat, args.src_len).long()
    valid_enc_inputs = make_data(valid_feat, args.src_len).long().to(device)
    test_enc_inputs = make_data(test_feat, args.src_len).long().to(device)

    train_loader = Data.DataLoader(MyDataSet(train_enc_inputs, train_label), args.batch_size, shuffle=True)

    # Define model, optimizer, and loss function with improvements
    model = Transformer(args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, num_training_steps)
    
    loss_kl = nn.KLDivLoss(reduction='batchmean')

    best_valid_kl = float('inf')
    patience_counter = 0

    # Ensure the directory for saving models exists
    os.makedirs(args.base_dir, exist_ok=True)

    # Initialize logging DataFrame
    log_data = {
        'epoch': [],
        'train_loss': [],
        'valid_kl': [],
        'learning_rate': [],
        'best_valid_kl': [],
        'patience_counter': [],
        'max_prob_error': [],
        'avg_grad_norm': [],
        'avg_param_norm': []
    }

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        grad_norm_sum = 0
        param_norm_sum = 0
        num_batches = 0
        
        for enc_inputs, labels in train_loader:
            enc_inputs = enc_inputs.to(device)
            labels = labels.to(device)

            outputs = model(enc_inputs)
            outputs = F.softmax(outputs, dim=-1) + epsilon

            # Check for invalid values
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Invalid values detected in model outputs!")
                return

            loss = loss_kl(F.log_softmax(outputs, dim=-1), labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print("Invalid loss detected!")
                return

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            param_norm = torch.norm(torch.stack([torch.norm(p) for p in model.parameters()]))
            
            grad_norm_sum += grad_norm
            param_norm_sum += param_norm
            num_batches += 1

            optimizer.step()
            scheduler.step()

        # Calculate average metrics
        train_loss /= len(train_loader)
        avg_grad_norm = grad_norm_sum / num_batches
        avg_param_norm = param_norm_sum / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            predict = model(valid_enc_inputs)
            predict = F.softmax(predict, dim=-1) + epsilon
            valid_kl_tensor = loss_kl(F.log_softmax(predict, dim=-1), valid_label)

            if torch.isnan(valid_kl_tensor) or torch.isinf(valid_kl_tensor):
                print("Invalid validation KL detected!")
                return

            valid_kl = valid_kl_tensor.item()
            max_prob_error = (predict - valid_label).abs().max().item()

        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'  Valid KL: {valid_kl:.6f}')
        print(f'  Max Prob Error: {max_prob_error:.6f}')
        print(f'  Avg Gradient Norm: {avg_grad_norm:.6f}')
        print(f'  Avg Parameter Norm: {avg_param_norm:.6f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('  ---')

        # Log metrics
        log_data['epoch'].append(epoch + 1)
        log_data['train_loss'].append(train_loss)
        log_data['valid_kl'].append(valid_kl)
        log_data['learning_rate'].append(optimizer.param_groups[0]["lr"])
        log_data['best_valid_kl'].append(best_valid_kl)
        log_data['patience_counter'].append(patience_counter)
        log_data['max_prob_error'].append(max_prob_error)
        log_data['avg_grad_norm'].append(avg_grad_norm)
        log_data['avg_param_norm'].append(avg_param_norm)

        # Early stopping and model saving
        if valid_kl < best_valid_kl:
            best_valid_kl = valid_kl
            patience_counter = 0
            kl_model_path = os.path.join(args.base_dir, f'{args.model}_KL_rebin_best_improved.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_valid_kl': best_valid_kl,
                'config': args.__dict__
            }, kl_model_path)
            print(f"Saved best model with Valid KL: {best_valid_kl:.6f} at {kl_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Test evaluation
    kl_model_path = os.path.join(args.base_dir, f'{args.model}_KL_rebin_best_improved.pt')
    if not os.path.exists(kl_model_path):
        print(f"Model file not found: {kl_model_path}")
        return

    model.load_state_dict(torch.load(kl_model_path)['model_state_dict'])
    model.eval()
    with torch.no_grad():
        outputs = model(test_enc_inputs)
        outputs = F.softmax(outputs, dim=-1) + epsilon
        test_kl_tensor = loss_kl(F.log_softmax(outputs, dim=-1), test_label)

        if torch.isnan(test_kl_tensor) or torch.isinf(test_kl_tensor):
            print("Invalid test KL detected!")
            return

        test_kl = test_kl_tensor.item()

    print(f"Test KL Divergence: {test_kl:.6f}")

    # Save test results
    predict = outputs.cpu().numpy().tolist()
    labels = test_label.cpu().numpy().tolist()

    df_test_save = pd.DataFrame()
    df_test_save['feature'] = df_test['Feature']
    df_test_save['predict'] = predict
    df_test_save['label'] = labels
    os.makedirs('results_seq_dist', exist_ok=True)
    df_test_save.to_csv(os.path.join('results_seq_dist', 'test_results_KL_rebin_improved.csv'), index=False)

if __name__ == '__main__':
    main() 
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

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='Distribution', choices=['Distribution'])
parser.add_argument('--dist_dim', type=int, default=6, help='Dimension of probability distribution.')
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.5, help='Initial learning rate.')
parser.add_argument('--src_vocab_size', type=int, default=21, help='Number of amino acids + "Empty".')
parser.add_argument('--src_len', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model', type=str, default='Transformer', choices=['Transformer'])
parser.add_argument('--model_path', type=str, default='model_val_best.pt')
parser.add_argument('--base_dir', type=str, default='/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide')
parser.add_argument('--d_ff_scaler', type=int, default=4, help='Scaler factor for d_ff computed as d_model * d_ff_scaler')

args = parser.parse_args()

# Transformer Parameters
args.d_model = 256    # Embedding size
args.d_k = args.d_v = 32  # Dimension of K(=Q), V
# Compute n_heads based on d_model and d_k
args.n_heads = args.d_model // args.d_k
# Compute d_ff based on d_model and d_ff_scaler
args.d_ff = args.d_model * args.d_ff_scaler
args.n_layers = 4   # Number of Encoder and Decoder layers

args.model_path = os.path.join(args.base_dir, '{}_lr_{}_bs_{}.pt'.format(args.model, args.lr, args.batch_size))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

def main(): 
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

    # Check for NaN values and print their indices, values, and corresponding features
    if torch.isnan(train_label).any():
        nan_indices = torch.nonzero(torch.isnan(train_label), as_tuple=True)[0].unique()
        print("NaN values detected in train_label at indices:", nan_indices.tolist())
        print("NaN values:", train_label[nan_indices])
        print("Corresponding features:", df_train.iloc[nan_indices.tolist()]['Feature'].tolist())
        raise AssertionError("Train labels contain NaN values!")

    # Check for Inf values and print their indices, values, and corresponding features
    if torch.isinf(train_label).any():
        inf_indices = torch.nonzero(torch.isinf(train_label), as_tuple=True)[0].unique()
        print("Inf values detected in train_label at indices:", inf_indices.tolist())
        print("Inf values:", train_label[inf_indices])
        print("Corresponding features:", df_train.iloc[inf_indices.tolist()]['Feature'].tolist())
        raise AssertionError("Train labels contain Inf values!")

    # Ensure labels are valid probability distributions
    train_label = train_label / (train_label.sum(dim=-1, keepdim=True) + epsilon)  # Add epsilon to denominator
    valid_label = valid_label / (valid_label.sum(dim=-1, keepdim=True) + epsilon)  # Add epsilon to denominator
    test_label = test_label / (test_label.sum(dim=-1, keepdim=True) + epsilon)  # Add epsilon to denominator

    # Debugging: Check for NaN or Inf after normalization
    if torch.isnan(train_label).any():
        nan_indices = torch.nonzero(torch.isnan(train_label), as_tuple=True)
        print("NaN values detected in normalized train_label at indices:", nan_indices)
        print("NaN values:", train_label[nan_indices])
        raise AssertionError("Normalized train labels contain NaN values!")

    if torch.isinf(train_label).any():
        inf_indices = torch.nonzero(torch.isinf(train_label), as_tuple=True)
        print("Inf values detected in normalized train_label at indices:", inf_indices)
        print("Inf values:", train_label[inf_indices])
        raise AssertionError("Normalized train labels contain Inf values!")

    assert (train_label >= 0).all(), "Train labels contain negative values!"
    assert (valid_label >= 0).all(), "Valid labels contain negative values!"
    assert (test_label >= 0).all(), "Test labels contain negative values!"
    assert torch.allclose(train_label.sum(dim=-1), torch.ones_like(train_label.sum(dim=-1)), atol=1e-6), "Train labels do not sum to 1!"
    assert torch.allclose(valid_label.sum(dim=-1), torch.ones_like(valid_label.sum(dim=-1)), atol=1e-6), "Valid labels do not sum to 1!"
    assert torch.allclose(test_label.sum(dim=-1), torch.ones_like(test_label.sum(dim=-1)), atol=1e-6), "Test labels do not sum to 1!"

    assert train_label.shape[-1] == valid_label.shape[-1] == test_label.shape[-1] == args.dist_dim

    train_feat = np.array(df_train["Feature"])
    valid_feat = np.array(df_valid["Feature"])
    test_feat = np.array(df_test["Feature"])

    train_enc_inputs = make_data(train_feat, args.src_len).long()
    valid_enc_inputs = make_data(valid_feat, args.src_len).long().to(device)
    test_enc_inputs = make_data(test_feat, args.src_len).long().to(device)

    train_loader = Data.DataLoader(MyDataSet(train_enc_inputs, train_label), args.batch_size, shuffle=True)

    # Define model, optimizer, and loss function
    model = Transformer(args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_kl = nn.KLDivLoss(reduction='batchmean')

    best_valid_kl = float('inf')

    # Ensure the directory for saving models exists
    os.makedirs(args.base_dir, exist_ok=True)

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
            outputs = F.softmax(outputs, dim=-1) + epsilon  # Normalize outputs and add epsilon

            # Check for invalid values in outputs and labels
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Invalid values detected in model outputs!")
                return
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print("Invalid values detected in labels!")
                return

            loss = loss_kl(F.log_softmax(outputs, dim=-1), labels)

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("Invalid loss detected!")
                return

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            # Calculate gradient and parameter norms
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            param_norm = torch.norm(torch.stack([torch.norm(p) for p in model.parameters()]))
            
            grad_norm_sum += grad_norm
            param_norm_sum += param_norm
            num_batches += 1

            optimizer.step()

        # Calculate average metrics
        train_loss /= len(train_loader)
        avg_grad_norm = grad_norm_sum / num_batches
        avg_param_norm = param_norm_sum / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            predict = model(valid_enc_inputs)
            predict = F.softmax(predict, dim=-1) + epsilon  # Normalize predictions and add epsilon
            valid_kl_tensor = loss_kl(F.log_softmax(predict, dim=-1), valid_label)

            # Check for invalid validation KL
            if torch.isnan(valid_kl_tensor) or torch.isinf(valid_kl_tensor):
                print("Invalid validation KL detected!")
                return

            valid_kl = valid_kl_tensor.item()

        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'  Valid KL: {valid_kl:.6f}')
        print(f'  Avg Gradient Norm: {avg_grad_norm:.6f}')
        print(f'  Avg Parameter Norm: {avg_param_norm:.6f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('  ---')

        # Save the best model
        if valid_kl < best_valid_kl:
            best_valid_kl = valid_kl
            kl_model_path = os.path.join(args.base_dir, f'{args.model}_KL_rebin_best.pt')
            torch.save(model.state_dict(), kl_model_path)
            print(f"Saved best model with Valid KL: {best_valid_kl:.6f} at {kl_model_path}")

    # Test evaluation
    kl_model_path = os.path.join(args.base_dir, f'{args.model}_KL_rebin_best.pt')
    if not os.path.exists(kl_model_path):
        print(f"Model file not found: {kl_model_path}")
        return

    model.load_state_dict(torch.load(kl_model_path))
    model.eval()
    with torch.no_grad():
        outputs = model(test_enc_inputs)
        outputs = F.softmax(outputs, dim=-1) + epsilon  # Normalize outputs and add epsilon
        test_kl_tensor = loss_kl(F.log_softmax(outputs, dim=-1), test_label)

        # Check for invalid test KL
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
    os.makedirs('results_seq_dist', exist_ok=True)  # Ensure results directory exists
    df_test_save.to_csv(os.path.join('results_seq_dist', 'test_results_KL_rebin.csv'), index=False)

if __name__ == '__main__':
    main()

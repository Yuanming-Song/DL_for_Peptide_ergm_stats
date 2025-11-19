"""
Script to generate test results from a trained generative model.

This script loads a saved model checkpoint and generates test results
including predicted dEdge vs true dEdge, sequence uniqueness, etc.

Usage:
    python run_test_evaluation.py [--model_path MODEL_PATH]
    
    Or submit via SLURM:
    sbatch run_test_evaluation.slurm
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse
import mlflow

# Define base paths
PROJECT_ROOT = '/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide'
BASE_DIR = os.path.join(PROJECT_ROOT, 'ML_dEdge_gen', 'v1+2')
TRAINING_DIR = os.path.join(BASE_DIR, 'scripts', 'training')
STRATIFIED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'stratified')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')

# Add paths
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'OG_util_py'))
from utils_seq import *
from models_seq_OG import get_attn_subsequence_mask
sys.path.append(TRAINING_DIR)
from models_gen import *

# Import dataset class and helper functions from training script
exec(open(os.path.join(TRAINING_DIR, 'train_generative_model.py')).read().split('if __name__')[0])

def load_ml_dedge_critic(model_path, device, src_len=10, dEdge_min=None, dEdge_max=None):
    """Load pre-trained ML_dEdge model as frozen critic."""
    from models_seq_OG import Transformer
    
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
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        critic_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        critic_model.load_state_dict(checkpoint)
    
    for param in critic_model.parameters():
        param.requires_grad = False
    critic_model.eval()
    
    return critic_model

def parse_args():
    parser = argparse.ArgumentParser(description='Generate test results from trained model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (default: latest model)')
    parser.add_argument('--ml_dedge_model_path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'ML_dEdge', 'models', 'iteration2', 'Transformer_curriculum_lr_0.2_bs_1024.pt'),
                        help='Path to ML_dEdge critic model')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for test results (default: auto-detect from MLflow)')
    return parser.parse_args()

def main():
    print("="*60)
    print("Test Evaluation Script")
    print("="*60)
    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find latest model if not specified
    if args.model_path is None:
        model_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith('.pt')]
        if not model_files:
            raise FileNotFoundError("No model files found in models directory")
        # Get most recently modified
        model_files_with_time = [(f, os.path.getmtime(os.path.join(MODEL_SAVE_DIR, f))) 
                                 for f in model_files]
        latest_model = max(model_files_with_time, key=lambda x: x[1])[0]
        args.model_path = os.path.join(MODEL_SAVE_DIR, latest_model)
        print(f"Using latest model: {latest_model}")
    else:
        print(f"Using specified model: {args.model_path}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract model args and normalization parameters
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        dEdge_min = checkpoint.get('dedge_min', None)
        dEdge_max = checkpoint.get('dedge_max', None)
        dEdge_range = checkpoint.get('dedge_range', None)
        best_epoch = checkpoint.get('epoch', 0)
    else:
        # Default args if not in checkpoint
        class ModelArgs:
            src_vocab_size = 21
            src_len = 10
            d_model = 768
            d_ff = 3072
            d_k = 64
            d_v = 64
            n_layers = 8
            n_heads = 12
            dropout = 0.1
        model_args = ModelArgs()
        dEdge_min = None
        dEdge_max = None
        dEdge_range = None
        best_epoch = 0
    
    # Load test data
    print("Loading test data...")
    stratified_test_path = os.path.join(STRATIFIED_DATA_DIR, 'stratified_test_seqs.csv')
    if not os.path.exists(stratified_test_path):
        raise FileNotFoundError(f"Test data not found: {stratified_test_path}")
    
    df_test = pd.read_csv(stratified_test_path)
    print(f"Test set size: {len(df_test)}")
    
    # Get dEdge normalization if not in checkpoint
    if dEdge_min is None or dEdge_range is None:
        # Load all data to compute normalization
        df_train = pd.read_csv(os.path.join(STRATIFIED_DATA_DIR, 'stratified_train_seqs.csv'))
        df_valid = pd.read_csv(os.path.join(STRATIFIED_DATA_DIR, 'stratified_valid_seqs.csv'))
        all_dEdge = np.concatenate([
            df_train["Label"].values,
            df_valid["Label"].values,
            df_test["Label"].values
        ])
        dEdge_min = float(np.min(all_dEdge))
        dEdge_max = float(np.max(all_dEdge))
        dEdge_range = dEdge_max - dEdge_min
        print(f"Computed dEdge normalization: min={dEdge_min:.6f}, max={dEdge_max:.6f}, range={dEdge_range:.6f}")
    else:
        print(f"Using dEdge normalization from checkpoint: min={dEdge_min:.6f}, max={dEdge_max:.6f}, range={dEdge_range:.6f}")
    
    # Create test dataset
    test_dataset = GenerativeDataset(
        sequences=df_test['Feature'].values,
        dEdge_values=df_test['Label'].values,
        src_len=model_args.src_len,
        dEdge_min=dEdge_min,
        dEdge_range=dEdge_range
    )
    
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)
    
    # Create model
    print("Creating model...")
    model = ConditionalGenerator(model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load critic model (use actual dEdge min/max from data)
    print(f"Loading ML_dEdge critic model from {args.ml_dedge_model_path}")
    critic_model = load_ml_dedge_critic(args.ml_dedge_model_path, device, model_args.src_len, 
                                        dEdge_min=dEdge_min, dEdge_max=dEdge_max)
    print("Critic model loaded")
    
    # Create reverse vocabulary
    src_vocab_reverse = {v: k for k, v in src_vocab.items()}
    
    # Generate test results
    print("\nGenerating test results...")
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
        
        print(f"Found {len(unique_conditions)} unique (dEdge, length) combinations")
        sys.stdout.flush()
        
        # Generate sequences for each unique condition
        for idx, ((dEdge_norm, seq_len), cond_info) in enumerate(unique_conditions.items()):
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  Processing condition {idx+1}/{len(unique_conditions)}...")
                sys.stdout.flush()
            
            dEdge_orig = cond_info['dEdge_orig']
            n_generate = int(seq_len * (19/3) ** (seq_len - 1))
            if idx == 0:
                print(f"    Condition {idx+1}: dEdge={dEdge_orig:.4f}, seq_len={seq_len}, n_generate={n_generate}")
                sys.stdout.flush()
            
            condition_tensor = torch.FloatTensor([[dEdge_norm, seq_len]]).to(device)
            
            generated_seqs = []
            generated_dEdges = []
            
            for _ in range(n_generate):
                generated_tokens = model.generate(
                    condition_tensor,
                    max_len=model_args.src_len,
                    start_token=1,
                    temperature=1.0
                )
                seq_tokens = generated_tokens[0].cpu().numpy()
                seq_str = decode_sequence_from_tokens(seq_tokens, src_vocab_reverse)
                if len(seq_str) > 0:
                    generated_seqs.append(seq_str)
                    
                    # Predict dEdge using critic
                    gen_enc_inputs = make_data(np.array([seq_str]), model_args.src_len).to(device)
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
        # Determine output directory
        if args.output_dir:
            test_results_dir = args.output_dir
        else:
            # Try to find MLflow run directory
            mlflow_dir = os.path.join(BASE_DIR, 'out')
            experiment_dirs = [d for d in os.listdir(mlflow_dir) if os.path.isdir(os.path.join(mlflow_dir, d)) and d != '.trash']
            if experiment_dirs:
                latest_exp = max(experiment_dirs, key=lambda x: os.path.getmtime(os.path.join(mlflow_dir, x)))
                exp_dir = os.path.join(mlflow_dir, latest_exp)
                run_dirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
                if run_dirs:
                    latest_run = max(run_dirs, key=lambda x: os.path.getmtime(os.path.join(exp_dir, x)))
                    test_results_dir = os.path.join(exp_dir, latest_run, "test_results")
                else:
                    test_results_dir = os.path.join(BASE_DIR, 'out', 'test_results')
            else:
                test_results_dir = os.path.join(BASE_DIR, 'out', 'test_results')
        
        os.makedirs(test_results_dir, exist_ok=True)
        
        test_results_path = os.path.join(test_results_dir, f'test_results_epoch_{best_epoch+1}.csv')
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(test_results_path, index=False)
        
        print(f'\n{"="*60}')
        print(f'Test results saved: {test_results_path}')
        print(f'Total conditions evaluated: {len(test_results)}')
        print(f'Average unique fraction: {np.mean([r["unique_fraction"] for r in test_results]):.4f}')
        print(f'Average dEdge error (MSE): {np.mean([r["avg_dEdge_error"] for r in test_results]):.4f}')
        print(f'{"="*60}')
    else:
        print("No test results generated!")

if __name__ == '__main__':
    main()


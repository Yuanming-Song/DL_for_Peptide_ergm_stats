"""
Reinforcement Learning fine-tuning script for conditional generative model.

This script fine-tunes a pre-trained MLE generator using RL with the ML_dEdge iteration 2 model as critic.
The generator is trained to maximize the reward (negative dEdge error) while maintaining
sequence quality.

The critic model uses min-max normalization, so we need to ensure dEdge values are
properly normalized when computing rewards.

Usage:
    python train_generative_model_rl.py --model_path <path_to_mle_model> [options]
    
    Or submit via SLURM:
    sbatch train_generative_model_rl.slurm
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
import mlflow
import mlflow.pytorch

# Define base paths
PROJECT_ROOT = '/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide'
BASE_DIR = os.path.join(PROJECT_ROOT, 'ML_dEdge_gen', 'v1+2')
TRAINING_DIR = os.path.join(BASE_DIR, 'scripts', 'training')
DATA_DIR_ITER1 = os.path.join(PROJECT_ROOT, 'ML_dEdge', 'data', 'iteration1', 'training', 'Sequential_Peptides_edges')
DATA_DIR_ITER2 = os.path.join(PROJECT_ROOT, 'ML_dEdge', 'data', 'iteration2', 'training', 'Sequential_Peptides_edges')
STRATIFIED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'stratified')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_SAVE_DIR = os.path.join(BASE_DIR, 'data')

# Add paths to sys.path
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'OG_util_py'))
from utils_seq import *
from models_seq_OG import get_attn_subsequence_mask
sys.path.append(TRAINING_DIR)
from models_gen import *

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='RL fine-tuning for generative model')
    
    # Model path
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pre-trained MLE model checkpoint')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for RL fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    # RL parameters
    parser.add_argument('--reward_weight', type=float, default=1.0,
                        help='Weight for reward signal (dEdge matching)')
    parser.add_argument('--entropy_weight', type=float, default=0.01,
                        help='Weight for entropy bonus (encourages exploration)')
    parser.add_argument('--baseline_decay', type=float, default=0.99,
                        help='Decay factor for reward baseline (exponential moving average)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling during generation')
    
    # Critic model path (iteration 2 ML_dEdge model)
    parser.add_argument('--ml_dedge_model_path', type=str, 
                        default=os.path.join(PROJECT_ROOT, 'ML_dEdge', 'models', 'iteration2', 'Transformer_curriculum_lr_0.2_bs_1024.pt'),
                        help='Path to pre-trained ML_dEdge iteration 2 model (critic)')
    
    # Data loading
    parser.add_argument('--no_stratified', action='store_true', default=False,
                        help='Disable stratified splits')
    
    # MLflow
    parser.add_argument('--experiment_name', type=str, default='generative_rl',
                        help='MLflow experiment name')
    
    return parser.parse_args()

def load_ml_dedge_critic(model_path, device, src_len=10, dEdge_min=-1.0, dEdge_max=3.0):
    """
    Load pre-trained ML_dEdge model as frozen critic.
    
    The critic model uses min-max normalization for dEdge values.
    
    Args:
        model_path: Path to ML_dEdge model checkpoint
        device: Device to load model on
        src_len: Maximum sequence length (must match training)
        dEdge_min: Minimum dEdge value for normalization
        dEdge_max: Maximum dEdge value for normalization
    
    Returns:
        Frozen ML_dEdge model ready to use as critic
    """
    # Import models_seq_OG here to avoid import issues when adversarial training is not used
    from models_seq_OG import Transformer
    
    # Create args for ML_dEdge model (must match training configuration)
    class Args:
        def __init__(self, dEdge_min, dEdge_max):
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
            self.max = dEdge_max
            self.min = dEdge_min
    
    ml_dedge_args = Args(dEdge_min, dEdge_max)
    
    # Initialize and load model
    critic_model = Transformer(ml_dedge_args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        critic_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        critic_model.load_state_dict(checkpoint)
    
    # Freeze the model (no gradients)
    critic_model.eval()
    for param in critic_model.parameters():
        param.requires_grad = False
    
    print(f"Loaded ML_dEdge iteration 2 critic model from {model_path}")
    print(f"Critic model frozen (no gradients)")
    
    return critic_model

def decode_sequence_from_tokens(token_ids, src_vocab_reverse):
    """Convert token IDs back to amino acid sequence string."""
    # Remove padding tokens (0) and START token at position 0 only
    # Token 1 at position 0 is START, but token 1 elsewhere is 'A' (amino acid)
    valid_tokens = []
    for i, t in enumerate(token_ids):
        if t == 0:  # PAD token - skip
            continue
        if i == 0 and t == 1:  # START token at position 0 - skip
            continue
        valid_tokens.append(t)
    sequence = ''.join([src_vocab_reverse.get(t, '?') for t in valid_tokens])
    return sequence

def compute_reward(generated_sequences, target_dEdge, critic_model, dEdge_min, dEdge_range, device, src_len):
    """
    Compute reward based on dEdge matching.
    
    The critic model uses min-max normalization, so we need to:
    1. Predict dEdge using critic (output is in original scale)
    2. Normalize both predicted and target using the same normalization parameters
    3. Compute reward as negative MSE
    
    Reward = -MSE(predicted_dEdge_normalized, target_dEdge_normalized)
    
    Args:
        generated_sequences: List of generated sequence strings
        target_dEdge: Target dEdge value (original scale)
        critic_model: Frozen ML_dEdge critic model
        dEdge_min: Minimum for normalization (from training data)
        dEdge_range: Range for normalization (from training data)
        device: Device
        src_len: Maximum sequence length
    
    Returns:
        Tensor of rewards [batch_size]
    """
    if len(generated_sequences) == 0:
        return torch.tensor([0.0], device=device)
    
    # Encode sequences using make_data (same as in training)
    gen_enc_inputs = make_data(np.array(generated_sequences), src_len).to(device)
    
    # Predict dEdge using critic (output is in original scale, not normalized)
    with torch.no_grad():
        predicted_dEdge_original = critic_model(gen_enc_inputs).squeeze()
    
    # Normalize both predicted and target using the same normalization parameters
    # The critic model was trained with min-max normalization, so we need to normalize
    # the target_dEdge using the same parameters
    predicted_dEdge_normalized = (predicted_dEdge_original - dEdge_min) / dEdge_range
    target_dEdge_normalized = (target_dEdge - dEdge_min) / dEdge_range
    
    # Reward is negative MSE (higher is better)
    mse = F.mse_loss(predicted_dEdge_normalized, target_dEdge_normalized, reduction='none')
    reward = -mse
    
    return reward

def compute_policy_gradient_loss(log_probs, rewards, baseline=0.0, entropy_bonus=0.0, seq_lengths=None):
    """
    Compute REINFORCE policy gradient loss.
    
    Loss = -E[log_prob * (reward - baseline)] - entropy_bonus * entropy
    
    Args:
        log_probs: Log probabilities of generated sequences [batch_size, seq_len]
        rewards: Rewards for each sequence [batch_size]
        baseline: Baseline reward (for variance reduction)
        entropy_bonus: Weight for entropy bonus
        seq_lengths: Actual sequence lengths for normalization [batch_size] (optional)
    
    Returns:
        Policy gradient loss
    """
    # Step 1: Normalize by sequence length (MEAN per sequence)
    # For each sequence: compute mean log prob per token (normalized by actual sequence length)
    # This ensures sequences of different lengths contribute equally
    # log_probs: [batch_size, max_len] with padding (zeros)
    if seq_lengths is not None:
        # Use provided sequence lengths for accurate normalization
        seq_log_probs = log_probs.sum(dim=1) / seq_lengths.clamp(min=1.0)  # [batch_size] - mean log prob per sequence
    else:
        # Fallback: estimate from non-zero positions (less accurate)
        valid_mask = (log_probs != 0.0) | (torch.abs(log_probs) > 1e-8)  # [batch_size, max_len]
        seq_lengths_est = valid_mask.sum(dim=1).float().clamp(min=1.0)  # [batch_size]
        seq_log_probs = log_probs.sum(dim=1) / seq_lengths_est  # [batch_size] - mean log prob per sequence
    
    # Compute advantages
    # Convert baseline to tensor if it's a float
    if isinstance(baseline, (int, float)):
        baseline_tensor = torch.tensor(baseline, dtype=rewards.dtype, device=rewards.device)
    else:
        baseline_tensor = baseline
    # Detach advantages since rewards come from frozen critic model
    # We only want gradients through seq_log_probs, not through rewards
    advantages = (rewards - baseline_tensor).detach()
    
    # Step 2: Aggregate over batch (MEAN over sequences)
    # Policy gradient loss: -E[log_prob * advantage] where E[] is expectation (mean) over batch
    policy_loss = -(seq_log_probs * advantages).mean()  # scalar - mean over batch
    
    # Entropy bonus (encourages exploration)
    # Entropy = -sum(p * log(p)), but we have log_probs, so entropy = -sum(exp(log_prob) * log_prob)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=1).mean()  # Mean entropy per sequence
    entropy_loss = -entropy_bonus * entropy
    
    return policy_loss + entropy_loss

def create_combined_mask(pad_mask, causal_mask):
    """Combine padding mask and causal mask for decoder self-attention."""
    # pad_mask: [batch_size, seq_len] - True for PAD tokens, False for valid tokens
    # causal_mask: [batch_size, seq_len, seq_len] - True for positions to mask
    pad_mask_expanded = pad_mask.unsqueeze(1).expand(-1, pad_mask.size(1), -1)  # [batch_size, seq_len, seq_len]
    combined_mask = causal_mask | pad_mask_expanded
    return combined_mask

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load pre-trained MLE model
    print(f"Loading pre-trained MLE model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['args']
    
    # Initialize model
    model = ConditionalGenerator(model_args).to(device)
    
    # Convert old checkpoint keys to new architecture if needed
    old_state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    
    # Check if we need to convert (old architecture has 'transformer.layers', new has 'encoder'/'decoder')
    needs_conversion = any('transformer.layers' in k for k in old_state_dict.keys())
    
    if needs_conversion:
        print("Detected old model architecture. Converting checkpoint keys...")
        
        # Key mappings from old to new architecture
        for old_key, value in old_state_dict.items():
            new_key = old_key
            
            # Map positional encoding
            if old_key == 'pos_enc.pe':
                new_key = 'pos_emb.pe'
            
            # Map condition embedding
            elif old_key.startswith('cond_emb.'):
                new_key = old_key.replace('cond_emb.', 'cond_input.')
            
            # Map transformer layers to decoder layers
            # Old: transformer.layers.X -> New: decoder.layers.X
            elif old_key.startswith('transformer.layers.'):
                new_key = old_key.replace('transformer.layers.', 'decoder.layers.')
                # Note: The old model was decoder-only, so we'll map all layers to decoder
                # We'll need to create a dummy encoder layer
            
            # Keep other keys as-is (tok_emb, proj, etc.)
            else:
                new_key = old_key
            
            new_state_dict[new_key] = value
        
        # Create encoder layer from first decoder layer (since old model was decoder-only)
        # We'll use the first decoder layer weights for the encoder
        if 'decoder.layers.0.self_attn.in_proj_weight' in new_state_dict:
            for key in list(new_state_dict.keys()):
                if key.startswith('decoder.layers.0.'):
                    encoder_key = key.replace('decoder.layers.0.', 'encoder.layers.0.')
                    # For encoder, we only need self-attention (no cross-attention)
                    if 'multihead_attn' not in key:  # Skip cross-attention layers
                        new_state_dict[encoder_key] = new_state_dict[key].clone()
        
        print(f"Converted {len(old_state_dict)} keys to new architecture")
        
        # Try loading converted state dict
        try:
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"Warning: {len(missing_keys)} keys still missing after conversion")
                print(f"  First few: {missing_keys[:5]}")
            if unexpected_keys:
                print(f"Warning: {len(unexpected_keys)} unexpected keys")
                print(f"  First few: {unexpected_keys[:5]}")
            print("Model loaded with converted checkpoint")
        except Exception as e:
            print(f"Error loading converted checkpoint: {e}")
            print("Falling back to strict=False loading...")
            model.load_state_dict(new_state_dict, strict=False)
    else:
        # Try to load state dict directly
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("Loaded model state dict (strict=True)")
        except RuntimeError as e:
            print(f"Warning: Could not load with strict=True: {e}")
            print("Attempting to load with strict=False...")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"Missing keys (not loaded): {missing_keys[:10]}... ({len(missing_keys)} total)")
            if unexpected_keys:
                print(f"Unexpected keys (ignored): {unexpected_keys[:10]}... ({len(unexpected_keys)} total)")
            print("Model loaded with some keys missing/unexpected")
    
    model.train()
    print(f"Loaded MLE model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Get normalization parameters from checkpoint
    dEdge_min = checkpoint.get('dedge_min', None)
    dEdge_max = checkpoint.get('dedge_max', None)
    dEdge_range = checkpoint.get('dedge_range', None)
    
    if dEdge_min is None or dEdge_range is None:
        raise ValueError("Checkpoint must contain dedge_min and dedge_range for normalization")
    
    print(f"dEdge normalization: min={dEdge_min:.6f}, max={dEdge_max:.6f}, range={dEdge_range:.6f}")
    
    # Load critic model (ML_dEdge iteration 2)
    print(f"Loading ML_dEdge iteration 2 critic model from {args.ml_dedge_model_path}")
    critic_model = load_ml_dedge_critic(args.ml_dedge_model_path, device, model_args.src_len, dEdge_min, dEdge_max)
    
    # Create reverse vocabulary for decoding
    src_vocab_reverse = {v: k for k, v in src_vocab.items()}
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Load training data
    print("Loading training data...")
    if args.no_stratified:
        # Load from original data directories
        df_iter1 = pd.read_csv(os.path.join(DATA_DIR_ITER1, 'Sequential_Peptides_edges.csv'))
        df_iter2 = pd.read_csv(os.path.join(DATA_DIR_ITER2, 'Sequential_Peptides_edges.csv'))
        df_train = pd.concat([df_iter1, df_iter2], ignore_index=True)
    else:
        # Load stratified splits
        stratified_train_path = os.path.join(STRATIFIED_DATA_DIR, 'stratified_train_seqs.csv')
        if not os.path.exists(stratified_train_path):
            raise FileNotFoundError(f"Stratified training data not found: {stratified_train_path}")
        df_train = pd.read_csv(stratified_train_path)
    
    print(f"Loaded {len(df_train)} training sequences")
    
    # Setup MLflow with separate tracking URI for RL
    # RL outputs go to out/RL/ instead of out/
    rl_output_dir = os.path.join(BASE_DIR, 'out', 'RL')
    os.makedirs(rl_output_dir, exist_ok=True)
    
    # Set MLflow tracking URI to RL directory
    mlflow.set_tracking_uri(f"file://{rl_output_dir}")
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            'rl_lr': args.lr,
            'rl_batch_size': args.batch_size,
            'rl_epochs': args.epochs,
            'reward_weight': args.reward_weight,
            'entropy_weight': args.entropy_weight,
            'baseline_decay': args.baseline_decay,
            'temperature': args.temperature,
            'mle_model_path': args.model_path,
            'critic_model_path': args.ml_dedge_model_path,
            'dedge_min': dEdge_min,
            'dedge_max': dEdge_max,
            'dedge_range': dEdge_range
        })
        
        # Initialize baseline
        reward_baseline = 0.0
        
        # Training loop
        print("Starting RL training loop...")
        for epoch in range(args.epochs):
            model.train()
            epoch_rewards = []
            epoch_losses = []
            epoch_entropies = []
            
            # Ensure model is in training mode and parameters require grad
            model.train()
            for param in model.parameters():
                if not param.requires_grad:
                    print(f"WARNING: Parameter does not require grad! Enabling gradients...")
                    param.requires_grad = True
            
            # Shuffle data
            df_epoch = df_train.sample(frac=1.0, random_state=epoch).reset_index(drop=True)
            
            print(f"Epoch {epoch+1}/{args.epochs}: Processing {len(df_epoch)} samples in batches of {args.batch_size}")
            sys.stdout.flush()
            
            # Process in batches
            for batch_start in range(0, len(df_epoch), args.batch_size):
                if batch_start % (args.batch_size * 10) == 0:
                    print(f"  Processing batch {batch_start // args.batch_size + 1}...")
                    sys.stdout.flush()
                batch_end = min(batch_start + args.batch_size, len(df_epoch))
                batch_data = df_epoch.iloc[batch_start:batch_end]
                
                # Sample conditions from batch
                batch_conditions = []
                batch_target_dEdge = []
                
                for _, row in batch_data.iterrows():
                    seq = row['Feature']
                    dEdge = row['Label']
                    seq_length = len(seq)
                    
                    # Normalize dEdge
                    dEdge_normalized = (dEdge - dEdge_min) / dEdge_range
                    condition = torch.FloatTensor([dEdge_normalized, seq_length]).unsqueeze(0).to(device)
                    batch_conditions.append(condition)
                    batch_target_dEdge.append(dEdge)
                
                batch_conditions = torch.cat(batch_conditions, dim=0)  # [batch_size, 2]
                batch_target_dEdge = torch.tensor(batch_target_dEdge, device=device)  # [batch_size]
                
                # Generate sequences
                generated_sequences = []
                log_probs_list = []
                
                # Generate sequences for the batch
                for i in range(batch_conditions.size(0)):
                    if i % 8 == 0 and i > 0:
                        print(f"    Generated {i}/{batch_conditions.size(0)} sequences...")
                        sys.stdout.flush()
                    condition = batch_conditions[i:i+1]  # [1, 2]
                    target_seq_len = int(condition[0, 1].item())
                    
                    # Generate sequence
                    # Note: model.generate() sets model to eval mode, but we need train mode for gradients
                    # We'll set it back to train mode after generation
                    was_training = model.training
                    generated = model.generate(
                        conditions=condition,
                        max_len=model_args.src_len,
                        temperature=args.temperature
                    )
                    if was_training:
                        model.train()  # Set back to training mode for gradient computation
                    
                    # Decode sequence
                    generated_seq = decode_sequence_from_tokens(
                        generated[0].cpu().numpy(),
                        src_vocab_reverse
                    )
                    
                    # Truncate to target length
                    generated_seq = generated_seq[:target_seq_len]
                    generated_sequences.append(generated_seq)
                    
                    # Compute log probabilities for the generated sequence
                    # We need to recompute logits for the generated sequence
                    # generated_tokens = [START(1), token1, token2, ..., tokenN, PAD(0), ...]
                    dec_input = torch.full((1, 1), 1, dtype=torch.long, device=device)  # Start with START token
                    generated_tokens = generated[0].cpu().numpy()
                    
                    # Build decoder input step by step and compute log probs
                    # Skip START token at position 0, start from position 1
                    seq_log_probs = []
                    loop_end = min(len(generated_tokens), target_seq_len + 1)
                    for t in range(1, loop_end):  # Start from t=1 to skip START
                        if generated_tokens[t] == 0:  # PAD token
                            # Debug: log when we hit PAD early
                            if t == 1 and batch_start % (args.batch_size * 100) == 0:
                                print(f"    WARNING: Sequence {i} in batch {batch_start // args.batch_size + 1}: "
                                      f"generated_tokens[1] is PAD (0). Generated tokens: {generated_tokens[:5]}")
                                sys.stdout.flush()
                            break
                        
                        # Forward pass to get logits
                        pad_mask = torch.zeros(1, dec_input.size(1), dtype=torch.bool, device=device)
                        causal_mask = get_attn_subsequence_mask(dec_input).to(device)
                        dec_self_attn_mask = create_combined_mask(pad_mask, causal_mask)
                        dec_enc_attn_mask = torch.zeros(1, dec_input.size(1), 1, dtype=torch.bool, device=device)
                        
                        logits = model(dec_input, condition, dec_self_attn_mask, dec_enc_attn_mask)
                        log_probs_step = F.log_softmax(logits[:, -1, :], dim=-1)  # [1, vocab_size]
                        
                        # Get log prob of the generated token at position t
                        next_token = generated_tokens[t]
                        log_prob = log_probs_step[0, next_token]
                        seq_log_probs.append(log_prob)
                        
                        # Append token to decoder input for next iteration
                        dec_input = torch.cat([dec_input, torch.tensor([[next_token]], device=device)], dim=1)
                    
                    if len(seq_log_probs) > 0:
                        log_probs_list.append(torch.stack(seq_log_probs))
                    else:
                        # Debug: log when we have empty log_probs
                        if batch_start % (args.batch_size * 100) == 0:
                            print(f"    WARNING: Sequence {i} in batch {batch_start // args.batch_size + 1}: "
                                  f"seq_log_probs is EMPTY! generated_tokens: {generated_tokens[:min(10, len(generated_tokens))]}, "
                                  f"target_seq_len: {target_seq_len}, loop_end: {loop_end}")
                            sys.stdout.flush()
                        # Create a tensor with requires_grad=True to ensure gradients flow
                        # This should rarely happen, but if it does, we need gradients
                        dummy_log_prob = torch.tensor([0.0], device=device, requires_grad=True)
                        log_probs_list.append(dummy_log_prob)
                
                # Compute rewards
                rewards = compute_reward(
                    generated_sequences,
                    batch_target_dEdge,
                    critic_model,
                    dEdge_min,
                    dEdge_range,
                    device,
                    model_args.src_len
                )
                
                # Stack log probs and track actual sequence lengths
                max_len = max(len(lp) for lp in log_probs_list)
                padded_log_probs = []
                seq_lengths_list = []
                empty_count = 0
                for lp in log_probs_list:
                    actual_len = len(lp)
                    seq_lengths_list.append(actual_len)
                    if actual_len == 1 and lp[0].item() == 0.0:  # Check if it's a dummy [0.0] tensor
                        empty_count += 1
                    if actual_len < max_len:
                        padding = torch.zeros(max_len - actual_len, device=device)
                        lp = torch.cat([lp, padding])
                    padded_log_probs.append(lp)
                log_probs = torch.stack(padded_log_probs)  # [batch_size, max_len]
                seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.float, device=device)  # [batch_size]
                
                # Debug: log if many sequences have empty log_probs
                if empty_count > 0 and batch_start % (args.batch_size * 10) == 0:
                    print(f"    WARNING: {empty_count}/{len(log_probs_list)} sequences have empty log_probs (dummy [0.0] tensors)")
                    sys.stdout.flush()
                
                # Debug: Check if log_probs have gradients
                if not log_probs.requires_grad:
                    print(f"ERROR: log_probs does not require grad! Checking individual components...")
                    for i, lp in enumerate(log_probs_list):
                        if not lp.requires_grad:
                            print(f"  Sequence {i}: log_probs_list[{i}] does not require grad (len={len(lp)})")
                    # Check if model parameters require grad
                    model_params_require_grad = [p.requires_grad for p in model.parameters()]
                    print(f"  Model parameters requiring grad: {sum(model_params_require_grad)}/{len(model_params_require_grad)}")
                    raise RuntimeError("log_probs tensor does not require gradients. Cannot compute loss with gradients.")
                
                # Compute policy gradient loss
                loss = compute_policy_gradient_loss(
                    log_probs,
                    rewards,
                    baseline=reward_baseline,
                    entropy_bonus=args.entropy_weight,
                    seq_lengths=seq_lengths
                )
                
                # Check if loss has gradients before backward
                if not loss.requires_grad:
                    print(f"ERROR: Loss does not require grad! log_probs.requires_grad={log_probs.requires_grad}")
                    print(f"  log_probs shape: {log_probs.shape}, dtype: {log_probs.dtype}")
                    print(f"  rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
                    raise RuntimeError("Loss tensor does not require gradients. Cannot perform backward pass.")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track metrics
                epoch_rewards.extend(rewards.cpu().numpy())
                epoch_losses.append(loss.item())
                
                # Compute entropy
                probs = torch.exp(log_probs)
                entropy = -(probs * log_probs).sum(dim=1).mean().item()
                epoch_entropies.append(entropy)
            
            # Update baseline
            if epoch_rewards:
                reward_baseline = args.baseline_decay * reward_baseline + (1 - args.baseline_decay) * np.mean(epoch_rewards)
            
            # Log metrics
            avg_reward = np.mean(epoch_rewards)
            avg_loss = np.mean(epoch_losses)
            avg_entropy = np.mean(epoch_entropies)
            
            mlflow.log_metrics({
                'reward': avg_reward,
                'loss': avg_loss,
                'entropy': avg_entropy,
                'baseline': reward_baseline
            }, step=epoch)
            
            print(f'Epoch {epoch+1}/{args.epochs}: '
                  f'Reward: {avg_reward:.4f} | '
                  f'Baseline: {reward_baseline:.4f} | '
                  f'Loss: {avg_loss:.4f} | '
                  f'Entropy: {avg_entropy:.4f}')
            sys.stdout.flush()
            
            # Save checkpoint periodically
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    MODEL_SAVE_DIR,
                    f'ConditionalGenerator_v1v2_rl_epoch_{epoch+1}.pt'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': model_args,
                    'rl_args': args,
                    'dedge_min': dEdge_min,
                    'dedge_max': dEdge_max,
                    'dedge_range': dEdge_range,
                    'reward_baseline': reward_baseline
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save final model
        final_model_path = os.path.join(
            MODEL_SAVE_DIR,
            f'ConditionalGenerator_v1v2_rl_final.pt'
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': model_args,
            'rl_args': args,
            'dedge_min': dEdge_min,
            'dedge_max': dEdge_max,
            'dedge_range': dEdge_range,
            'reward_baseline': reward_baseline
        }, final_model_path)
        print(f"Saved final model to {final_model_path}")
        mlflow.log_artifact(final_model_path)
    
    print("RL fine-tuning completed!")

if __name__ == '__main__':
    main()

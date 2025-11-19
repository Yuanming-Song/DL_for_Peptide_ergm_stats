"""
Debug script to understand RL loss computation.

This script generates 10 sequences and prints detailed information:
- Sequence tokens and decoded sequence
- Log probabilities for each token
- Softmax probabilities for each token
- Sum of log probabilities
- Reward
- Advantage
- Loss contribution
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Define base paths
PROJECT_ROOT = '/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide'
BASE_DIR = os.path.join(PROJECT_ROOT, 'ML_dEdge_gen', 'v1+2')
TRAINING_DIR = os.path.join(BASE_DIR, 'scripts', 'training')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Add paths to sys.path
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'OG_util_py'))
from utils_seq import *
from models_seq_OG import get_attn_subsequence_mask
sys.path.append(TRAINING_DIR)
from models_gen import *

def load_ml_dedge_critic(model_path, device, src_len=10, dEdge_min=-1.0, dEdge_max=3.0):
    """Load pre-trained ML_dEdge model as frozen critic."""
    from models_seq_OG import Transformer
    
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
    critic_model = Transformer(ml_dedge_args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        critic_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        critic_model.load_state_dict(checkpoint)
    
    critic_model.eval()
    for param in critic_model.parameters():
        param.requires_grad = False
    
    return critic_model

def decode_sequence_from_tokens(token_ids, src_vocab_reverse):
    """Convert token IDs back to amino acid sequence string."""
    valid_tokens = [t for t in token_ids if t > 1]
    sequence = ''.join([src_vocab_reverse.get(t, '?') for t in valid_tokens])
    return sequence

def compute_reward(generated_sequences, target_dEdge, critic_model, dEdge_min, dEdge_range, device, src_len):
    """Compute reward based on dEdge matching."""
    if len(generated_sequences) == 0:
        return torch.tensor([0.0], device=device)
    
    gen_enc_inputs = make_data(np.array(generated_sequences), src_len).to(device)
    
    with torch.no_grad():
        predicted_dEdge_original = critic_model(gen_enc_inputs).squeeze()
    
    # Handle scalar vs tensor
    if predicted_dEdge_original.dim() == 0:
        predicted_dEdge_original = predicted_dEdge_original.unsqueeze(0)
    if target_dEdge.dim() == 0:
        target_dEdge = target_dEdge.unsqueeze(0)
    
    predicted_dEdge_normalized = (predicted_dEdge_original - dEdge_min) / dEdge_range
    target_dEdge_normalized = (target_dEdge - dEdge_min) / dEdge_range
    
    mse = F.mse_loss(predicted_dEdge_normalized, target_dEdge_normalized, reduction='none')
    reward = -mse
    
    return reward, predicted_dEdge_original

def create_combined_mask(pad_mask, causal_mask):
    """Combine padding mask and causal mask for decoder self-attention."""
    pad_mask_expanded = pad_mask.unsqueeze(1).expand(-1, pad_mask.size(1), -1)
    combined_mask = causal_mask | pad_mask_expanded
    return combined_mask

def main():
    print("="*80)
    print("RL Loss Debug Script")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Model paths
    MLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ConditionalGenerator_v1v2_minmax_lr_0.003_bs_512.pt')
    CRITIC_MODEL_PATH = os.path.join(PROJECT_ROOT, 'ML_dEdge', 'models', 'iteration2', 'Transformer_curriculum_lr_0.2_bs_1024.pt')
    N_SEQUENCES = 10
    
    # Load MLE model
    print("Loading MLE model...")
    checkpoint = torch.load(MLE_MODEL_PATH, map_location=device)
    model_args = checkpoint['args']
    model = ConditionalGenerator(model_args).to(device)
    
    # Convert checkpoint if needed
    old_state_dict = checkpoint['model_state_dict']
    needs_conversion = any('transformer.layers' in k for k in old_state_dict.keys())
    
    if needs_conversion:
        new_state_dict = {}
        for old_key, value in old_state_dict.items():
            new_key = old_key
            if old_key == 'pos_enc.pe':
                new_key = 'pos_emb.pe'
            elif old_key.startswith('cond_emb.'):
                new_key = old_key.replace('cond_emb.', 'cond_input.')
            elif old_key.startswith('transformer.layers.'):
                new_key = old_key.replace('transformer.layers.', 'decoder.layers.')
            else:
                new_key = old_key
            new_state_dict[new_key] = value
        
        if 'decoder.layers.0.self_attn.in_proj_weight' in new_state_dict:
            for key in list(new_state_dict.keys()):
                if key.startswith('decoder.layers.0.') and 'multihead_attn' not in key:
                    encoder_key = key.replace('decoder.layers.0.', 'encoder.layers.0.')
                    new_state_dict[encoder_key] = new_state_dict[key].clone()
        
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.train()
    
    # Get normalization parameters
    dEdge_min = checkpoint.get('dedge_min', None)
    dEdge_max = checkpoint.get('dedge_max', None)
    dEdge_range = checkpoint.get('dedge_range', None)
    print(f"dEdge normalization: min={dEdge_min:.6f}, max={dEdge_max:.6f}, range={dEdge_range:.6f}\n")
    
    # Load critic model
    print("Loading critic model...")
    critic_model = load_ml_dedge_critic(CRITIC_MODEL_PATH, device, model_args.src_len, dEdge_min, dEdge_max)
    print("Critic model loaded\n")
    
    # Create reverse vocabulary
    src_vocab_reverse = {v: k for k, v in src_vocab.items()}
    
    # Generate test sequences
    print(f"Generating {N_SEQUENCES} test sequences...\n")
    
    all_sequences = []
    all_log_probs = []
    all_rewards = []
    all_target_dEdges = []
    all_predicted_dEdges = []
    
    for seq_idx in range(N_SEQUENCES):
        # Random condition
        dEdge_norm = np.random.uniform(0.2, 0.8)
        seq_len = np.random.randint(4, 10)
        condition = torch.FloatTensor([[dEdge_norm, seq_len]]).to(device)
        target_dEdge = dEdge_norm * dEdge_range + dEdge_min
        
        print(f"{'='*80}")
        print(f"Sequence {seq_idx + 1}/{N_SEQUENCES}")
        print(f"{'='*80}")
        print(f"Condition: dEdge_norm={dEdge_norm:.4f}, seq_len={seq_len}")
        print(f"Target dEdge (original): {target_dEdge:.4f}\n")
        
        # Generate sequence
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                conditions=condition,
                max_len=model_args.src_len,
                temperature=1.0
            )
        model.train()
        
        generated_tokens = generated[0].cpu().numpy()
        generated_seq = decode_sequence_from_tokens(generated_tokens, src_vocab_reverse)
        generated_seq = generated_seq[:seq_len]
        
        print(f"Generated sequence: {generated_seq}")
        print(f"Sequence length: {len(generated_seq)}")
        print(f"Generated tokens: {generated_tokens[:seq_len+2]}\n")
        
        # Compute log probabilities step by step
        dec_input = torch.full((1, 1), 1, dtype=torch.long, device=device)
        seq_log_probs = []
        seq_probs = []
        seq_token_probs = []
        
        print("Token-by-token analysis:")
        print("-" * 80)
        print(f"{'Token':<10} {'Token ID':<10} {'Log Prob':<15} {'Prob':<15} {'Top 3 Probs':<40}")
        print("-" * 80)
        
        for t in range(min(len(generated_tokens), seq_len)):
            if generated_tokens[t] == 0:
                break
            
            # Forward pass
            pad_mask = torch.zeros(1, dec_input.size(1), dtype=torch.bool, device=device)
            causal_mask = get_attn_subsequence_mask(dec_input).to(device)
            dec_self_attn_mask = create_combined_mask(pad_mask, causal_mask)
            dec_enc_attn_mask = torch.zeros(1, dec_input.size(1), 1, dtype=torch.bool, device=device)
            
            logits = model(dec_input, condition, dec_self_attn_mask, dec_enc_attn_mask)
            log_probs_step = F.log_softmax(logits[:, -1, :], dim=-1)  # [1, vocab_size]
            probs_step = F.softmax(logits[:, -1, :], dim=-1)  # [1, vocab_size]
            
            next_token = generated_tokens[t]
            log_prob = log_probs_step[0, next_token].item()
            prob = probs_step[0, next_token].item()
            
            seq_log_probs.append(log_prob)
            seq_probs.append(prob)
            
            # Get top 3 probabilities
            top3_probs, top3_indices = torch.topk(probs_step[0], 3)
            top3_tokens = [src_vocab_reverse.get(idx.item(), '?') for idx in top3_indices]
            top3_str = ', '.join([f"{tok}({p:.3f})" for tok, p in zip(top3_tokens, top3_probs.detach().cpu().numpy())])
            
            token_char = src_vocab_reverse.get(next_token, '?')
            
            print(f"{token_char:<10} {next_token:<10} {log_prob:<15.6f} {prob:<15.6f} {top3_str:<40}")
            
            dec_input = torch.cat([dec_input, torch.tensor([[next_token]], device=device)], dim=1)
        
        seq_log_probs_tensor = torch.tensor(seq_log_probs, device=device)
        sum_log_probs = seq_log_probs_tensor.sum().item()
        
        print("-" * 80)
        print(f"Sum of log probabilities: {sum_log_probs:.6f}")
        print(f"Mean log probability: {sum_log_probs / len(seq_log_probs):.6f}")
        print(f"Product of probabilities: {np.prod(seq_probs):.6f}")
        print()
        
        # Compute reward
        rewards, predicted_dEdge = compute_reward(
            [generated_seq],
            torch.tensor([target_dEdge], device=device),
            critic_model,
            dEdge_min,
            dEdge_range,
            device,
            model_args.src_len
        )
        reward = rewards[0].item() if rewards.dim() > 0 else rewards.item()
        if predicted_dEdge.dim() > 0:
            pred_dEdge = predicted_dEdge[0].item()
        else:
            pred_dEdge = predicted_dEdge.item()
        
        print(f"Reward computation:")
        print(f"  Predicted dEdge (original): {pred_dEdge:.6f}")
        print(f"  Target dEdge (original): {target_dEdge:.6f}")
        print(f"  Reward: {reward:.6f}")
        print()
        
        # Compute advantage (assuming baseline = 0 for now)
        baseline = 0.0
        advantage = reward - baseline
        
        print(f"Loss computation:")
        print(f"  Sum log_probs: {sum_log_probs:.6f}")
        print(f"  Advantage (reward - baseline): {advantage:.6f}")
        print(f"  log_prob * advantage: {sum_log_probs * advantage:.6f}")
        print(f"  Loss contribution: {-sum_log_probs * advantage:.6f}")
        print()
        
        all_sequences.append(generated_seq)
        all_log_probs.append(sum_log_probs)
        all_rewards.append(reward)
        all_target_dEdges.append(target_dEdge)
        all_predicted_dEdges.append(pred_dEdge)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Seq':<5} {'Length':<8} {'Sum LogProb':<15} {'Reward':<12} {'Loss Contrib':<15}")
    print("-" * 80)
    for i, (seq, log_prob, reward) in enumerate(zip(all_sequences, all_log_probs, all_rewards)):
        advantage = reward - 0.0
        loss_contrib = -log_prob * advantage
        print(f"{i+1:<5} {len(seq):<8} {log_prob:<15.6f} {reward:<12.6f} {loss_contrib:<15.6f}")
    
    print("-" * 80)
    print(f"Mean sum log_probs: {np.mean(all_log_probs):.6f}")
    print(f"Mean reward: {np.mean(all_rewards):.6f}")
    print(f"Mean loss contribution: {np.mean([-lp * (r - 0.0) for lp, r in zip(all_log_probs, all_rewards)]):.6f}")
    print()

if __name__ == '__main__':
    main()


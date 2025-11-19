"""
Debug script to check what sequences are actually being generated during RL training.
This will help identify why log_probs are empty (leading to loss=0.0).
"""

import torch
import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.training.models_gen import ConditionalGenerator
from scripts.training.train_generative_model_rl import (
    load_model, load_ml_dedge_critic, decode_sequence_from_tokens,
    load_training_data, get_attn_subsequence_mask, create_combined_mask
)
import torch.nn.functional as F

def debug_sequence_generation():
    """
    Load the RL model checkpoint and generate sequences to see what's happening.
    """
    print("=" * 80)
    print("DEBUGGING RL SEQUENCE GENERATION")
    print("=" * 80)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    # Find the latest RL checkpoint
    model_dir = '/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2/models'
    rl_checkpoints = [f for f in os.listdir(model_dir) if 'rl' in f.lower() and f.endswith('.pt')]
    
    if not rl_checkpoints:
        print("No RL checkpoints found. Checking for MLE checkpoint...")
        mle_checkpoints = [f for f in os.listdir(model_dir) if 'mle' in f.lower() and f.endswith('.pt')]
        if mle_checkpoints:
            checkpoint_path = os.path.join(model_dir, sorted(mle_checkpoints)[-1])
            print(f"Using MLE checkpoint: {checkpoint_path}")
        else:
            print("ERROR: No checkpoints found!")
            return
    else:
        checkpoint_path = os.path.join(model_dir, sorted(rl_checkpoints)[-1])
        print(f"Using RL checkpoint: {checkpoint_path}")
    
    print()
    
    # Load model
    print("Loading model...")
    model, model_args, src_vocab_reverse = load_model(checkpoint_path, device)
    model.eval()
    print(f"Model loaded. src_len={model_args.src_len}, vocab_size={model_args.tgt_vocab_size}")
    print()
    
    # Load a sample batch from training data
    print("Loading training data...")
    train_loader = load_training_data(
        batch_size=4,  # Small batch for debugging
        src_len=model_args.src_len
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    batch_conditions = batch['condition'].to(device)  # [batch_size, 2]
    batch_target_dEdge = batch['target_dEdge'].to(device)  # [batch_size]
    
    print(f"Batch size: {batch_conditions.size(0)}")
    print(f"Conditions (dEdge, seq_len):")
    for i in range(batch_conditions.size(0)):
        dEdge = batch_conditions[i, 0].item()
        seq_len = int(batch_conditions[i, 1].item())
        print(f"  Sequence {i}: dEdge={dEdge:.4f}, seq_len={seq_len}")
    print()
    
    # Generate sequences
    print("=" * 80)
    print("GENERATING SEQUENCES")
    print("=" * 80)
    print()
    
    for i in range(batch_conditions.size(0)):
        condition = batch_conditions[i:i+1]  # [1, 2]
        target_seq_len = int(condition[0, 1].item())
        target_dEdge = condition[0, 0].item()
        
        print(f"Sequence {i}: target_dEdge={target_dEdge:.4f}, target_seq_len={target_seq_len}")
        
        # Generate
        generated = model.generate(
            conditions=condition,
            max_len=model_args.src_len,
            temperature=1.0
        )
        
        generated_tokens = generated[0].cpu().numpy()
        print(f"  Generated tokens (first 10): {generated_tokens[:10]}")
        print(f"  Generated tokens length: {len(generated_tokens)}")
        print(f"  Non-zero tokens: {np.sum(generated_tokens != 0)}")
        print(f"  Tokens == 1 (START): {np.sum(generated_tokens == 1)}")
        print(f"  Tokens == 0 (PAD): {np.sum(generated_tokens == 0)}")
        print()
        
        # Decode
        generated_seq = decode_sequence_from_tokens(generated_tokens, src_vocab_reverse)
        print(f"  Decoded sequence: '{generated_seq}'")
        print(f"  Decoded length: {len(generated_seq)}")
        print()
        
        # Truncate to target length
        generated_seq_truncated = generated_seq[:target_seq_len]
        print(f"  Truncated sequence: '{generated_seq_truncated}'")
        print(f"  Truncated length: {len(generated_seq_truncated)}")
        print()
        
        # Now simulate the log_prob computation
        print(f"  Computing log probabilities...")
        dec_input = torch.full((1, 1), 1, dtype=torch.long, device=device)
        seq_log_probs = []
        
        loop_range = range(1, min(len(generated_tokens), target_seq_len + 1))
        print(f"  Loop range: range(1, min({len(generated_tokens)}, {target_seq_len}+1)) = range(1, {min(len(generated_tokens), target_seq_len + 1)})")
        print(f"  Loop will iterate over positions: {list(loop_range)}")
        print()
        
        for t in loop_range:
            if generated_tokens[t] == 0:  # PAD token
                print(f"    Position {t}: PAD token (0) detected, breaking loop")
                break
            
            print(f"    Position {t}: token={generated_tokens[t]}, computing log_prob...")
            
            # Forward pass
            pad_mask = torch.zeros(1, dec_input.size(1), dtype=torch.bool, device=device)
            causal_mask = get_attn_subsequence_mask(dec_input).to(device)
            dec_self_attn_mask = create_combined_mask(pad_mask, causal_mask)
            dec_enc_attn_mask = torch.zeros(1, dec_input.size(1), 1, dtype=torch.bool, device=device)
            
            with torch.no_grad():
                logits = model(dec_input, condition, dec_self_attn_mask, dec_enc_attn_mask)
                log_probs_step = F.log_softmax(logits[:, -1, :], dim=-1)
            
            next_token = generated_tokens[t]
            log_prob = log_probs_step[0, next_token]
            seq_log_probs.append(log_prob)
            
            print(f"      log_prob = {log_prob.item():.4f}")
            
            # Append token to decoder input
            dec_input = torch.cat([dec_input, torch.tensor([[next_token]], device=device)], dim=1)
        
        print(f"  Result: seq_log_probs has {len(seq_log_probs)} entries")
        if len(seq_log_probs) > 0:
            print(f"  Log probs: {[lp.item() for lp in seq_log_probs]}")
            print(f"  Mean log prob: {torch.stack(seq_log_probs).mean().item():.4f}")
        else:
            print(f"  WARNING: seq_log_probs is EMPTY! This will lead to dummy [0.0] tensor!")
        print()
        print("-" * 80)
        print()

if __name__ == "__main__":
    debug_sequence_generation()


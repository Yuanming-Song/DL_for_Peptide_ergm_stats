"""
Debug script to show step-by-step error calculation for RL workflow.

This script:
1. Loads the MLE model
2. Loads the critic model
3. Generates sequences
4. Shows step-by-step dEdge error calculation
5. Writes detailed debug info to a text file

Usage:
    python test_rl_workflow.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from datetime import datetime

# Define base paths
PROJECT_ROOT = '/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide'
BASE_DIR = os.path.join(PROJECT_ROOT, 'ML_dEdge_gen', 'v1+2')
TRAINING_DIR = os.path.join(BASE_DIR, 'scripts', 'training')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DEBUG_OUTPUT_FILE = os.path.join(TRAINING_DIR, 'test_debug', 'rl_error_calculation_debug.txt')

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
    # Remove padding tokens (0) and START token at position 0 only
    valid_tokens = []
    for i, t in enumerate(token_ids):
        if t == 0:  # PAD token - skip
            continue
        if i == 0 and t == 1:  # START token at position 0 - skip
            continue
        valid_tokens.append(t)
    sequence = ''.join([src_vocab_reverse.get(t, '?') for t in valid_tokens])
    return sequence

def compute_reward_detailed(generated_sequences, target_dEdge, critic_model, dEdge_min, dEdge_range, device, src_len, debug_file):
    """Compute reward based on dEdge matching with detailed step-by-step output."""
    if len(generated_sequences) == 0:
        return torch.tensor([0.0], device=device), {}
    
    debug_file.write("\n" + "="*80 + "\n")
    debug_file.write("STEP-BY-STEP dEdge ERROR CALCULATION\n")
    debug_file.write("="*80 + "\n\n")
    
    # Step 1: Encode sequences
    debug_file.write("Step 1: Encode generated sequences\n")
    debug_file.write(f"  Generated sequences: {generated_sequences}\n")
    gen_enc_inputs = make_data(np.array(generated_sequences), src_len).to(device)
    debug_file.write(f"  Encoded input shape: {gen_enc_inputs.shape}\n\n")
    
    # Step 2: Predict dEdge using critic
    debug_file.write("Step 2: Predict dEdge using critic model\n")
    with torch.no_grad():
        predicted_dEdge_original = critic_model(gen_enc_inputs).squeeze()
    
    # Handle scalar vs tensor
    if predicted_dEdge_original.dim() == 0:
        predicted_dEdge_original = predicted_dEdge_original.unsqueeze(0)
    if target_dEdge.dim() == 0:
        target_dEdge = target_dEdge.unsqueeze(0)
    
    debug_file.write(f"  Predicted dEdge (original scale): {predicted_dEdge_original.cpu().numpy()}\n")
    debug_file.write(f"  Target dEdge (original scale): {target_dEdge.cpu().numpy()}\n")
    debug_file.write(f"  dEdge normalization params: min={dEdge_min:.6f}, range={dEdge_range:.6f}\n\n")
    
    # Step 3: Normalize both predicted and target
    debug_file.write("Step 3: Normalize dEdge values\n")
    predicted_dEdge_normalized = (predicted_dEdge_original - dEdge_min) / dEdge_range
    target_dEdge_normalized = (target_dEdge - dEdge_min) / dEdge_range
    debug_file.write(f"  Predicted dEdge (normalized): {predicted_dEdge_normalized.cpu().numpy()}\n")
    debug_file.write(f"  Target dEdge (normalized): {target_dEdge_normalized.cpu().numpy()}\n")
    debug_file.write(f"  Formula: normalized = (original - min) / range\n\n")
    
    # Step 4: Compute MSE
    debug_file.write("Step 4: Compute MSE (Mean Squared Error)\n")
    mse = F.mse_loss(predicted_dEdge_normalized, target_dEdge_normalized, reduction='none')
    debug_file.write(f"  MSE per sequence: {mse.cpu().numpy()}\n")
    debug_file.write(f"  MSE formula: (predicted_norm - target_norm)^2\n")
    for i, (pred_norm, tgt_norm, mse_val) in enumerate(zip(
        predicted_dEdge_normalized.cpu().numpy(),
        target_dEdge_normalized.cpu().numpy(),
        mse.cpu().numpy()
    )):
        error = pred_norm - tgt_norm
        debug_file.write(f"    Seq {i+1}: ({pred_norm:.6f} - {tgt_norm:.6f})^2 = {error:.6f}^2 = {mse_val:.6f}\n")
    debug_file.write("\n")
    
    # Step 5: Compute reward
    debug_file.write("Step 5: Compute reward (negative MSE)\n")
    reward = -mse
    debug_file.write(f"  Reward per sequence: {reward.cpu().numpy()}\n")
    debug_file.write(f"  Reward formula: -MSE (higher is better)\n")
    debug_file.write(f"  Mean reward: {reward.mean().item():.6f}\n\n")
    
    # Step 6: Compute absolute error in original scale
    debug_file.write("Step 6: Absolute error in original scale\n")
    abs_error_original = torch.abs(predicted_dEdge_original - target_dEdge)
    debug_file.write(f"  Absolute error (original): {abs_error_original.cpu().numpy()}\n")
    debug_file.write(f"  Mean absolute error: {abs_error_original.mean().item():.6f}\n\n")
    
    debug_info = {
        'predicted_dEdge_original': predicted_dEdge_original.cpu().numpy(),
        'target_dEdge_original': target_dEdge.cpu().numpy(),
        'predicted_dEdge_normalized': predicted_dEdge_normalized.cpu().numpy(),
        'target_dEdge_normalized': target_dEdge_normalized.cpu().numpy(),
        'mse': mse.cpu().numpy(),
        'reward': reward.cpu().numpy(),
        'abs_error_original': abs_error_original.cpu().numpy()
    }
    
    return reward, debug_info

def create_combined_mask(pad_mask, causal_mask):
    """Combine padding mask and causal mask for decoder self-attention."""
    pad_mask_expanded = pad_mask.unsqueeze(1).expand(-1, pad_mask.size(1), -1)
    combined_mask = causal_mask | pad_mask_expanded
    return combined_mask

def main():
    # Create debug output directory
    os.makedirs(os.path.dirname(DEBUG_OUTPUT_FILE), exist_ok=True)
    
    # Open debug file
    with open(DEBUG_OUTPUT_FILE, 'w') as debug_file:
        debug_file.write("="*80 + "\n")
        debug_file.write("RL Error Calculation Debug Output\n")
        debug_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        debug_file.write("="*80 + "\n\n")
        
        print("="*60)
        print("RL Error Calculation Debug Script")
        print("="*60)
        print(f"Debug output will be written to: {DEBUG_OUTPUT_FILE}\n")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        debug_file.write(f"Using device: {device}\n\n")
        print(f"Using device: {device}")
        
        # Test parameters
        MLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ConditionalGenerator_v1v2_minmax_lr_0.003_bs_512.pt')
        CRITIC_MODEL_PATH = os.path.join(PROJECT_ROOT, 'ML_dEdge', 'models', 'iteration2', 'Transformer_curriculum_lr_0.2_bs_1024.pt')
        N_TEST_SEQUENCES = 10  # Test with 10 sequences
        
        # Step 1: Load MLE model
        debug_file.write("[1/5] Loading MLE model...\n")
        print("\n[1/5] Loading MLE model...")
        checkpoint = torch.load(MLE_MODEL_PATH, map_location=device)
        model_args = checkpoint['args']
        model = ConditionalGenerator(model_args).to(device)
        
        # Convert checkpoint if needed
        old_state_dict = checkpoint['model_state_dict']
        needs_conversion = any('transformer.layers' in k for k in old_state_dict.keys())
        
        if needs_conversion:
            debug_file.write("  Converting checkpoint keys...\n")
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
        
        model.eval()  # Set to eval mode for generation
        debug_file.write(f"  MLE model loaded with {sum(p.numel() for p in model.parameters()):,} parameters\n")
        print(f"  ✓ MLE model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Get normalization parameters
        dEdge_min = checkpoint.get('dedge_min', None)
        dEdge_max = checkpoint.get('dedge_max', None)
        dEdge_range = checkpoint.get('dedge_range', None)
        debug_file.write(f"  dEdge normalization: min={dEdge_min:.6f}, max={dEdge_max:.6f}, range={dEdge_range:.6f}\n\n")
        print(f"  ✓ dEdge normalization: min={dEdge_min:.6f}, max={dEdge_max:.6f}, range={dEdge_range:.6f}")
        
        # Step 2: Load critic model
        debug_file.write("[2/5] Loading critic model...\n")
        print("\n[2/5] Loading critic model...")
        critic_model = load_ml_dedge_critic(CRITIC_MODEL_PATH, device, model_args.src_len, dEdge_min, dEdge_max)
        debug_file.write("  Critic model loaded and frozen\n\n")
        print(f"  ✓ Critic model loaded and frozen")
        
        # Step 3: Create reverse vocabulary
        debug_file.write("[3/5] Creating reverse vocabulary...\n")
        print("\n[3/5] Creating reverse vocabulary...")
        src_vocab_reverse = {v: k for k, v in src_vocab.items()}
        debug_file.write(f"  Vocabulary size: {len(src_vocab_reverse)}\n\n")
        print(f"  ✓ Vocabulary size: {len(src_vocab_reverse)}")
        
        # Step 4: Generate test sequences
        debug_file.write(f"[4/5] Generating {N_TEST_SEQUENCES} test sequences...\n\n")
        print(f"\n[4/5] Generating {N_TEST_SEQUENCES} test sequences...")
        test_conditions = []
        test_target_dEdges = []
        
        # Create some test conditions
        for i in range(N_TEST_SEQUENCES):
            # Random dEdge in normalized range [0, 1]
            dEdge_norm = np.random.uniform(0.2, 0.8)
            seq_len = np.random.randint(4, 10)
            condition = torch.FloatTensor([[dEdge_norm, seq_len]]).to(device)
            test_conditions.append(condition)
            
            # Original dEdge for reward computation
            dEdge_orig = dEdge_norm * dEdge_range + dEdge_min
            test_target_dEdges.append(dEdge_orig)
            
            debug_file.write(f"  Condition {i+1}: dEdge_norm={dEdge_norm:.4f}, seq_len={seq_len}, dEdge_orig={dEdge_orig:.4f}\n")
        
        debug_file.write("\n")
        
        generated_sequences = []
        generated_tokens_list = []
        
        for i, condition in enumerate(test_conditions):
            debug_file.write(f"\n{'='*80}\n")
            debug_file.write(f"SEQUENCE {i+1}/{N_TEST_SEQUENCES}\n")
            debug_file.write(f"{'='*80}\n")
            debug_file.write(f"Condition: dEdge_norm={condition[0,0].item():.4f}, seq_len={int(condition[0,1].item())}\n")
            debug_file.write(f"Target dEdge (original): {test_target_dEdges[i]:.4f}\n\n")
            
            print(f"  Generating sequence {i+1}/{N_TEST_SEQUENCES}...")
            target_seq_len = int(condition[0, 1].item())
            
            # Generate sequence
            with torch.no_grad():
                generated = model.generate(
                    conditions=condition,
                    max_len=model_args.src_len,
                    temperature=1.0
                )
            
            generated_tokens = generated[0].cpu().numpy()
            generated_tokens_list.append(generated_tokens)
            
            debug_file.write(f"Generated tokens (full, length {len(generated_tokens)}): {generated_tokens}\n")
            debug_file.write(f"  Token breakdown:\n")
            debug_file.write(f"    Position 0: token {generated_tokens[0]} = START token (will be removed)\n")
            for pos in range(1, min(len(generated_tokens), target_seq_len + 1)):
                token_id = generated_tokens[pos]
                token_char = src_vocab_reverse.get(token_id, '?')
                debug_file.write(f"    Position {pos}: token {token_id} = '{token_char}'\n")
            debug_file.write(f"  Note: Generated {len(generated_tokens)} tokens total (START + {len(generated_tokens)-1} sequence tokens)\n\n")
            
            # Decode sequence
            generated_seq = decode_sequence_from_tokens(generated_tokens, src_vocab_reverse)
            debug_file.write(f"After removing START token (position 0): {generated_seq}\n")
            debug_file.write(f"  Length before truncation: {len(generated_seq)}\n")
            generated_seq = generated_seq[:target_seq_len]
            generated_sequences.append(generated_seq)
            
            debug_file.write(f"After truncation to target_seq_len={target_seq_len}: {generated_seq}\n")
            debug_file.write(f"Final decoded sequence length: {len(generated_seq)}\n\n")
            print(f"    Generated: {generated_seq} (len={len(generated_seq)})")
        
        debug_file.write(f"\n✓ Generated {len(generated_sequences)} sequences\n\n")
        print(f"  ✓ Generated {len(generated_sequences)} sequences")
        
        # Step 5: Compute log probabilities with detailed step-by-step
        debug_file.write("\n" + "="*80 + "\n")
        debug_file.write("STEP-BY-STEP LOG PROBABILITY COMPUTATION\n")
        debug_file.write("="*80 + "\n\n")
        print("\n[5/6] Computing log probabilities with step-by-step calculation...")
        
        log_probs_list = []
        model.train()  # Set to train mode for log prob computation
        
        for i, (condition, generated_tokens, target_seq_len) in enumerate(zip(
            test_conditions, generated_tokens_list, [int(c[0,1].item()) for c in test_conditions]
        )):
            debug_file.write(f"\n{'='*80}\n")
            debug_file.write(f"LOG PROB COMPUTATION FOR SEQUENCE {i+1}\n")
            debug_file.write(f"{'='*80}\n")
            debug_file.write(f"Sequence: {generated_sequences[i]}\n")
            debug_file.write(f"Generated tokens: {generated_tokens[:target_seq_len+2]}\n")
            debug_file.write(f"Target sequence length: {target_seq_len}\n\n")
            
            # Start with START token
            dec_input = torch.full((1, 1), 1, dtype=torch.long, device=device)
            seq_log_probs = []
            
            debug_file.write("Step-by-step log probability computation:\n")
            debug_file.write(f"{'Step':<6} {'Token':<8} {'Token ID':<10} {'Log Prob':<15} {'Prob':<15} {'Note':<30}\n")
            debug_file.write("-"*80 + "\n")
            
            # Skip START token at position 0, start from position 1
            for t in range(1, min(len(generated_tokens), target_seq_len + 1)):
                if generated_tokens[t] == 0:  # PAD token
                    debug_file.write(f"  Position {t}: PAD token (0) - stopping\n")
                    break
                
                # Forward pass to get logits
                pad_mask = torch.zeros(1, dec_input.size(1), dtype=torch.bool, device=device)
                causal_mask = get_attn_subsequence_mask(dec_input).to(device)
                dec_self_attn_mask = create_combined_mask(pad_mask, causal_mask)
                dec_enc_attn_mask = torch.zeros(1, dec_input.size(1), 1, dtype=torch.bool, device=device)
                
                logits = model(dec_input, condition, dec_self_attn_mask, dec_enc_attn_mask)
                log_probs_step = F.log_softmax(logits[:, -1, :], dim=-1)  # [1, vocab_size]
                probs_step = F.softmax(logits[:, -1, :], dim=-1)  # [1, vocab_size]
                
                # Get log prob of the generated token at position t
                next_token = generated_tokens[t]
                log_prob = log_probs_step[0, next_token].item()
                prob = probs_step[0, next_token].item()
                
                token_char = src_vocab_reverse.get(next_token, '?')
                note = f"Position {t} in generated_tokens"
                
                debug_file.write(f"{t-1:<6} {token_char:<8} {next_token:<10} {log_prob:<15.6f} {prob:<15.6f} {note:<30}\n")
                
                seq_log_probs.append(log_prob)
                
                # Append token to decoder input for next iteration
                dec_input = torch.cat([dec_input, torch.tensor([[next_token]], device=device)], dim=1)
            
            if len(seq_log_probs) > 0:
                seq_log_probs_tensor = torch.tensor(seq_log_probs, device=device)
                sum_log_probs = seq_log_probs_tensor.sum().item()
                mean_log_probs = seq_log_probs_tensor.mean().item()
                
                debug_file.write("-"*80 + "\n")
                debug_file.write(f"Sum of log probabilities: {sum_log_probs:.6f}\n")
                debug_file.write(f"Mean log probability (normalized by length): {mean_log_probs:.6f}\n")
                debug_file.write(f"Number of tokens: {len(seq_log_probs)}\n")
                debug_file.write(f"Note: Mean = sum / length = {sum_log_probs:.6f} / {len(seq_log_probs)} = {mean_log_probs:.6f}\n")
                debug_file.write(f"      Using MEAN (not sum) to normalize by sequence length in loss computation.\n\n")
                
                log_probs_list.append(seq_log_probs_tensor)
            else:
                debug_file.write("  No valid tokens found!\n\n")
                log_probs_list.append(torch.tensor([0.0], device=device))
        
        model.eval()  # Set back to eval mode
        
        debug_file.write(f"\n✓ Computed log probabilities for {len(log_probs_list)} sequences\n\n")
        print(f"  ✓ Computed log probabilities")
        
        # Step 6: Compute rewards with detailed step-by-step
        debug_file.write("[6/6] Computing rewards with step-by-step error calculation...\n")
        print("\n[6/6] Computing rewards with step-by-step error calculation...")
        batch_target_dEdge = torch.tensor(test_target_dEdges, device=device)
        rewards, debug_info = compute_reward_detailed(
            generated_sequences,
            batch_target_dEdge,
            critic_model,
            dEdge_min,
            dEdge_range,
            device,
            model_args.src_len,
            debug_file
        )
        
        # Summary
        debug_file.write("\n" + "="*80 + "\n")
        debug_file.write("SUMMARY\n")
        debug_file.write("="*80 + "\n")
        # Add log prob info to summary (using MEAN, not sum, to normalize by length)
        debug_file.write(f"{'Seq':<5} {'Sequence':<15} {'Length':<8} {'Mean LogProb':<15} {'Target dEdge':<15} {'Pred dEdge':<15} {'Abs Error':<15} {'Reward':<15}\n")
        debug_file.write("-"*100 + "\n")
        
        for i, (seq, lp, tgt, pred, err, rew) in enumerate(zip(
            generated_sequences,
            [lp.mean().item() if len(lp) > 0 else 0.0 for lp in log_probs_list],
            debug_info['target_dEdge_original'],
            debug_info['predicted_dEdge_original'],
            debug_info['abs_error_original'],
            debug_info['reward']
        )):
            debug_file.write(f"{i+1:<5} {seq:<15} {len(seq):<8} {lp:<15.6f} {tgt:<15.6f} {pred:<15.6f} {err:<15.6f} {rew:<15.6f}\n")
        
        debug_file.write("-"*100 + "\n")
        mean_mean_logprob = np.mean([lp.mean().item() if len(lp) > 0 else 0.0 for lp in log_probs_list])
        debug_file.write(f"Mean log_prob (normalized by length): {mean_mean_logprob:.6f}\n")
        debug_file.write(f"Mean absolute error: {debug_info['abs_error_original'].mean():.6f}\n")
        debug_file.write(f"Mean reward: {debug_info['reward'].mean():.6f}\n")
        debug_file.write(f"Mean MSE (normalized): {debug_info['mse'].mean():.6f}\n\n")
        
        # Show how loss would be computed
        debug_file.write("\n" + "="*80 + "\n")
        debug_file.write("HOW LOSS WOULD BE COMPUTED (for reference)\n")
        debug_file.write("="*80 + "\n")
        debug_file.write("Loss formula: -E[log_prob * (reward - baseline)]\n")
        debug_file.write("Where:\n")
        debug_file.write("  Step 1 (per sequence): log_prob = MEAN of log probabilities over tokens\n")
        debug_file.write("    - Normalizes by sequence length: sum(log_probs) / seq_length\n")
        debug_file.write("    - Ensures sequences of different lengths contribute equally\n")
        debug_file.write("  Step 2 (over batch): E[] = MEAN over batch sequences\n")
        debug_file.write("    - loss = -mean(seq_log_prob * advantage) over batch\n")
        debug_file.write("  Other terms:\n")
        debug_file.write("    - reward = -MSE(predicted_dEdge_norm, target_dEdge_norm)\n")
        debug_file.write("    - baseline = mean reward (for variance reduction)\n\n")
        
        baseline = debug_info['reward'].mean()
        debug_file.write(f"Baseline (mean reward): {baseline:.6f}\n\n")
        debug_file.write(f"{'Seq':<5} {'Mean LogProb':<15} {'Reward':<15} {'Advantage':<15} {'log_prob*adv':<15}\n")
        debug_file.write("-"*80 + "\n")
        
        for i, (lp, rew) in enumerate(zip(
            log_probs_list,
            debug_info['reward']
        )):
            mean_lp = lp.mean().item() if len(lp) > 0 else 0.0
            advantage = rew - baseline
            product = mean_lp * advantage
            debug_file.write(f"{i+1:<5} {mean_lp:<15.6f} {rew:<15.6f} {advantage:<15.6f} {product:<15.6f}\n")
        
        debug_file.write("-"*80 + "\n")
        mean_product = np.mean([(lp.mean().item() if len(lp) > 0 else 0.0) * (rew - baseline) 
                                for lp, rew in zip(log_probs_list, debug_info['reward'])])
        debug_file.write(f"Mean (log_prob * advantage): {mean_product:.6f}\n")
        debug_file.write(f"Loss would be: {-mean_product:.6f}\n")
        debug_file.write(f"\nNote: Using MEAN log_prob (normalized by sequence length) instead of SUM\n")
        debug_file.write(f"      This ensures sequences of different lengths contribute equally to the loss.\n\n")
        
        print(f"  ✓ Rewards computed: {rewards.cpu().numpy()}")
        print(f"    Mean reward: {rewards.mean().item():.4f}")
        print(f"    Mean absolute error: {debug_info['abs_error_original'].mean():.6f}")
        
        debug_file.write("\n" + "="*80 + "\n")
        debug_file.write("Debug output complete.\n")
        debug_file.write("="*80 + "\n")
    
    print(f"\n✓ Debug output written to: {DEBUG_OUTPUT_FILE}")
    print("\n" + "="*60)
    print("Debug script complete!")
    print("="*60)

if __name__ == '__main__':
    main()

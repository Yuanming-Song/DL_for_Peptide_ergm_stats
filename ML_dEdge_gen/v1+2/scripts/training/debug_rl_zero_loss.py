"""
Debug script to investigate why loss and entropy become 0.0000 in RL training.

This script simulates the exact conditions that could lead to zero loss/entropy:
1. Empty log_probs (loop never executes)
2. All sequences generating only START/PAD tokens
3. Issues with target_seq_len calculation
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.training.models_gen import ConditionalGenerator

def investigate_zero_loss_scenario():
    """
    Investigate scenarios that could cause loss=0.0000 and entropy=0.0000
    """
    print("=" * 80)
    print("INVESTIGATING ZERO LOSS/ENTROPY SCENARIOS")
    print("=" * 80)
    print()
    
    # Scenario 1: Empty log_probs list (loop never executes)
    print("Scenario 1: Empty log_probs (loop never executes)")
    print("-" * 80)
    
    # Simulate what happens when all sequences have empty log_probs
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # All sequences get dummy [0.0] tensors
    log_probs_list = [torch.tensor([0.0], device=device, requires_grad=True) for _ in range(batch_size)]
    
    print(f"log_probs_list: {len(log_probs_list)} sequences, all with dummy [0.0] tensors")
    print(f"  Example: {log_probs_list[0]}")
    print()
    
    # Stack them
    max_len = max(len(lp) for lp in log_probs_list)  # max_len = 1
    padded_log_probs = []
    seq_lengths_list = []
    for lp in log_probs_list:
        actual_len = len(lp)  # actual_len = 1
        seq_lengths_list.append(actual_len)
        if actual_len < max_len:
            padding = torch.zeros(max_len - actual_len, device=device)
            lp = torch.cat([lp, padding])
        padded_log_probs.append(lp)
    
    log_probs = torch.stack(padded_log_probs)  # [32, 1] - all zeros
    seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.float, device=device)  # [32] - all 1.0
    
    print(f"After stacking:")
    print(f"  log_probs shape: {log_probs.shape}")
    print(f"  log_probs (first 3):\n{log_probs[:3]}")
    print(f"  seq_lengths: {seq_lengths[:5]}...")
    print()
    
    # Simulate rewards (from critic)
    rewards = torch.tensor([-0.0252] * batch_size, device=device)
    baseline = -0.0004
    
    print(f"Rewards: {rewards[:5]}... (all = {rewards[0].item():.4f})")
    print(f"Baseline: {baseline}")
    print()
    
    # Compute what the loss would be
    # Step 1: Normalize by sequence length
    seq_log_probs = log_probs.sum(dim=1) / seq_lengths.clamp(min=1.0)  # [32] - all 0.0
    print(f"seq_log_probs (normalized): {seq_log_probs[:5]}... (all = {seq_log_probs[0].item():.4f})")
    print()
    
    # Step 2: Compute advantages
    baseline_tensor = torch.tensor(baseline, dtype=rewards.dtype, device=rewards.device)
    advantages = (rewards - baseline_tensor).detach()  # [32]
    print(f"advantages: {advantages[:5]}... (all = {advantages[0].item():.4f})")
    print()
    
    # Step 3: Policy loss
    policy_loss = -(seq_log_probs * advantages).mean()  # -(0.0 * advantages).mean() = 0.0
    print(f"policy_loss: {policy_loss.item():.4f}")
    print()
    
    # Step 4: Entropy
    probs = torch.exp(log_probs)  # exp(0.0) = 1.0
    entropy = -(probs * log_probs).sum(dim=1).mean()  # -(1.0 * 0.0).sum() = 0.0
    print(f"entropy: {entropy.item():.4f}")
    print()
    
    print("=" * 80)
    print("CONCLUSION: If all sequences have empty log_probs, loss=0.0 and entropy=0.0")
    print("=" * 80)
    print()
    
    # Scenario 2: Check when the loop would not execute
    print("Scenario 2: When does the log_prob computation loop NOT execute?")
    print("-" * 80)
    
    # The loop is: for t in range(1, min(len(generated_tokens), target_seq_len + 1)):
    # It won't execute if: range(1, min(len(generated_tokens), target_seq_len + 1)) is empty
    # This happens if: min(len(generated_tokens), target_seq_len + 1) <= 1
    
    test_cases = [
        ("target_seq_len=0", 0, 10),
        ("target_seq_len=1, generated_len=10", 1, 10),
        ("target_seq_len=1, generated_len=1", 1, 1),
        ("target_seq_len=1, generated_len=2", 1, 2),
        ("target_seq_len=5, generated_len=10", 5, 10),
    ]
    
    for desc, target_seq_len, generated_len in test_cases:
        range_start = 1
        range_end = min(generated_len, target_seq_len + 1)
        loop_range = list(range(range_start, range_end))
        executes = len(loop_range) > 0
        
        print(f"{desc}:")
        print(f"  range(1, min({generated_len}, {target_seq_len}+1)) = range(1, {range_end})")
        print(f"  Loop executes: {executes} (range: {loop_range})")
        print()
    
    print("=" * 80)
    print("KEY INSIGHT: Loop doesn't execute if target_seq_len=0 OR")
    print("             if generated_tokens has only START token (len=1) AND target_seq_len=0")
    print("=" * 80)
    print()
    
    # Scenario 3: What if generated sequence is just [START, PAD, PAD, ...]?
    print("Scenario 3: Generated sequence is [START, PAD, PAD, ...]")
    print("-" * 80)
    
    generated_tokens = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # START, then all PAD
    target_seq_len = 5
    
    print(f"generated_tokens: {generated_tokens}")
    print(f"target_seq_len: {target_seq_len}")
    print()
    
    seq_log_probs = []
    for t in range(1, min(len(generated_tokens), target_seq_len + 1)):  # range(1, 6) = [1,2,3,4,5]
        if generated_tokens[t] == 0:  # PAD token
            print(f"  Position {t}: PAD token detected, breaking loop")
            break
        print(f"  Position {t}: token={generated_tokens[t]}, would compute log_prob")
        seq_log_probs.append(0.0)  # dummy
    
    print(f"Result: seq_log_probs has {len(seq_log_probs)} entries (empty!)")
    print()
    
    print("=" * 80)
    print("CONCLUSION: If generated sequence starts with [START, PAD, ...],")
    print("            the loop breaks immediately, resulting in empty log_probs")
    print("=" * 80)
    print()
    
    # Scenario 4: What if model.generate() returns sequences that are too short?
    print("Scenario 4: Model generates sequences shorter than expected")
    print("-" * 80)
    print("This could happen if:")
    print("  1. Model generates END token early (but we don't have END token)")
    print("  2. Model generates only START token (generated = [START])")
    print("  3. Model generates [START, token1, PAD, PAD, ...] where token1 is invalid")
    print()
    print("Need to check:")
    print("  - What does model.generate() actually return?")
    print("  - Are there any early stopping conditions?")
    print("  - Could the model be collapsing to generate only START/PAD?")
    print()

if __name__ == "__main__":
    investigate_zero_loss_scenario()



"""
Overfitting diagnostic script for the conditional generator.
Goal: drive CE loss < 0.5 on 100 sequences (if the data/architecture allow).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Define paths
PROJECT_ROOT = '/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide'
BASE_DIR = os.path.join(PROJECT_ROOT, 'ML_dEdge_gen', 'v1+2')
TRAINING_DIR = os.path.join(BASE_DIR, 'scripts', 'training')
TEST_DIR = os.path.join(TRAINING_DIR, 'test_debug')
STRATIFIED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'stratified')

# Add paths
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'OG_util_py'))
from utils_seq import *
sys.path.append(TRAINING_DIR)
from models_gen import *

# Import dataset class and helper functions from main training script
exec(open(os.path.join(TRAINING_DIR, 'train_generative_model.py')).read().split('if __name__')[0])

def main():
    """Test overfitting on 100 sequences with all bugs fixed"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output file for monitoring
    output_file = os.path.join(TEST_DIR, 'overfit_results.txt')
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Overfitting Test - All Bugs Fixed\n")
        f.write("="*80 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"Target: CE loss < 0.5 within 200 epochs\n")
        f.write("="*80 + "\n\n")
    
    # Hyperparameters - conservative for stability
    class Args:
        src_vocab_size = 21
        src_len = 10
        d_model = 256  # Increased capacity
        d_ff = 1024
        d_k = 64
        d_v = 64
        n_layers = 4  # More layers
        n_heads = 8
        dropout = 0.0  # No dropout for overfitting
        lr = 0.001  # Conservative LR for stability
        batch_size = 8  # Small batch
        epochs = 400  # More epochs
        warmup_epochs = 10
    
    args = Args()
    
    # Load small subset of data (100 sequences)
    print("Loading data...")
    stratified_train_path = os.path.join(STRATIFIED_DATA_DIR, 'stratified_train_seqs.csv')
    df_train = pd.read_csv(stratified_train_path)
    
    # Take only first 100 sequences
    df_train_small = df_train.head(100).copy()
    
    print(f"Training on {len(df_train_small)} sequences")
    
    # Get dEdge statistics for normalization
    dEdge_min = df_train_small['Label'].min()
    dEdge_max = df_train_small['Label'].max()
    dEdge_range = dEdge_max - dEdge_min
    
    print(f"dEdge range: [{dEdge_min:.4f}, {dEdge_max:.4f}]")
    
    # Create dataset
    train_dataset = GenerativeDataset(
        sequences=df_train_small['Feature'].values,
        dEdge_values=df_train_small['Label'].values,
        src_len=args.src_len,
        dEdge_min=dEdge_min,
        dEdge_range=dEdge_range
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = ConditionalGenerator(args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ce_criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print("Starting training...")
    print("="*80)
    
    with open(output_file, 'a') as f:
        f.write("Epoch | Train Loss | LR | Status\n")
        f.write("-"*80 + "\n")
    
    best_loss = float('inf')
    
    # Sanity check: verify architecture and alignment
    print("\n=== ARCHITECTURE VERIFICATION ===")
    enc_inputs, dec_inputs, conditions, seq_masks = next(iter(train_loader))
    print(f"Batch size: {enc_inputs.size(0)}")
    print(f"Sequence length: {enc_inputs.size(1)}")
    print(f"\n1. Input Format:")
    print(f"   dec_inputs[0, :6]: {dec_inputs[0, :6].tolist()}")
    print(f"   Format: [START(1), token1, token2, ..., tokenN, PAD(0), ...]")
    print(f"\n2. Condition Format:")
    print(f"   conditions[0]: {conditions[0].tolist()}")
    print(f"   Format: [dEdge_normalized, sequence_length]")
    print(f"   Condition is embedded and added to every position via cross-attention")
    print(f"\n3. Target Format:")
    print(f"   enc_inputs[0, :6]: {enc_inputs[0, :6].tolist()}")
    print(f"   Format: [token1, token2, ..., tokenN, PAD(0), ...]")
    print(f"\n4. Alignment Check:")
    seq_len = int(seq_masks[0].sum().item())
    print(f"   Sequence length (from mask): {seq_len}")
    print(f"   Sequence length (from condition): {int(conditions[0, 1].item())}")
    print(f"   dec_inputs[0, 0] = START(1): {dec_inputs[0, 0].item() == 1}")
    print(f"   dec_inputs[1:{seq_len+1}] == enc_inputs[:{seq_len}]: {torch.equal(dec_inputs[0, 1:seq_len+1], enc_inputs[0, :seq_len])}")
    print(f"\n5. Prediction Alignment:")
    print(f"   logits[0] (after START) → predicts enc_inputs[0] = {enc_inputs[0, 0].item()}")
    print(f"   logits[1] (after token1) → predicts enc_inputs[1] = {enc_inputs[0, 1].item()}")
    print(f"   logits[i] → predicts enc_inputs[i]")
    print(f"\n6. PAD Token Usage:")
    print(f"   PAD(0) is used for: (1) Padding sequences to fixed length for batching")
    print(f"                      (2) Currently used to stop generation (problematic)")
    print(f"   Sequence length from condition should be used to stop generation")
    print(f"\n7. Architecture Summary:")
    print(f"   - dEdge and seq_length are encoded via condition embedding (NOT as tokens)")
    print(f"   - Condition is embedded: Linear(2 -> d_model)")
    print(f"   - Embedded condition added to every position via cross-attention")
    print(f"   - This is a VALID approach (condition embedding vs token embedding)")
    if dec_inputs[0, 0].item() == 1 and torch.equal(dec_inputs[0, 1:seq_len+1], enc_inputs[0, :seq_len]) and int(conditions[0, 1].item()) == seq_len:
        print(f"\n✓ Architecture is CORRECT")
    else:
        print(f"\n✗ Architecture has issues!")
    print("="*80 + "\n")
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        
        for batch_idx, (enc_inputs, dec_inputs, conditions, seq_masks) in enumerate(train_loader):
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            conditions = conditions.to(device)
            seq_masks = seq_masks.to(device)
            
            # FIX #1: Create proper padding mask
            # seq_masks: True = valid token, False = PAD
            # pad_mask: True = PAD (should be masked), False = valid
            pad_mask = ~seq_masks  # [batch, seq_len]
            
            # FIX #2: Create causal mask
            causal_mask = get_attn_subsequence_mask(dec_inputs).to(device)  # [batch, seq_len, seq_len]
            
            # FIX #3: Combine masks properly for self-attention
            # Expand pad_mask to [batch, seq_len, seq_len]
            pad_mask_expanded = pad_mask.unsqueeze(1).expand(-1, pad_mask.size(1), -1)  # [batch, seq_len, seq_len]
            # Position (i,j) is masked if EITHER causal OR padding says so
            dec_self_attn_mask = causal_mask | pad_mask_expanded
            
            # FIX #4: Create proper cross-attention mask
            # Cross-attention: decoder attends to condition embedding
            # Condition is [batch, 1, d_model], so we need [batch, dec_len, 1]
            # But actually, we want decoder to always attend to condition, so no mask needed
            # However, shape must be correct: [batch, dec_len, enc_len] where enc_len=1
            dec_enc_attn_mask = torch.zeros(dec_inputs.size(0), dec_inputs.size(1), 1).bool().to(device)
            
            # Forward pass
            dec_logits = model(dec_inputs, conditions, dec_self_attn_mask, dec_enc_attn_mask)
            # dec_logits: [batch, seq_len, vocab_size]
            
            # FIX #5: Proper target alignment
            # Architecture: Condition [dEdge, seq_length] is embedded and added to every position
            # dec_inputs = [START(1), token1, token2, ..., tokenN, PAD(0), ...]
            # enc_inputs = [token1, token2, ..., tokenN, PAD(0), ...]
            # For teacher forcing:
            #   Position 0 (START): predicts token1 = enc_inputs[0]
            #   Position 1 (token1): predicts token2 = enc_inputs[1]
            #   Position i: predicts enc_inputs[i]
            # So logits[i] should predict enc_inputs[i]
            targets = enc_inputs  # [batch, seq_len]
            
            # Flatten for loss computation and mask out PAD positions using seq_masks
            logits_flat = dec_logits.view(-1, dec_logits.size(-1))  # [batch*seq_len, vocab_size]
            targets_flat = targets.view(-1)  # [batch*seq_len]
            mask_flat = seq_masks.view(-1)   # [batch*seq_len], True = valid, False = PAD
            
            # Keep only valid (non-PAD) positions
            logits_flat = logits_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]
            
            # CrossEntropyLoss now operates only on real tokens
            loss = ce_criterion(logits_flat, targets_flat)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Update learning rate (warmup)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        avg_loss = np.mean(train_losses)
        
        # Determine status
        if avg_loss < 0.5:
            status = "✓ PASS"
        elif avg_loss < best_loss:
            status = "↓ Improving"
            best_loss = avg_loss
        else:
            status = "→ Plateau"
        
        # Log to file
        with open(output_file, 'a') as f:
            f.write(f"{epoch+1:5d} | {avg_loss:10.4f} | {current_lr:.6f} | {status}\n")
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0 or avg_loss < 0.5:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss = {avg_loss:.4f} | LR = {current_lr:.6f} | {status}")
        
        # FIX #8: Log sample predictions every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                # Get one sample
                sample_enc, sample_dec, sample_cond, sample_mask = next(iter(train_loader))
                sample_enc = sample_enc[0:1].to(device)
                sample_dec = sample_dec[0:1].to(device)
                sample_cond = sample_cond[0:1].to(device)
                sample_mask = sample_mask[0:1].to(device)
                
                pad_mask = ~sample_mask
                causal_mask = get_attn_subsequence_mask(sample_dec).to(device)
                pad_mask_exp = pad_mask.unsqueeze(1).expand(-1, pad_mask.size(1), -1)
                dec_self_mask = causal_mask | pad_mask_exp
                dec_enc_mask = torch.zeros(sample_dec.size(0), sample_dec.size(1), 1).bool().to(device)
                
                sample_logits = model(sample_dec, sample_cond, dec_self_mask, dec_enc_mask)
                # logits[i] predicts what comes after dec_inputs[i]
                # dec_inputs = [START, token1, token2, ..., tokenN, PAD, ...]
                # enc_inputs = [token1, token2, ..., tokenN, PAD, ...]
                # So logits[0] predicts enc_inputs[0], logits[1] predicts enc_inputs[1], etc.
                sample_preds = torch.argmax(sample_logits, dim=-1)[0]  # [seq_len]
                
                seq_len = int(sample_mask[0].sum().item())
                # Compare predictions with targets
                # sample_preds[i] should equal sample_enc[0, i]
                true_tokens = sample_enc[0, :seq_len].cpu()
                pred_tokens = sample_preds[:seq_len].cpu()
                
                with open(output_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} Sample Predictions:\n")
                    f.write(f"  dec_input: {sample_dec[0, :seq_len+1].cpu().tolist()}\n")
                    f.write(f"  True:      {true_tokens.tolist()}\n")
                    f.write(f"  Pred:      {pred_tokens.tolist()}\n")
                    matches = (true_tokens == pred_tokens).sum().item()
                    f.write(f"  Matches:   {matches}/{seq_len}\n")
                    f.write(f"  Full Match: {torch.equal(true_tokens, pred_tokens)}\n\n")
            
            model.train()
        
        # Early stopping if target reached
        if avg_loss < 0.5:
            print(f"\n{'='*80}")
            print(f"SUCCESS! Model can overfit.")
            print(f"Final loss: {avg_loss:.4f} at epoch {epoch+1}")
            print(f"{'='*80}")
            
            with open(output_file, 'a') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"SUCCESS! Model can overfit.\n")
                f.write(f"Final loss: {avg_loss:.4f} at epoch {epoch+1}\n")
                f.write("="*80 + "\n")
            
            break
    else:
        # Loop completed without breaking
        print(f"\n{'='*80}")
        print(f"WARNING: Did not reach target loss < 0.5")
        print(f"Best loss achieved: {best_loss:.4f}")
        print(f"{'='*80}")
        
        with open(output_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"WARNING: Did not reach target loss < 0.5\n")
            f.write(f"Best loss achieved: {best_loss:.4f}\n")
            f.write("="*80 + "\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()

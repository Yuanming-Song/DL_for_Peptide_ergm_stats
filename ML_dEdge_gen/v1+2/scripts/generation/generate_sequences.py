# Generative model script for generating peptide sequences with target dEdge values
import numpy as np
import torch
import sys
import argparse
import os
import pandas as pd

# Define base path once at the beginning
PROJECT_ROOT = '/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide'
BASE_DIR = os.path.join(PROJECT_ROOT, 'ML_dEdge_gen', 'v1+2')
TRAINING_DIR = os.path.join(BASE_DIR, 'scripts', 'training')
OUTPUT_DIR = os.path.join(BASE_DIR, 'out')

# Add path to sys.path
sys.path.append(TRAINING_DIR)
from models_gen import *

# Reverse vocabulary for decoding
id_to_aa = {0: 'Empty', 1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
             9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 
             17: 'T', 18: 'V', 19: 'W', 20: 'Y'}

def parse_args():
    parser = argparse.ArgumentParser(description='Generate peptide sequences with target dEdge values')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained generative model checkpoint')
    parser.add_argument('--dEdge_min', type=float, required=True,
                        help='Minimum dEdge value')
    parser.add_argument('--dEdge_max', type=float, required=True,
                        help='Maximum dEdge value')
    parser.add_argument('--seq_length_min', type=int, required=True,
                        help='Minimum sequence length')
    parser.add_argument('--seq_length_max', type=int, required=True,
                        help='Maximum sequence length')
    parser.add_argument('--num_sequences', type=int, required=True,
                        help='Number of sequences to generate')
    parser.add_argument('--src_len', type=int, default=10,
                        help='Maximum sequence length for model input')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling (higher = more random)')
    
    # Model parameters (should match training)
    parser.add_argument('--src_vocab_size', type=int, default=21)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    return parser.parse_args()

def decode_sequence(token_ids):
    """Convert token IDs to amino acid sequence"""
    sequence = []
    for token_id in token_ids:
        if token_id == 0:  # Padding token
            break
        if token_id.item() in id_to_aa:
            aa = id_to_aa[token_id.item()]
            if aa != 'Empty':
                sequence.append(aa)
    return ''.join(sequence)

def generate_sequences(model, conditions, max_len, temperature=1.0, device='cuda'):
    """Generate sequences given conditions"""
    model.eval()
    with torch.no_grad():
        generated = model.generate(conditions, max_len, start_token=1, temperature=temperature)
    return generated

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Validate inputs
    if args.dEdge_min >= args.dEdge_max:
        raise ValueError("dEdge_min must be less than dEdge_max")
    if args.seq_length_min > args.seq_length_max:
        raise ValueError("seq_length_min must be less than or equal to seq_length_max")
    if args.seq_length_max > args.src_len:
        raise ValueError(f"seq_length_max ({args.seq_length_max}) cannot exceed src_len ({args.src_len})")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Initialize model
    model = ConditionalGenerator(args).to(device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded successfully")
    print(f"Generating {args.num_sequences} sequences")
    print(f"dEdge range: [{args.dEdge_min}, {args.dEdge_max}]")
    print(f"Sequence length range: [{args.seq_length_min}, {args.seq_length_max}]")
    
    # Generate random conditions
    dEdge_values = np.random.uniform(args.dEdge_min, args.dEdge_max, args.num_sequences)
    seq_lengths = np.random.randint(args.seq_length_min, args.seq_length_max + 1, args.num_sequences)
    
    # Create condition tensor: [batch_size, 2] where [:, 0] is dEdge, [:, 1] is seq_length
    conditions = torch.FloatTensor(np.column_stack([dEdge_values, seq_lengths])).to(device)
    
    # Generate sequences
    print("Generating sequences...")
    generated_tokens = generate_sequences(
        model, 
        conditions, 
        max_len=args.seq_length_max,
        temperature=args.temperature,
        device=device
    )
    
    # Decode sequences
    sequences = []
    for i in range(generated_tokens.size(0)):
        seq = decode_sequence(generated_tokens[i])
        sequences.append(seq)
    
    # Create output DataFrame
    results_df = pd.DataFrame({
        'Sequence': sequences,
        'dEdge_target': dEdge_values,
        'SeqLength_target': seq_lengths,
        'SeqLength_actual': [len(seq) for seq in sequences]
    })
    
    # Remove empty sequences
    results_df = results_df[results_df['Sequence'].str.len() > 0].reset_index(drop=True)
    
    print(f"Generated {len(results_df)} valid sequences")
    
    # Save results
    if args.output_file is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        args.output_file = os.path.join(OUTPUT_DIR, 
            f'generated_sequences_dEdge_{args.dEdge_min}_{args.dEdge_max}_len{args.seq_length_min}_{args.seq_length_max}_n{len(results_df)}.csv')
    
    results_df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to: {args.output_file}")
    print(f"\nSummary:")
    print(f"  Total sequences generated: {len(results_df)}")
    print(f"  dEdge range: [{results_df['dEdge_target'].min():.4f}, {results_df['dEdge_target'].max():.4f}]")
    print(f"  Sequence length range: [{results_df['SeqLength_actual'].min()}, {results_df['SeqLength_actual'].max()}]")
    print(f"  Mean sequence length: {results_df['SeqLength_actual'].mean():.2f}")

if __name__ == '__main__':
    main()

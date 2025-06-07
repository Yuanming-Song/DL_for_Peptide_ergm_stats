import numpy as np
import torch
import pandas as pd
import sys
import argparse
from tqdm import tqdm
import random
sys.path.append('/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/')
from utils_seq import *
from models_seq_OG import *

AA_VOCAB = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def parse_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--src_vocab_size', type=int, default=21)
    parser.add_argument('--src_len', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=32)
    parser.add_argument('--d_v', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max', type=float, default=3.0)
    parser.add_argument('--min', type=float, default=-1.0)
    parser.add_argument('--task_type', type=str, default='Regression')
    parser.add_argument('--model_type', type=str, default='Transformer',
                        choices=['Transformer', 'LSTM', 'Bi-LSTM', 'RNN'])
    
    # Generation parameters
    parser.add_argument('--target_label', type=float, required=True,
                        help='Target label value to generate sequences for')
    parser.add_argument('--comparison', type=str, choices=['greater', 'less', 'equal'],
                        help='Generate sequences with labels greater/less than target')
    parser.add_argument('--tolerance', type=float, default=0.1,
                        help='Tolerance range for label matching')
    parser.add_argument('--num_sequences', type=int, default=10,
                        help='Number of sequences to generate')
    parser.add_argument('--max_attempts', type=int, default=1000,
                        help='Maximum number of generation attempts')
    
    # File paths
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_file', type=str, default='generated_sequences.csv',
                        help='Output file for generated sequences')
    
    return parser.parse_args()

def sequence_to_features(sequence):
    """Convert amino acid sequence to feature string format"""
    return ','.join(sequence)

def generate_random_sequence(length):
    """Generate a random amino acid sequence"""
    return [random.choice(AA_VOCAB) for _ in range(length)]

def meets_criteria(predicted_label, target_label, comparison, tolerance):
    """Check if predicted label meets the specified criteria"""
    if comparison == 'greater':
        return predicted_label > target_label
    elif comparison == 'less':
        return predicted_label < target_label
    else:  # equal
        return abs(predicted_label - target_label) <= tolerance

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    if args.model_type == 'Transformer':
        model = Transformer(args).to(device)
    elif args.model_type == 'LSTM':
        model = LSTM(args).to(device)
    elif args.model_type == 'Bi-LSTM':
        model = BidirectionalLSTM(args).to(device)
    else:  # RNN
        model = RNN(args).to(device)
    
    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    generated_sequences = []
    generated_labels = []
    attempts = 0
    pbar = tqdm(total=args.num_sequences)
    
    while len(generated_sequences) < args.num_sequences and attempts < args.max_attempts:
        # Generate a batch of random sequences
        batch_size = min(100, args.num_sequences - len(generated_sequences))
        sequences = [generate_random_sequence(args.src_len) for _ in range(batch_size)]
        features = [sequence_to_features(seq) for seq in sequences]
        
        # Convert to model input format
        enc_inputs = make_data(np.array(features), args.src_len).to(device)
        
        # Get predictions
        with torch.no_grad():
            predictions = model(enc_inputs)
            predicted_labels = predictions.cpu().numpy().flatten()
        
        # Filter sequences that meet criteria
        for seq, pred_label in zip(sequences, predicted_labels):
            if meets_criteria(pred_label, args.target_label, args.comparison, args.tolerance):
                generated_sequences.append(seq)
                generated_labels.append(pred_label)
                pbar.update(1)
                
                if len(generated_sequences) >= args.num_sequences:
                    break
        
        attempts += batch_size
    
    pbar.close()
    
    # Save results
    results = pd.DataFrame({
        'Feature': [sequence_to_features(seq) for seq in generated_sequences],
        'Sequence': [''.join(seq) for seq in generated_sequences],
        'Predicted_Label': generated_labels
    })
    
    results.to_csv(args.output_file, index=False)
    print(f"\nGenerated {len(generated_sequences)} sequences")
    print(f"Results saved to {args.output_file}")
    
    if len(generated_sequences) < args.num_sequences:
        print(f"\nWarning: Could only generate {len(generated_sequences)} sequences "
              f"meeting criteria after {attempts} attempts")

if __name__ == '__main__':
    main() 
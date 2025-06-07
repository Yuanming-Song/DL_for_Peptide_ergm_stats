import numpy as np
import torch
import pandas as pd
import sys
import gc
sys.path.append('/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/')
from utils_seq import *
from models_seq_OG import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # Model parameters (must match training parameters)
    parser.add_argument('--task_type', type=str, default='Regression')
    parser.add_argument('--src_vocab_size', type=int, default=21)
    parser.add_argument('--src_len', type=int, default=10)
    parser.add_argument('--model', type=str, default='Transformer')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max', type=float, default=3.0)
    parser.add_argument('--min', type=float, default=-1.0)
    parser.add_argument('--batch_size', type=int, default=50)  # Reduced default batch size
    
    # File paths
    parser.add_argument('--model_path', type=str, default='results_transformer/best_model.pt')
    parser.add_argument('--input_file', type=str, default='unknown_seq.csv')
    parser.add_argument('--output_file', type=str, default='predict_unknown_seq.csv')
    
    args = parser.parse_args()
    
    # Set Transformer Parameters exactly as in main_seq_OG.py
    if args.model == 'Transformer':
        args.d_model = 512  # Embedding size
        args.d_ff = 2048  # FeedForward dimension
        args.d_k = args.d_v = 64  # dimension of K(=Q), V
        args.n_layers = 6  # number of Encoder and Decoder Layer
        args.n_heads = 8  # number of heads in Multi-Head Attention
    else:
        args.d_model = 512  # Embedding size
    
    return args

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the sequences
    print(f"Loading sequences from {args.input_file}")
    df_unknown = pd.read_csv(args.input_file)
    
    if df_unknown.empty:
        print(f"Error: Input file {args.input_file} is empty!")
        return
    
    print(f"Processing {len(df_unknown)} sequences")
    
    # Process features
    unknown_feat = np.array(df_unknown["Feature"])
    
    # Initialize model
    model = Transformer(args).to(device)
    
    # Load trained weights
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Process in batches
    all_predictions = []
    batch_size = args.batch_size
    
    for i in range(0, len(unknown_feat), batch_size):
        batch_feat = unknown_feat[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(unknown_feat) + batch_size - 1)//batch_size}")
        
        try:
            # Process batch
            batch_enc_inputs = make_data(batch_feat, args.src_len).to(device)
            
            # Make predictions
            with torch.no_grad():
                batch_predictions = model(batch_enc_inputs)
                all_predictions.extend(batch_predictions.cpu().numpy().flatten())
            
            # Clear GPU memory
            del batch_enc_inputs
            del batch_predictions
            torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU OOM in batch {i//batch_size + 1}. Trying with smaller batch size...")
                # Try with half the batch size
                half_batch = batch_size // 2
                if half_batch < 1:
                    print("Error: Batch size too small to process")
                    return
                
                # Process first half
                first_half = batch_feat[:half_batch]
                first_half_inputs = make_data(first_half, args.src_len).to(device)
                with torch.no_grad():
                    first_half_preds = model(first_half_inputs)
                    all_predictions.extend(first_half_preds.cpu().numpy().flatten())
                
                # Process second half
                second_half = batch_feat[half_batch:]
                second_half_inputs = make_data(second_half, args.src_len).to(device)
                with torch.no_grad():
                    second_half_preds = model(second_half_inputs)
                    all_predictions.extend(second_half_preds.cpu().numpy().flatten())
                
                # Clear memory
                del first_half_inputs, second_half_inputs
                del first_half_preds, second_half_preds
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e
    
    # Save results
    results = pd.DataFrame({
        'Feature': df_unknown['Feature'],
        'Prediction': all_predictions
    })
    
    print(f"Saving predictions to {args.output_file}")
    results.to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")

if __name__ == '__main__':
    main() 
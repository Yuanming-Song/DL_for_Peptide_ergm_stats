"""
Create stratified train/valid/test splits for generative model training.

This script restructures the data to ensure that (dEdge, length) combinations
are held out from training, allowing proper evaluation of generalization to
unseen condition combinations.

The stratified split:
1. Groups data by (dEdge_bin, sequence_length) combinations
2. Splits at the group level (not individual sequences)
3. Ensures validation/test sets contain (dEdge, length) combinations not seen in training
"""

import pandas as pd
import numpy as np
import os
import argparse

# Define base paths
PROJECT_ROOT = '/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide'
DATA_DIR_ITER1 = os.path.join(PROJECT_ROOT, 'ML_dEdge', 'data', 'iteration1', 'training', 'Sequential_Peptides_edges')
DATA_DIR_ITER2 = os.path.join(PROJECT_ROOT, 'ML_dEdge', 'data', 'iteration2', 'training', 'Sequential_Peptides_edges')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'ML_dEdge_gen', 'v1+2', 'data', 'stratified')

def parse_args():
    parser = argparse.ArgumentParser(description='Create stratified splits for generative model')
    parser.add_argument('--dEdge_bin_size', type=float, default=0.01,
                        help='Bin size for dEdge values (default: 0.01)')
    parser.add_argument('--train_ratio', type=float, default=0.75,
                        help='Training set ratio (default: 0.75)')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def create_stratified_split(df, dEdge_bin_size=0.01, train_ratio=0.75, valid_ratio=0.15, test_ratio=0.1, seed=42):
    """
    Create stratified split by splitting (dEdge, seq_length) pairs separately for each seq_length.
    
    For each sequence length:
    1. Group sequences by (dEdge_bin, seq_length) pairs
    2. Split these pairs with train_ratio/valid_ratio/test_ratio
    3. Ensures validation/test contain (dEdge, length) combinations not seen in training
    
    Args:
        df: DataFrame with 'Feature' (sequences) and 'Label' (dEdge values)
        dEdge_bin_size: Size of bins for dEdge values
        train_ratio: Ratio for training set (default: 0.75)
        valid_ratio: Ratio for validation set (default: 0.15)
        test_ratio: Ratio for test set (default: 0.1)
        seed: Random seed
    
    Returns:
        tuple: (df_train, df_valid, df_test)
    """
    np.random.seed(seed)
    
    # Convert Label to float (remove quotes if present)
    df['Label'] = df['Label'].astype(str).str.replace('"', '').astype(float)
    
    # Add sequence length
    df['SeqLength'] = df['Feature'].str.len()
    
    # Create dEdge bins
    dEdge_min = df['Label'].min()
    dEdge_max = df['Label'].max()
    df['dEdge_bin'] = (df['Label'] // dEdge_bin_size).astype(int) * dEdge_bin_size
    
    # Create combination groups: (dEdge_bin, SeqLength)
    df['Group'] = df['dEdge_bin'].astype(str) + '_' + df['SeqLength'].astype(str)
    
    print(f"Total sequences: {len(df)}")
    print(f"Unique (dEdge_bin, length) combinations: {df['Group'].nunique()}")
    print(f"dEdge range: [{dEdge_min:.4f}, {dEdge_max:.4f}]")
    print(f"Sequence length range: [{df['SeqLength'].min()}, {df['SeqLength'].max()}]")
    
    # Split separately for each sequence length
    train_groups = set()
    valid_groups = set()
    test_groups = set()
    
    unique_lengths = sorted(df['SeqLength'].unique())
    print(f"\nSplitting by sequence length (75/15/10 for each length):")
    
    for seq_len in unique_lengths:
        # Get all groups for this sequence length
        length_df = df[df['SeqLength'] == seq_len]
        length_groups = length_df['Group'].unique()
        n_groups = len(length_groups)
        
        if n_groups == 0:
            continue
        
        # Shuffle groups for this length
        length_groups_list = list(length_groups)
        np.random.shuffle(length_groups_list)
        
        # Split groups for this length
        train_size = int(train_ratio * n_groups)
        valid_size = int(valid_ratio * n_groups)
        
        length_train = set(length_groups_list[:train_size])
        length_valid = set(length_groups_list[train_size:train_size + valid_size])
        length_test = set(length_groups_list[train_size + valid_size:])
        
        train_groups.update(length_train)
        valid_groups.update(length_valid)
        test_groups.update(length_test)
        
        print(f"  Length {seq_len}: {n_groups} groups -> train:{len(length_train)}, valid:{len(length_valid)}, test:{len(length_test)}")
    
    # Assign sequences to splits based on their group
    df_train = df[df['Group'].isin(train_groups)].copy()
    df_valid = df[df['Group'].isin(valid_groups)].copy()
    df_test = df[df['Group'].isin(test_groups)].copy()
    
    # Remove helper columns
    for df_split in [df_train, df_valid, df_test]:
        df_split.drop(columns=['SeqLength', 'dEdge_bin', 'Group'], inplace=True)
    
    print(f"\nOverall stratified split results:")
    print(f"  Training: {len(df_train)} sequences from {len(train_groups)} groups")
    print(f"  Validation: {len(df_valid)} sequences from {len(valid_groups)} groups")
    print(f"  Test: {len(df_test)} sequences from {len(test_groups)} groups")
    
    return df_train, df_valid, df_test

def main():
    args = parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("Creating stratified splits for generative model training")
    print("="*60)
    
    # Load data from both iterations
    print("\nLoading data from iteration1 (v1)...")
    df_v1 = pd.read_csv(f'{DATA_DIR_ITER1}/ddedge_train_seqs.csv')
    df_v1_valid = pd.read_csv(f'{DATA_DIR_ITER1}/ddedge_valid_seqs.csv')
    df_v1_test = pd.read_csv(f'{DATA_DIR_ITER1}/ddedge_test_seqs.csv')
    
    print("Loading data from iteration2 (v2)...")
    df_v2 = pd.read_csv(f'{DATA_DIR_ITER2}/dedge_train_seqs.csv')
    df_v2_valid = pd.read_csv(f'{DATA_DIR_ITER2}/dedge_valid_seqs.csv')
    df_v2_test = pd.read_csv(f'{DATA_DIR_ITER2}/dedge_test_seqs.csv')
    
    # Combine all data from both iterations
    print("\nCombining all data...")
    df_all = pd.concat([
        df_v1, df_v1_valid, df_v1_test,
        df_v2, df_v2_valid, df_v2_test
    ], ignore_index=True)
    
    print(f"Total sequences from both iterations: {len(df_all)}")
    
    # Create stratified split
    print(f"\nCreating stratified split (dEdge_bin_size={args.dEdge_bin_size})...")
    df_train, df_valid, df_test = create_stratified_split(
        df_all,
        dEdge_bin_size=args.dEdge_bin_size,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Reset indices
    df_train['Index'] = range(len(df_train))
    df_valid['Index'] = range(len(df_valid))
    df_test['Index'] = range(len(df_test))
    
    # Save stratified splits
    print(f"\nSaving stratified splits to {OUTPUT_DIR}...")
    df_train.to_csv(os.path.join(OUTPUT_DIR, 'stratified_train_seqs.csv'), index=False)
    df_valid.to_csv(os.path.join(OUTPUT_DIR, 'stratified_valid_seqs.csv'), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, 'stratified_test_seqs.csv'), index=False)
    
    print("\n" + "="*60)
    print("Stratified splits created successfully!")
    print("="*60)
    print(f"\nFiles saved:")
    print(f"  {OUTPUT_DIR}/stratified_train_seqs.csv")
    print(f"  {OUTPUT_DIR}/stratified_valid_seqs.csv")
    print(f"  {OUTPUT_DIR}/stratified_test_seqs.csv")
    print(f"\nThese files ensure (dEdge, length) combinations are held out from training.")

if __name__ == '__main__':
    main()


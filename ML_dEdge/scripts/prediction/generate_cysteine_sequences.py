"""
Generate cysteine-containing peptide sequences.

This script generates all possible peptide sequences of a specified length that contain
cysteine (C) at specified positions. For each position where C is placed, all other positions
will contain all possible combinations of non-C amino acids.

Usage:
    python generate_cysteine_sequences.py [options]

Options:
    --length INT              Length of peptide sequences (default: 4)
    --include_positions STR   Comma-separated list of positions to include C (0-based, default: all)
    --exclude_positions STR   Comma-separated list of positions to exclude C (0-based, default: none)
    --output_file STR         Output file path (default: Sequential_Peptides_edges/dEdge_predict_seqs_{length}mer.csv)
    --batch_size INT          Number of sequences to write in each batch (default: 1000)

Examples:
    # Generate all possible 4-mer sequences with C at any position
    python generate_cysteine_sequences.py --length 4

    # Generate 5-mer sequences with C only at positions 0 and 2
    python generate_cysteine_sequences.py --length 5 --include_positions 0,2

    # Generate 4-mer sequences with C at any position except position 1
    python generate_cysteine_sequences.py --length 4 --exclude_positions 1

    # Generate 6-mer sequences with C at positions 0 and 3, save to custom file
    python generate_cysteine_sequences.py --length 6 --include_positions 0,3 --output_file custom_sequences.csv
"""

import itertools
import pandas as pd
import argparse
import os

# All amino acids except C
other_aas = ['A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def parse_args():
    parser = argparse.ArgumentParser(description='Generate cysteine-containing peptide sequences')
    parser.add_argument('--length', type=int, default=4,
                      help='Length of peptide sequences (default: 4)')
    parser.add_argument('--include_positions', type=str, default='all',
                      help='Comma-separated list of positions to include C (0-based, default: all)')
    parser.add_argument('--exclude_positions', type=str, default='',
                      help='Comma-separated list of positions to exclude C (0-based, default: none)')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Output file path (default: Sequential_Peptides_edges/dEdge_predict_seqs_{length}mer.csv)')
    parser.add_argument('--batch_size', type=int, default=1000,
                      help='Number of sequences to write in each batch (default: 1000)')
    return parser.parse_args()

def generate_all_sequences_for_position(length, c_pos, batch_size):
    """
    Generate all possible sequences with C at a specific position.
    
    Args:
        length (int): Length of peptide sequences
        c_pos (int): Position to place C (0-based)
        batch_size (int): Number of sequences to write in each batch
    
    Returns:
        None: Writes sequences directly to file
    """
    # Calculate total number of sequences
    total_combinations = len(other_aas) ** (length - 1)
    print(f"Generating {total_combinations} sequences with C at position {c_pos}...")
    
    # Create output file if it doesn't exist
    output_file = f'Sequential_Peptides_edges/dEdge_predict_seqs_{length}mer_pos{c_pos}.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Generate all possible combinations for non-C positions
    non_c_positions = list(range(length))
    non_c_positions.remove(c_pos)
    
    # Write sequences in batches
    with open(output_file, 'w') as f:
        f.write('Feature\n')  # Write header
        
        # Generate all possible combinations
        sequences_written = 0
        for combo in itertools.product(other_aas, repeat=length-1):
            # Create sequence with C at specified position
            seq_list = list(combo)
            seq_list.insert(c_pos, 'C')
            sequence = ''.join(seq_list)
            
            # Write sequence
            f.write(f'{sequence}\n')
            
            # Update progress
            sequences_written += 1
            if sequences_written % 10000 == 0:
                print(f"Generated {sequences_written}/{total_combinations} sequences...")
    
    print(f"Generated {total_combinations} sequences for position {c_pos}")
    print(f"Saved to {output_file}")

def main():
    args = parse_args()
    
    # Validate length
    if args.length < 2:
        raise ValueError("Peptide length must be at least 2")
    
    # Validate positions
    if args.include_positions != 'all':
        positions = [int(pos) for pos in args.include_positions.split(',')]
        if any(pos < 0 or pos >= args.length for pos in positions):
            raise ValueError(f"Positions must be between 0 and {args.length-1}")
    
    if args.exclude_positions:
        positions = [int(pos) for pos in args.exclude_positions.split(',')]
        if any(pos < 0 or pos >= args.length for pos in positions):
            raise ValueError(f"Positions must be between 0 and {args.length-1}")
    
    # Convert position strings to lists
    if args.include_positions == 'all':
        include_positions = list(range(args.length))
    else:
        include_positions = [int(pos) for pos in args.include_positions.split(',')]
    
    exclude_positions = [int(pos) for pos in args.exclude_positions.split(',')] if args.exclude_positions else []
    
    # Filter out excluded positions
    valid_positions = [pos for pos in include_positions if pos not in exclude_positions]
    
    if not valid_positions:
        raise ValueError("No valid positions left after filtering!")
    
    # Generate sequences for each position
    for c_pos in valid_positions:
        generate_all_sequences_for_position(args.length, c_pos, args.batch_size)

if __name__ == '__main__':
    main() 
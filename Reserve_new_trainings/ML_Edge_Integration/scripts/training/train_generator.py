import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import sys
import os
import yaml
import argparse
from typing import Dict, List, Tuple

sys.path.append('/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/')
from utils_seq import make_data, src_vocab
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.generative_model import EdgeConditionedGenerator

class EdgePropertyDataset(Dataset):
    """Dataset for training the generative model."""
    def __init__(
        self,
        sequences: List[str],
        monomer_props: torch.Tensor,
        dimer_props: torch.Tensor,
        max_len: int = 10
    ):
        self.sequences = sequences
        self.monomer_props = monomer_props
        self.dimer_props = dimer_props
        self.max_len = max_len
        
        # Convert sequences to indices
        self.sequence_indices = make_data(sequences, max_len)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return {
            'sequence': self.sequence_indices[idx],
            'monomer_props': self.monomer_props[idx],
            'dimer_props': self.dimer_props[idx]
        }

def load_data(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare training data."""
    # Load sequence data
    train_data = pd.read_csv(config['data']['train_path'])
    valid_data = pd.read_csv(config['data']['valid_path'])
    test_data = pd.read_csv(config['data']['test_path'])
    
    # Create datasets
    train_dataset = EdgePropertyDataset(
        sequences=train_data['sequence'].tolist(),
        monomer_props=torch.tensor(train_data[config['data']['monomer_cols']].values).float(),
        dimer_props=torch.tensor(train_data[config['data']['dimer_cols']].values).float(),
        max_len=config['model']['max_len']
    )
    
    valid_dataset = EdgePropertyDataset(
        sequences=valid_data['sequence'].tolist(),
        monomer_props=torch.tensor(valid_data[config['data']['monomer_cols']].values).float(),
        dimer_props=torch.tensor(valid_data[config['data']['dimer_cols']].values).float(),
        max_len=config['model']['max_len']
    )
    
    test_dataset = EdgePropertyDataset(
        sequences=test_data['sequence'].tolist(),
        monomer_props=torch.tensor(test_data[config['data']['monomer_cols']].values).float(),
        dimer_props=torch.tensor(test_data[config['data']['dimer_cols']].values).float(),
        max_len=config['model']['max_len']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    return train_loader, valid_loader, test_loader

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    teacher_forcing_ratio: float = 0.5
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {'total_loss': 0., 'seq_loss': 0., 'mono_loss': 0., 'dimer_loss': 0.}
    n_batches = len(train_loader)
    
    for batch in train_loader:
        # Move to device
        sequences = batch['sequence'].to(device)
        monomer_props = batch['monomer_props'].to(device)
        dimer_props = batch['dimer_props'].to(device)
        
        # Forward pass
        use_teacher_forcing = np.random.random() < teacher_forcing_ratio
        output_probs, encoded_props = model(
            monomer_props,
            dimer_props,
            sequences,
            teacher_forcing=use_teacher_forcing
        )
        
        # Compute loss
        losses = model.compute_loss(
            output_probs=output_probs,
            target_sequences=sequences,
            monomer_pred=None,  # Add property predictions if available
            dimer_pred=None,
            monomer_target=monomer_props,
            dimer_target=dimer_props
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        # Accumulate losses
        for k, v in losses.items():
            total_losses[k] += v.item()
    
    # Average losses
    return {k: v/n_batches for k, v in total_losses.items()}

def validate(
    model: nn.Module,
    valid_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_losses = {'total_loss': 0., 'seq_loss': 0., 'mono_loss': 0., 'dimer_loss': 0.}
    n_batches = len(valid_loader)
    
    with torch.no_grad():
        for batch in valid_loader:
            sequences = batch['sequence'].to(device)
            monomer_props = batch['monomer_props'].to(device)
            dimer_props = batch['dimer_props'].to(device)
            
            output_probs, encoded_props = model(
                monomer_props,
                dimer_props,
                sequences,
                teacher_forcing=True
            )
            
            losses = model.compute_loss(
                output_probs=output_probs,
                target_sequences=sequences,
                monomer_pred=None,
                dimer_pred=None,
                monomer_target=monomer_props,
                dimer_target=dimer_props
            )
            
            for k, v in losses.items():
                total_losses[k] += v.item()
    
    return {k: v/n_batches for k, v in total_losses.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, valid_loader, test_loader = load_data(config)
    
    # Initialize model
    model = EdgeConditionedGenerator(
        monomer_prop_dim=len(config['data']['monomer_cols']),
        dimer_prop_dim=len(config['data']['dimer_cols']),
        **config['model']
    ).to(device)
    
    # Setup training
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Training loop
    best_valid_loss = float('inf')
    patience = config['training']['patience']
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        # Train
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config['training']['teacher_forcing_ratio']
        )
        
        # Validate
        valid_losses = validate(model, valid_loader, device)
        
        # Print progress
        print(f'Epoch {epoch+1}:')
        print(f'Train losses: {train_losses}')
        print(f'Valid losses: {valid_losses}')
        
        # Save best model
        if valid_losses['total_loss'] < best_valid_loss:
            best_valid_loss = valid_losses['total_loss']
            patience_counter = 0
            torch.save(model.state_dict(), config['training']['model_save_path'])
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    print('Training completed!')

if __name__ == '__main__':
    main() 
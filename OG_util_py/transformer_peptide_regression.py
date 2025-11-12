import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd
from pathlib import Path
import argparse
import random

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerRegressor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Embedding(args.src_vocab_size, args.d_model)
        self.pos_encoder = PositionalEncoding(args.d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.n_heads,
            dim_feedforward=args.d_ff,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, args.n_layers)
        self.output_layer = nn.Linear(args.d_model, 1)
            
    def forward(self, src):
        # src shape: [batch_size, seq_len]
        src = self.embedding(src) * np.sqrt(args.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Global average pooling
        return self.output_layer(output)

class PeptideDataset(Data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer for Peptide Property Regression')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--src_vocab_size', type=int, default=21)  # 20 amino acids + padding
    parser.add_argument('--src_len', type=int, default=10)
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def process_sequence(sequence, args):
    # Convert amino acid sequence to indices
    aa_to_idx = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    aa_to_idx['Empty'] = len(aa_to_idx)  # Add padding token
    
    indices = []
    for aa in sequence:
        indices.append(aa_to_idx.get(aa, aa_to_idx['Empty']))
    
    # Pad sequence if necessary
    if len(indices) < args.src_len:
        indices.extend([aa_to_idx['Empty']] * (args.src_len - len(indices)))
    return torch.tensor(indices[:args.src_len])

def load_data(data_dir='Sequential_Peptides_edges'):
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / 'train_seqs.csv')
    valid_df = pd.read_csv(data_dir / 'valid_seqs.csv')
    test_df = pd.read_csv(data_dir / 'test_seqs.csv')
    return train_df, valid_df, test_df

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    
    # Load and process data
    train_df, valid_df, test_df = load_data()
    
    # Process features and labels
    train_features = torch.stack([process_sequence(seq, args) for seq in train_df['Feature']])
    valid_features = torch.stack([process_sequence(seq, args) for seq in valid_df['Feature']]).to(device)
    test_features = torch.stack([process_sequence(seq, args) for seq in test_df['Feature']]).to(device)
    
    train_labels = torch.tensor(train_df['Label'].values).float().unsqueeze(1).to(device)
    valid_labels = torch.tensor(valid_df['Label'].values).float().unsqueeze(1).to(device)
    test_labels = torch.tensor(test_df['Label'].values).float().unsqueeze(1).to(device)
    
    # Create data loader
    train_dataset = PeptideDataset(train_features, train_labels)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model, optimizer, and loss function
    model = TransformerRegressor(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training loop
    best_valid_loss = float('inf')
    best_epoch = 0
    model_save_path = Path('saved_models')
    model_save_path.mkdir(exist_ok=True)
    
    print(f"Training on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            valid_outputs = model(valid_features)
            valid_loss = criterion(valid_outputs, valid_labels).item()
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                }, model_save_path / 'best_model.pt')
                
                print(f'Epoch {epoch+1}: New best model saved!')
                print(f'Validation MSE: {valid_loss:.6f}')
    
    # Load best model and evaluate on test set
    checkpoint = torch.load(model_save_path / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        test_outputs = model(test_features)
        test_loss = criterion(test_outputs, test_labels)
        
        # Calculate additional metrics
        test_mae = nn.L1Loss()(test_outputs, test_labels)
        test_predictions = test_outputs.cpu().numpy()
        test_true = test_labels.cpu().numpy()
        
        # Save results
        results_dir = Path('results_transformer')
        results_dir.mkdir(exist_ok=True)
        
        results = pd.DataFrame({
            'Feature': test_df['Feature'],
            'Prediction': test_predictions.flatten(),
            'True_Value': test_true.flatten(),
            'Absolute_Error': np.abs(test_predictions.flatten() - test_true.flatten())
        })
        
        results.to_csv(results_dir / 'test_results.csv', index=False)
        
        print('\nTest Results:')
        print(f'Best model from epoch {best_epoch + 1}')
        print(f'Test MSE: {test_loss:.6f}')
        print(f'Test MAE: {test_mae:.6f}')

if __name__ == '__main__':
    main() 
import torch
import torch.nn as nn
import math
import sys
sys.path.append('/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/')
from utils_seq import src_vocab

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.prop_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prop_encoding, tgt_mask=None):
        # Self attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Property attention
        prop_out, _ = self.prop_attn(x, prop_encoding, prop_encoding)
        x = self.norm2(x + self.dropout(prop_out))
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x

class SequenceDecoder(nn.Module):
    """
    Autoregressive decoder that generates sequences conditioned on property encoding.
    """
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(len(src_vocab), d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, len(src_vocab))
        self.dropout = nn.Dropout(dropout)
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(
        self,
        tgt_seq: torch.Tensor,  # [batch_size, seq_len]
        prop_encoding: torch.Tensor,  # [batch_size, d_model]
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate sequence probabilities autoregressively.
        
        Args:
            tgt_seq: Input sequence indices
            prop_encoding: Encoded property conditions
            tgt_mask: Optional attention mask for autoregressive generation
            
        Returns:
            output_probs: [batch_size, seq_len, vocab_size]
        """
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq.size(1)).to(tgt_seq.device)
        
        # Embed and add positional encoding
        x = self.token_embedding(tgt_seq)
        x = self.positional_encoding(x)
        
        # Expand property encoding to match sequence length
        prop_encoding = prop_encoding.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, prop_encoding, tgt_mask)
        
        # Project to vocabulary
        output_logits = self.output_projection(x)
        output_probs = torch.log_softmax(output_logits, dim=-1)
        
        return output_probs
    
    def generate(
        self,
        prop_encoding: torch.Tensor,
        max_len: int = 10,
        temperature: float = 1.0,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate sequences from scratch given property encoding.
        
        Args:
            prop_encoding: [batch_size, d_model]
            max_len: Maximum sequence length
            temperature: Sampling temperature
            
        Returns:
            sequences: [batch_size, max_len]
        """
        batch_size = prop_encoding.size(0)
        
        # Start with empty sequences
        sequences = torch.zeros(batch_size, 1).long().to(device)
        
        # Generate autoregressively
        for _ in range(max_len - 1):
            # Get predictions
            with torch.no_grad():
                logits = self.forward(sequences, prop_encoding)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample next tokens
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)
                
                # Append to sequences
                sequences = torch.cat([sequences, next_tokens], dim=1)
        
        return sequences 
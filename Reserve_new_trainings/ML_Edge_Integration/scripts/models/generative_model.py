import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import sys
sys.path.append('/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/')
from utils_seq import src_vocab, make_data
from .property_encoder import PropertyEncoder
from .sequence_decoder import SequenceDecoder

class EdgeConditionedGenerator(nn.Module):
    """
    Complete generative model that generates sequences conditioned on desired edge properties.
    """
    def __init__(
        self,
        monomer_prop_dim: int,
        dimer_prop_dim: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Property encoder
        self.property_encoder = PropertyEncoder(
            input_dim=monomer_prop_dim + dimer_prop_dim,
            hidden_dims=[512, 256, 128],
            output_dim=d_model,
            dropout=dropout
        )
        
        # Sequence decoder
        self.sequence_decoder = SequenceDecoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # Save dimensions
        self.monomer_prop_dim = monomer_prop_dim
        self.dimer_prop_dim = dimer_prop_dim
        self.max_len = max_len
        
    def forward(
        self,
        monomer_props: torch.Tensor,
        dimer_props: torch.Tensor,
        target_sequences: torch.Tensor = None,
        teacher_forcing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            monomer_props: [batch_size, monomer_prop_dim]
            dimer_props: [batch_size, dimer_prop_dim]
            target_sequences: Optional [batch_size, seq_len] for teacher forcing
            teacher_forcing: Whether to use teacher forcing during training
            
        Returns:
            output_probs: [batch_size, seq_len, vocab_size]
            encoded_props: [batch_size, d_model]
        """
        # Encode properties
        encoded_props = self.property_encoder(monomer_props, dimer_props)
        
        if teacher_forcing and target_sequences is not None:
            # Use teacher forcing
            output_probs = self.sequence_decoder(target_sequences, encoded_props)
        else:
            # Generate autoregressively
            sequences = self.sequence_decoder.generate(
                encoded_props,
                max_len=self.max_len,
                device=monomer_props.device
            )
            output_probs = self.sequence_decoder(sequences, encoded_props)
        
        return output_probs, encoded_props
    
    def generate_sequences(
        self,
        monomer_props: torch.Tensor,
        dimer_props: torch.Tensor,
        n_sequences: int = 1,
        temperature: float = 1.0,
        device: str = 'cuda'
    ) -> List[str]:
        """
        Generate sequences given target properties.
        
        Args:
            monomer_props: [batch_size, monomer_prop_dim]
            dimer_props: [batch_size, dimer_prop_dim]
            n_sequences: Number of sequences to generate per property pair
            temperature: Sampling temperature
            
        Returns:
            sequences: List of generated sequences
        """
        # Encode properties
        encoded_props = self.property_encoder(monomer_props, dimer_props)
        
        # Expand encodings if generating multiple sequences per property
        if n_sequences > 1:
            encoded_props = encoded_props.repeat_interleave(n_sequences, dim=0)
        
        # Generate sequences
        sequences = self.sequence_decoder.generate(
            encoded_props,
            max_len=self.max_len,
            temperature=temperature,
            device=device
        )
        
        # Convert to amino acid sequences
        rev_vocab = {v: k for k, v in src_vocab.items()}
        sequence_list = []
        
        for seq in sequences:
            amino_acids = [rev_vocab[idx.item()] for idx in seq if idx.item() != 0]  # Skip padding
            sequence_list.append(''.join(amino_acids))
            
        return sequence_list
    
    def compute_loss(
        self,
        output_probs: torch.Tensor,
        target_sequences: torch.Tensor,
        monomer_pred: torch.Tensor = None,
        dimer_pred: torch.Tensor = None,
        monomer_target: torch.Tensor = None,
        dimer_target: torch.Tensor = None,
        lambda_mono: float = 1.0,
        lambda_dimer: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.
        
        Args:
            output_probs: [batch_size, seq_len, vocab_size]
            target_sequences: [batch_size, seq_len]
            monomer_pred/target: Optional property predictions and targets
            dimer_pred/target: Optional property predictions and targets
            
        Returns:
            losses: Dictionary of loss components and total loss
        """
        # Sequence reconstruction loss
        seq_loss = nn.NLLLoss(ignore_index=0)(
            output_probs.view(-1, len(src_vocab)),
            target_sequences.view(-1)
        )
        
        losses = {'seq_loss': seq_loss}
        total_loss = seq_loss
        
        # Property prediction losses if available
        if all(x is not None for x in [monomer_pred, monomer_target]):
            mono_loss = nn.MSELoss()(monomer_pred, monomer_target)
            losses['mono_loss'] = mono_loss
            total_loss += lambda_mono * mono_loss
            
        if all(x is not None for x in [dimer_pred, dimer_target]):
            dimer_loss = nn.MSELoss()(dimer_pred, dimer_target)
            losses['dimer_loss'] = dimer_loss
            total_loss += lambda_dimer * dimer_loss
            
        losses['total_loss'] = total_loss
        return losses 
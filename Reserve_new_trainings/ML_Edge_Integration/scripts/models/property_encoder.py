import torch
import torch.nn as nn

class PropertyEncoder(nn.Module):
    """
    Encodes target edge properties (monomer and dimer) into a latent representation.
    """
    def __init__(
        self,
        input_dim: int,  # Total number of edge properties (monomer + dimer)
        hidden_dims: list = [512, 256, 128],
        output_dim: int = 512,  # Should match decoder's d_model
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final projection to output dimension
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, monomer_props: torch.Tensor, dimer_props: torch.Tensor) -> torch.Tensor:
        """
        Encode properties into latent representation.
        
        Args:
            monomer_props: [batch_size, monomer_prop_dim]
            dimer_props: [batch_size, dimer_prop_dim]
            
        Returns:
            encoded: [batch_size, output_dim]
        """
        # Concatenate properties
        combined_props = torch.cat([monomer_props, dimer_props], dim=-1)
        
        # Encode
        encoded = self.encoder(combined_props)
        
        return encoded 
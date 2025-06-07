import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import yaml
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils_seq import src_vocab, make_data

class EdgePredictor:
    """
    Handles loading and prediction from both monomer and dimer edge prediction models.
    """
    def __init__(
        self,
        monomer_model_path: str,
        dimer_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.monomer_model = self._load_model(monomer_model_path)
        self.dimer_model = self._load_model(dimer_model_path)
        self.src_len = 10  # Maximum sequence length, same as in original training
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load a trained model from path."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model
    
    def predict(
        self,
        sequences: List[str],
        batch_size: int = 32
    ) -> Dict[str, torch.Tensor]:
        """
        Get edge predictions for both monomer and dimer states.
        
        Args:
            sequences: List of peptide sequences
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary containing predictions for both states
        """
        with torch.no_grad():
            monomer_preds = self._batch_predict(self.monomer_model, sequences, batch_size)
            dimer_preds = self._batch_predict(self.dimer_model, sequences, batch_size)
            
        return {
            "monomer": monomer_preds,
            "dimer": dimer_preds
        }
    
    def _batch_predict(
        self,
        model: nn.Module,
        sequences: List[str],
        batch_size: int
    ) -> torch.Tensor:
        """Run prediction in batches."""
        predictions = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            batch_input = self._preprocess_sequences(batch_seqs)
            batch_preds = model(batch_input.to(self.device))
            predictions.append(batch_preds)
        
        return torch.cat(predictions, dim=0)
    
    def _preprocess_sequences(self, sequences: List[str]) -> torch.Tensor:
        """
        Preprocess sequences using the same method as in training.
        Uses utils_seq.make_data() for consistency.
        """
        return make_data(sequences, self.src_len)
    
    def get_edge_properties(
        self,
        sequences: List[str],
        target_properties: Optional[Dict] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Get edge properties and calculate score based on target properties.
        
        Args:
            sequences: List of peptide sequences
            target_properties: Optional dict of target properties for scoring
            
        Returns:
            Tuple of (predictions dict, score)
        """
        predictions = self.predict(sequences)
        
        if target_properties is None:
            score = 0.0
        else:
            score = self._calculate_property_score(predictions, target_properties)
            
        return predictions, score
    
    def _calculate_property_score(
        self,
        predictions: Dict[str, torch.Tensor],
        target_properties: Dict
    ) -> float:
        """
        Calculate how well predictions match target properties.
        
        Args:
            predictions: Dictionary of predictions for both states
            target_properties: Dictionary of target properties
            
        Returns:
            Score indicating match quality (higher is better)
        """
        # TODO: Implement scoring based on your specific requirements
        # This might include:
        # - MSE between predictions and targets
        # - Custom scoring function
        # - Weighted combination of multiple metrics
        raise NotImplementedError("Implement scoring based on your requirements") 
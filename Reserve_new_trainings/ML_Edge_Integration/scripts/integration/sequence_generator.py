import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import yaml
from .edge_predictor import EdgePredictor

class SequenceGenerator:
    """
    Generates peptide sequences optimized for desired edge properties in both monomer and dimer states.
    """
    def __init__(
        self,
        predictor: EdgePredictor,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.predictor = predictor
        self.device = device
        self.config = self._load_config(config_path)
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def generate(
        self,
        n_sequences: int = 10,
        target_properties: Optional[Dict] = None,
        temperature: float = 1.0,
        max_iterations: int = 100
    ) -> List[str]:
        """
        Generate optimized sequences using iterative refinement.
        
        Args:
            n_sequences: Number of sequences to generate
            target_properties: Target properties for optimization
            temperature: Temperature for sampling (higher = more diverse)
            max_iterations: Maximum optimization iterations
            
        Returns:
            List of generated sequences
        """
        # Initialize random sequences
        sequences = self._initialize_sequences(n_sequences)
        
        # Iteratively optimize
        for _ in range(max_iterations):
            # Get current predictions and scores
            predictions, scores = self.predictor.get_edge_properties(
                sequences, target_properties
            )
            
            # Generate candidates and select best
            candidates = self._generate_candidates(sequences, scores, temperature)
            candidate_predictions, candidate_scores = self.predictor.get_edge_properties(
                candidates, target_properties
            )
            
            # Update sequences if better candidates found
            sequences = self._select_best_sequences(
                sequences, candidates,
                scores, candidate_scores,
                n_sequences
            )
            
        return sequences
    
    def _initialize_sequences(self, n_sequences: int) -> List[str]:
        """Initialize random sequences."""
        # TODO: Implement sequence initialization
        # This should:
        # - Generate valid peptide sequences
        # - Include necessary constraints (e.g., one C)
        # - Match your sequence format
        raise NotImplementedError("Implement sequence initialization")
    
    def _generate_candidates(
        self,
        sequences: List[str],
        scores: Union[float, List[float]],
        temperature: float
    ) -> List[str]:
        """
        Generate candidate sequences by mutating current ones.
        
        Args:
            sequences: Current sequences
            scores: Current sequence scores
            temperature: Sampling temperature
            
        Returns:
            List of candidate sequences
        """
        candidates = []
        for seq in sequences:
            # Generate multiple candidates per sequence
            for _ in range(self.config.get('candidates_per_sequence', 5)):
                mutated = self._mutate_sequence(seq, temperature)
                candidates.append(mutated)
        return candidates
    
    def _mutate_sequence(self, sequence: str, temperature: float) -> str:
        """
        Mutate a single sequence.
        
        Args:
            sequence: Input sequence
            temperature: Sampling temperature
            
        Returns:
            Mutated sequence
        """
        # TODO: Implement sequence mutation
        # This should:
        # - Make valid mutations
        # - Maintain constraints
        # - Use temperature for exploration
        raise NotImplementedError("Implement sequence mutation")
    
    def _select_best_sequences(
        self,
        current_sequences: List[str],
        candidate_sequences: List[str],
        current_scores: Union[float, List[float]],
        candidate_scores: Union[float, List[float]],
        n_select: int
    ) -> List[str]:
        """
        Select best sequences from current and candidates.
        
        Args:
            current_sequences: Current sequence list
            candidate_sequences: Candidate sequence list
            current_scores: Current sequence scores
            candidate_scores: Candidate sequence scores
            n_select: Number of sequences to select
            
        Returns:
            List of selected sequences
        """
        # Combine current and candidates
        all_sequences = current_sequences + candidate_sequences
        all_scores = (
            current_scores if isinstance(current_scores, list) else [current_scores] * len(current_sequences)
        ) + (
            candidate_scores if isinstance(candidate_scores, list) else [candidate_scores] * len(candidate_sequences)
        )
        
        # Sort by score and select top n
        sorted_indices = sorted(
            range(len(all_sequences)),
            key=lambda i: all_scores[i],
            reverse=True
        )
        return [all_sequences[i] for i in sorted_indices[:n_select]] 
"""Multi-head output layer for Octuple prediction."""

import torch
import torch.nn as nn
from typing import Dict


class MultiHeadOutput(nn.Module):
    """8 output heads for predicting each Octuple component."""
    
    COMPONENT_NAMES = [
        "time_sig", "tempo", "bar", "position",
        "instrument", "pitch", "duration", "velocity"
    ]
    
    def __init__(
        self,
        d_model: int = 512,
        vocab_sizes: Dict[str, int] = None,
    ):
        super().__init__()
        
        if vocab_sizes is None:
            vocab_sizes = {
                "time_sig": 16,
                "tempo": 64,
                "bar": 512,
                "position": 128,
                "instrument": 128,
                "pitch": 128,
                "duration": 128,
                "velocity": 32,
            }
        
        self.heads = nn.ModuleDict({
            name: nn.Linear(d_model, size)
            for name, size in vocab_sizes.items()
        })
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]
            
        Returns:
            Dict of logits for each component: [batch, seq_len, vocab_size]
        """
        return {
            name: head(hidden_states)
            for name, head in self.heads.items()
        }

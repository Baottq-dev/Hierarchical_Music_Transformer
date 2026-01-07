"""Embedding layers for Octuple tokens."""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional


class OctupleEmbeddingLayer(nn.Module):
    """Embedding layer for Octuple MIDI tokens.
    
    Each token has 8 components:
    (TimeSig, Tempo, Bar, Position, Instrument, Pitch, Duration, Velocity)
    """
    
    COMPONENT_NAMES = [
        "time_sig", "tempo", "bar", "position",
        "instrument", "pitch", "duration", "velocity"
    ]
    
    def __init__(
        self,
        d_model: int = 512,
        vocab_sizes: Optional[Dict[str, int]] = None,
        dropout: float = 0.1,
        fusion_method: str = "concat_project",
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
        
        self.d_model = d_model
        self.vocab_sizes = vocab_sizes
        self.fusion_method = fusion_method
        self.num_components = 8
        
        # Individual embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(size, d_model)
            for name, size in vocab_sizes.items()
        })
        
        # Fusion layer
        if fusion_method == "concat_project":
            self.fusion = nn.Sequential(
                nn.Linear(d_model * self.num_components, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
        elif fusion_method == "gated":
            self.gates = nn.Linear(d_model * self.num_components, self.num_components)
            self.fusion = nn.Linear(d_model * self.num_components, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0, std=0.02)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [batch_size, seq_len, 8] - Octuple token indices
            
        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        component_embs = []
        for i, name in enumerate(self.COMPONENT_NAMES):
            comp_tokens = tokens[:, :, i]
            comp_emb = self.embeddings[name](comp_tokens)
            component_embs.append(comp_emb)
        
        if self.fusion_method == "sum":
            output = sum(component_embs)
        else:
            concat = torch.cat(component_embs, dim=-1)
            
            if self.fusion_method == "gated":
                gates = torch.sigmoid(self.gates(concat))
                gated_embs = [
                    g.unsqueeze(-1) * e 
                    for g, e in zip(gates.unbind(dim=-1), component_embs)
                ]
                concat = torch.cat(gated_embs, dim=-1)
            
            output = self.fusion(concat)
        
        return self.dropout(output)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

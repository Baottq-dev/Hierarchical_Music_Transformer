"""Hybrid Mamba-Transformer model placeholder."""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .components.embeddings import OctupleEmbeddingLayer, PositionalEncoding
from .components.attention import HierarchicalAttention
from .components.output import MultiHeadOutput


class HybridBlock(nn.Module):
    """Hybrid block: N Mamba layers + 1 Attention layer.
    
    Note: Full Mamba integration requires Phase 2 implementation.
    This is a placeholder that uses Attention only.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_mamba: int = 7,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        # TODO: Add Mamba layers in Phase 2
        self.num_mamba = num_mamba
        
        # Attention layer
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = HierarchicalAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_flash=use_flash_attention,
        )
        
        # FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention layer
        x = x + self.attention(self.attn_norm(x), mask=mask)
        
        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        
        return x


class HybridMambaTransformer(nn.Module):
    """Hybrid Mamba-Transformer model.
    
    Note: This is Phase 1 implementation without Mamba blocks.
    Full Mamba integration will be added in Phase 2.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        num_mamba_per_block: int = 7,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        vocab_sizes: Dict[str, int] = None,
    ):
        super().__init__()
        
        # Embedding
        self.embedding = OctupleEmbeddingLayer(
            d_model=d_model,
            vocab_sizes=vocab_sizes,
            dropout=dropout,
        )
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Layers
        self.layers = nn.ModuleList([
            HybridBlock(
                d_model=d_model,
                num_mamba=num_mamba_per_block,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.output = MultiHeadOutput(d_model=d_model, vocab_sizes=vocab_sizes)
    
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: [batch, seq_len, 8] Octuple tokens
            mask: [batch, seq_len] attention mask
            
        Returns:
            Dict of logits for each component
        """
        x = self.embedding(tokens)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        x = self.final_norm(x)
        logits = self.output(x)
        
        return logits

"""Hierarchical attention mechanism."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HierarchicalAttention(nn.Module):
    """Multi-head attention with hierarchical structure awareness."""
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_flash: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash = use_flash
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        hierarchical_info: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] or [batch, 1, seq_len, seq_len]
            hierarchical_info: Optional dict with bar/beat positions
            
        Returns:
            [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use PyTorch 2.0+ scaled_dot_product_attention if available
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Convert mask to attention mask format
            if mask is not None:
                if mask.dim() == 2:
                    attn_mask = mask.unsqueeze(1).unsqueeze(2)
                    attn_mask = attn_mask.expand(-1, self.num_heads, seq_len, -1)
                else:
                    attn_mask = mask
            else:
                attn_mask = None
            
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention
            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        
        return self.out_proj(attn_output)

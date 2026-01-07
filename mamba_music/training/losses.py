"""Custom loss functions for multi-head Octuple prediction."""

import torch
import torch.nn as nn
from typing import Dict


class MultiHeadCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for multi-head Octuple prediction."""
    
    COMPONENT_NAMES = [
        "time_sig", "tempo", "bar", "position",
        "instrument", "pitch", "duration", "velocity"
    ]
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
        label_smoothing: float = 0.0,
        ignore_index: int = 0,
    ):
        super().__init__()
        
        if weights is None:
            weights = {name: 1.0 for name in self.COMPONENT_NAMES}
        
        self.weights = weights
        self.ce_losses = nn.ModuleDict({
            name: nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                ignore_index=ignore_index,
                reduction="mean",
            )
            for name in self.COMPONENT_NAMES
        })
    
    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: Dict of [batch, seq_len, vocab_size] for each component
            targets: [batch, seq_len, 8] target token indices
            mask: [batch, seq_len] attention mask
            
        Returns:
            Dict with 'loss' (total) and individual component losses
        """
        losses = {}
        total_loss = 0.0
        
        for i, name in enumerate(self.COMPONENT_NAMES):
            component_logits = logits[name]
            component_targets = targets[:, :, i]
            
            batch, seq_len, vocab = component_logits.shape
            logits_flat = component_logits.view(-1, vocab)
            targets_flat = component_targets.view(-1)
            
            loss = self.ce_losses[name](logits_flat, targets_flat)
            losses[f"loss_{name}"] = loss
            total_loss = total_loss + self.weights[name] * loss
        
        losses["loss"] = total_loss
        return losses

"""Training module for Mamba Music."""

from .trainer import Trainer
from .losses import MultiHeadCrossEntropyLoss

__all__ = [
    "Trainer",
    "MultiHeadCrossEntropyLoss",
]

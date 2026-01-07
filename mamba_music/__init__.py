"""
Mamba Music - Hybrid Mamba-Transformer for Symbolic Music Generation

A PyTorch library for text-to-music generation using:
- OctupleMIDI tokenization
- Hybrid Mamba-Transformer architecture
- Hierarchical attention mechanisms
"""

__version__ = "0.1.0"

from . import data
from . import models
from . import training
from . import evaluation
from . import generation
from . import utils

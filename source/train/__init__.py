"""
Train Module - Model Training Components
Handles model architecture, training, and generation
"""

from .model import MusicTransformer
from .trainer import ModelTrainer
from .generator import MusicGenerator

__all__ = [
    'MusicTransformer',
    'ModelTrainer',
    'MusicGenerator'
] 
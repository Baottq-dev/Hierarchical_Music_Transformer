"""
Train Module - Train music generation models
"""

from amt.train.model import MusicTransformer
from amt.train.trainer import ModelTrainer
from amt.train.training_loop import TrainingLoop
from amt.train.create_training_data import create_advanced_training_data as create_training_data

__all__ = [
    "MusicTransformer",
    "ModelTrainer",
    "TrainingLoop",
    "create_training_data",
] 
"""
Model module for AMT.
Contains modules for text embedding clustering and model training.
"""

from .clustering import cluster_embeddings, determine_optimal_k

__all__ = [
    'cluster_embeddings',
    'determine_optimal_k'
] 
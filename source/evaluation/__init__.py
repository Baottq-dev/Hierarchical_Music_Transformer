"""
Evaluation module for AMT.
"""

from .metrics import (
    calculate_note_density_ratio,
    calculate_velocity_similarity,
    calculate_note_range_similarity,
    calculate_time_signature_match,
    calculate_tempo_similarity,
    evaluate_generated_music,
    evaluate_batch
)

__all__ = [
    'calculate_note_density_ratio',
    'calculate_velocity_similarity',
    'calculate_note_range_similarity',
    'calculate_time_signature_match',
    'calculate_tempo_similarity',
    'evaluate_generated_music',
    'evaluate_batch'
] 
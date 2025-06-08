"""
Evaluation metrics module for AMT.
Contains functions for evaluating the quality of generated music.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import mido
from ..data_processing.midi_processor import analyze_midi_file

def calculate_note_density_ratio(original: Dict[str, Any], generated: Dict[str, Any]) -> float:
    """
    Calculate the ratio of note densities between original and generated music.
    Args:
        original: Features of original MIDI file
        generated: Features of generated MIDI file
    Returns:
        Ratio of note densities (closer to 1.0 is better)
    """
    if not original["note_density"] or not generated["note_density"]:
        return 0.0
    return min(original["note_density"] / generated["note_density"],
               generated["note_density"] / original["note_density"])

def calculate_velocity_similarity(original: Dict[str, Any], generated: Dict[str, Any]) -> float:
    """
    Calculate the similarity of velocity distributions.
    Args:
        original: Features of original MIDI file
        generated: Features of generated MIDI file
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not original["velocity_mean"] or not generated["velocity_mean"]:
        return 0.0
    
    # Calculate mean and std differences
    mean_diff = abs(original["velocity_mean"] - generated["velocity_mean"]) / 127.0
    std_diff = abs(original["velocity_std"] - generated["velocity_std"]) / 127.0
    
    # Combine differences (lower is better)
    return 1.0 - (mean_diff + std_diff) / 2.0

def calculate_note_range_similarity(original: Dict[str, Any], generated: Dict[str, Any]) -> float:
    """
    Calculate the similarity of note ranges.
    Args:
        original: Features of original MIDI file
        generated: Features of generated MIDI file
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not original["note_range"] or not generated["note_range"]:
        return 0.0
    
    # Calculate range overlaps
    min_overlap = max(original["note_range"]["min"], generated["note_range"]["min"])
    max_overlap = min(original["note_range"]["max"], generated["note_range"]["max"])
    
    if max_overlap < min_overlap:
        return 0.0
    
    overlap_size = max_overlap - min_overlap + 1
    original_size = original["note_range"]["max"] - original["note_range"]["min"] + 1
    generated_size = generated["note_range"]["max"] - generated["note_range"]["min"] + 1
    
    # Calculate Jaccard similarity
    return overlap_size / (original_size + generated_size - overlap_size)

def calculate_time_signature_match(original: Dict[str, Any], generated: Dict[str, Any]) -> float:
    """
    Calculate the match between time signatures.
    Args:
        original: Features of original MIDI file
        generated: Features of generated MIDI file
    Returns:
        Match score (0.0 to 1.0)
    """
    if not original["time_signatures"] or not generated["time_signatures"]:
        return 0.0
    
    # Calculate intersection of time signatures
    common = set(original["time_signatures"]) & set(generated["time_signatures"])
    total = set(original["time_signatures"]) | set(generated["time_signatures"])
    
    return len(common) / len(total) if total else 0.0

def calculate_tempo_similarity(original: Dict[str, Any], generated: Dict[str, Any]) -> float:
    """
    Calculate the similarity of tempos.
    Args:
        original: Features of original MIDI file
        generated: Features of generated MIDI file
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not original["tempo_mean"] or not generated["tempo_mean"]:
        return 0.0
    
    # Calculate tempo ratio (closer to 1.0 is better)
    tempo_ratio = min(original["tempo_mean"] / generated["tempo_mean"],
                     generated["tempo_mean"] / original["tempo_mean"])
    
    return tempo_ratio

def evaluate_generated_music(original_file: str, generated_file: str) -> Dict[str, float]:
    """
    Evaluate the quality of generated music compared to original.
    Args:
        original_file: Path to original MIDI file
        generated_file: Path to generated MIDI file
    Returns:
        Dictionary containing evaluation metrics
    """
    # Analyze both files
    original_features = analyze_midi_file(original_file)
    generated_features = analyze_midi_file(generated_file)
    
    if not original_features or not generated_features:
        return {
            "note_density_ratio": 0.0,
            "velocity_similarity": 0.0,
            "note_range_similarity": 0.0,
            "time_signature_match": 0.0,
            "tempo_similarity": 0.0,
            "overall_score": 0.0
        }
    
    # Calculate individual metrics
    metrics = {
        "note_density_ratio": calculate_note_density_ratio(original_features, generated_features),
        "velocity_similarity": calculate_velocity_similarity(original_features, generated_features),
        "note_range_similarity": calculate_note_range_similarity(original_features, generated_features),
        "time_signature_match": calculate_time_signature_match(original_features, generated_features),
        "tempo_similarity": calculate_tempo_similarity(original_features, generated_features)
    }
    
    # Calculate overall score (weighted average)
    weights = {
        "note_density_ratio": 0.3,
        "velocity_similarity": 0.2,
        "note_range_similarity": 0.2,
        "time_signature_match": 0.15,
        "tempo_similarity": 0.15
    }
    
    metrics["overall_score"] = sum(metrics[k] * weights[k] for k in weights)
    
    return metrics

def evaluate_batch(original_files: List[str], generated_files: List[str]) -> Dict[str, float]:
    """
    Evaluate a batch of generated music files.
    Args:
        original_files: List of paths to original MIDI files
        generated_files: List of paths to generated MIDI files
    Returns:
        Dictionary containing average evaluation metrics
    """
    if len(original_files) != len(generated_files):
        raise ValueError("Number of original and generated files must match")
    
    all_metrics = []
    for orig, gen in zip(original_files, generated_files):
        metrics = evaluate_generated_music(orig, gen)
        all_metrics.append(metrics)
    
    # Calculate averages
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics 
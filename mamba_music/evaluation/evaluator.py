"""
Evaluator for Mamba Music models.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional

from .metrics import FrechetMusicDistance, OverlappedArea
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluator for music generation models."""
    
    def __init__(self):
        self.fmd = FrechetMusicDistance()
        self.oa = OverlappedArea()
        self.results = {}
    
    def evaluate(
        self,
        real_samples: List[np.ndarray],
        generated_samples: List[np.ndarray],
    ) -> Dict[str, float]:
        """Evaluate generated samples against real samples.
        
        Args:
            real_samples: List of real token sequences [seq_len, 8]
            generated_samples: List of generated token sequences
            
        Returns:
            Dictionary of metrics
        """
        results = {}
        
        # Fréchet Music Distance
        try:
            fmd_score = self.fmd.calculate(real_samples, generated_samples)
            results["fmd"] = fmd_score
        except Exception as e:
            logger.warning(f"FMD calculation failed: {e}")
            results["fmd"] = None
        
        # Pitch distribution overlap
        try:
            real_pitches = np.concatenate([s[:, 5] for s in real_samples])
            gen_pitches = np.concatenate([s[:, 5] for s in generated_samples])
            results["pitch_oa"] = self.oa.calculate(real_pitches, gen_pitches)
        except Exception as e:
            logger.warning(f"Pitch OA calculation failed: {e}")
            results["pitch_oa"] = None
        
        # Duration distribution overlap
        try:
            real_durations = np.concatenate([s[:, 6] for s in real_samples])
            gen_durations = np.concatenate([s[:, 6] for s in generated_samples])
            results["duration_oa"] = self.oa.calculate(real_durations, gen_durations)
        except Exception as e:
            logger.warning(f"Duration OA calculation failed: {e}")
            results["duration_oa"] = None
        
        # Velocity distribution overlap
        try:
            real_velocities = np.concatenate([s[:, 7] for s in real_samples])
            gen_velocities = np.concatenate([s[:, 7] for s in generated_samples])
            results["velocity_oa"] = self.oa.calculate(real_velocities, gen_velocities)
        except Exception as e:
            logger.warning(f"Velocity OA calculation failed: {e}")
            results["velocity_oa"] = None
        
        # Statistics
        results["num_real_samples"] = len(real_samples)
        results["num_generated_samples"] = len(generated_samples)
        results["avg_real_length"] = np.mean([len(s) for s in real_samples])
        results["avg_generated_length"] = np.mean([len(s) for s in generated_samples])
        
        self.results = results
        return results
    
    def save_results(self, output_path: str):
        """Save evaluation results to JSON file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to: {output_path}")
    
    def print_summary(self):
        """Print evaluation summary."""
        logger.info("=" * 50)
        logger.info("Evaluation Summary")
        logger.info("=" * 50)
        
        if self.results.get("fmd") is not None:
            logger.info(f"  FMD (lower is better): {self.results['fmd']:.4f}")
        
        if self.results.get("pitch_oa") is not None:
            logger.info(f"  Pitch OA (higher is better): {self.results['pitch_oa']:.4f}")
        
        if self.results.get("duration_oa") is not None:
            logger.info(f"  Duration OA (higher is better): {self.results['duration_oa']:.4f}")
        
        if self.results.get("velocity_oa") is not None:
            logger.info(f"  Velocity OA (higher is better): {self.results['velocity_oa']:.4f}")
        
        logger.info("=" * 50)

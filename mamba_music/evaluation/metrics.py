"""Evaluation metrics for symbolic music generation."""

import numpy as np
from scipy import linalg
from typing import List, Tuple, Optional


class FrechetMusicDistance:
    """Fréchet Music Distance for evaluating generated music."""
    
    def __init__(self, feature_extractor=None):
        self.feature_extractor = feature_extractor
    
    def extract_statistics(
        self,
        samples: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mean and covariance from samples."""
        if self.feature_extractor is not None:
            features = [self.feature_extractor(s) for s in samples]
        else:
            features = [self._extract_simple_features(s) for s in samples]
        
        features = np.array(features)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    def _extract_simple_features(self, tokens: np.ndarray) -> np.ndarray:
        """Extract simple statistics as features."""
        if len(tokens.shape) == 1:
            tokens = tokens.reshape(-1, 8)
        
        pitches = tokens[:, 5]
        durations = tokens[:, 6]
        velocities = tokens[:, 7]
        
        features = []
        
        # Pitch statistics
        features.extend([
            np.mean(pitches),
            np.std(pitches),
            np.min(pitches),
            np.max(pitches),
        ])
        
        # Duration statistics
        features.extend([
            np.mean(durations),
            np.std(durations),
        ])
        
        # Velocity statistics
        features.extend([
            np.mean(velocities),
            np.std(velocities),
        ])
        
        # Pitch intervals
        if len(pitches) > 1:
            intervals = np.diff(pitches)
            features.extend([
                np.mean(np.abs(intervals)),
                np.std(intervals),
            ])
        else:
            features.extend([0, 0])
        
        return np.array(features)
    
    def calculate(
        self,
        real_samples: List[np.ndarray],
        generated_samples: List[np.ndarray],
    ) -> float:
        """Calculate FMD between real and generated samples."""
        mu_real, sigma_real = self.extract_statistics(real_samples)
        mu_gen, sigma_gen = self.extract_statistics(generated_samples)
        
        diff = mu_real - mu_gen
        covmean = linalg.sqrtm(sigma_real @ sigma_gen)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fmd = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
        
        return float(fmd)


class OverlappedArea:
    """Overlapped Area metric for distribution comparison."""
    
    @staticmethod
    def calculate_histogram(
        values: np.ndarray,
        bins: int = 50,
        range: Tuple[float, float] = None,
    ) -> np.ndarray:
        hist, _ = np.histogram(values, bins=bins, range=range, density=True)
        return hist / (hist.sum() + 1e-10)
    
    @staticmethod
    def calculate(
        real_values: np.ndarray,
        gen_values: np.ndarray,
        bins: int = 50,
    ) -> float:
        min_val = min(real_values.min(), gen_values.min())
        max_val = max(real_values.max(), gen_values.max())
        range_ = (min_val, max_val)
        
        real_hist = OverlappedArea.calculate_histogram(real_values, bins, range_)
        gen_hist = OverlappedArea.calculate_histogram(gen_values, bins, range_)
        
        oa = np.sum(np.minimum(real_hist, gen_hist))
        
        return float(oa)

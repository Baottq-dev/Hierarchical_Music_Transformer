#!/usr/bin/env python3
"""
Evaluation script for Mamba Music Transformer.
"""

import os
import sys
import argparse
from pathlib import Path

import torch

from mamba_music.utils.logging import get_logger, setup_logging
from mamba_music.data import OctupleTokenizer
from mamba_music.evaluation import FrechetMusicDistance, OverlappedArea, Evaluator

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="experiments/evaluation", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()
    
    setup_logging("INFO")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # TODO: Implement full evaluation pipeline
    # 1. Load model
    # 2. Generate samples
    # 3. Compute FMD
    # 4. Compute pitch/duration distributions
    # 5. Save results
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

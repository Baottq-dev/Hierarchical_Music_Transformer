#!/usr/bin/env python3
"""
Music generation script for Mamba Music Transformer.
"""

import os
import sys
import argparse
from pathlib import Path

import torch

from mamba_music.utils.logging import get_logger, setup_logging
from mamba_music.data import OctupleTokenizer
from mamba_music.generation import MusicGenerator

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate music with trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="experiments/generated", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()
    
    setup_logging("INFO")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # TODO: Implement full generation pipeline
    # 1. Load model
    # 2. Create tokenizer
    # 3. Generate tokens
    # 4. Convert to MIDI
    # 5. Save MIDI files
    
    logger.info(f"Generated {args.num_samples} samples to {args.output_dir}")


if __name__ == "__main__":
    main()

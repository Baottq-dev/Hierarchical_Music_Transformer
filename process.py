#!/usr/bin/env python3
"""
Processing script for AMT data
Converts paired MIDI and text data into format suitable for training
"""

import os
import json
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from amt.process.data_preparer import DataPreparer
from amt.process.midi_processor import MidiProcessor
from amt.process.text_processor import TextProcessor
from amt.utils.logging import get_logger
from amt.config import get_settings

# Set up logger and settings
logger = get_logger(__name__)
settings = get_settings()

def apply_optimal_settings(args):
    """Apply optimal transfer learning settings to arguments"""
    args.use_pretrained_text_model = True
    args.pretrained_text_model_path = args.pretrained_text_model_path or "roberta-base"
    args.enable_text_fine_tuning = True
    args.use_pretrained_music_model = True
    args.pretrained_music_model_path = args.pretrained_music_model_path or "m-a-p/MERT-v1-95M"
    args.feature_fusion_method = "attention"
    args.use_hierarchical_encoding = True
    args.use_relative_attention = True
    args.max_seq_len = 1024
    
    logger.info("Using optimal transfer learning settings with MERT and RoBERTa-base")
    return args

def detect_kaggle_environment():
    """Detect if running in Kaggle environment and log appropriate information"""
    if os.path.exists("/kaggle/input"):
        logger.info("Kaggle environment detected!")
        
        # Look for MIDI files in standard Kaggle locations
        possible_paths = [
            "/kaggle/input/midi-dataset/midi",
            "/kaggle/input/your-dataset/midi",
            "/kaggle/input/lakh-midi-dataset/midi",
            "/kaggle/input/midi-files/midi",
            "/kaggle/working/data/midi"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                file_count = sum(1 for _ in Path(path).glob('**/*.mid'))
                logger.info(f"Found {file_count} MIDI files in {path}")
                
        return True
    return False

def main(args):
    """Main entry point"""
    start_time = time.time()
    
    print("Starting process.py main function...")
    print(f"Python version: {sys.version}")
    print(f"Arguments: {args}")
    
    # Check for Kaggle environment
    is_kaggle = os.path.exists("/kaggle/input")
    if is_kaggle:
        logger.info("Kaggle environment detected!")
        logger.info("Running in Kaggle compatibility mode")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load paired data
    logger.info(f"Loading paired data from {args.paired_data_file}")
    with open(args.paired_data_file, 'rb') as f:
        paired_data = json.loads(f.read().decode('utf-8'))
    logger.info(f"Loaded {len(paired_data if isinstance(paired_data, list) else paired_data.get('pairs', []))} paired data items")
    
    # Limit number of samples if specified
    if args.max_samples is not None and args.max_samples > 0:
        logger.info(f"Limiting to {args.max_samples} samples for testing")
        paired_data = paired_data[:args.max_samples]
    
    # Apply optimal settings if requested
    if args.optimal_transfer_learning:
        args = apply_optimal_settings(args)
    
    # Initialize MIDI processor
    midi_processor = MidiProcessor(
        max_sequence_length=args.max_seq_len,
        use_pretrained_model=args.use_pretrained_music_model,
        pretrained_model_path=args.pretrained_music_model_path if args.use_pretrained_music_model else None,
        use_hierarchical_encoding=args.use_hierarchical_encoding,
        device=args.device,
        use_mixed_precision=args.use_mixed_precision
    )
    
    # Initialize text processor
    text_processor = TextProcessor(
        pretrained_model_path=args.pretrained_text_model_path if args.use_pretrained_text_model else None,
        enable_fine_tuning=args.enable_text_fine_tuning,
        use_gpu=(args.device == "cuda")
    )
    
    # Initialize data preparer
    data_preparer = DataPreparer(
        midi_processor=midi_processor,
        text_processor=text_processor,
        feature_fusion_method=args.feature_fusion_method,
        output_dir=args.output_dir,
        is_kaggle=is_kaggle
    )
    
    # Process paired data
    processed_data = data_preparer.process_paired_data(paired_data, batch_size=args.batch_size)
    
    # Save processed data
    output_file = os.path.join(args.output_dir, f"{args.dataset_name}.json")
    with open(output_file, 'w') as f:
        json.dump(processed_data, f)
    
    logger.info(f"Processed {len(processed_data)} items")
    logger.info(f"Saved processed data to {output_file}")
    logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process paired MIDI and text data")
    
    # Input/output arguments
    parser.add_argument("--paired-data-file", type=str, required=True, 
                        help="Path to paired data JSON file")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--dataset-name", type=str, default="processed_dataset",
                        help="Name of the processed dataset")
    
    # Model arguments
    parser.add_argument("--use-pretrained-text-model", action="store_true",
                        help="Use pretrained text model")
    parser.add_argument("--pretrained-text-model-path", type=str, default=None,
                        help="Path to pretrained text model")
    parser.add_argument("--enable-text-fine-tuning", action="store_true",
                        help="Enable fine-tuning of text model")
    parser.add_argument("--use-pretrained-music-model", action="store_true",
                        help="Use pretrained music model")
    parser.add_argument("--pretrained-music-model-path", type=str, default=None,
                        help="Path to pretrained music model")
    
    # Processing arguments
    parser.add_argument("--feature-fusion-method", type=str, default="attention",
                        choices=["none", "concat", "attention", "gated"],
                        help="Method to fuse text and music features")
    parser.add_argument("--use-hierarchical-encoding", action="store_true",
                        help="Use hierarchical encoding for MIDI")
    parser.add_argument("--use-relative-attention", action="store_true",
                        help="Use relative attention for MIDI")
    parser.add_argument("--max-seq-len", type=int, default=1024,
                        help="Maximum sequence length for MIDI")
    
    # Optimization arguments
    parser.add_argument("--optimal-transfer-learning", action="store_true",
                        help="Use optimal transfer learning settings")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for processing (cpu, cuda)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for processing paired data")
    parser.add_argument("--use-mixed-precision", action="store_true",
                        help="Use mixed precision (FP16) for faster processing on GPU")
    
    args = parser.parse_args()
    main(args) 
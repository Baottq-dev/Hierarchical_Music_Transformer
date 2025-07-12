#!/usr/bin/env python3
"""
Processing script for AMT data
Converts paired MIDI and text data into format suitable for training
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from amt.process.data_preparer import DataPreparer
from amt.process.midi_processor import MIDIProcessor
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
    args.pretrained_music_model_path = args.pretrained_music_model_path or "sander-wood/midi-bert"
    args.feature_fusion_method = "attention"
    args.use_hierarchical_encoding = True
    args.use_relative_attention = True
    args.max_seq_len = 1024
    
    logger.info("Using optimal transfer learning settings with MIDI-BERT and RoBERTa-base")
    return args

def main():
    """Main entry point for the processing script"""
    parser = argparse.ArgumentParser(
        description="Process paired MIDI and text data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output arguments
    parser.add_argument("--paired-data-file", type=str, help="Path to JSON file with paired MIDI and text data")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--dataset-name", type=str, default="music_dataset", help="Name of the dataset")
    
    # Processing arguments
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--midi-resolution", type=int, default=480, help="MIDI resolution (ticks per quarter note)")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable data augmentation")
    parser.add_argument("--no-hierarchical-encoding", dest="use_hierarchical_encoding", action="store_false", 
                      help="Disable hierarchical token encoding")
    parser.add_argument("--no-relative-attention", dest="use_relative_attention", action="store_false",
                      help="Disable relative position attention")
    
    # Transfer learning options
    transfer_group = parser.add_argument_group("Transfer learning options")
    transfer_group.add_argument("--optimal-transfer-learning", action="store_true",
                              help="Enable optimal transfer learning settings across all steps")
    transfer_group.add_argument("--use-optimal-models", action="store_true",
                              help="Use optimal models (MIDI-BERT and RoBERTa) without other optimal settings")
    transfer_group.add_argument("--use-pretrained-text-model", action="store_true",
                              help="Use a pretrained text model for fine-tuning")
    transfer_group.add_argument("--pretrained-text-model-path", type=str,
                              help="Path to a pretrained text model for fine-tuning")
    transfer_group.add_argument("--enable-text-fine-tuning", action="store_true",
                              help="Enable fine-tuning of the text model")
    transfer_group.add_argument("--use-pretrained-music-model", action="store_true",
                              help="Use a pretrained music model")
    transfer_group.add_argument("--pretrained-music-model-path", type=str,
                              help="Path to a pretrained music model")
    transfer_group.add_argument("--feature-fusion-method", type=str, choices=["concat", "attention", "gated"],
                              default="concat", help="Method for fusing features")
    
    args = parser.parse_args()
    
    if args.paired_data_file is None:
        parser.error("--paired-data-file is required")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If optimal transfer learning is enabled, set optimal settings
    if args.optimal_transfer_learning:
        args = apply_optimal_settings(args)
    
    # If only optimal models are requested
    if args.use_optimal_models:
        args.use_pretrained_text_model = True
        args.pretrained_text_model_path = args.pretrained_text_model_path or "roberta-base"
        args.use_pretrained_music_model = True
        args.pretrained_music_model_path = args.pretrained_music_model_path or "sander-wood/midi-bert"
        logger.info("Using optimal models: MIDI-BERT and RoBERTa-base")
    
    # Load paired data
    logger.info(f"Loading paired data from {args.paired_data_file}")
    with open(args.paired_data_file, "r") as f:
        paired_data = json.load(f)
    
    # Initialize processors
    midi_processor = MIDIProcessor(
        resolution=args.midi_resolution,
        max_length=args.max_seq_len,
        use_hierarchical_encoding=args.use_hierarchical_encoding,
        use_pretrained_model=args.use_pretrained_music_model,
        pretrained_model_path=args.pretrained_music_model_path
    )
    
    text_processor = TextProcessor(
        max_length=args.max_seq_len,
        use_pretrained_model=args.use_pretrained_text_model,
        pretrained_model_path=args.pretrained_text_model_path,
        enable_fine_tuning=args.enable_text_fine_tuning
    )
    
    data_preparer = DataPreparer(
        midi_processor=midi_processor,
        text_processor=text_processor,
        max_sequence_length=args.max_seq_len,
        vocab_size=args.vocab_size,
        feature_fusion_method=args.feature_fusion_method
    )
    
    # Process data
    logger.info("Processing paired data")
    processed_data = data_preparer.prepare_data(paired_data)
    
    # Save processed data
    output_file = os.path.join(args.output_dir, f"{args.dataset_name}_processed.json")
    logger.info(f"Saving processed data to {output_file}")
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    # Save transfer learning configuration
    transfer_config = {
        "optimal_transfer_learning": args.optimal_transfer_learning,
        "use_optimal_models": args.use_optimal_models,
        "use_pretrained_text_model": args.use_pretrained_text_model,
        "pretrained_text_model_path": args.pretrained_text_model_path,
        "enable_text_fine_tuning": args.enable_text_fine_tuning,
        "use_pretrained_music_model": args.use_pretrained_music_model,
        "pretrained_music_model_path": args.pretrained_music_model_path,
        "feature_fusion_method": args.feature_fusion_method,
        "use_hierarchical_encoding": args.use_hierarchical_encoding,
        "use_relative_attention": args.use_relative_attention,
        "max_seq_len": args.max_seq_len
    }

    # Lưu cấu hình vào file JSON
    transfer_config_path = os.path.join(args.output_dir, "transfer_config.json")
    with open(transfer_config_path, "w") as f:
        json.dump(transfer_config, f, indent=2)

    logger.info(f"Transfer learning configuration saved to {transfer_config_path}")
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main() 
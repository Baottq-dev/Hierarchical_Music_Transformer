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
import numpy as np
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

# Hàm chuyển đổi numpy array để có thể serialize JSON
def convert_numpy_types(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy_types(i) for i in obj]
    return obj

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
        
        # Automatically disable caching on Kaggle to save disk space
        if not hasattr(args, 'disable_cache') or not args.disable_cache:
            logger.info("Automatically disabling cache on Kaggle to save disk space")
            args.disable_cache = True
            
        # Limit memory usage on Kaggle
        if args.limit_memory_usage:
            logger.info("Limiting memory usage for Kaggle environment")
            import gc
            gc.collect()
            
            # Try to limit TensorFlow memory growth if available
            try:
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Set TensorFlow memory growth")
            except:
                logger.info("Could not configure TensorFlow memory growth")
                
            # Try to limit PyTorch memory usage
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("Cleared PyTorch CUDA cache")
            except:
                pass
            
            # Process in smaller chunks if needed
            if not args.memory_efficient_batch_size and args.batch_size > 8:
                logger.info(f"Reducing batch size from {args.batch_size} to 8 to save memory")
                args.batch_size = 8
    
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
        use_mixed_precision=args.use_mixed_precision,
        use_cache=not args.disable_cache,
        cache_dir=os.path.join(args.output_dir, "midi_cache")
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
        is_kaggle=is_kaggle,
        disable_cache=args.disable_cache
    )
    
    # Process paired data, potentially in chunks to save memory
    processed_data = []
    if args.process_in_chunks:
        logger.info(f"Processing data in chunks of {args.chunk_size} to save memory")
        
        # Get total number of items to process
        total_items = len(paired_data if isinstance(paired_data, list) else paired_data.get('pairs', []))
        
        # Process in chunks
        for chunk_start in range(0, total_items, args.chunk_size):
            chunk_end = min(chunk_start + args.chunk_size, total_items)
            logger.info(f"Processing chunk {chunk_start+1}-{chunk_end} of {total_items}")
            
            # Extract chunk
            chunk_data = paired_data[chunk_start:chunk_end]
            
            # Process chunk
            chunk_processed = data_preparer.process_paired_data(chunk_data, batch_size=args.batch_size)
            processed_data.extend(chunk_processed)
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
            # Try to clear PyTorch cache
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
            
            logger.info(f"Completed chunk {chunk_start+1}-{chunk_end}, processed {len(chunk_processed)} items")
            
            # Save intermediate results
            if args.save_intermediate_chunks:
                chunk_output_file = os.path.join(args.output_dir, f"{args.dataset_name}_chunk_{chunk_start+1}_{chunk_end}.json")
                with open(chunk_output_file, 'w') as f:
                    json.dump(convert_numpy_types(chunk_processed), f)
                logger.info(f"Saved intermediate chunk to {chunk_output_file}")
    else:
        # Process all data at once
        processed_data = data_preparer.process_paired_data(paired_data, batch_size=args.batch_size)
    
    # Save processed data
    output_file = os.path.join(args.output_dir, f"{args.dataset_name}.json")
    with open(output_file, 'w') as f:
        # Sử dụng hàm convert_numpy_types để chuyển đổi numpy types trước khi lưu
        json.dump(convert_numpy_types(processed_data), f)
    
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
    parser.add_argument("--disable-cache", action="store_true",
                        help="Disable caching of processed files to save disk space")
    parser.add_argument("--process-in-chunks", action="store_true",
                        help="Process data in smaller chunks to avoid memory issues")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Size of chunks when processing in chunks")
    parser.add_argument("--save-intermediate-chunks", action="store_true",
                        help="Save intermediate chunks to disk")
    
    # Memory management arguments
    parser.add_argument("--limit-memory-usage", action="store_true",
                      help="Limit memory usage for environments with restricted RAM (like Kaggle)")
    parser.add_argument("--memory-efficient-batch-size", action="store_true",
                      help="Don't automatically reduce batch size even when limiting memory")
    
    args = parser.parse_args()
    main(args) 
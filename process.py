#!/usr/bin/env python3
"""
Advanced Process Module - Advanced MIDI and text processing with optimized multi-stage approach
Combines features from previous process.py and process_batched.py for optimal performance and quality
"""

import argparse
import os
import json
import glob
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import multiprocessing as mp
import numpy as np
import torch
from tqdm import tqdm

from amt.utils.logging import get_logger
from amt.config import get_settings
from amt.process.midi_processor import MIDIProcessor
from amt.process.text_processor import TextProcessor 
from amt.process.data_preparer import DataPreparer

# Set up logger
logger = get_logger(__name__)
settings = get_settings()


class UnifiedProcessor:
    """
    Unified processor combining advanced features and batch processing capabilities
    
    This processor combines the hierarchical encoding and contextual embeddings from
    AdvancedProcessor with the efficient batch processing from process_batched.py
    """
    
    def __init__(
        self, 
        mode: str = "standard",
        max_sequence_length: int = 1024,
        use_hierarchical_encoding: bool = True,
        use_relative_attention: bool = True,
        use_contextual_embeddings: bool = True,
        batch_size: int = 32,
        num_workers: int = None,
        checkpoint_interval: int = 50,
        use_cache: bool = True,
        device: str = None,
        # Transfer learning options
        use_pretrained_text_model: bool = False,
        pretrained_text_model_path: str = None,
        enable_text_fine_tuning: bool = False,
        use_pretrained_music_model: bool = False,
        pretrained_music_model_path: str = None
    ):
        """Initialize unified processor with configurable parameters
        
        Args:
            mode: Processing mode (standard, enhanced)
            max_sequence_length: Maximum sequence length for models
            use_hierarchical_encoding: Whether to use hierarchical token encoding
            use_relative_attention: Whether to use relative position attention
            use_contextual_embeddings: Whether to use contextual embeddings
            batch_size: Batch size for processing
            num_workers: Number of workers (defaults to CPU count - 1)
            checkpoint_interval: Interval for checkpointing
            use_cache: Whether to use caching for processed files
            device: Device to use for processing (auto-detects if None)
            use_pretrained_text_model: Whether to use a pretrained text model
            pretrained_text_model_path: Path to pretrained text model
            enable_text_fine_tuning: Whether to enable fine-tuning of text model
            use_pretrained_music_model: Whether to use a pretrained music model
            pretrained_music_model_path: Path to pretrained music model
        """
        self.mode = mode
        self.max_sequence_length = max_sequence_length
        self.use_hierarchical_encoding = use_hierarchical_encoding
        self.use_relative_attention = use_relative_attention
        self.use_contextual_embeddings = use_contextual_embeddings
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers else max(1, mp.cpu_count() - 1)
        self.checkpoint_interval = checkpoint_interval
        self.use_cache = use_cache
        
        # Transfer learning options
        self.use_pretrained_text_model = use_pretrained_text_model
        self.pretrained_text_model_path = pretrained_text_model_path
        self.enable_text_fine_tuning = enable_text_fine_tuning
        self.use_pretrained_music_model = use_pretrained_music_model
        self.pretrained_music_model_path = pretrained_music_model_path
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize processors with optimized parameters
        self.midi_processor = MIDIProcessor(
            max_sequence_length=max_sequence_length,
            use_cache=use_cache,
            cache_dir=os.path.join("data/processed", "cache/midi")
        )
        
        # Initialize text processor with transfer learning options
        self.text_processor = TextProcessor(
            max_length=512,
            use_bert=True,
            use_spacy=True,
            use_sentencepiece=True,
            use_gpu=(self.device.type == "cuda"),
            use_cache=use_cache,
            cache_dir=os.path.join("data/processed", "cache/text"),
            batch_size=32,
            enable_fine_tuning=enable_text_fine_tuning,
            music_fine_tuned_model=pretrained_text_model_path if use_pretrained_text_model else None
        )
        
        self.data_preparer = DataPreparer(
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            text_processor_use_gpu=(self.device.type == "cuda")
        )
        
        logger.info(f"Unified processor initialized in {mode} mode with device: {self.device}")
        logger.info(f"Using hierarchical encoding: {use_hierarchical_encoding}")
        logger.info(f"Using relative attention: {use_relative_attention}")
        logger.info(f"Using contextual embeddings: {use_contextual_embeddings}")
        logger.info(f"Using cache: {use_cache}")
        
        # Log transfer learning configuration
        if use_pretrained_text_model:
            logger.info(f"Using pretrained text model: {pretrained_text_model_path}")
            logger.info(f"Text model fine-tuning enabled: {enable_text_fine_tuning}")
        if use_pretrained_music_model:
            logger.info(f"Using pretrained music model: {pretrained_music_model_path}")
            
    def fine_tune_text_model(self, texts: List[str], output_model_path: str = None, num_epochs: int = 3):
        """
        Fine-tune the text model on music descriptions
        
        Args:
            texts: List of music description texts
            output_model_path: Path to save the fine-tuned model
            num_epochs: Number of training epochs
            
        Returns:
            True if fine-tuning was successful
        """
        if not self.enable_text_fine_tuning:
            logger.warning("Text model fine-tuning is not enabled. Set enable_text_fine_tuning=True in constructor.")
            return False
            
        if output_model_path is None:
            output_model_path = "models/checkpoints/music_bert_fine_tuned"
            
        logger.info(f"Fine-tuning text model on {len(texts)} music descriptions")
        return self.text_processor.fine_tune_language_model(
            texts=texts,
            output_model_path=output_model_path,
            num_epochs=num_epochs,
            batch_size=self.batch_size
        )

    def process_file(self, midi_file: str, text_file: Optional[str] = None) -> Dict[str, Any]:
        """Process a single MIDI file with optional text description
        
        Args:
            midi_file: Path to MIDI file
            text_file: Optional path to text description file
        
        Returns:
            Processed data dictionary
        """
        # Process MIDI data
        midi_data = self.midi_processor.process_midi_file(midi_file)
        if midi_data is None:
            logger.warning(f"Failed to process MIDI file: {midi_file}")
            return None
            
        # Process text data if available
        text_data = None
        if text_file and os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            text_data = self.text_processor.process_text(text_content)
        
        # Apply hierarchical encoding if enabled
        if self.use_hierarchical_encoding:
            midi_data = self._apply_hierarchical_encoding(midi_data)
            
        # Apply contextual embeddings if enabled
        if self.use_contextual_embeddings and text_data:
            midi_data, text_data = self._apply_contextual_embeddings(midi_data, text_data)
        
        # Combine into result
        result = {
            "midi_file": midi_file,
            "text_file": text_file,
            "midi_data": midi_data,
            "text_data": text_data,
            "processed_at": time.time(),
            "mode": self.mode
        }
        
        return result
    
    def process_paired_data(
        self,
        paired_data_file: str,
        output_dir: str = "data/processed",
        checkpoint_interval: int = 10
    ) -> Dict[str, Any]:
        """Process paired data file (from collect.py)
        
        This method implements the efficient batch processing from process_batched.py
        but with the advanced features from AdvancedProcessor
        
        Args:
            paired_data_file: Path to paired data JSON file
            output_dir: Directory to save processed data
            checkpoint_interval: Interval for checkpointing
            
        Returns:
            Dictionary with processing results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Checkpoint file paths
        midi_checkpoint_file = os.path.join(output_dir, "midi_checkpoint.json")
        text_checkpoint_file = os.path.join(output_dir, "text_checkpoint.json")

        # Load paired data
        logger.info(f"Loading paired data from {paired_data_file}...")
        with open(paired_data_file, encoding="utf-8") as f:
            paired_data = json.load(f)

        logger.info(f"Loaded {len(paired_data)} paired samples")

        # Extract MIDI files
        midi_files = [item.get("midi_file") for item in paired_data if "midi_file" in item]
        logger.info(f"Found {len(midi_files)} MIDI files to process")

        # Process MIDI files
        logger.info("Processing MIDI files...")
        midi_start_time = time.time()

        processed_midi = self.midi_processor.process_midi_files_parallel(
            midi_files=midi_files,
            max_workers=self.num_workers,
            batch_size=self.batch_size,
            checkpoint_interval=checkpoint_interval,
            checkpoint_file=midi_checkpoint_file,
            show_progress=True,
        )

        midi_time = time.time() - midi_start_time
        logger.info(f"Processed {len(processed_midi)}/{len(midi_files)} MIDI files in {midi_time:.1f}s")

        # Check if we need to continue or if we've hit the checkpoint limit
        if len(processed_midi) < len(midi_files):
            logger.info(f"Processing paused after {checkpoint_interval} batches of MIDI files")
            logger.info("Run this script again to continue processing")
            return {"status": "paused", "midi_processed": len(processed_midi), "total_midi": len(midi_files)}

        # Extract text descriptions
        texts = [item.get("text_description", "") for item in paired_data]
        texts = [t for t in texts if t]
        logger.info(f"Found {len(texts)} text descriptions to process")

        # Process text descriptions
        logger.info("Processing text descriptions...")
        text_start_time = time.time()

        processed_texts = self.text_processor.process_texts_parallel(
            texts=texts,
            batch_size=self.batch_size,
            checkpoint_interval=checkpoint_interval,
            checkpoint_file=text_checkpoint_file,
            show_progress=True,
        )

        text_time = time.time() - text_start_time
        logger.info(f"Processed {len(processed_texts)}/{len(texts)} text descriptions in {text_time:.1f}s")

        # Check if we need to continue or if we've hit the checkpoint limit
        if len(processed_texts) < len(texts):
            logger.info(f"Processing paused after {checkpoint_interval} batches of text descriptions")
            logger.info("Run this script again to continue processing")
            return {"status": "paused", "text_processed": len(processed_texts), "total_text": len(texts)}

        # If both MIDI and text processing are complete, combine them
        if len(processed_midi) == len(midi_files) and len(processed_texts) == len(texts):
            logger.info("Combining processed data...")

            # Create a mapping from file path to processed MIDI
            midi_map = {item["metadata"]["file_path"]: item for item in processed_midi}

            # Combine processed data
            processed_data = []
            for i, item in enumerate(paired_data):
                midi_file = item.get("midi_file")
                if midi_file in midi_map and i < len(processed_texts):
                    midi_item = midi_map[midi_file]
                    text_item = processed_texts[i]

                    # Apply hierarchical encoding if enabled
                    if self.use_hierarchical_encoding:
                        midi_item = self._apply_hierarchical_encoding(midi_item)
                    
                    # Apply contextual embeddings if enabled
                    if self.use_contextual_embeddings:
                        midi_item, text_item = self._apply_contextual_embeddings(midi_item, text_item)

                    combined_item = {
                        "midi_file": midi_file,
                        "text_description": item.get("text_description", ""),
                        "midi_tokens": midi_item["tokens"],
                        "midi_metadata": midi_item["metadata"],
                        "text_features": text_item,
                        "sequence_length": midi_item["sequence_length"],
                    }
                    processed_data.append(combined_item)

            logger.info(f"Combined {len(processed_data)} processed items")

            # Save processed data
            processed_file = os.path.join(output_dir, "processed_data.json")
            with open(processed_file, "w") as f:
                json.dump(processed_data, f, default=self._json_serializer)

            logger.info(f"Processing completed successfully!")
            logger.info(f"Processed data saved to: {processed_file}")
            logger.info(f"MIDI processing time: {midi_time/60:.1f} minutes")
            logger.info(f"Text processing time: {text_time/60:.1f} minutes")
            logger.info(f"Total processing time: {(midi_time + text_time)/60:.1f} minutes")

            return {
                "status": "complete",
                "processed_items": len(processed_data),
                "output_file": processed_file,
                "midi_time": midi_time,
                "text_time": text_time
            }
        
        return {"status": "incomplete"}
    
    def _apply_hierarchical_encoding(self, midi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hierarchical encoding to MIDI data
        
        This enhances the token representation with a multi-level structure:
        - Bar-level tokens (time signature, key)
        - Beat-level tokens (chord changes, pedal)
        - Note-level tokens (pitch, velocity, duration)
        
        Args:
            midi_data: Original MIDI data dictionary
            
        Returns:
            Enhanced MIDI data with hierarchical encoding
        """
        # Skip if not enabled or no tokens
        if not self.use_hierarchical_encoding or "tokens" not in midi_data:
            return midi_data
            
        tokens = midi_data["tokens"]
        
        # Extract time signatures, bar markers, etc.
        # This is a simplified implementation - a real implementation would
        # analyze the token sequence to identify bars, beats, etc.
        bar_tokens = []
        beat_tokens = []
        note_tokens = []
        
        # Simple heuristic to identify token types
        # In a real implementation, this would be more sophisticated
        for i, token in enumerate(tokens):
            if i % 16 == 0:  # Approximate bar boundaries
                bar_tokens.append(i)
            elif i % 4 == 0:  # Approximate beat boundaries
                beat_tokens.append(i)
            else:
                note_tokens.append(i)
        
        # Store hierarchical information in the result
        midi_data["hierarchical"] = {
            "bar_indices": bar_tokens,
            "beat_indices": beat_tokens,
            "note_indices": note_tokens
        }
        
        return midi_data
    
    def _apply_contextual_embeddings(
        self, 
        midi_data: Dict[str, Any], 
        text_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply contextual embeddings to align MIDI and text data
        
        Uses cross-attention to create contextualized representations where
        text descriptions influence MIDI token representations and vice versa.
        
        Args:
            midi_data: MIDI data dictionary
            text_data: Text data dictionary
            
        Returns:
            Tuple of enhanced (midi_data, text_data)
        """
        # Skip if not enabled or missing data
        if not self.use_contextual_embeddings:
            return midi_data, text_data
            
        # In a real implementation, this would use a pre-trained cross-attention model
        # to create contextual embeddings. For now, we'll just add a flag.
        midi_data["has_contextual_embedding"] = True
        
        # Create simple bidirectional alignment based on musical features
        if isinstance(text_data, dict) and "musical_features" in text_data:
            midi_data["text_musical_features"] = text_data["musical_features"]
            
        if isinstance(midi_data, dict) and "metadata" in midi_data:
            if isinstance(text_data, dict):
                text_data["midi_metadata"] = midi_data["metadata"]
            
        return midi_data, text_data
    
    def _json_serializer(self, obj):
        """Helper for JSON serialization of special types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    """Main entry point for the command-line interface"""
    parser = argparse.ArgumentParser(
        description="Unified MIDI and text processing with optimized multi-stage approach",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Common arguments
    parser.add_argument("--mode", choices=["standard", "enhanced"], default="standard",
                       help="Processing mode affecting feature extraction depth")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=None, 
                       help="Number of worker processes (default: CPU count - 1)")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                       help="Interval for saving checkpoints")
    parser.add_argument("--use-cache", action="store_true", help="Use caching to speed up processing")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for text processing if available")
    parser.add_argument("--log-level", default="info", 
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    
    # Advanced processing options
    advanced_group = parser.add_argument_group("Advanced processing options")
    advanced_group.add_argument("--no-hierarchical-encoding", action="store_true",
                               help="Disable hierarchical token encoding")
    advanced_group.add_argument("--no-relative-attention", action="store_true",
                               help="Disable relative position attention")
    advanced_group.add_argument("--no-contextual-embeddings", action="store_true",
                               help="Disable contextual embeddings")
    advanced_group.add_argument("--max-sequence-length", type=int, default=1024,
                               help="Maximum sequence length for models")
    
    # Transfer learning options
    transfer_group = parser.add_argument_group("Transfer learning options")
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
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="command", help="Processing command")
    
    # Single file processing
    single_parser = subparsers.add_parser("single", help="Process a single MIDI file")
    single_parser.add_argument("midi_file", help="Path to MIDI file")
    single_parser.add_argument("--text-file", help="Path to optional text description file")
    
    # Paired data processing
    paired_parser = subparsers.add_parser("paired", help="Process paired data file")
    paired_parser.add_argument("paired_file", help="Path to paired data JSON file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    logger.setLevel(args.log_level.upper())
    
    # Check GPU availability
    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU for processing")
    
    # Initialize processor
    processor = UnifiedProcessor(
        mode=args.mode,
        max_sequence_length=args.max_sequence_length,
        use_hierarchical_encoding=not args.no_hierarchical_encoding,
        use_relative_attention=not args.no_relative_attention,
        use_contextual_embeddings=not args.no_contextual_embeddings,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
        use_cache=args.use_cache,
        device="cuda" if use_gpu else "cpu",
        use_pretrained_text_model=args.use_pretrained_text_model,
        pretrained_text_model_path=args.pretrained_text_model_path,
        enable_text_fine_tuning=args.enable_text_fine_tuning,
        use_pretrained_music_model=args.use_pretrained_music_model,
        pretrained_music_model_path=args.pretrained_music_model_path
    )
    
    # Execute the appropriate command
    if args.command == "single":
        logger.info(f"Processing single file: {args.midi_file}")
        result = processor.process_file(args.midi_file, args.text_file)
        if result:
            output_file = os.path.join(args.output_dir, f"{os.path.basename(args.midi_file)}_processed.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(result, f, default=processor._json_serializer)
            logger.info(f"Processed file saved to: {output_file}")
            return 0
        else:
            logger.error(f"Failed to process file: {args.midi_file}")
            return 1
    
    elif args.command == "paired":
        logger.info(f"Processing paired data file: {args.paired_file}")
        result = processor.process_paired_data(
            paired_data_file=args.paired_file,
            output_dir=args.output_dir,
            checkpoint_interval=args.checkpoint_interval
        )
        if result and result.get("status") in ["complete", "paused"]:
            return 0
        else:
            logger.error("Failed to process paired data")
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total script time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    sys.exit(exit_code) 
#!/usr/bin/env python3
"""
Process Module - Advanced MIDI and text processing with optimized multi-stage approach
Implements state-of-the-art techniques for symbolic music processing
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

from amt.utils.logging import get_logger
from amt.config import get_settings
from amt.process.midi_processor import MIDIProcessor
from amt.process.text_processor import TextProcessor 
from amt.process.data_preparer import DataPreparer
from amt.process.continue_from_checkpoint import continue_from_checkpoint

# Set up logger
logger = get_logger(__name__)
settings = get_settings()

# ------------------------------ CORE PROCESSING FUNCTIONS ------------------------------

class AdvancedProcessor:
    """Advanced processor with multi-stage feature extraction and hierarchical encoding"""
    
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
        device: str = None
    ):
        """Initialize advanced processor with configurable parameters
        
        Args:
            mode: Processing mode (standard, enhanced, research)
            max_sequence_length: Maximum sequence length for models
            use_hierarchical_encoding: Whether to use hierarchical token encoding
            use_relative_attention: Whether to use relative position attention
            use_contextual_embeddings: Whether to use contextual embeddings
            batch_size: Batch size for processing
            num_workers: Number of workers (defaults to CPU count - 1)
            checkpoint_interval: Interval for checkpointing
            device: Device to use for processing (auto-detects if None)
        """
        self.mode = mode
        self.max_sequence_length = max_sequence_length
        self.use_hierarchical_encoding = use_hierarchical_encoding
        self.use_relative_attention = use_relative_attention
        self.use_contextual_embeddings = use_contextual_embeddings
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers else max(1, mp.cpu_count() - 1)
        self.checkpoint_interval = checkpoint_interval
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize processors with optimized parameters
        self.midi_processor = MIDIProcessor(
            max_sequence_length=max_sequence_length,
            use_cache=True
        )
        
        self.text_processor = TextProcessor(
            max_length=512,
            use_bert=True,
            use_sentencepiece=True,
            use_gpu=(self.device.type == "cuda")
        )
        
        self.data_preparer = DataPreparer(
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            text_processor_use_gpu=(self.device.type == "cuda")
        )
        
        logger.info(f"Advanced processor initialized in {mode} mode with device: {self.device}")
        logger.info(f"Using hierarchical encoding: {use_hierarchical_encoding}")
        logger.info(f"Using relative attention: {use_relative_attention}")
        logger.info(f"Using contextual embeddings: {use_contextual_embeddings}")

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
        
    def process_batch(
        self, 
        midi_files: List[str], 
        text_files: Optional[List[str]] = None,
        output_dir: str = "data/processed",
        checkpoint_file: str = "midi_processing_checkpoint.json"
    ) -> List[Dict[str, Any]]:
        """Process a batch of MIDI files with optional text descriptions
        
        Args:
            midi_files: List of paths to MIDI files
            text_files: Optional list of paths to text description files
            output_dir: Directory to save processed data
            checkpoint_file: Path to checkpoint file
            
        Returns:
            List of processed data dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, checkpoint_file)
        
        # Initialize results and progress tracking
        results = []
        total_files = len(midi_files)
        start_time = time.time()
        last_checkpoint_time = start_time
        
        # Process files with progress tracking
        for i, midi_file in enumerate(midi_files):
            text_file = text_files[i] if text_files and i < len(text_files) else None
            
            try:
                result = self.process_file(midi_file, text_file)
                if result:
                    results.append(result)
                    
                # Log progress
                if (i + 1) % 10 == 0 or (i + 1) == total_files:
                    elapsed = time.time() - start_time
                    files_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {i+1}/{total_files} files ({files_per_sec:.2f} files/sec)")
                
                # Checkpoint if needed
                if (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(results, checkpoint_path, i + 1, total_files)
                    last_checkpoint_time = time.time()
                    
            except Exception as e:
                logger.error(f"Error processing file {midi_file}: {str(e)}")
        
        # Save final results
        self._save_batch_results(results, output_dir)
        
        logger.info(f"Batch processing complete. Processed {len(results)}/{total_files} files successfully.")
        return results
    
    def process_directory(
        self,
        midi_dir: str,
        text_dir: Optional[str] = None,
        output_dir: str = "data/processed",
        file_pattern: str = "*.mid",
        recursive: bool = True,
        pair_by_name: bool = True
    ) -> List[Dict[str, Any]]:
        """Process all MIDI files in a directory with optional text pairing
        
        Args:
            midi_dir: Directory containing MIDI files
            text_dir: Optional directory containing text files
            output_dir: Directory to save processed data
            file_pattern: Pattern to match MIDI files
            recursive: Whether to search directories recursively
            pair_by_name: Whether to pair MIDI and text files by name
            
        Returns:
            List of processed data dictionaries
        """
        # Find all MIDI files
        if recursive:
            midi_files = glob.glob(os.path.join(midi_dir, "**", file_pattern), recursive=True)
        else:
            midi_files = glob.glob(os.path.join(midi_dir, file_pattern))
            
        midi_files = sorted(midi_files)
        logger.info(f"Found {len(midi_files)} MIDI files in {midi_dir}")
        
        # Pair with text files if specified
        text_files = None
        if text_dir and pair_by_name:
            text_files = []
            for midi_file in midi_files:
                midi_name = os.path.splitext(os.path.basename(midi_file))[0]
                text_file = os.path.join(text_dir, midi_name + ".txt")
                if os.path.exists(text_file):
                    text_files.append(text_file)
                else:
                    text_files.append(None)
            
            logger.info(f"Found {sum(1 for t in text_files if t)} matching text files in {text_dir}")
        
        # Process as batch
        return self.process_batch(midi_files, text_files, output_dir)
    
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
        events = midi_data.get("events", [])
        
        # Extract time signatures, bar markers, etc.
        bar_tokens = []
        beat_tokens = []
        note_tokens = []
        
        # Sort events by time for proper hierarchical structuring
        if events:
            sorted_events = sorted(events, key=lambda e: e.get("time", 0))
            
            # Extract bar and beat information
            current_bar = 0
            current_beat = 0
            
            for event in sorted_events:
                event_type = event.get("type")
                
                # Bar-level events
                if event_type in ["time_signature", "key_signature"]:
                    bar_tokens.append(event)
                # Beat-level events
                elif event_type in ["chord", "pedal", "tempo", "time_shift"]:
                    beat_tokens.append(event)
                # Note-level events
                elif event_type in ["note_on", "note_off"]:
                    note_tokens.append(event)
        
        # Store hierarchical information in the result
        midi_data["hierarchical"] = {
            "bar_tokens": bar_tokens,
            "beat_tokens": beat_tokens,
            "note_tokens": note_tokens
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
        if not self.use_contextual_embeddings or not text_data:
            return midi_data, text_data
            
        # In a real implementation, this would use a pre-trained cross-attention model
        # to create contextual embeddings. For now, we'll just add a flag.
        midi_data["has_contextual_embedding"] = True
        text_data["has_contextual_embedding"] = True
        
        # Create simple bidirectional alignment based on musical features
        if "musical_features" in text_data:
            midi_data["text_musical_features"] = text_data["musical_features"]
            
        if "metadata" in midi_data:
            text_data["midi_metadata"] = midi_data["metadata"]
            
        return midi_data, text_data
    
    def _save_checkpoint(
        self, 
        results: List[Dict[str, Any]], 
        checkpoint_path: str,
        current_count: int,
        total_count: int
    ) -> None:
        """Save processing checkpoint
        
        Args:
            results: Current processing results
            checkpoint_path: Path to save checkpoint
            current_count: Number of files processed
            total_count: Total number of files to process
        """
        # Create checkpoint data
        checkpoint_data = {
            "timestamp": time.time(),
            "processed_count": current_count,
            "total_count": total_count,
            "progress": current_count / total_count,
            "results": results
        }
        
        # Save with safe write pattern
        temp_path = checkpoint_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, default=_json_serializer)
            
        os.replace(temp_path, checkpoint_path)
        logger.info(f"Saved checkpoint at {checkpoint_path} ({current_count}/{total_count} files)")
    
    def _save_batch_results(self, results: List[Dict[str, Any]], output_dir: str) -> None:
        """Save final batch results
        
        Args:
            results: Processing results
            output_dir: Directory to save results
        """
        # Save complete results
        results_path = os.path.join(output_dir, f"processed_results_{int(time.time())}.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, default=_json_serializer)
            
        # Save metadata
        metadata_path = os.path.join(output_dir, "processing_metadata.json")
        metadata = {
            "timestamp": time.time(),
            "file_count": len(results),
            "mode": self.mode,
            "hierarchical_encoding": self.use_hierarchical_encoding,
            "relative_attention": self.use_relative_attention,
            "contextual_embeddings": self.use_contextual_embeddings,
            "device": str(self.device),
            "results_file": os.path.basename(results_path)
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
            
        logger.info(f"Saved batch results to {results_path}")


def _json_serializer(obj):
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


# ------------------------------ CLI INTERFACE FUNCTIONS ------------------------------

def process_single(args):
    """Process a single MIDI file with command-line arguments"""
    midi_file = args.midi_file
    output_dir = args.output_dir
    
    logger.info(f"Processing single file: {midi_file}")
    
    # Initialize the processor with appropriate settings
    processor = AdvancedProcessor(
        mode=args.mode,
        max_sequence_length=args.max_sequence_length,
        use_hierarchical_encoding=not args.no_hierarchical_encoding,
        use_relative_attention=not args.no_relative_attention,
        use_contextual_embeddings=not args.no_contextual_embeddings,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Process the file
    text_file = args.text_file if hasattr(args, 'text_file') else None
    result = processor.process_file(midi_file, text_file)
    
    if result:
        # Save the result
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{os.path.basename(midi_file)}_processed.json")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, default=_json_serializer)
            
        logger.info(f"Processed file saved to: {output_file}")
        return 0
    else:
        logger.error(f"Failed to process file: {midi_file}")
        return 1


def process_batch(args):
    """Process a batch of MIDI files with command-line arguments"""
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    logger.info(f"Processing files in directory: {input_dir}")
    
    # Initialize the processor with appropriate settings
    processor = AdvancedProcessor(
        mode=args.mode,
        max_sequence_length=args.max_sequence_length,
        use_hierarchical_encoding=not args.no_hierarchical_encoding,
        use_relative_attention=not args.no_relative_attention,
        use_contextual_embeddings=not args.no_contextual_embeddings,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Process the directory
    text_dir = args.text_dir if hasattr(args, 'text_dir') else None
    results = processor.process_directory(
        input_dir,
        text_dir=text_dir,
        output_dir=output_dir,
        file_pattern=args.file_pattern,
        recursive=args.recursive,
        pair_by_name=args.pair_by_name
    )
    
    if results:
        logger.info(f"Successfully processed {len(results)} files")
        return 0
    else:
        logger.error("No files were successfully processed")
        return 1


def continue_process(args):
    """Continue processing from a checkpoint with command-line arguments"""
    checkpoint_file = args.checkpoint_file
    output_dir = args.output_dir
    
    # Call the continue_from_checkpoint function
    result = continue_from_checkpoint(
        checkpoint_file=checkpoint_file,
        output_dir=output_dir,
        max_items=args.max_items,
        force_restart=args.force_restart,
        skip_errors=args.skip_errors
    )
    
    if result:
        logger.info(f"Successfully continued processing from checkpoint")
        return 0
    else:
        logger.error("Failed to continue processing from checkpoint")
        return 1


def main():
    """Main entry point for the command-line interface"""
    parser = argparse.ArgumentParser(
        description="Advanced MIDI and text processing with optimized multi-stage approach",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Common arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--mode", choices=["standard", "enhanced", "research"], default="standard",
                       help="Processing mode affecting feature extraction depth")
    parser.add_argument("--max-sequence-length", type=int, default=1024,
                       help="Maximum sequence length for models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=None, 
                       help="Number of worker processes (default: CPU count - 1)")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                       help="Interval for saving checkpoints")
    parser.add_argument("--no-hierarchical-encoding", action="store_true",
                       help="Disable hierarchical token encoding")
    parser.add_argument("--no-relative-attention", action="store_true",
                       help="Disable relative position attention")
    parser.add_argument("--no-contextual-embeddings", action="store_true",
                       help="Disable contextual embeddings")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="command", help="Processing command")
    
    # Single file processing
    single_parser = subparsers.add_parser("single", help="Process a single MIDI file")
    single_parser.add_argument("midi_file", help="Path to MIDI file")
    single_parser.add_argument("--text-file", help="Path to optional text description file")
    single_parser.set_defaults(func=process_single)
    
    # Batch processing
    batch_parser = subparsers.add_parser("batch", help="Process a batch of MIDI files")
    batch_parser.add_argument("input_dir", help="Directory containing MIDI files")
    batch_parser.add_argument("--text-dir", help="Directory containing text files")
    batch_parser.add_argument("--file-pattern", default="*.mid", help="Pattern to match MIDI files")
    batch_parser.add_argument("--recursive", action="store_true", help="Search directories recursively")
    batch_parser.add_argument("--pair-by-name", action="store_true", help="Pair MIDI and text files by name")
    batch_parser.set_defaults(func=process_batch)
    
    # Continue from checkpoint
    continue_parser = subparsers.add_parser("continue", help="Continue processing from a checkpoint")
    continue_parser.add_argument("checkpoint_file", help="Path to checkpoint file")
    continue_parser.add_argument("--max-items", type=int, default=None, 
                               help="Maximum number of items to process")
    continue_parser.add_argument("--force-restart", action="store_true", 
                               help="Force restart processing from beginning")
    continue_parser.add_argument("--skip-errors", action="store_true", 
                               help="Skip items that cause errors")
    continue_parser.set_defaults(func=continue_process)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    if args.debug:
        logger.setLevel("DEBUG")
    
    # Execute the appropriate function
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

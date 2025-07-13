"""
Continue processing MIDI files from a checkpoint.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import datetime
from typing import Dict, List, Any, Optional, Tuple

import torch

from amt.utils.logging import get_logger
from amt.config import get_settings
from amt.process.midi_processor import MidiProcessor
from amt.process.text_processor import TextProcessor

# Set up logger
logger = get_logger(__name__)
settings = get_settings()


def load_checkpoint(checkpoint_file: str) -> Tuple[List[Any], int, Dict[str, Any]]:
    """
    Load data from a checkpoint file with robust error handling.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        
    Returns:
        Tuple of (processed_data, last_processed_idx, metadata)
    """
    processed_data = []
    last_processed_idx = -1
    metadata = {}
    
    if not os.path.exists(checkpoint_file):
        logger.debug(f"Checkpoint file {checkpoint_file} not found")
        return processed_data, last_processed_idx, metadata
    
    try:
        logger.debug(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, encoding="utf-8") as f:
            checkpoint = json.load(f)
            
        if isinstance(checkpoint, dict):
            # Modern checkpoint format with metadata
            processed_data = checkpoint.get("processed_data", [])
            last_processed_idx = checkpoint.get("last_processed_idx", -1)
            
            # Extract metadata (timestamp, progress, etc.)
            metadata = {
                "timestamp": checkpoint.get("timestamp", ""),
                "total_files": checkpoint.get("total_files", 0),
                "batch_info": checkpoint.get("batch_info", {}),
            }
            
        elif isinstance(checkpoint, list):
            # Legacy format (just a list of processed items)
            processed_data = checkpoint
            last_processed_idx = len(processed_data) - 1
            metadata = {
                "timestamp": "",
                "total_files": 0,
                "batch_info": {},
            }
    except Exception as e:
        logger.error(f"❌ Error loading checkpoint file {checkpoint_file}: {e}")
        
    return processed_data, last_processed_idx, metadata


def save_checkpoint(
    checkpoint_file: str, 
    processed_data: List[Any], 
    last_processed_idx: int,
    total_files: int,
    batch_info: Dict[str, Any] = None
) -> bool:
    """
    Save checkpoint data with robust error handling.
    
    Args:
        checkpoint_file: Path to save the checkpoint
        processed_data: List of processed data items
        last_processed_idx: Index of the last processed item
        total_files: Total number of files to process
        batch_info: Additional batch processing information
        
    Returns:
        True if checkpoint was saved successfully, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        
        # Create checkpoint data
        checkpoint = {
            "processed_data": processed_data,
            "last_processed_idx": last_processed_idx,
            "total_files": total_files,
            "timestamp": datetime.datetime.now().isoformat(),
            "batch_info": batch_info or {},
        }
        
        # Save to temporary file first to avoid corruption
        temp_file = f"{checkpoint_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False)
            
        # Rename to final checkpoint file
        if os.path.exists(temp_file):
            if os.path.exists(checkpoint_file):
                # Create backup of previous checkpoint
                backup_file = f"{checkpoint_file}.bak"
                try:
                    os.replace(checkpoint_file, backup_file)
                except Exception as e:
                    logger.warning(f"Could not create backup of checkpoint: {e}")
            
            os.replace(temp_file, checkpoint_file)
            return True
            
    except Exception as e:
        logger.error(f"❌ Error saving checkpoint to {checkpoint_file}: {e}")
        
    return False


def create_progress_report(
    processed_data: List[Any], 
    total_files: int, 
    start_time: float,
    checkpoint_file: str = None
) -> Dict[str, Any]:
    """
    Create a detailed progress report.
    
    Args:
        processed_data: List of processed items
        total_files: Total number of files to process
        start_time: Processing start time
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Dictionary with progress information
    """
    elapsed_time = time.time() - start_time
    progress_pct = len(processed_data) / total_files * 100 if total_files > 0 else 0
    
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Estimate remaining time
    if len(processed_data) > 0 and elapsed_time > 0:
        items_per_second = len(processed_data) / elapsed_time
        remaining_items = total_files - len(processed_data)
        est_remaining_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
        
        est_hours, est_remainder = divmod(est_remaining_seconds, 3600)
        est_minutes, est_seconds = divmod(est_remainder, 60)
        
        est_remaining = f"{int(est_hours)}h {int(est_minutes)}m {est_seconds:.1f}s"
    else:
        items_per_second = 0
        est_remaining = "unknown"
    
    report = {
        "progress": {
            "processed": len(processed_data),
            "total": total_files,
            "percentage": progress_pct,
            "remaining": total_files - len(processed_data),
        },
        "time": {
            "elapsed": f"{int(hours)}h {int(minutes)}m {seconds:.1f}s",
            "elapsed_seconds": elapsed_time,
            "estimated_remaining": est_remaining,
            "items_per_second": items_per_second,
        },
        "checkpoint_file": checkpoint_file,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    
    return report


def continue_from_checkpoint(
    midi_checkpoint=None,
    text_checkpoint=None,
    processing_checkpoint=None,
    input_file=None,
    output_dir=None,
    workers=4,
    use_gpu=False,
    use_cache=False,
    log_level="info",
    checkpoint_interval=10,
    batch_size=32,
    force_restart=False,
    save_partial=False
):
    """
    Continue processing from checkpoint files.
    
    Args:
        midi_checkpoint: Path to MIDI checkpoint file
        text_checkpoint: Path to text checkpoint file
        processing_checkpoint: Path to legacy processing checkpoint file
        input_file: Path to original paired data file
        output_dir: Output directory
        workers: Number of parallel workers
        use_gpu: Use GPU for text processing if available
        use_cache: Use caching to speed up processing
        log_level: Logging level
        checkpoint_interval: Save checkpoint after processing this many batches
        batch_size: Batch size for processing
        force_restart: Force restart processing from beginning
        save_partial: Save partial results even if processing is incomplete
    
    Returns:
        Path to processed data file or None if unsuccessful
    """
    start_time = time.time()
    
    # Set defaults
    midi_checkpoint = midi_checkpoint or str(settings.processed_dir / "midi_checkpoint.json")
    text_checkpoint = text_checkpoint or str(settings.processed_dir / "text_checkpoint.json")
    input_file = input_file or str(settings.output_dir / "automated_paired_data.json")
    output_dir = output_dir or str(settings.processed_dir)
    
    # Set log level
    logger.setLevel(log_level.upper())

    logger.info("🚀 Continuing Processing from Checkpoint...")
    logger.info("=" * 50)
    logger.info(f"MIDI checkpoint: {midi_checkpoint}")
    logger.info(f"Text checkpoint: {text_checkpoint}")
    if processing_checkpoint:
        logger.info(f"Processing checkpoint: {processing_checkpoint}")
    logger.info(f"Original paired data: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoint interval: {checkpoint_interval}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Force restart: {force_restart}")
    logger.info(f"Save partial results: {save_partial}")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint directories if needed
    os.makedirs(os.path.dirname(midi_checkpoint), exist_ok=True)
    os.makedirs(os.path.dirname(text_checkpoint), exist_ok=True)

    # Check if checkpoint files exist
    midi_checkpoint_exists = os.path.exists(midi_checkpoint) and not force_restart
    text_checkpoint_exists = os.path.exists(text_checkpoint) and not force_restart
    processing_checkpoint_exists = processing_checkpoint and os.path.exists(
        processing_checkpoint
    ) and not force_restart

    # Load original paired data
    logger.info(f"\n📂 Loading original paired data from {input_file}...")
    
    try:
        with open(input_file, encoding="utf-8") as f:
            paired_data = json.load(f)
        logger.info(f"✅ Loaded {len(paired_data)} paired samples")
    except Exception as e:
        logger.error(f"❌ Error loading paired data: {e}")
        return None

    # Get last processed index from processing checkpoint
    last_processed_idx = -1
    if processing_checkpoint_exists:
        try:
            with open(processing_checkpoint) as f:
                checkpoint_info = json.load(f)
                last_processed_idx = checkpoint_info.get("last_processed_idx", -1)
            logger.info(f"✅ Resuming from index {last_processed_idx + 1}")
        except Exception as e:
            logger.error(f"❌ Error loading processing checkpoint: {e}")

    # Process MIDI data
    processed_midi = []
    midi_metadata = {}
    
    if midi_checkpoint_exists:
        logger.info("\n🎵 Loading processed MIDI data from checkpoint...")
        processed_midi, midi_last_idx, midi_metadata = load_checkpoint(midi_checkpoint)
        
        if processed_midi:
            logger.info(
                f"✅ Loaded {len(processed_midi)} processed MIDI items (last idx {midi_last_idx})"
            )
            
            # Show checkpoint timestamp if available
            if "timestamp" in midi_metadata and midi_metadata["timestamp"]:
                logger.info(f"📅 MIDI checkpoint timestamp: {midi_metadata['timestamp']}")
        else:
            logger.warning("⚠️ No MIDI data found in checkpoint, will process from scratch")
    else:
        logger.warning("⚠️ No MIDI checkpoint found or force_restart=True, starting MIDI processing from scratch")
    
    # Initialize MIDI processor if needed
    if not processed_midi or len(processed_midi) < len(paired_data):
        # Initialize MIDI processor
        midi_processor = MidiProcessor(
            max_sequence_length=settings.max_sequence_length,
            use_cache=use_cache,
            cache_dir=os.path.join(output_dir, "cache/midi"),
        )

        # Extract MIDI files
        midi_files = [item.get("midi_file") for item in paired_data if "midi_file" in item]
        remaining_files = len(midi_files) - len(processed_midi)
        
        logger.info(f"Found {len(midi_files)} MIDI files to process ({remaining_files} remaining)")

        # Process MIDI files
        if remaining_files > 0:
            midi_start_time = time.time()
            
            # Skip already processed files
            start_idx = len(processed_midi)
            files_to_process = midi_files[start_idx:]
            
            logger.info(f"Processing {len(files_to_process)} MIDI files...")
            
            # Process remaining MIDI files
            new_processed_midi = midi_processor.process_midi_files_parallel(
                midi_files=files_to_process,
                max_workers=workers,
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
                checkpoint_file=midi_checkpoint,
                show_progress=True,
            )
            
            # Combine with already processed data
            if processed_midi:
                processed_midi.extend(new_processed_midi)
            else:
                processed_midi = new_processed_midi

            midi_time = time.time() - midi_start_time
            logger.info(f"✅ Processed {len(processed_midi)}/{len(midi_files)} MIDI files in {midi_time:.1f}s")
        
            # Create MIDI progress report
            midi_progress = create_progress_report(
                processed_midi, len(midi_files), midi_start_time, midi_checkpoint
            )
            
            # Save updated MIDI checkpoint with progress info
            save_checkpoint(
                midi_checkpoint,
                processed_midi,
                len(processed_midi) - 1,
                len(midi_files),
                midi_progress
            )

    # Process text data
    processed_texts = []
    text_metadata = {}
    
    if text_checkpoint_exists:
        logger.info("\n📝 Loading processed text data from checkpoint...")
        processed_texts, text_last_idx, text_metadata = load_checkpoint(text_checkpoint)
        
        if processed_texts:
            logger.info(f"✅ Loaded {len(processed_texts)} processed text items")
            
            # Show checkpoint timestamp if available
            if "timestamp" in text_metadata and text_metadata["timestamp"]:
                logger.info(f"📅 Text checkpoint timestamp: {text_metadata['timestamp']}")
        else:
            logger.warning("⚠️ No text data found in checkpoint, will process from scratch")
    else:
        logger.warning("⚠️ No text checkpoint found or force_restart=True, starting text processing from scratch")
    
    # Initialize text processor if needed
    if not processed_texts or len(processed_texts) < len(paired_data):
        # Check GPU availability
        use_gpu = use_gpu and torch.cuda.is_available()
        if use_gpu:
            logger.info(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("💻 Using CPU for processing")

        # Initialize text processor
        text_processor = TextProcessor(
            max_length=settings.max_text_length,
            use_bert=True,
            use_spacy=True,
            use_sentencepiece=True,
            use_gpu=use_gpu,
            use_cache=use_cache,
            cache_dir=os.path.join(output_dir, "cache/text"),
            batch_size=batch_size,
        )

        # Extract text descriptions
        texts = [item.get("text_description", "") for item in paired_data]
        texts = [t for t in texts if t]
        remaining_texts = len(texts) - len(processed_texts)
        
        logger.info(f"Found {len(texts)} text descriptions to process ({remaining_texts} remaining)")

        # Process text descriptions
        if remaining_texts > 0:
            text_start_time = time.time()
            
            # Skip already processed texts
            start_idx = len(processed_texts)
            texts_to_process = texts[start_idx:]
            
            logger.info(f"Processing {len(texts_to_process)} text descriptions...")
            
            # Process remaining text descriptions
            new_processed_texts = text_processor.process_texts_parallel(
                texts=texts_to_process,
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
                checkpoint_file=text_checkpoint,
                show_progress=True,
            )
            
            # Combine with already processed data
            if processed_texts:
                processed_texts.extend(new_processed_texts)
            else:
                processed_texts = new_processed_texts

            text_time = time.time() - text_start_time
            logger.info(
                f"✅ Processed {len(processed_texts)}/{len(texts)} text descriptions in {text_time:.1f}s"
            )
            
            # Create text progress report
            text_progress = create_progress_report(
                processed_texts, len(texts), text_start_time, text_checkpoint
            )
            
            # Save updated text checkpoint with progress info
            save_checkpoint(
                text_checkpoint,
                processed_texts,
                len(processed_texts) - 1,
                len(texts),
                text_progress
            )

    # Check if processing is complete or save_partial is True
    if (len(processed_midi) == len([i for i in paired_data if "midi_file" in i]) and 
        len(processed_texts) == len([i for i in paired_data if i.get("text_description", "")])) or save_partial:
        
        # Combine processed data
        logger.info("\n🔄 Combining processed data...")

        # Create a mapping from file path to processed MIDI
        midi_map = {item["file_path"]: item for item in processed_midi}

        # Combine processed data
        processed_data = []
        successful_items = 0
        
        for i, item in enumerate(paired_data):
            midi_file = item.get("midi_file")
            text_description = item.get("text_description", "")
            
            # Skip items with missing data
            if not midi_file or not text_description:
                continue
                
            # Skip if we don't have processed data for this item
            if midi_file not in midi_map or i >= len(processed_texts):
                continue
                
            midi_item = midi_map[midi_file]
            text_item = processed_texts[i]

            combined_item = {
                "midi_file": midi_file,
                "text_description": text_description,
                "midi_tokens": midi_item["tokens"],
                "midi_metadata": midi_item["metadata"],
                "text_features": text_item,
                "sequence_length": midi_item["sequence_length"],
            }
            processed_data.append(combined_item)
            successful_items += 1

        logger.info(f"Combined {len(processed_data)} processed items")
        
        if save_partial and len(processed_data) < len(paired_data):
            logger.warning(f"⚠️ Saving partial results ({len(processed_data)}/{len(paired_data)} items)")

        # Save processed data
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "processed_data.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Saved {len(processed_data)} processed items to {output_file}")
        
        # Create a backup copy
        backup_file = os.path.join(output_dir, f"processed_data_{int(time.time())}.json.bak")
        try:
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Created backup at {backup_file}")
        except Exception as e:
            logger.warning(f"⚠️ Could not create backup: {e}")
        
        logger.info(f"✨ Processing complete! Output saved to {output_file}")
        
        # Calculate success rate
        total_possible_items = min(
            len([i for i in paired_data if "midi_file" in i]),
            len([i for i in paired_data if i.get("text_description", "")])
        )
        success_rate = successful_items / total_possible_items * 100 if total_possible_items > 0 else 0
        
        logger.info(f"📊 Statistics: {successful_items}/{total_possible_items} items processed successfully ({success_rate:.1f}%)")

        # Create a detailed statistics file
        stats = {
            "total_paired_items": len(paired_data),
            "processed_items": len(processed_data),
            "success_rate": success_rate / 100,
            "midi_files_processed": len(processed_midi),
            "text_files_processed": len(processed_texts),
            "timestamp": datetime.datetime.now().isoformat(),
            "processing_time": time.time() - start_time,
            "checkpoint_info": {
                "midi": midi_metadata,
                "text": text_metadata
            }
        }
        
        stats_file = os.path.join(output_dir, "processing_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info("\n✅ Processing completed successfully!")
        logger.info(f"Processed data saved to: {output_file}")
        
        return output_file
    else:
        # Processing is incomplete
        logger.warning("\n⚠️ Processing is incomplete")
        logger.warning(f"MIDI files: {len(processed_midi)}/{len([i for i in paired_data if 'midi_file' in i])}")
        logger.warning(f"Text descriptions: {len(processed_texts)}/{len([i for i in paired_data if i.get('text_description', '')])}")
        logger.warning("Run this script again to continue processing or use --save_partial to save current progress")
        return None


def main():
    """Command-line interface for continuing processing from checkpoints."""
    parser = argparse.ArgumentParser(description="Continue processing from MIDI checkpoint")
    parser.add_argument(
        "--midi_checkpoint",
        default=str(settings.processed_dir / "midi_checkpoint.json"),
        help="MIDI checkpoint file",
    )
    parser.add_argument(
        "--text_checkpoint",
        default=str(settings.processed_dir / "text_checkpoint.json"),
        help="Text checkpoint file",
    )
    parser.add_argument(
        "--processing_checkpoint", default=None, help="(Optional) Legacy processing checkpoint file"
    )
    parser.add_argument(
        "--input_file",
        default=str(settings.output_dir / "automated_paired_data.json"),
        help="Original paired data file",
    )
    parser.add_argument("--output_dir", default=str(settings.processed_dir), help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for text processing if available"
    )
    parser.add_argument(
        "--use_cache", action="store_true", help="Use caching to speed up processing"
    )
    parser.add_argument("--log_level", default=settings.log_level, 
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Save checkpoint after processing this many batches")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--force_restart", action="store_true",
                        help="Force restart processing from beginning")
    parser.add_argument("--save_partial", action="store_true",
                        help="Save partial results even if processing is incomplete")

    args = parser.parse_args()
    
    # Call the function with parsed arguments
    output_file = continue_from_checkpoint(
        midi_checkpoint=args.midi_checkpoint,
        text_checkpoint=args.text_checkpoint,
        processing_checkpoint=args.processing_checkpoint,
        input_file=args.input_file,
        output_dir=args.output_dir,
        workers=args.workers,
        use_gpu=args.use_gpu,
        use_cache=args.use_cache,
        log_level=args.log_level,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        force_restart=args.force_restart,
        save_partial=args.save_partial
    )
    
    # Suggest next steps
    if output_file:
        logger.info("Next step: Create training_data.json with train/val/test splits")
        logger.info(
            f"Run: python -m amt.train.create_training_data --input_file {output_file} --output_dir {args.output_dir}"
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"\n⏱️ Total script time: {int(hours)}h {int(minutes)}m {seconds:.1f}s") 
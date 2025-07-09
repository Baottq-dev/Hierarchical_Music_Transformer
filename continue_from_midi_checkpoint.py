#!/usr/bin/env python3
"""
Continue Processing Script - Continues processing from MIDI checkpoint and creates processed_data.json
"""

import argparse
import json
import os
import time

import torch

from source.process import MIDIProcessor, TextProcessor


def main():
    parser = argparse.ArgumentParser(description="Continue processing from MIDI checkpoint")
    parser.add_argument(
        "--midi_checkpoint",
        default="data/processed/midi_checkpoint.json",
        help="MIDI checkpoint file",
    )
    parser.add_argument(
        "--text_checkpoint",
        default="data/processed/text_checkpoint.json",
        help="Text checkpoint file",
    )
    parser.add_argument(
        "--processing_checkpoint", default=None, help="(Optional) Legacy processing checkpoint file"
    )
    parser.add_argument(
        "--input_file",
        default="data/output/automated_paired_data.json",
        help="Original paired data file",
    )
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for text processing if available"
    )
    parser.add_argument(
        "--use_cache", action="store_true", help="Use caching to speed up processing"
    )

    args = parser.parse_args()

    print("🚀 Continuing Processing from Checkpoint...")
    print("=" * 50)
    print(f"MIDI checkpoint: {args.midi_checkpoint}")
    print(f"Text checkpoint: {args.text_checkpoint}")
    if args.processing_checkpoint:
        print(f"Processing checkpoint: {args.processing_checkpoint}")
    print(f"Original paired data: {args.input_file}")
    print(f"Output directory: {args.output_dir}")

    # Check if checkpoint files exist
    midi_checkpoint_exists = os.path.exists(args.midi_checkpoint)
    text_checkpoint_exists = os.path.exists(args.text_checkpoint)
    processing_checkpoint_exists = args.processing_checkpoint and os.path.exists(
        args.processing_checkpoint
    )

    if not midi_checkpoint_exists and not text_checkpoint_exists:
        print("❌ No checkpoint files found. Please run run_process_batched.py first.")
        return

    # Load original paired data
    print(f"\n📂 Loading original paired data from {args.input_file}...")
    with open(args.input_file, encoding="utf-8") as f:
        paired_data = json.load(f)

    print(f"✅ Loaded {len(paired_data)} paired samples")

    # Get last processed index
    last_processed_idx = -1
    if processing_checkpoint_exists:
        with open(args.processing_checkpoint) as f:
            checkpoint_info = json.load(f)
            last_processed_idx = checkpoint_info.get("last_processed_idx", -1)
        print(f"✅ Resuming from index {last_processed_idx + 1}")

    # Process MIDI data
    processed_midi = []
    if midi_checkpoint_exists:
        print("\n🎵 Loading processed MIDI data from checkpoint...")
        with open(args.midi_checkpoint, encoding="utf-8") as f:
            checkpoint = json.load(f)
            if isinstance(checkpoint, dict):
                processed_midi = checkpoint.get("processed_data", [])
                last_processed_idx = checkpoint.get("last_processed_idx", last_processed_idx)
            else:
                # fallback if file contains list directly
                processed_midi = checkpoint
            print(
                f"✅ Loaded {len(processed_midi)} processed MIDI items (last idx {last_processed_idx})"
            )
    else:
        print("⚠️ No MIDI checkpoint found, starting MIDI processing from scratch")
        # Initialize MIDI processor
        midi_processor = MIDIProcessor(
            max_sequence_length=1024,
            use_cache=args.use_cache,
            cache_dir=os.path.join(args.output_dir, "cache/midi"),
        )

        # Extract MIDI files
        midi_files = [item.get("midi_file") for item in paired_data if "midi_file" in item]
        print(f"Found {len(midi_files)} MIDI files to process")

        # Process MIDI files
        midi_start_time = time.time()
        processed_midi = midi_processor.process_midi_files_parallel(
            midi_files=midi_files,
            max_workers=args.workers,
            batch_size=100,
            checkpoint_interval=0,  # Process all files
            checkpoint_file=args.midi_checkpoint,
            show_progress=True,
        )

        midi_time = time.time() - midi_start_time
        print(f"✅ Processed {len(processed_midi)}/{len(midi_files)} MIDI files in {midi_time:.1f}s")

    # Process text data
    processed_texts = []
    if text_checkpoint_exists:
        print("\n📝 Loading processed text data from checkpoint...")
        with open(args.text_checkpoint, encoding="utf-8") as f:
            checkpoint_t = json.load(f)
            if isinstance(checkpoint_t, dict):
                processed_texts = checkpoint_t.get("processed_data", [])
            else:
                processed_texts = checkpoint_t
            print(f"✅ Loaded {len(processed_texts)} processed text items")
    else:
        print("⚠️ No text checkpoint found, starting text processing from scratch")
        # Check GPU availability
        use_gpu = args.use_gpu and torch.cuda.is_available()
        if use_gpu:
            print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Using CPU for processing")

        # Initialize text processor
        text_processor = TextProcessor(
            max_length=512,
            use_bert=True,
            use_spacy=True,
            use_gpu=use_gpu,
            use_cache=args.use_cache,
            cache_dir=os.path.join(args.output_dir, "cache/text"),
            batch_size=32,
        )

        # Extract text descriptions
        texts = [item.get("text_description", "") for item in paired_data]
        texts = [t for t in texts if t]
        print(f"Found {len(texts)} text descriptions to process")

        # Process text descriptions
        text_start_time = time.time()
        processed_texts = text_processor.process_texts_parallel(
            texts=texts,
            batch_size=100,
            checkpoint_interval=0,  # Process all files
            checkpoint_file=args.text_checkpoint,
            show_progress=True,
        )

        text_time = time.time() - text_start_time
        print(
            f"✅ Processed {len(processed_texts)}/{len(texts)} text descriptions in {text_time:.1f}s"
        )

    # Combine processed data
    print("\n🔄 Combining processed data...")

    # Create a mapping from file path to processed MIDI
    midi_map = {item["file_path"]: item for item in processed_midi}

    # Combine processed data
    processed_data = []
    for i, item in enumerate(paired_data):
        midi_file = item.get("midi_file")
        if midi_file in midi_map and i < len(processed_texts):
            midi_item = midi_map[midi_file]
            text_item = processed_texts[i]

            combined_item = {
                "midi_file": midi_file,
                "text_description": item.get("text_description", ""),
                "midi_tokens": midi_item["tokens"],
                "midi_metadata": midi_item["metadata"],
                "text_features": text_item,
                "sequence_length": midi_item["sequence_length"],
            }
            processed_data.append(combined_item)

    print(f"Combined {len(processed_data)} processed items")

    # Save processed data
    processed_file = os.path.join(args.output_dir, "processed_data.json")
    with open(processed_file, "w") as f:
        json.dump(processed_data, f)

    print("\n✅ Processing completed successfully!")
    print(f"Processed data saved to: {processed_file}")
    print("Next step: Create training_data.json with train/val/test splits")
    print(
        f"Run: python create_training_data.py --input_file {processed_file} --output_dir {args.output_dir}"
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n⏱️ Total script time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")

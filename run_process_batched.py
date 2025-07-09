#!/usr/bin/env python3
"""
Batched Processing Script - Process MIDI and text data in batches with checkpointing
"""

import argparse
import json
import os
import time

import torch

from source.process import MIDIProcessor, TextProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Process MIDI and text data in batches with checkpointing"
    )
    parser.add_argument(
        "--input_file",
        default="data/output/automated_paired_data.json",
        help="Input paired data file",
    )
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=100, help="Files per batch")
    parser.add_argument(
        "--checkpoint_interval", type=int, default=10, help="Stop after processing N batches"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for text processing if available"
    )
    parser.add_argument(
        "--use_cache", action="store_true", help="Use caching to speed up processing"
    )

    args = parser.parse_args()

    # Check GPU availability
    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU for processing")

    print("🚀 Starting Batched Processing...")
    print("=" * 50)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Checkpoint interval: {args.checkpoint_interval} batches")
    print(f"Workers: {args.workers}")
    print(f"Using GPU: {use_gpu}")
    print(f"Using cache: {args.use_cache}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Checkpoint file paths
    midi_checkpoint_file = os.path.join(args.output_dir, "midi_checkpoint.json")
    text_checkpoint_file = os.path.join(args.output_dir, "text_checkpoint.json")

    # Load paired data
    print(f"\n📂 Loading paired data from {args.input_file}...")
    with open(args.input_file, encoding="utf-8") as f:
        paired_data = json.load(f)

    print(f"✅ Loaded {len(paired_data)} paired samples")

    # Initialize processors
    midi_processor = MIDIProcessor(
        max_sequence_length=1024,
        use_cache=args.use_cache,
        cache_dir=os.path.join(args.output_dir, "cache/midi"),
    )

    text_processor = TextProcessor(
        max_length=512,
        use_bert=True,
        use_spacy=True,
        use_gpu=use_gpu,
        use_cache=args.use_cache,
        cache_dir=os.path.join(args.output_dir, "cache/text"),
        batch_size=32,
    )

    # Extract MIDI files
    midi_files = [item.get("midi_file") for item in paired_data if "midi_file" in item]
    print(f"Found {len(midi_files)} MIDI files to process")

    # Process MIDI files
    print("\n🎵 Processing MIDI files...")
    midi_start_time = time.time()

    processed_midi = midi_processor.process_midi_files_parallel(
        midi_files=midi_files,
        max_workers=args.workers,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_file=midi_checkpoint_file,
        show_progress=True,
    )

    midi_time = time.time() - midi_start_time
    print(f"✅ Processed {len(processed_midi)}/{len(midi_files)} MIDI files in {midi_time:.1f}s")

    # Check if we need to continue or if we've hit the checkpoint limit
    if len(processed_midi) < len(midi_files):
        print(f"\n⏸️ Processing paused after {args.checkpoint_interval} batches of MIDI files")
        print("Run this script again to continue processing")
        return

    # Extract text descriptions
    texts = [item.get("text_description", "") for item in paired_data]
    texts = [t for t in texts if t]
    print(f"Found {len(texts)} text descriptions to process")

    # Process text descriptions
    print("\n📝 Processing text descriptions...")
    text_start_time = time.time()

    processed_texts = text_processor.process_texts_parallel(
        texts=texts,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_file=text_checkpoint_file,
        show_progress=True,
    )

    text_time = time.time() - text_start_time
    print(f"✅ Processed {len(processed_texts)}/{len(texts)} text descriptions in {text_time:.1f}s")

    # Check if we need to continue or if we've hit the checkpoint limit
    if len(processed_texts) < len(texts):
        print(
            f"\n⏸️ Processing paused after {args.checkpoint_interval} batches of text descriptions"
        )
        print("Run this script again to continue processing")
        return

    # If both MIDI and text processing are complete, combine them
    if len(processed_midi) == len(midi_files) and len(processed_texts) == len(texts):
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
        print(f"MIDI processing time: {midi_time/60:.1f} minutes")
        print(f"Text processing time: {text_time/60:.1f} minutes")
        print(f"Total processing time: {(midi_time + text_time)/60:.1f} minutes")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n⏱️ Total script time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")

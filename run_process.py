#!/usr/bin/env python3
"""
Process Module Runner - Processes MIDI and text data for training
"""

import argparse
import json
import multiprocessing
import os
import time

import numpy as np
import torch

from source.process import DataPreparer, MIDIProcessor, TextProcessor


# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def main():
    parser = argparse.ArgumentParser(description="Process MIDI and text data for training")
    parser.add_argument("--input_file", required=True, help="Input paired data file")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument(
        "--max_sequence_length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument("--max_text_length", type=int, default=512, help="Maximum text length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data ratio")
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--batch_processing", type=int, default=100, help="Number of files to process in each batch"
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=10, help="Stop after processing N batches"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for text processing if available"
    )
    parser.add_argument(
        "--no_gpu", action="store_true", help="Force CPU usage even if GPU is available"
    )
    parser.add_argument(
        "--use_cache", action="store_true", help="Use caching to speed up processing"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable caching even if enabled by default"
    )
    parser.add_argument(
        "--cache_dir", default="data/processed/cache", help="Directory for cache files"
    )

    args = parser.parse_args()

    # Set default workers based on CPU count
    if args.workers is None:
        args.workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free

    # Check GPU availability
    use_gpu = args.use_gpu and not args.no_gpu
    if use_gpu and torch.cuda.is_available():
        print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"🧠 CUDA Version: {torch.version.cuda}")
    else:
        if args.use_gpu and not torch.cuda.is_available():
            print("⚠️ GPU requested but not available, falling back to CPU")
        use_gpu = False

    # Check caching settings
    use_cache = args.use_cache and not args.no_cache

    print("🚀 Starting Data Processing...")
    print("=" * 50)
    print(f"Using {args.workers} parallel workers")
    print(f"Processing in batches of {args.batch_processing} files")
    print(f"Checkpoint interval: {args.checkpoint_interval} batches")
    print(f"Using GPU for text processing: {use_gpu}")
    print(f"Using caching: {use_cache}")
    if use_cache:
        print(f"Cache directory: {args.cache_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Checkpoint file paths
    midi_checkpoint_file = os.path.join(args.output_dir, "midi_checkpoint.json")
    text_checkpoint_file = os.path.join(args.output_dir, "text_checkpoint.json")

    # Step 1: Load paired data
    print(f"\n📂 Step 1: Loading paired data from {args.input_file}...")
    with open(args.input_file, encoding="utf-8") as f:
        paired_data = json.load(f)

    print(f"✅ Loaded {len(paired_data)} paired samples")

    # Initialize processors with new features
    midi_processor = MIDIProcessor(
        max_sequence_length=args.max_sequence_length,
        use_cache=use_cache,
        cache_dir=os.path.join(args.cache_dir, "midi"),
    )

    text_processor = TextProcessor(
        max_length=args.max_text_length,
        use_bert=True,
        use_spacy=True,
        use_gpu=use_gpu,
        use_cache=use_cache,
        cache_dir=os.path.join(args.cache_dir, "text"),
        batch_size=32,  # BERT batch size
    )

    # Step 2: Process MIDI files with new parallel processing
    print("\n🎵 Step 2: Processing MIDI files...")

    # Extract MIDI files
    midi_files = [item.get("midi_file") for item in paired_data if "midi_file" in item]
    print(f"Found {len(midi_files)} MIDI files to process")

    # Process MIDI files with new optimized method
    processed_midi = midi_processor.process_midi_files_parallel(
        midi_files=midi_files,
        max_workers=args.workers,
        batch_size=args.batch_processing,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_file=midi_checkpoint_file,
        show_progress=True,
    )

    print(f"✅ Processed {len(processed_midi)} MIDI files")

    # Step 3: Process text descriptions
    print("\n📝 Step 3: Processing text descriptions...")

    # Extract text descriptions
    texts = [item.get("text_description", "") for item in paired_data]
    texts = [t for t in texts if t]  # Filter empty texts
    print(f"Found {len(texts)} text descriptions to process")

    # Process texts with new optimized method
    processed_texts = text_processor.process_texts_parallel(
        texts=texts,
        batch_size=args.batch_processing,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_file=text_checkpoint_file,
        show_progress=True,
    )

    print(f"✅ Processed {len(processed_texts)} text descriptions")

    # Step 4: Prepare training data
    print("\n🔧 Step 4: Preparing training data...")
    data_preparer = DataPreparer(
        max_sequence_length=args.max_sequence_length,
        max_text_length=args.max_text_length,
        batch_size=args.batch_size,
    )

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

    # Split data
    train_data, val_data, test_data = data_preparer.split_data(
        processed_data, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    # Step 5: Save processed data
    print("\n💾 Step 5: Saving processed data...")

    # Save training data
    training_data = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "vocab_size": midi_processor.vocab_size,
        "max_sequence_length": args.max_sequence_length,
        "max_text_length": args.max_text_length,
        "total_samples": len(processed_data),
    }

    training_file = os.path.join(args.output_dir, "training_data.json")
    with open(training_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, cls=NumpyEncoder)

    # Save processed data
    processed_file = os.path.join(args.output_dir, "processed_data.json")
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, cls=NumpyEncoder)

    # Step 6: Print statistics
    print("\n📊 Step 6: Processing statistics...")

    # Get data statistics
    stats = data_preparer.get_data_statistics(processed_data)

    print(f"Total samples: {stats['total_samples']}")
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print(
        f"Sequence length: min={stats['sequence_length']['min']}, max={stats['sequence_length']['max']}, mean={stats['sequence_length']['mean']:.1f}"
    )
    print(
        f"Duration: min={stats['duration']['min']:.1f}s, max={stats['duration']['max']:.1f}s, mean={stats['duration']['mean']:.1f}s"
    )
    print(
        f"Top instruments: {', '.join([f'{k}({v})' for k, v in list(stats['top_instruments'].items())[:5]])}"
    )

    print("\n✅ Processing completed successfully!")
    print(f"Training data saved to: {training_file}")
    print(f"Processed data saved to: {processed_file}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n⏱️ Total processing time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")

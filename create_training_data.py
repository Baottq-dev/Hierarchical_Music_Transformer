#!/usr/bin/env python3
"""
Create Training Data Script - Creates training_data.json from processed_data.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from source.process import DataPreparer


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
    parser = argparse.ArgumentParser(description="Create training data with train/val/test splits")
    parser.add_argument(
        "--input_file", default="data/processed/processed_data.json", help="Processed data file"
    )
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument(
        "--max_sequence_length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument("--max_text_length", type=int, default=512, help="Maximum text length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data ratio")
    parser.add_argument(
        "--vocab_size", type=int, default=1000, help="Vocabulary size (if not specified in data)"
    )

    args = parser.parse_args()

    print("🚀 Creating Training Data...")
    print("=" * 50)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Test ratio: {1 - args.train_ratio - args.val_ratio}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load processed data
    print(f"\n📂 Step 1: Loading processed data from {args.input_file}...")
    with open(args.input_file, encoding="utf-8") as f:
        processed_data = json.load(f)

    if not processed_data:
        print(
            "❗ Processed data is empty. Please run continue_from_midi_checkpoint.py or run_process_batched.py until processed_data.json is populated."
        )
        sys.exit(1)

    print(f"✅ Loaded {len(processed_data)} processed samples")

    # Step 2: Prepare training data
    print("\n🔧 Step 2: Preparing training data...")
    data_preparer = DataPreparer(
        max_sequence_length=args.max_sequence_length,
        max_text_length=args.max_text_length,
        batch_size=args.batch_size,
    )

    # Split data
    train_data, val_data, test_data = data_preparer.split_data(
        processed_data, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    # Step 3: Save training data
    print("\n💾 Step 3: Saving training data...")

    # Try to determine vocab_size from data
    vocab_size = args.vocab_size
    if processed_data and "midi_tokens" in processed_data[0]:
        # Find the maximum token value and add 1 for vocab size
        max_token = max(
            max(item["midi_tokens"])
            for item in processed_data
            if "midi_tokens" in item and item["midi_tokens"]
        )
        vocab_size = max_token + 1
        print(f"Detected vocabulary size: {vocab_size}")

    # Save training data
    training_data = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "vocab_size": vocab_size,
        "max_sequence_length": args.max_sequence_length,
        "max_text_length": args.max_text_length,
        "total_samples": len(processed_data),
    }

    training_file = os.path.join(args.output_dir, "training_data.json")
    with open(training_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, cls=NumpyEncoder)

    # Step 4: Print statistics
    print("\n📊 Step 4: Data statistics...")

    # Get data statistics
    stats = data_preparer.get_data_statistics(processed_data)

    print(f"Total samples: {stats['total_samples']}")
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print(
        f"Sequence length: min={stats['sequence_length']['min']}, max={stats['sequence_length']['max']}, mean={stats['sequence_length']['mean']:.1f}"
    )

    if "duration" in stats:
        print(
            f"Duration: min={stats['duration']['min']:.1f}s, max={stats['duration']['max']:.1f}s, mean={stats['duration']['mean']:.1f}s"
        )

    if "top_instruments" in stats:
        print(
            f"Top instruments: {', '.join([f'{k}({v})' for k, v in list(stats['top_instruments'].items())[:5]])}"
        )

    print("\n✅ Training data creation completed successfully!")
    print(f"Training data saved to: {training_file}")
    print("Next step: Train the model")
    print(f"Run: python run_train.py --data_file {training_file}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n⏱️ Total processing time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")

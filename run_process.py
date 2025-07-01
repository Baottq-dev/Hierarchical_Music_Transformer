#!/usr/bin/env python3
"""
Process Module Runner - Processes MIDI and text data for training
"""

import argparse
import sys
import os
from source.process import MIDIProcessor, TextProcessor, DataPreparer

def main():
    parser = argparse.ArgumentParser(description="Process MIDI and text data for training")
    parser.add_argument("--input_file", required=True, help="Input paired data file")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument("--max_sequence_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max_text_length", type=int, default=512, help="Maximum text length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data ratio")
    
    args = parser.parse_args()
    
    print("🚀 Starting Data Processing...")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load paired data
    print(f"\n📂 Step 1: Loading paired data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        import json
        paired_data = json.load(f)
    
    print(f"✅ Loaded {len(paired_data)} paired samples")
    
    # Step 2: Process MIDI data
    print(f"\n🎵 Step 2: Processing MIDI data...")
    midi_processor = MIDIProcessor(max_sequence_length=args.max_sequence_length)
    
    processed_midi = []
    for item in paired_data:
        midi_file = item.get('midi_file')
        if midi_file and os.path.exists(midi_file):
            processed = midi_processor.process_midi_file(midi_file)
            if processed:
                processed_midi.append(processed)
    
    print(f"✅ Processed {len(processed_midi)} MIDI files")
    
    # Step 3: Process text data
    print(f"\n📝 Step 3: Processing text data...")
    text_processor = TextProcessor(max_length=args.max_text_length)
    
    processed_texts = []
    for item in paired_data:
        text = item.get('text_description', '')
        if text:
            processed = text_processor.process_text(text)
            processed_texts.append(processed)
    
    print(f"✅ Processed {len(processed_texts)} text descriptions")
    
    # Step 4: Prepare training data
    print(f"\n🔧 Step 4: Preparing training data...")
    data_preparer = DataPreparer(
        max_sequence_length=args.max_sequence_length,
        max_text_length=args.max_text_length,
        batch_size=args.batch_size
    )
    
    # Combine processed data
    processed_data = []
    for i, midi_item in enumerate(processed_midi):
        if i < len(processed_texts):
            combined_item = {
                'midi_file': midi_item['file_path'],
                'text_description': paired_data[i].get('text_description', ''),
                'midi_tokens': midi_item['tokens'],
                'midi_metadata': midi_item['metadata'],
                'text_features': processed_texts[i],
                'sequence_length': midi_item['sequence_length']
            }
            processed_data.append(combined_item)
    
    # Create dataset and dataloaders
    dataset = data_preparer.create_dataset(processed_data)
    train_loader = data_preparer.create_dataloader(dataset, shuffle=True)
    
    # Split data
    train_data, val_data, test_data = data_preparer.split_data(
        processed_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    # Create dataloaders for each split
    train_dataset = data_preparer.create_dataset(train_data)
    val_dataset = data_preparer.create_dataset(val_data)
    test_dataset = data_preparer.create_dataset(test_data)
    
    train_loader = data_preparer.create_dataloader(train_dataset, shuffle=True)
    val_loader = data_preparer.create_dataloader(val_dataset, shuffle=False)
    test_loader = data_preparer.create_dataloader(test_dataset, shuffle=False)
    
    # Step 5: Save processed data
    print(f"\n💾 Step 5: Saving processed data...")
    
    # Save training data
    training_data = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'vocab_size': midi_processor.vocab_size,
        'max_sequence_length': args.max_sequence_length,
        'max_text_length': args.max_text_length,
        'total_samples': len(processed_data)
    }
    
    training_file = os.path.join(args.output_dir, "training_data.json")
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # Save processed data
    processed_file = os.path.join(args.output_dir, "processed_data.json")
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    # Step 6: Print statistics
    print(f"\n📊 Step 6: Processing statistics...")
    stats = data_preparer.get_data_statistics(processed_data)
    
    print("\n📈 Processing Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n📁 Data splits:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Vocabulary size: {midi_processor.vocab_size}")
    
    print(f"\n🎉 Data processing completed!")
    print(f"📁 Output files:")
    print(f"  - Processed data: {processed_file}")
    print(f"  - Training data: {training_file}")

if __name__ == "__main__":
    main() 
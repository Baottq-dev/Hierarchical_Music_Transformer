#!/usr/bin/env python3
"""
Collect Module Runner - Collects MIDI files and text descriptions
"""

import argparse
import os

from source.collect import DataPairing, MIDICollector, TextCollector


def main():
    parser = argparse.ArgumentParser(description="Collect MIDI files and text descriptions")
    parser.add_argument("--midi_dir", default="data/midi", help="Directory containing MIDI files")
    parser.add_argument("--output_dir", default="data/output", help="Output directory")
    parser.add_argument("--filter_quality", action="store_true", help="Filter data by quality")
    parser.add_argument("--min_text_length", type=int, default=20, help="Minimum text length")
    parser.add_argument("--min_duration", type=float, default=10.0, help="Minimum MIDI duration")

    args = parser.parse_args()

    print("🎵 Starting Data Collection...")
    print("=" * 50)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Collect MIDI metadata
    print("\n📊 Step 1: Collecting MIDI metadata...")
    midi_collector = MIDICollector(args.midi_dir)
    midi_metadata = midi_collector.collect_all_metadata()

    metadata_file = os.path.join(args.output_dir, "midi_metadata.json")
    midi_collector.save_metadata(metadata_file)

    print(f"✅ Collected metadata for {len(midi_metadata)} MIDI files")

    # Step 2: Collect text descriptions
    print("\n📝 Step 2: Collecting text descriptions...")
    text_collector = TextCollector()
    paired_data = text_collector.collect_text_for_all_midi(midi_metadata)

    paired_file = os.path.join(args.output_dir, "paired_data.json")
    with open(paired_file, "w", encoding="utf-8") as f:
        import json

        json.dump(paired_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Paired {len(paired_data)} MIDI files with text descriptions")

    # Step 3: Create complete dataset
    print("\n🔗 Step 3: Creating complete dataset...")

    # Initialize data pairing
    data_pairing = DataPairing(args.midi_dir)

    # Use the paired_data we already created in Step 2
    if args.filter_quality:
        # Filter the existing paired_data
        filtered_dataset = data_pairing.filter_paired_data(
            paired_data, min_text_length=args.min_text_length, min_duration=args.min_duration
        )

        # Save filtered dataset
        complete_file = os.path.join(args.output_dir, "complete_dataset_filtered.json")
        with open(complete_file, "w", encoding="utf-8") as f:
            json.dump(filtered_dataset, f, indent=2, ensure_ascii=False)

        dataset = filtered_dataset
    else:
        # Use the existing paired_data
        complete_file = os.path.join(args.output_dir, "complete_dataset.json")
        with open(complete_file, "w", encoding="utf-8") as f:
            json.dump(paired_data, f, indent=2, ensure_ascii=False)

        dataset = paired_data

    # Step 4: Validate dataset
    print("\n✅ Step 4: Validating dataset...")
    stats = data_pairing.validate_paired_data(dataset)

    print("\n📈 Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n🎉 Data collection completed!")
    print("📁 Output files:")
    print(f"  - MIDI metadata: {metadata_file}")
    print(f"  - Paired data: {paired_file}")
    print(f"  - Complete dataset: {complete_file}")


if __name__ == "__main__":
    main()

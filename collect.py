#!/usr/bin/env python3
"""
Collect Module - Collects MIDI files and text descriptions
"""

import argparse
import os
import json
from pathlib import Path

from amt.utils.logging import get_logger
from amt.config import get_settings
from amt.collect import DataPairing, MIDICollector, TextCollector

# Set up logger
logger = get_logger(__name__)


def main():
    # Get settings
    settings = get_settings()
    
    parser = argparse.ArgumentParser(description="Collect MIDI files and text descriptions")
    parser.add_argument("--midi_dir", default=str(settings.midi_dir), help="Directory containing MIDI files")
    parser.add_argument("--output_dir", default=str(settings.output_dir), help="Output directory")
    parser.add_argument("--filter_quality", action="store_true", help="Filter data by quality")
    parser.add_argument("--min_text_length", type=int, default=20, help="Minimum text length")
    parser.add_argument("--min_duration", type=float, default=10.0, help="Minimum MIDI duration")
    parser.add_argument("--require_wikipedia", action="store_true", help="Only use descriptions from Wikipedia")
    parser.add_argument("--log_level", default=settings.log_level, 
                        choices=["debug", "info", "warning", "error", "critical"], 
                        help="Logging level")

    args = parser.parse_args()
    
    # Set log level
    logger.setLevel(args.log_level.upper())

    logger.info("🎵 Starting Data Collection...")
    logger.info("=" * 50)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Collect MIDI metadata
    logger.info("\n📊 Step 1: Collecting MIDI metadata...")
    midi_collector = MIDICollector(args.midi_dir)
    midi_metadata = midi_collector.collect_all_metadata()

    metadata_file = os.path.join(args.output_dir, "midi_metadata.json")
    midi_collector.save_metadata(metadata_file)

    logger.info(f"✅ Collected metadata for {len(midi_metadata)} MIDI files")

    # Step 2: Collect text descriptions
    logger.info("\n📝 Step 2: Collecting text descriptions...")
    text_collector = TextCollector()
    paired_data = text_collector.collect_text_for_all_midi(midi_metadata, require_wikipedia=args.require_wikipedia)

    paired_file = os.path.join(args.output_dir, "paired_data.json")
    with open(paired_file, "w", encoding="utf-8") as f:
        json.dump(paired_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Paired {len(paired_data)} MIDI files with text descriptions")

    # Step 3: Create complete dataset
    logger.info("\n🔗 Step 3: Creating complete dataset...")

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
    logger.info("\n✅ Step 4: Validating dataset...")
    stats = data_pairing.validate_paired_data(dataset)

    logger.info("\n📈 Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n🎉 Data collection completed!")
    logger.info("📁 Output files:")
    logger.info(f"  - MIDI metadata: {metadata_file}")
    logger.info(f"  - Paired data: {paired_file}")
    logger.info(f"  - Complete dataset: {complete_file}")


if __name__ == "__main__":
    main()

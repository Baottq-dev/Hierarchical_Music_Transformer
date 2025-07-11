"""
Main collector module for AMT data collection
"""

import os
import json
from typing import Dict, List, Any, Optional

from amt.utils.logging import get_logger
from amt.collect.midi_collector import MIDICollector
from amt.collect.text_collector import TextCollector
from amt.collect.data_pairing import DataPairing

logger = get_logger(__name__)


def collect_data(
    midi_dir: str = "data/midi",
    output_dir: str = "data/output",
    filter_quality: bool = False,
    min_text_length: int = 20,
    min_duration: float = 10.0,
) -> Dict[str, Any]:
    """
    Collect MIDI files and text descriptions.
    
    Args:
        midi_dir: Directory containing MIDI files
        output_dir: Output directory
        filter_quality: Filter data by quality
        min_text_length: Minimum text length
        min_duration: Minimum MIDI duration
        
    Returns:
        Dictionary containing dataset statistics
    """
    logger.info("Starting data collection process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Collect MIDI metadata
    logger.info("Step 1: Collecting MIDI metadata...")
    midi_collector = MIDICollector(midi_dir)
    midi_metadata = midi_collector.collect_all_metadata()

    metadata_file = os.path.join(output_dir, "midi_metadata.json")
    midi_collector.save_metadata(metadata_file)

    logger.info(f"Collected metadata for {len(midi_metadata)} MIDI files")

    # Step 2: Collect text descriptions
    logger.info("Step 2: Collecting text descriptions...")
    text_collector = TextCollector()
    paired_data = text_collector.collect_text_for_all_midi(midi_metadata)

    paired_file = os.path.join(output_dir, "paired_data.json")
    with open(paired_file, "w", encoding="utf-8") as f:
        json.dump(paired_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Paired {len(paired_data)} MIDI files with text descriptions")

    # Step 3: Create complete dataset
    logger.info("Step 3: Creating complete dataset...")

    # Initialize data pairing
    data_pairing = DataPairing(midi_dir)

    # Use the paired_data we already created in Step 2
    if filter_quality:
        # Filter the existing paired_data
        filtered_dataset = data_pairing.filter_paired_data(
            paired_data, min_text_length=min_text_length, min_duration=min_duration
        )

        # Save filtered dataset
        complete_file = os.path.join(output_dir, "complete_dataset_filtered.json")
        with open(complete_file, "w", encoding="utf-8") as f:
            json.dump(filtered_dataset, f, indent=2, ensure_ascii=False)

        dataset = filtered_dataset
    else:
        # Use the existing paired_data
        complete_file = os.path.join(output_dir, "complete_dataset.json")
        with open(complete_file, "w", encoding="utf-8") as f:
            json.dump(paired_data, f, indent=2, ensure_ascii=False)

        dataset = paired_data

    # Step 4: Validate dataset
    logger.info("Step 4: Validating dataset...")
    stats = data_pairing.validate_paired_data(dataset)

    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("Data collection completed!")
    logger.info(f"Output files:")
    logger.info(f"  - MIDI metadata: {metadata_file}")
    logger.info(f"  - Paired data: {paired_file}")
    logger.info(f"  - Complete dataset: {complete_file}")
    
    return stats 
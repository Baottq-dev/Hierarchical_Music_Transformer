"""
Data Pairing - Pairs MIDI files with text descriptions
"""

import json
import os
from typing import Any, Dict, List

from .midi_collector import MIDICollector
from .text_collector import TextCollector


class DataPairing:
    """Pairs MIDI files with text descriptions."""

    def __init__(self, midi_dir: str = "data/midi"):
        self.midi_dir = midi_dir
        self.midi_collector = MIDICollector(midi_dir)
        self.text_collector = TextCollector()

    def create_paired_dataset(
        self, output_file: str = "data/output/paired_data.json"
    ) -> List[Dict[str, Any]]:
        """Create a paired dataset of MIDI files and text descriptions."""
        print("Step 1: Collecting MIDI metadata...")
        midi_metadata = self.midi_collector.collect_all_metadata()

        print(f"Found {len(midi_metadata)} MIDI files")

        print("Step 2: Collecting text descriptions...")
        paired_data = self.text_collector.collect_text_for_all_midi(midi_metadata)

        print("Step 3: Saving paired data...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(paired_data, f, indent=2, ensure_ascii=False)

        print(f"Paired dataset saved to {output_file}")
        print(f"Total pairs: {len(paired_data)}")

        return paired_data

    def validate_paired_data(self, paired_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the paired dataset."""
        stats = {
            "total_pairs": len(paired_data),
            "wikipedia_sources": 0,
            "generated_sources": 0,
            "valid_midi_files": 0,
            "valid_text_descriptions": 0,
        }

        for pair in paired_data:
            # Count sources
            if pair.get("source") == "wikipedia":
                stats["wikipedia_sources"] += 1
            elif pair.get("source") == "generated":
                stats["generated_sources"] += 1

            # Validate MIDI file exists
            if os.path.exists(pair.get("midi_file", "")):
                stats["valid_midi_files"] += 1

            # Validate text description
            if pair.get("text_description") and len(pair["text_description"]) > 10:
                stats["valid_text_descriptions"] += 1

        return stats

    def filter_paired_data(
        self,
        paired_data: List[Dict[str, Any]],
        min_text_length: int = 20,
        min_duration: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Filter paired data based on quality criteria."""
        filtered_data = []

        for pair in paired_data:
            text = pair.get("text_description", "")
            metadata = pair.get("metadata", {})
            duration = metadata.get("duration", 0)

            # Check text length
            if len(text) < min_text_length:
                continue

            # Check duration
            if duration < min_duration:
                continue

            filtered_data.append(pair)

        print(f"Filtered {len(paired_data)} -> {len(filtered_data)} pairs")
        return filtered_data


def create_complete_dataset(
    midi_dir: str = "data/midi",
    output_file: str = "data/output/complete_dataset.json",
    filter_quality: bool = True,
) -> List[Dict[str, Any]]:
    """Create a complete paired dataset with optional quality filtering."""
    pairing = DataPairing(midi_dir)

    # Create paired dataset
    paired_data = pairing.create_paired_dataset(output_file)

    # Validate
    stats = pairing.validate_paired_data(paired_data)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Filter if requested
    if filter_quality:
        filtered_data = pairing.filter_paired_data(paired_data)

        # Save filtered data
        filtered_output = output_file.replace(".json", "_filtered.json")
        with open(filtered_output, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)

        print(f"Filtered dataset saved to {filtered_output}")
        return filtered_data

    return paired_data

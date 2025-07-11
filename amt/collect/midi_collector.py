"""
MIDI Collector - Collects and processes MIDI files
"""

import os
import json
import glob
from typing import List, Dict, Any, Optional

import pretty_midi

from amt.utils.logging import get_logger
from amt.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class MIDICollector:
    """Collects MIDI files and extracts their metadata."""

    def __init__(self, midi_dir: str = "data/midi"):
        self.midi_dir = midi_dir
        self.metadata = []

    def collect_midi_files(self) -> List[str]:
        """Collect all MIDI files from the directory."""
        midi_files = []
        for root, dirs, files in os.walk(self.midi_dir):
            for file in files:
                if file.lower().endswith(".mid") or file.lower().endswith(".midi"):
                    midi_files.append(os.path.join(root, file))
        return midi_files

    def extract_metadata(self, midi_file: str) -> Dict[str, Any]:
        """Extract metadata from a MIDI file."""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)

            metadata = {
                "file_path": midi_file,
                "file_name": os.path.basename(midi_file),
                "duration": midi_data.get_end_time(),
                "tempo": midi_data.estimate_tempo(),
                "key_signature": self._extract_key_signature(midi_data),
                "time_signature": self._extract_time_signature(midi_data),
                "instruments": self._extract_instruments(midi_data),
                "note_count": self._count_notes(midi_data),
                "total_notes": sum(len(track.notes) for track in midi_data.instruments),
            }

            return metadata
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
            return None

    def _extract_key_signature(self, midi_data: pretty_midi.PrettyMIDI) -> str:
        """Extract key signature from MIDI data."""
        if midi_data.key_signature_changes:
            key_change = midi_data.key_signature_changes[0]
            return f"{key_change.key_number}"
        return "C"  # Default to C major

    def _extract_time_signature(self, midi_data: pretty_midi.PrettyMIDI) -> str:
        """Extract time signature from MIDI data."""
        if midi_data.time_signature_changes:
            ts_change = midi_data.time_signature_changes[0]
            return f"{ts_change.numerator}/{ts_change.denominator}"
        return "4/4"  # Default to 4/4

    def _extract_instruments(self, midi_data: pretty_midi.PrettyMIDI) -> List[str]:
        """Extract instrument names from MIDI data."""
        instruments = []
        for instrument in midi_data.instruments:
            if instrument.name:
                instruments.append(instrument.name)
            else:
                instruments.append(f"Instrument_{instrument.program}")
        return instruments

    def _count_notes(self, midi_data: pretty_midi.PrettyMIDI) -> int:
        """Count total notes in MIDI data."""
        return sum(len(track.notes) for track in midi_data.instruments)

    def collect_all_metadata(self) -> List[Dict[str, Any]]:
        """Collect metadata from all MIDI files."""
        midi_files = self.collect_midi_files()
        metadata_list = []

        for midi_file in midi_files:
            metadata = self.extract_metadata(midi_file)
            if metadata:
                metadata_list.append(metadata)

        self.metadata = metadata_list
        return metadata_list

    def save_metadata(self, output_file: str = "data/output/midi_metadata.json"):
        """Save collected metadata to file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"Metadata saved to {output_file}")


def list_midi_files_and_metadata(midi_dir: str = "data/midi") -> List[Dict[str, Any]]:
    """Convenience function to list MIDI files and extract metadata."""
    collector = MIDICollector(midi_dir)
    return collector.collect_all_metadata()


def save_metadata(metadata: List[Dict[str, Any]], output_file: str):
    """Convenience function to save metadata."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to {output_file}")

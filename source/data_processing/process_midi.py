"""
Script to process MIDI files from Lakh MIDI Clean dataset.
"""

import os
import json
from typing import List, Dict
from midi_processor import midi_to_event_sequence, analyze_midi_file

def process_midi_files(midi_dir: str, output_file: str):
    """
    Process MIDI files and save event sequences.
    Args:
        midi_dir: Directory containing MIDI files
        output_file: Path to output JSON file
    """
    # Get all MIDI files
    midi_files = []
    for root, _, files in os.walk(midi_dir):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    
    # Process MIDI files
    midi_data = []
    for midi_file in midi_files:
        try:
            # Convert to event sequence
            event_sequence = midi_to_event_sequence(midi_file)
            
            # Analyze MIDI file
            analysis = analyze_midi_file(midi_file)
            
            midi_data.append({
                "midi_file": midi_file,
                "event_sequence": event_sequence,
                "analysis": analysis
            })
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(midi_data, f, indent=2)

if __name__ == "__main__":
    # Set paths
    midi_dir = "data/midi"
    output_file = "data/processed/midi_sequences.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process MIDI files
    process_midi_files(midi_dir, output_file) 
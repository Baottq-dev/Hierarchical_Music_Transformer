"""
Script to collect text descriptions from Wikipedia for Lakh MIDI Clean dataset.
"""

import os
import json
from typing import List, Dict
from text_processor import scrape_wikipedia

def get_midi_info(midi_file: str) -> Dict[str, str]:
    """
    Extract artist and song information from MIDI filename.
    Args:
        midi_file: Path to MIDI file
    Returns:
        Dictionary containing artist and song information
    """
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(midi_file))[0]
    
    # Split by common separators
    parts = filename.replace('_', ' ').replace('-', ' ').split()
    
    # Assume first part is artist, rest is song
    if len(parts) >= 2:
        artist = parts[0]
        song = ' '.join(parts[1:])
        return {
            "artist": artist,
            "song": song
        }
    return None

def collect_text_descriptions(midi_dir: str, output_file: str):
    """
    Collect text descriptions for MIDI files.
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
    
    # Collect text descriptions
    text_data = []
    for midi_file in midi_files:
        # Get artist and song info
        info = get_midi_info(midi_file)
        if not info:
            continue
        
        # Scrape Wikipedia
        text = scrape_wikipedia(info["artist"], info["song"])
        if text:
            text_data.append({
                "midi_file": midi_file,
                "artist": info["artist"],
                "song": info["song"],
                "text": text
            })
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(text_data, f, indent=2)

if __name__ == "__main__":
    # Set paths
    midi_dir = "data/midi"
    output_file = "data/text/wikipedia_descriptions.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Collect text descriptions
    collect_text_descriptions(midi_dir, output_file)
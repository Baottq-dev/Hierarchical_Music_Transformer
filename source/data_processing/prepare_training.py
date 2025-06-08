"""
Script to prepare training data for AMT model.
"""

import os
import json
from typing import List, Dict
from text_processor import create_training_examples

def prepare_training_data(midi_file: str, text_file: str, output_file: str):
    """
    Prepare training data by combining MIDI and text data.
    Args:
        midi_file: Path to MIDI sequences JSON file
        text_file: Path to text embeddings JSON file
        output_file: Path to output JSON file
    """
    # Load MIDI data
    with open(midi_file, 'r') as f:
        midi_data = json.load(f)
    
    # Load text data
    with open(text_file, 'r') as f:
        text_data = json.load(f)
    
    # Create training examples
    training_data = []
    for midi_item in midi_data:
        midi_file = midi_item["midi_file"]
        
        # Find matching text description
        for text_item in text_data["processed_data"]:
            if text_item["original_text"] in midi_file:
                # Create training example
                example = create_training_examples(
                    midi_file=midi_file,
                    text_description=text_item["original_text"]
                )
                training_data.append(example)
                break
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)

if __name__ == "__main__":
    # Set paths
    midi_file = "data/processed/midi_sequences.json"
    text_file = "data/processed/text_embeddings.json"
    output_file = "data/processed/training_data.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Prepare training data
    prepare_training_data(midi_file, text_file, output_file) 
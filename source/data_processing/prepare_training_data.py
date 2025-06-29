"""
Script to prepare training data for AMT model.
Combines MIDI event sequences with semantic tokens.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from midi_processor import midi_to_event_sequence
from config import CLUSTERED_DATA_FILE, TRAINING_DATA_FILE, OUTPUT_DIR

def load_clustered_data(file_path: str) -> List[Dict[str, Any]]:
    """Load clustered text data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, file_path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def prepare_training_data(clustered_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare training data by combining MIDI sequences with semantic tokens."""
    print("Preparing training data...")
    
    training_data = []
    
    for item in clustered_data:
        midi_file = item['file_path']
        semantic_token = item['semantic_token']
        
        try:
            # Convert MIDI to event sequence
            event_sequence = midi_to_event_sequence(midi_file)
            
            if event_sequence:
                # Create semantic token string
                semantic_token_str = f"SEMANTIC_TOKEN_{semantic_token}"
                
                # Combine semantic token with event sequence
                combined_sequence = [semantic_token_str] + event_sequence
                
                training_example = {
                    'midi_file': midi_file,
                    'artist': item.get('artist', ''),
                    'title': item.get('title', ''),
                    'text_description': item.get('text_description', ''),
                    'semantic_token': semantic_token,
                    'semantic_token_str': semantic_token_str,
                    'midi_event_sequence': event_sequence,
                    'combined_sequence_for_amt': combined_sequence
                }
                
                training_data.append(training_example)
                
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
            continue
    
    return training_data

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Prepare AMT training data')
    parser.add_argument('--clustered_file', default=CLUSTERED_DATA_FILE,
                       help='Path to clustered data JSON file')
    parser.add_argument('--output_file', default=TRAINING_DATA_FILE,
                       help='Path to output training data JSON file')
    args = parser.parse_args()
    
    print("Starting training data preparation...")
    
    # Load clustered data
    print(f"Loading clustered data from {args.clustered_file}")
    clustered_data = load_clustered_data(args.clustered_file)
    print(f"Loaded {len(clustered_data)} clustered items")
    
    # Prepare training data
    training_data = prepare_training_data(clustered_data)
    
    # Save training data
    save_json(training_data, args.output_file)
    print(f"Saved {len(training_data)} training examples to {args.output_file}")
    
    print("Training data preparation completed successfully!")

if __name__ == "__main__":
    main() 
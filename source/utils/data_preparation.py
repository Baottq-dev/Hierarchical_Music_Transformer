import json
import os
from ..data_processing.midi_processor import midi_to_event_sequence

def prepare_training_data(clustered_data_path, output_path):
    """
    Prepares training data for the AMT model by combining MIDI event sequences
    with semantic tokens.
    Args:
        clustered_data_path: Path to JSON file containing clustered data
        output_path: Path to save prepared training data
    """
    try:
        with open(clustered_data_path, "r") as f:
            clustered_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Clustered data file {clustered_data_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {clustered_data_path}.")
        return

    if not clustered_data:
        print("No data found in the clustered data file.")
        return

    amt_training_data = []
    processed_count = 0
    error_count = 0

    print(f"Starting to prepare training data for {len(clustered_data)} entries...")

    for item in clustered_data:
        midi_file_path = item.get("file_path")
        semantic_token_id = item.get("semantic_token")

        if midi_file_path is None or semantic_token_id is None:
            print(f"Skipping item due to missing MIDI path or semantic token: {item.get('title', 'Unknown title')}")
            error_count += 1
            continue
        
        if not os.path.exists(midi_file_path):
            print(f"Skipping item as MIDI file not found: {midi_file_path}")
            error_count += 1
            continue

        semantic_token_str = f"SEMANTIC_TOKEN_{semantic_token_id}"
        midi_event_sequence = midi_to_event_sequence(midi_file_path)

        if midi_event_sequence is None:
            print(f"Skipping item as MIDI event sequence could not be generated for: {midi_file_path}")
            error_count += 1
            continue
        
        combined_sequence = [semantic_token_str] + midi_event_sequence
        
        training_item = {
            "file_path": midi_file_path,
            "artist": item.get("artist", ""),
            "title": item.get("title", ""),
            "text_description": item.get("text_description", ""),
            "semantic_token_id": semantic_token_id,
            "semantic_token_str": semantic_token_str,
            "midi_event_sequence": midi_event_sequence,
            "combined_sequence_for_amt": combined_sequence
        }
        amt_training_data.append(training_item)
        processed_count += 1
        if processed_count % 1 == 0:
             print(f"Processed {processed_count}/{len(clustered_data)} items. Last processed: {item.get('title', 'N/A')}")

    try:
        with open(output_path, "w") as f:
            json.dump(amt_training_data, f, indent=4)
        print(f"Successfully prepared and saved AMT training data to {output_path}")
        print(f"Total items processed: {processed_count}, Errors/Skipped: {error_count}")
    except IOError:
        print(f"Error: Could not write AMT training data to {output_path}.") 
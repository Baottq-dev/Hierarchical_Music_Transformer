import json
import os
# Assuming preprocess_midi.py is in the same directory and its function can be imported
# If not, copy the relevant functions here or adjust PYTHONPATH
from preprocess_midi import midi_to_event_sequence # This line might need adjustment based on actual file structure

def prepare_training_data(clustered_data_path, output_path):
    """
    Loads clustered text data, processes corresponding MIDI files into event sequences,
    prepends the semantic token to each MIDI sequence, and saves the result.
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
        semantic_token_id = item.get("semantic_token") # This is an integer, e.g., 0, 1, 2

        if midi_file_path is None or semantic_token_id is None:
            print(f"Skipping item due to missing MIDI path or semantic token: {item.get('title', 'Unknown title')}")
            error_count += 1
            continue
        
        if not os.path.exists(midi_file_path):
            print(f"Skipping item as MIDI file not found: {midi_file_path}")
            error_count += 1
            continue

        # Convert semantic token ID to the string representation like "SEMANTIC_TOKEN_X"
        semantic_token_str = f"SEMANTIC_TOKEN_{semantic_token_id}"

        # Generate MIDI event sequence
        midi_event_sequence = midi_to_event_sequence(midi_file_path)

        if midi_event_sequence is None:
            print(f"Skipping item as MIDI event sequence could not be generated for: {midi_file_path}")
            error_count += 1
            continue
        
        # Prepend the semantic token string to the MIDI event sequence
        combined_sequence = [semantic_token_str] + midi_event_sequence
        
        # Create a new dictionary for the training data
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
        if processed_count % 1 == 0: # Print progress for every item given the small sample size
             print(f"Processed {processed_count}/{len(clustered_data)} items. Last processed: {item.get('title', 'N/A')}")

    try:
        with open(output_path, "w") as f:
            json.dump(amt_training_data, f, indent=4)
        print(f"Successfully prepared and saved AMT training data to {output_path}")
        print(f"Total items processed: {processed_count}, Errors/Skipped: {error_count}")
    except IOError:
        print(f"Error: Could not write AMT training data to {output_path}.")

if __name__ == "__main__":
    clustered_json_path = "./data/output/clustered_text_data.json"
    amt_data_output_path = "./data/output/amt_training_data.json"
    prepare_training_data(clustered_json_path, amt_data_output_path)


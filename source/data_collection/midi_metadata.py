import os
import json

def list_midi_files_and_metadata(base_dir):
    """
    Scans a directory for MIDI files and extracts metadata.
    Returns a list of dictionaries containing file paths and metadata.
    """
    midi_metadata_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".mid") or file.lower().endswith(".midi"):
                file_path = os.path.join(root, file)
                # Attempt to extract artist from directory name
                artist = os.path.basename(root)
                # Attempt to extract title from filename (simple cleaning)
                title = os.path.splitext(file)[0].replace("_", " ").replace(".", " ")
                # Remove potential numbering like .1, .2 at the end of titles
                parts = title.split(" ")
                if len(parts) > 1 and parts[-1].isdigit():
                    title = " ".join(parts[:-1])
                
                midi_metadata_list.append({
                    "file_path": file_path,
                    "artist": artist,
                    "title": title.strip()
                })
    return midi_metadata_list

def save_metadata(metadata_list, output_file):
    """Saves metadata to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(metadata_list, f, indent=4)
    print(f"Found {len(metadata_list)} MIDI files. Metadata saved to {output_file}") 
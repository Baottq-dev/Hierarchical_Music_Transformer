import os
import json

def list_midi_files_and_metadata(base_dir):
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

if __name__ == "__main__":
    INPUT_MIDI_DIR = "./data/midi"
    OUTPUT_METADATA_FILE = "./data/output/midi_metadata_list.json"
    metadata = list_midi_files_and_metadata(INPUT_MIDI_DIR)
    with open(OUTPUT_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Found {len(metadata)} MIDI files. Metadata saved to {OUTPUT_METADATA_FILE}")


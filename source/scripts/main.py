import os
import sys
import json

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.midi_metadata import list_midi_files_and_metadata, save_metadata
from data_collection.wikipedia_collector import pair_midi_with_wikipedia
from data_processing.text_processor import get_bert_embeddings
from model.clustering import cluster_embeddings
from utils.data_preparation import prepare_training_data
from utils.environment import verify_environment

def main():
    """Main function to run the entire AMT pipeline."""
    
    # Verify environment
    print("Verifying environment...")
    verify_environment()
    
    # Step 1: Extract MIDI metadata
    print("\nStep 1: Extracting MIDI metadata...")
    midi_dir = "./data/midi"
    metadata_file = "./data/output/midi_metadata_list.json"
    metadata = list_midi_files_and_metadata(midi_dir)
    save_metadata(metadata, metadata_file)
    
    # Step 2: Pair with Wikipedia descriptions
    print("\nStep 2: Pairing with Wikipedia descriptions...")
    paired_data_file = "./data/output/automated_paired_data.json"
    pair_midi_with_wikipedia(metadata_file, paired_data_file)
    
    # Step 3: Generate text embeddings
    print("\nStep 3: Generating text embeddings...")
    with open(paired_data_file, 'r') as f:
        paired_data = json.load(f)
    text_descriptions = [item.get("text_description", "") for item in paired_data]
    embeddings = get_bert_embeddings(text_descriptions)
    
    # Save embeddings
    embeddings_file = "./data/output/text_embeddings.json"
    output_data = []
    for i, item in enumerate(paired_data):
        output_data.append({
            "file_path": item["file_path"],
            "artist": item["artist"],
            "title": item["title"],
            "text_description": item["text_description"],
            "embedding": embeddings[i].tolist()
        })
    with open(embeddings_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # Step 4: Cluster embeddings
    print("\nStep 4: Clustering embeddings...")
    clustered_file = "./data/output/clustered_text_data.json"
    cluster_embeddings(embeddings_file, clustered_file)
    
    # Step 5: Prepare training data
    print("\nStep 5: Preparing training data...")
    training_data_file = "./data/output/amt_training_data.json"
    prepare_training_data(clustered_file, training_data_file)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 
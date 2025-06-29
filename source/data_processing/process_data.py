"""
Main data processing script for AMT project.
Handles text embeddings, clustering, and data preparation.
"""

import os
import json
import argparse
from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from text_processor import get_bert_embeddings, clean_text
from clustering import cluster_embeddings
from data_preparation import prepare_training_data
from config import (
    PAIRED_DATA_FILE, 
    TEXT_EMBEDDINGS_FILE, 
    CLUSTERED_DATA_FILE,
    TRAINING_DATA_FILE,
    OUTPUT_DIR
)

def load_paired_data(file_path: str) -> List[Dict[str, Any]]:
    """Load paired MIDI and text data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, file_path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_text_embeddings(paired_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process text descriptions and generate BERT embeddings."""
    print("Processing text embeddings...")
    
    # Extract text descriptions
    texts = []
    for item in paired_data:
        if 'text_description' in item and item['text_description']:
            texts.append(item['text_description'])
        else:
            texts.append("instrumental music")
    
    # Clean texts
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Generate BERT embeddings
    embeddings = get_bert_embeddings(cleaned_texts)
    
    # Create output data
    text_data = []
    for i, item in enumerate(paired_data):
        text_data.append({
            'file_path': item['file_path'],
            'artist': item.get('artist', ''),
            'title': item.get('title', ''),
            'text_description': item.get('text_description', ''),
            'cleaned_text': cleaned_texts[i],
            'embedding': embeddings[i].tolist()
        })
    
    return text_data

def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(description='Process AMT data')
    parser.add_argument('--paired_file', default=PAIRED_DATA_FILE, 
                       help='Path to paired data JSON file')
    parser.add_argument('--output_dir', default=OUTPUT_DIR,
                       help='Output directory')
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of clusters for K-means')
    args = parser.parse_args()
    
    print("Starting AMT data processing...")
    
    # Step 1: Load paired data
    print(f"Loading paired data from {args.paired_file}")
    paired_data = load_paired_data(args.paired_file)
    print(f"Loaded {len(paired_data)} paired items")
    
    # Step 2: Process text embeddings
    text_data = process_text_embeddings(paired_data)
    text_embeddings_path = os.path.join(args.output_dir, 'text_embeddings.json')
    save_json(text_data, text_embeddings_path)
    print(f"Saved text embeddings to {text_embeddings_path}")
    
    # Step 3: Perform clustering
    print("Performing clustering...")
    embeddings = np.array([item['embedding'] for item in text_data])
    clustered_data = cluster_embeddings(embeddings, text_data, n_clusters=args.n_clusters)
    
    clustered_data_path = os.path.join(args.output_dir, 'clustered_text_data.json')
    save_json(clustered_data, clustered_data_path)
    print(f"Saved clustered data to {clustered_data_path}")
    
    # Step 4: Prepare training data
    print("Preparing training data...")
    training_data = prepare_training_data(clustered_data)
    
    training_data_path = os.path.join(args.output_dir, 'amt_training_data.json')
    save_json(training_data, training_data_path)
    print(f"Saved training data to {training_data_path}")
    
    print("Data processing completed successfully!")

if __name__ == "__main__":
    main() 
"""
Script to process text descriptions from Wikipedia.
"""

import os
import json
from typing import List, Dict
from text_processor import process_text_descriptions, get_bert_embedding

def process_text_data(input_file: str, output_file: str):
    """
    Process text descriptions and generate embeddings.
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    # Load text data
    with open(input_file, 'r') as f:
        text_data = json.load(f)
    
    # Extract text descriptions
    text_list = [item["text"] for item in text_data]
    
    # Process text descriptions
    processed_data = process_text_descriptions(text_list)
    
    # Generate BERT embeddings
    for item in processed_data["processed_data"]:
        item["embedding"] = get_bert_embedding(item["cleaned_text"]).tolist()
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)

if __name__ == "__main__":
    # Set paths
    input_file = "data/text/wikipedia_descriptions.json"
    output_file = "data/processed/text_embeddings.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process text data
    process_text_data(input_file, output_file) 
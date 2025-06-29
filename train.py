#!/usr/bin/env python3
"""
Training Script for AMT
Processes data and trains the model
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add source to path
sys.path.append('source')

from source.data_processing.text_processor import get_bert_embeddings
from source.model.clustering import cluster_embeddings
from source.utils.data_preparation import prepare_training_data
from source.model.training import train_model

def process_data(input_file, output_dir):
    """Process data: embeddings -> clustering -> training data"""
    print("🔄 Processing data...")
    
    # Load paired data
    with open(input_file, 'r') as f:
        paired_data = json.load(f)
    
    # Filter valid descriptions
    valid_data = []
    text_descriptions = []
    for item in paired_data:
        text_desc = item.get("text_description", "")
        if text_desc and text_desc != "Not Found":
            text_descriptions.append(text_desc)
            valid_data.append(item)
    
    if not text_descriptions:
        print("❌ No valid text descriptions found!")
        return False
    
    print(f"📝 Processing {len(text_descriptions)} descriptions...")
    
    # Generate embeddings
    print("🧠 Generating BERT embeddings...")
    embeddings = get_bert_embeddings(text_descriptions)
    
    # Save embeddings
    embeddings_file = os.path.join(output_dir, "text_embeddings.json")
    output_data = []
    for i, item in enumerate(valid_data):
        output_data.append({
            "file_path": item["file_path"],
            "artist": item["artist"],
            "title": item["title"],
            "text_description": item["text_description"],
            "embedding": embeddings[i].tolist()
        })
    
    with open(embeddings_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"✅ Embeddings saved to: {embeddings_file}")
    
    # Cluster embeddings
    print("🎯 Clustering embeddings...")
    clustered_file = os.path.join(output_dir, "clustered_text_data.json")
    cluster_embeddings(embeddings_file, clustered_file)
    print(f"✅ Clustered data saved to: {clustered_file}")
    
    # Prepare training data
    print("📊 Preparing training data...")
    training_file = os.path.join(output_dir, "amt_training_data.json")
    prepare_training_data(clustered_file, training_file)
    print(f"✅ Training data saved to: {training_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Train AMT model")
    parser.add_argument("--paired_file", default="./data/output/automated_paired_data.json", 
                       help="Paired data JSON file")
    parser.add_argument("--output_dir", default="./data/output", help="Output directory")
    parser.add_argument("--model_dir", default="./models/checkpoints", help="Model checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--skip_processing", action="store_true", help="Skip data processing")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    print("🎵 AMT Training")
    print("=" * 50)
    print(f"Paired File: {args.paired_file}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Model Directory: {args.model_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    
    # Check input file
    if not os.path.exists(args.paired_file):
        print(f"❌ Error: Paired data file {args.paired_file} does not exist!")
        print("Please run collect_data.py first")
        return
    
    # Step 1: Process data
    if not args.skip_processing:
        print("\n📊 Step 1: Processing data...")
        if not process_data(args.paired_file, args.output_dir):
            print("❌ Data processing failed!")
            return
    else:
        print("\n⏭️ Skipping data processing")
    
    # Step 2: Train model
    if not args.skip_training:
        print("\n🤖 Step 2: Training model...")
        training_file = os.path.join(args.output_dir, "amt_training_data.json")
        
        if not os.path.exists(training_file):
            print(f"❌ Error: Training data file {training_file} does not exist!")
            print("Please run data processing first")
            return
        
        try:
            train_model(
                data_file=training_file,
                output_dir=args.model_dir,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.lr
            )
            print("✅ Model training completed!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return
    else:
        print("\n⏭️ Skipping model training")
    
    print("\n🎉 Training pipeline completed!")
    print(f"📄 Training data: {os.path.join(args.output_dir, 'amt_training_data.json')}")
    print(f"🤖 Model checkpoints: {args.model_dir}")

if __name__ == "__main__":
    main() 
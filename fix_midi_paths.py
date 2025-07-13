#!/usr/bin/env python3
"""
Fix MIDI paths in paired JSON data files to use Kaggle paths.

This script helps when running the AMT project on Kaggle by updating MIDI file paths
in the paired data JSON file to point to the correct location in the Kaggle environment.

Usage:
    python fix_midi_paths.py --input-file data/output/wikipedia_only_dataset.json \
                           --output-file data/output/wikipedia_kaggle_dataset.json \
                           --kaggle-path /kaggle/input/your-dataset/midi
"""

import os
import json
import argparse
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def fix_midi_paths(
    data: List[Dict[str, Any]], 
    old_prefix: str = "data/midi", 
    new_prefix: str = "/kaggle/input/your-dataset/midi"
) -> List[Dict[str, Any]]:
    """
    Update MIDI paths in the paired data.
    
    Args:
        data: List of paired data items
        old_prefix: Original MIDI path prefix to replace
        new_prefix: New MIDI path prefix to use
        
    Returns:
        Updated data with fixed MIDI paths
    """
    midi_path_keys = ["midi_path", "file_path", "path", "midi_file"]
    paths_updated = 0
    
    # Process each item in the data
    for item in data:
        # Check for each possible key that might contain a MIDI path
        for key in list(item.keys()):
            if isinstance(item[key], str) and old_prefix in item[key] and (
                key in midi_path_keys or 
                ("midi" in key.lower() and "path" in key.lower()) or
                key.lower().endswith("path") or 
                "file" in key.lower()
            ):
                old_path = item[key]
                new_path = old_path.replace(old_prefix, new_prefix)
                item[key] = new_path
                paths_updated += 1
                logger.info(f"Updated path: {old_path} → {new_path}")
                
        # Check nested dictionaries in metadata
        if "metadata" in item and isinstance(item["metadata"], dict):
            for key in list(item["metadata"].keys()):
                if isinstance(item["metadata"][key], str) and old_prefix in item["metadata"][key] and (
                    key in midi_path_keys or 
                    ("midi" in key.lower() and "path" in key.lower()) or 
                    key.lower().endswith("path") or 
                    "file" in key.lower()
                ):
                    old_path = item["metadata"][key]
                    new_path = old_path.replace(old_prefix, new_prefix)
                    item["metadata"][key] = new_path
                    paths_updated += 1
                    logger.info(f"Updated metadata path: {old_path} → {new_path}")
    
    logger.info(f"Total paths updated: {paths_updated}")
    return data

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fix MIDI paths in paired data JSON files")
    parser.add_argument("--input-file", type=str, required=True, 
                        help="Input JSON file with paired data")
    parser.add_argument("--output-file", type=str, required=True, 
                        help="Output JSON file to save fixed paths")
    parser.add_argument("--old-prefix", type=str, default="data/midi",
                        help="Original MIDI path prefix to replace")
    parser.add_argument("--kaggle-path", type=str, default="/kaggle/input/your-dataset/midi",
                        help="Kaggle path prefix for MIDI files")
    
    args = parser.parse_args()
    
    # Ensure input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load input data
    logger.info(f"Loading paired data from {args.input_file}")
    try:
        with open(args.input_file, "r") as f:
            paired_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return False
    
    # Fix MIDI paths
    logger.info(f"Fixing MIDI paths: {args.old_prefix} → {args.kaggle_path}")
    fixed_data = fix_midi_paths(paired_data, args.old_prefix, args.kaggle_path)
    
    # Save output data
    logger.info(f"Saving fixed data to {args.output_file}")
    try:
        with open(args.output_file, "w") as f:
            json.dump(fixed_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save output file: {e}")
        return False
    
    logger.info(f"✅ Successfully fixed MIDI paths! You can now use this file with process.py")
    logger.info(f"Example command:")
    logger.info(f"python process.py --paired-data-file {args.output_file} --output-dir data/processed --dataset-name kaggle_dataset ...")
    
    return True

if __name__ == "__main__":
    main() 
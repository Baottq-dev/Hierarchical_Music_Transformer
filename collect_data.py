#!/usr/bin/env python3
"""
Data Collection Script for AMT
Collects MIDI metadata and Wikipedia descriptions
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add source to path
sys.path.append('source')

from source.data_collection.midi_metadata import list_midi_files_and_metadata, save_metadata
from source.data_collection.wikipedia_collector import pair_midi_with_wikipedia

def main():
    parser = argparse.ArgumentParser(description="Collect MIDI and Wikipedia data")
    parser.add_argument("--midi_dir", default="./data/midi", help="MIDI files directory")
    parser.add_argument("--output_dir", default="./data/output", help="Output directory")
    parser.add_argument("--skip_wikipedia", action="store_true", help="Skip Wikipedia collection")
    parser.add_argument("--delay", type=float, default=1.0, help="Wikipedia request delay")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎵 AMT Data Collection")
    print("=" * 50)
    print(f"MIDI Directory: {args.midi_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Skip Wikipedia: {args.skip_wikipedia}")
    print(f"Request Delay: {args.delay}s")
    
    # Step 1: Collect MIDI metadata
    print("\n📁 Step 1: Collecting MIDI metadata...")
    if not os.path.exists(args.midi_dir):
        print(f"Error: MIDI directory {args.midi_dir} does not exist!")
        return
    
    metadata = list_midi_files_and_metadata(args.midi_dir)
    if not metadata:
        print("No MIDI files found!")
        return
    
    metadata_file = os.path.join(args.output_dir, "midi_metadata_list.json")
    save_metadata(metadata, metadata_file)
    print(f"Found {len(metadata)} MIDI files")
    
    # Step 2: Collect Wikipedia data 
    if not args.skip_wikipedia:
        print("\nStep 2: Collecting Wikipedia descriptions...")
        paired_file = os.path.join(args.output_dir, "automated_paired_data.json")
        pair_midi_with_wikipedia(metadata_file, paired_file, request_delay=args.delay)
        print("Wikipedia data collected")
    else:
        print("\nSkipping Wikipedia collection")
    
    print("\nData collection completed!")
    print(f"Metadata saved to: {metadata_file}")
    if not args.skip_wikipedia:
        print(f"Paired data saved to: {os.path.join(args.output_dir, 'automated_paired_data.json')}")

if __name__ == "__main__":
    main() 
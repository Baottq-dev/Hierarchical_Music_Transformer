"""
Script to run the entire AMT pipeline.
"""

import os
import subprocess
from typing import List

def run_command(command: List[str]):
    """
    Run a command and print output.
    Args:
        command: Command to run
    """
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)

def main():
    # Create necessary directories
    os.makedirs("data/midi", exist_ok=True)
    os.makedirs("data/text", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Step 1: Collect text descriptions
    print("\nStep 1: Collecting text descriptions...")
    run_command(["python", "source/data_processing/collect_text.py"])
    
    # Step 2: Process MIDI files
    print("\nStep 2: Processing MIDI files...")
    run_command(["python", "source/data_processing/process_midi.py"])
    
    # Step 3: Process text data
    print("\nStep 3: Processing text data...")
    run_command(["python", "source/data_processing/process_text.py"])
    
    # Step 4: Prepare training data
    print("\nStep 4: Preparing training data...")
    run_command(["python", "source/data_processing/prepare_training.py"])
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 
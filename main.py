#!/usr/bin/env python3
"""
AMT (Audio Music Transformer) Pipeline Main Entry
Controls the complete pipeline: collect, process, prepare, train, generate, evaluate
"""

import argparse
import subprocess
import sys
import os
from typing import List, Tuple

# Đường dẫn các script/module
COLLECT_CMD = [sys.executable, 'collect_data.py']
PROCESS_CMD = [sys.executable, 'source/data_processing/process_data.py']
PREPARE_CMD = [sys.executable, 'source/data_processing/prepare_training_data.py']
TRAIN_CMD = [sys.executable, 'source/model/training.py']
GENERATE_CMD = [sys.executable, 'source/model/generation.py']
EVALUATE_CMD = [sys.executable, 'source/evaluation/metrics.py']

# Định nghĩa pipeline với tên và lệnh
PIPELINE = [
    ('collect', COLLECT_CMD, 'Collect MIDI metadata and Wikipedia descriptions'),
    ('process', PROCESS_CMD, 'Process text embeddings and perform clustering'),
    ('prepare', PREPARE_CMD, 'Prepare training data from MIDI and embeddings'),
    ('train', TRAIN_CMD, 'Train GPT-2 model with MIDI sequences'),
    ('generate', GENERATE_CMD, 'Generate music from text descriptions'),
    ('evaluate', EVALUATE_CMD, 'Evaluate generated music quality'),
]

def print_banner():
    """Print AMT pipeline banner."""
    print("=" * 60)
    print("🎵 AMT (Audio Music Transformer) Pipeline")
    print("   Symbolic Music Generation with Text Controls")
    print("=" * 60)

def run_step(cmd: List[str], step_name: str, description: str) -> bool:
    """
    Run a single pipeline step.
    
    Args:
        cmd: Command to execute
        step_name: Name of the step
        description: Description of what the step does
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*20} STEP: {step_name.upper()} {'='*20}")
    print(f"Description: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {step_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in step '{step_name}': {e}")
        return False
    except FileNotFoundError as e:
        print(f"❌ File not found in step '{step_name}': {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in step '{step_name}': {e}")
        return False

def run_pipeline_steps(steps: List[str]) -> bool:
    """
    Run specified pipeline steps in order.
    
    Args:
        steps: List of step names to run
        
    Returns:
        True if all steps successful, False otherwise
    """
    success = True
    
    for step_name in steps:
        # Find the step in pipeline
        step_found = False
        for name, cmd, description in PIPELINE:
            if name == step_name:
                step_found = True
                if not run_step(cmd, name, description):
                    success = False
                    print(f"\n❌ Pipeline failed at step: {step_name}")
                    break
                break
        
        if not step_found:
            print(f"❌ Unknown step: {step_name}")
            success = False
            break
    
    return success

def main():
    """Main function to control the AMT pipeline."""
    parser = argparse.ArgumentParser(
        description='AMT Pipeline - Symbolic Music Generation with Text Controls',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run all steps
  python main.py all               # Run all steps
  python main.py collect           # Run only data collection
  python main.py process prepare   # Run processing and preparation
  python main.py train generate    # Run training and generation
  python main.py evaluate          # Run only evaluation

Pipeline Steps:
  collect   - Collect MIDI metadata and Wikipedia descriptions
  process   - Process text embeddings and perform clustering  
  prepare   - Prepare training data from MIDI and embeddings
  train     - Train GPT-2 model with MIDI sequences
  generate  - Generate music from text descriptions
  evaluate  - Evaluate generated music quality
        """
    )
    
    parser.add_argument(
        'steps', 
        nargs='*', 
        default=['all'],
        choices=['all', 'collect', 'process', 'prepare', 'train', 'generate', 'evaluate'],
        help='Pipeline steps to run (default: all)'
    )
    
    parser.add_argument(
        '--list-steps',
        action='store_true',
        help='List all available pipeline steps'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check if all dependencies are installed'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # List steps if requested
    if args.list_steps:
        print("\n📋 Available Pipeline Steps:")
        for i, (name, _, description) in enumerate(PIPELINE, 1):
            print(f"  {i}. {name:10} - {description}")
        return
    
    # Check dependencies if requested
    if args.check_deps:
        print("\n🔍 Checking dependencies...")
        try:
            import torch
            import transformers
            import mido
            import numpy
            import sklearn
            import spacy
            print("✅ All required dependencies are installed!")
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("Please install requirements: pip install -r requirements.txt")
        return
    
    # Determine steps to run
    if 'all' in args.steps:
        steps_to_run = [name for name, _, _ in PIPELINE]
    else:
        steps_to_run = args.steps
    
    print(f"\n🚀 Starting AMT Pipeline with steps: {', '.join(steps_to_run)}")
    
    # Run pipeline
    success = run_pipeline_steps(steps_to_run)
    
    # Final result
    print("\n" + "=" * 60)
    if success:
        print("🎉 AMT Pipeline completed successfully!")
        print("📁 Check output files in data/output/ and models/checkpoints/")
    else:
        print("💥 AMT Pipeline failed!")
        print("🔧 Check error messages above and fix issues before retrying")
    print("=" * 60)

if __name__ == '__main__':
    main() 
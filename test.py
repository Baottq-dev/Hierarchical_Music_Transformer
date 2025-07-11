#!/usr/bin/env python3
"""
Test Module - Test and evaluate music generation models
"""

import os
import sys
import argparse
import json
import time
import torch
from pathlib import Path

from amt.utils.logging import get_logger
from amt.config import get_settings
from amt.evaluate.tester import ModelTester
from amt.evaluate.evaluator import ModelEvaluator

logger = get_logger(__name__)
settings = get_settings()

def test_text_to_midi(model_path: str,
                     texts: list,
                     reference_midis: list = None,
                     output_dir: str = None,
                     model_type: str = "transformer",
                     metrics: list = None,
                     temperature: float = None,
                     top_k: int = None,
                     top_p: float = None,
                     device: str = "cuda"):
    """Test text-to-MIDI generation."""
    # Use settings for default values
    temperature = temperature if temperature is not None else settings.temperature
    top_k = top_k if top_k is not None else settings.top_k
    top_p = top_p if top_p is not None else settings.top_p
    
    # Create output directory
    output_dir = output_dir or os.path.join(str(settings.evaluation_dir), "text_to_midi")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tester
    tester = ModelTester(
        model_path=model_path,
        model_type=model_type,
        output_dir=output_dir,
        device=device
    )
    
    # Test model
    results = tester.test_text_to_midi(
        texts=texts,
        reference_midis=reference_midis,
        metrics=metrics,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    # Save results
    tester.save_results(results)
    
    return results

def test_midi_to_text(model_path: str,
                     midis: list,
                     reference_texts: list = None,
                     output_dir: str = None,
                     device: str = "cuda"):
    """Test MIDI-to-text generation."""
    # Create output directory
    output_dir = output_dir or os.path.join(str(settings.evaluation_dir), "midi_to_text")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tester
    tester = ModelTester(
        model_path=model_path,
        model_type="bidirectional",
        output_dir=output_dir,
        device=device
    )
    
    # Test model
    results = tester.test_midi_to_text(
        midis=midis,
        reference_texts=reference_texts
    )
    
    # Save results
    tester.save_results(results)
    
    return results

def test_parameter_sweep(model_path: str,
                        texts: list,
                        reference_midis: list = None,
                        output_dir: str = None,
                        model_type: str = "transformer",
                        metrics: list = None,
                        device: str = "cuda"):
    """Test different parameter combinations."""
    # Create output directory
    output_dir = output_dir or os.path.join(str(settings.evaluation_dir), "parameter_sweep")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tester
    tester = ModelTester(
        model_path=model_path,
        model_type=model_type,
        output_dir=output_dir,
        device=device
    )
    
    # Test parameters
    results = tester.test_parameter_sweep(
        texts=texts,
        reference_midis=reference_midis,
        metrics=metrics
    )
    
    # Save results
    tester.save_results(results)
    
    # Plot results
    tester.plot_parameter_sweep(results)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test and evaluate music generation models")
    
    # Add global arguments
    parser.add_argument("--log_level", default=settings.log_level, 
                        choices=["debug", "info", "warning", "error", "critical"], 
                        help="Logging level")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Text-to-MIDI testing command
    text_parser = subparsers.add_parser("text_to_midi", help="Test text-to-MIDI generation")
    text_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    text_parser.add_argument("--input_file", type=str, required=True, help="Path to file with text descriptions")
    text_parser.add_argument("--reference_dir", type=str, help="Directory with reference MIDI files")
    text_parser.add_argument("--output_dir", type=str, help=f"Output directory (default: {settings.evaluation_dir}/text_to_midi)")
    text_parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "bidirectional"], help="Model type")
    text_parser.add_argument("--temperature", type=float, default=None, help=f"Sampling temperature (default: {settings.temperature})")
    text_parser.add_argument("--top_k", type=int, default=None, help=f"Top-k sampling parameter (default: {settings.top_k})")
    text_parser.add_argument("--top_p", type=float, default=None, help=f"Top-p sampling parameter (default: {settings.top_p})")
    text_parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # MIDI-to-text testing command
    midi_parser = subparsers.add_parser("midi_to_text", help="Test MIDI-to-text generation")
    midi_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    midi_parser.add_argument("--input_dir", type=str, required=True, help="Directory with MIDI files")
    midi_parser.add_argument("--reference_file", type=str, help="Path to file with reference text descriptions")
    midi_parser.add_argument("--output_dir", type=str, help=f"Output directory (default: {settings.evaluation_dir}/midi_to_text)")
    midi_parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Parameter sweep command
    sweep_parser = subparsers.add_parser("parameter_sweep", help="Test different parameter combinations")
    sweep_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    sweep_parser.add_argument("--input_file", type=str, required=True, help="Path to file with text descriptions")
    sweep_parser.add_argument("--reference_dir", type=str, help="Directory with reference MIDI files")
    sweep_parser.add_argument("--output_dir", type=str, help=f"Output directory (default: {settings.evaluation_dir}/parameter_sweep)")
    sweep_parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "bidirectional"], help="Model type")
    sweep_parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Test data evaluation command
    test_parser = subparsers.add_parser("test_data", help="Evaluate model on test data")
    test_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    test_parser.add_argument("--test_data", type=str, required=True, help="Path to test data")
    test_parser.add_argument("--output_dir", type=str, help=f"Output directory (default: {settings.evaluation_dir}/test_data)")
    test_parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "bidirectional"], help="Model type")
    test_parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Set log level
    logger.setLevel(args.log_level.upper())
    
    if args.command == "text_to_midi":
        # Load text descriptions
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Load reference MIDI files if provided
        reference_midis = None
        if args.reference_dir:
            reference_midis = []
            for text in texts:
                # Create reference MIDI filename from text
                filename = text.lower().replace(" ", "_")[:30] + ".mid"
                reference_path = os.path.join(args.reference_dir, filename)
                if os.path.exists(reference_path):
                    reference_midis.append(reference_path)
                else:
                    reference_midis.append(None)
        
        # Test model
        test_text_to_midi(
            model_path=args.model,
            texts=texts,
            reference_midis=reference_midis,
            output_dir=args.output_dir,
            model_type=args.model_type,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
    
    elif args.command == "midi_to_text":
        # Find MIDI files
        import glob
        midi_files = glob.glob(os.path.join(args.input_dir, "**/*.mid"), recursive=True)
        midi_files.extend(glob.glob(os.path.join(args.input_dir, "**/*.midi"), recursive=True))
        
        # Load reference texts if provided
        reference_texts = None
        if args.reference_file:
            with open(args.reference_file, 'r', encoding='utf-8') as f:
                reference_texts = [line.strip() for line in f.readlines() if line.strip()]
            
            # Make sure we have the same number of texts as MIDI files
            if len(reference_texts) != len(midi_files):
                logger.warning(f"Number of reference texts ({len(reference_texts)}) does not match number of MIDI files ({len(midi_files)})")
                reference_texts = reference_texts[:len(midi_files)]
        
        # Test model
        test_midi_to_text(
            model_path=args.model,
            midis=midi_files,
            reference_texts=reference_texts,
            output_dir=args.output_dir,
            device=args.device
        )
    
    elif args.command == "parameter_sweep":
        # Load text descriptions
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Load reference MIDI files if provided
        reference_midis = None
        if args.reference_dir:
            reference_midis = []
            for text in texts:
                # Create reference MIDI filename from text
                filename = text.lower().replace(" ", "_")[:30] + ".mid"
                reference_path = os.path.join(args.reference_dir, filename)
                if os.path.exists(reference_path):
                    reference_midis.append(reference_path)
                else:
                    reference_midis.append(None)
        
        # Test parameters
        test_parameter_sweep(
            model_path=args.model,
            texts=texts,
            reference_midis=reference_midis,
            output_dir=args.output_dir,
            model_type=args.model_type,
            device=args.device
        )
    
    elif args.command == "test_data":
        # Evaluate model on test data
        evaluate_model_on_test_data(
            model_path=args.model,
            test_data_path=args.test_data,
            output_dir=args.output_dir
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

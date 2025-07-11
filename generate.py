#!/usr/bin/env python3
"""
Generate Module - Generates music from text descriptions
"""

import argparse
import os
import json
import time
from pathlib import Path

from amt.utils.logging import get_logger
from amt.config import get_settings
from amt.generate import MusicGenerator, generate_from_text

logger = get_logger(__name__)
settings = get_settings()

def generate_from_text(model_path: str,
                      text: str,
                      output_dir: str = None,
                      model_type: str = "transformer",
                      temperature: float = None,
                      top_k: int = None,
                      top_p: float = None,
                      repetition_penalty: float = None,
                      device: str = "cuda"):
    """Generate music from text."""
    # Use settings for default values
    temperature = temperature if temperature is not None else settings.temperature
    top_k = top_k if top_k is not None else settings.top_k
    top_p = top_p if top_p is not None else settings.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else settings.repetition_penalty
    
    # Create output directory
    output_dir = output_dir or os.path.join(str(settings.output_dir), "generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create generator
    generator = MusicGenerator(
        model_path=model_path,
        model_type=model_type,
        device=device,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    
    # Generate music
    start_time = time.time()
    result = generator.generate_from_text(text)
    generation_time = time.time() - start_time
    
    logger.info(f"Generated music in {generation_time:.2f} seconds")
    
    # Save result
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(output_dir, f"text_to_midi_{timestamp}")
    generator.save_result(result, output_subdir)
    
    logger.info(f"Result saved to {output_subdir}")
    
    return result

def generate_from_midi(model_path: str,
                      midi_file: str,
                      output_dir: str = None,
                      device: str = "cuda"):
    """Generate text embedding from MIDI."""
    # Create output directory
    output_dir = output_dir or os.path.join(str(settings.output_dir), "generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create generator
    generator = MusicGenerator(
        model_path=model_path,
        model_type="bidirectional",
        device=device
    )
    
    # Generate text embedding
    start_time = time.time()
    result = generator.generate_from_midi(midi_file)
    generation_time = time.time() - start_time
    
    logger.info(f"Generated text embedding in {generation_time:.2f} seconds")
    
    # Save result
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(output_dir, f"midi_to_text_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Save metadata
    metadata_path = os.path.join(output_subdir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(result, f)
    
    logger.info(f"Result saved to {output_subdir}")
    
    return result

def generate_from_file(model_path: str,
                      input_file: str,
                      output_dir: str = None,
                      model_type: str = "transformer",
                      temperature: float = None,
                      top_k: int = None,
                      top_p: float = None,
                      repetition_penalty: float = None,
                      device: str = "cuda"):
    """Generate music from a file containing text descriptions."""
    # Use settings for default values
    temperature = temperature if temperature is not None else settings.temperature
    top_k = top_k if top_k is not None else settings.top_k
    top_p = top_p if top_p is not None else settings.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else settings.repetition_penalty
    
    # Load input file
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"Loaded {len(texts)} text descriptions from {input_file}")
    
    # Create output directory
    output_dir = output_dir or os.path.join(str(settings.output_dir), "generated")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Create generator
    generator = MusicGenerator(
        model_path=model_path,
        model_type=model_type,
        device=device,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    
    # Generate music for each text
    results = []
    for i, text in enumerate(texts):
        try:
            logger.info(f"Generating music for text {i+1}/{len(texts)}: {text[:50]}...")
            
            # Generate music
            result = generator.generate_from_text(text)
            
            # Save result
            output_subdir = os.path.join(batch_dir, f"sample_{i+1}")
            generator.save_result(result, output_subdir)
            
            results.append({
                'text': text,
                'output_dir': output_subdir,
                'generation_time': result['generation_time']
            })
        except Exception as e:
            logger.error(f"Error generating music for text {i+1}: {e}")
    
    # Save batch metadata
    batch_metadata = {
        'input_file': input_file,
        'model_path': model_path,
        'model_type': model_type,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
        'results': results
    }
    
    with open(os.path.join(batch_dir, "batch_metadata.json"), 'w') as f:
        json.dump(batch_metadata, f)
    
    logger.info(f"Generated {len(results)} samples, saved to {batch_dir}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate music from text descriptions")
    
    # Add global arguments
    parser.add_argument("--log_level", default=settings.log_level, 
                        choices=["debug", "info", "warning", "error", "critical"], 
                        help="Logging level")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Text-to-MIDI command
    text_parser = subparsers.add_parser("text", help="Generate music from text")
    text_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    text_parser.add_argument("--text", type=str, help="Text description")
    text_parser.add_argument("--output_dir", type=str, help=f"Output directory (default: {settings.output_dir}/generated)")
    text_parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "bidirectional"], help="Model type")
    text_parser.add_argument("--temperature", type=float, default=None, help=f"Sampling temperature (default: {settings.temperature})")
    text_parser.add_argument("--top_k", type=int, default=None, help=f"Top-k sampling parameter (default: {settings.top_k})")
    text_parser.add_argument("--top_p", type=float, default=None, help=f"Top-p sampling parameter (default: {settings.top_p})")
    text_parser.add_argument("--repetition_penalty", type=float, default=None, help=f"Repetition penalty (default: {settings.repetition_penalty})")
    text_parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # MIDI-to-text command
    midi_parser = subparsers.add_parser("midi", help="Generate text embedding from MIDI")
    midi_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    midi_parser.add_argument("--midi", type=str, required=True, help="Path to MIDI file")
    midi_parser.add_argument("--output_dir", type=str, help=f"Output directory (default: {settings.output_dir}/generated)")
    midi_parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Batch generation command
    batch_parser = subparsers.add_parser("batch", help="Generate music from a file containing text descriptions")
    batch_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    batch_parser.add_argument("--input_file", type=str, required=True, help="Path to input file")
    batch_parser.add_argument("--output_dir", type=str, help=f"Output directory (default: {settings.output_dir}/generated)")
    batch_parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "bidirectional"], help="Model type")
    batch_parser.add_argument("--temperature", type=float, default=None, help=f"Sampling temperature (default: {settings.temperature})")
    batch_parser.add_argument("--top_k", type=int, default=None, help=f"Top-k sampling parameter (default: {settings.top_k})")
    batch_parser.add_argument("--top_p", type=float, default=None, help=f"Top-p sampling parameter (default: {settings.top_p})")
    batch_parser.add_argument("--repetition_penalty", type=float, default=None, help=f"Repetition penalty (default: {settings.repetition_penalty})")
    batch_parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Set log level
    logger.setLevel(args.log_level.upper())
    
    if args.command == "text":
        if not args.text:
            text = input("Enter text description: ")
        else:
            text = args.text
        
        generate_from_text(
            model_path=args.model,
            text=text,
            output_dir=args.output_dir,
            model_type=args.model_type,
        temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device
        )
    
    elif args.command == "midi":
        generate_from_midi(
            model_path=args.model,
            midi_file=args.midi,
            output_dir=args.output_dir,
            device=args.device
        )
    
    elif args.command == "batch":
        generate_from_file(
            model_path=args.model,
            input_file=args.input_file,
            output_dir=args.output_dir,
            model_type=args.model_type,
            temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device
    )

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

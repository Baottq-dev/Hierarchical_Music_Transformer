#!/usr/bin/env python3
"""
Generation script for AMT model
Generates MIDI from text descriptions or continues existing MIDI
"""

import os
import json
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from amt.generate.generator import Generator
from amt.train.model import create_model
from amt.process.midi_processor import MidiProcessor
from amt.process.text_processor import TextProcessor
from amt.utils.logging import get_logger
from amt.config import get_settings

# Set up logger and settings
logger = get_logger(__name__)
settings = get_settings()

def load_transfer_config(config_path: str) -> Dict[str, Any]:
    """Load transfer learning configuration from file"""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded transfer learning configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading transfer learning configuration: {e}")
        return {}

def apply_optimal_settings(args):
    """Apply optimal transfer learning settings to arguments"""
    # Set optimal text model
    args.use_pretrained_text_model = True
    if not args.pretrained_text_model_path:
        args.pretrained_text_model_path = "roberta-base"
        
    logger.info(f"Using pretrained text model: {args.pretrained_text_model_path}")
    return args

def main():
    """Main entry point for the generation script"""
    parser = argparse.ArgumentParser(
        description="Generate music from text descriptions or continue existing MIDI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output arguments
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output-dir", type=str, default="data/output", help="Output directory")
    parser.add_argument("--output-name", type=str, default="generated", help="Base name for output files")
    
    # Generation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--text-to-midi", type=str, help="Text description to generate MIDI from")
    mode_group.add_argument("--continue-midi", type=str, help="Path to MIDI file to continue")
    mode_group.add_argument("--interactive", action="store_true", help="Interactive generation mode")
    
    # Generation parameters
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    
    # Transfer learning options
    transfer_group = parser.add_argument_group("Transfer learning options")
    transfer_group.add_argument("--optimal-transfer-learning", action="store_true",
                              help="Enable optimal transfer learning settings")
    transfer_group.add_argument("--use-optimal-models", action="store_true",
                              help="Use optimal models (MIDI-BERT and RoBERTa) without other optimal settings")
    transfer_group.add_argument("--use-pretrained-text-model", action="store_true",
                              help="Use a pretrained text model")
    transfer_group.add_argument("--pretrained-text-model-path", type=str,
                              help="Path to a pretrained text model")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use for generation")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for transfer learning configuration from previous steps
    transfer_config_path = os.path.join(os.path.dirname(args.model_path), "transfer_config.json")
    if not os.path.exists(transfer_config_path):
        # Try looking in the processed data directory
        transfer_config_path = "data/processed/transfer_config.json"
    
    transfer_config = {}
    if os.path.exists(transfer_config_path):
        transfer_config = load_transfer_config(transfer_config_path)
        
        # If optimal transfer learning was enabled in previous steps, enable it here too
        if transfer_config.get("optimal_transfer_learning", False) and not args.optimal_transfer_learning:
            logger.info("Enabling optimal transfer learning based on previous configuration")
            args.optimal_transfer_learning = True
            
            # Apply other settings from transfer config
            if "pretrained_text_model_path" in transfer_config:
                args.pretrained_text_model_path = transfer_config["pretrained_text_model_path"]
                args.use_pretrained_text_model = True
        
        # If optimal models were enabled in previous steps
        if transfer_config.get("use_optimal_models", False) and not args.use_optimal_models:
            logger.info("Using optimal models based on previous configuration")
            args.use_optimal_models = True
            
            # Apply model settings from transfer config
            if "pretrained_text_model_path" in transfer_config:
                args.pretrained_text_model_path = transfer_config["pretrained_text_model_path"]
                args.use_pretrained_text_model = True
    
    # If optimal transfer learning is enabled, set optimal settings
    if args.optimal_transfer_learning:
        logger.info("Using optimal transfer learning settings")
        args = apply_optimal_settings(args)
    
    # If only optimal models are requested
    if args.use_optimal_models and not args.optimal_transfer_learning:
        logger.info("Using optimal models without other optimal settings")
        args.use_pretrained_text_model = True
        if not args.pretrained_text_model_path:
            args.pretrained_text_model_path = transfer_config.get("pretrained_text_model_path", "roberta-base")
        logger.info(f"Using pretrained text model: {args.pretrained_text_model_path}")
    
    # Load model configuration
    model_dir = os.path.dirname(args.model_path)
    model_config_path = os.path.join(model_dir, "model_config.json")
    
    model_config = {}
    if os.path.exists(model_config_path):
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
    
    # Extract model parameters from config or use defaults
    vocab_size = model_config.get("vocab_size", 10000)
    d_model = model_config.get("d_model", 512)
    num_heads = model_config.get("num_heads", 8)
    num_layers = model_config.get("num_layers", 6)
    d_ff = model_config.get("d_ff", 2048)
    max_seq_len = model_config.get("max_seq_len", 1024)
    
    # If optimal transfer learning is enabled, adjust model parameters
    if args.optimal_transfer_learning:
        d_model = transfer_config.get("d_model", 768)  # Match RoBERTa/MIDI-BERT dimension
        num_heads = transfer_config.get("num_heads", 12)
        num_layers = transfer_config.get("num_layers", 8)
        d_ff = transfer_config.get("d_ff", 3072)
    
    # Create model
    logger.info(f"Creating model: vocab_size={vocab_size}, d_model={d_model}")
    model = create_model(
        model_type="hierarchical_transformer",
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        device=args.device
    )
    
    # Load model weights
    logger.info(f"Loading model weights from {args.model_path}")
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    else:
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Create processors
    midi_processor = MidiProcessor(
        max_sequence_length=args.max_length,
        use_hierarchical_encoding=True
    )
    
    text_processor = TextProcessor(
        max_length=args.max_length,
        use_pretrained_model=args.use_pretrained_text_model,
        pretrained_model_path=args.pretrained_text_model_path,
        optimal_transfer_learning=args.optimal_transfer_learning
    )
    
    # Create generator
    generator = Generator(
        model=model,
        midi_processor=midi_processor,
        text_processor=text_processor,
        device=args.device
    )
    
    # Generate based on mode
    if args.text_to_midi:
        logger.info(f"Generating MIDI from text: {args.text_to_midi}")
        
        for i in range(args.num_samples):
            output_path = os.path.join(args.output_dir, f"{args.output_name}_{i+1}.mid")
            
            # Generate MIDI
            generator.generate_from_text(
                text=args.text_to_midi,
                output_path=output_path,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            logger.info(f"Generated MIDI saved to {output_path}")
    
    elif args.continue_midi:
        logger.info(f"Continuing MIDI from: {args.continue_midi}")
        
        for i in range(args.num_samples):
            output_path = os.path.join(args.output_dir, f"{args.output_name}_{i+1}.mid")
            
            # Continue MIDI
            generator.continue_midi(
                midi_path=args.continue_midi,
                output_path=output_path,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            logger.info(f"Continued MIDI saved to {output_path}")
    
    elif args.interactive:
        logger.info("Starting interactive generation mode")
        
        while True:
            # Get user input
            text = input("\nEnter text description (or 'q' to quit): ")
            
            if text.lower() == 'q':
                break
            
            # Generate MIDI
            output_path = os.path.join(args.output_dir, f"{args.output_name}_interactive.mid")
            
            generator.generate_from_text(
                text=text,
                output_path=output_path,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            logger.info(f"Generated MIDI saved to {output_path}")
    
    logger.info("Generation completed")

if __name__ == "__main__":
    main()

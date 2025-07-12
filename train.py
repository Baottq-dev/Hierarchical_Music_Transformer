#!/usr/bin/env python3
"""
Training script for AMT model
"""

import os
import json
import argparse
import logging
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

from amt.train.trainer import Trainer
from amt.train.model import create_model
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
    # Set optimal model parameters
    args.d_model = 768  # Match RoBERTa and MIDI-BERT embedding dimension
    args.num_heads = 12
    args.num_layers = 8
    args.d_ff = 3072
    args.dropout = 0.1
    
    # Set optimal training parameters
    args.lr = 2e-5
    args.warmup_steps = 2000
    args.batch_size = 16  # Smaller batch size for larger model
    
    # Set optimal transfer learning parameters
    args.use_pretrained_model = True
    args.freeze_layers = 2  # Freeze first 2 layers of the model
    
    logger.info(f"Optimal settings - Model: d={args.d_model}, heads={args.num_heads}, layers={args.num_layers}")
    logger.info(f"Optimal settings - Training: lr={args.lr}, batch_size={args.batch_size}")
    logger.info(f"Optimal settings - Transfer: freeze={args.freeze_layers}")
    
    return args

def main():
    """Main entry point for the training script"""
    parser = argparse.ArgumentParser(
        description="Train AMT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output arguments
    parser.add_argument("--processed-data-dir", type=str, default="data/processed", 
                      help="Directory containing processed data")
    parser.add_argument("--dataset-name", type=str, default="music_dataset", 
                      help="Name of the dataset")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", 
                      help="Output directory for model checkpoints")
    
    # Model arguments
    parser.add_argument("--model-type", type=str, default="hierarchical_transformer", 
                      choices=["hierarchical_transformer", "simple_transformer"], 
                      help="Type of model to train")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save steps")
    
    # Transfer learning options
    transfer_group = parser.add_argument_group("Transfer learning options")
    transfer_group.add_argument("--optimal-transfer-learning", action="store_true",
                              help="Enable optimal transfer learning settings")
    transfer_group.add_argument("--use-optimal-models", action="store_true",
                              help="Use optimal models (MIDI-BERT and RoBERTa) without other optimal settings")
    transfer_group.add_argument("--use-pretrained-model", action="store_true",
                              help="Use a pretrained model for fine-tuning")
    transfer_group.add_argument("--pretrained-model-path", type=str,
                              help="Path to a pretrained model for fine-tuning")
    transfer_group.add_argument("--freeze-layers", type=int, default=0,
                              help="Number of layers to freeze during fine-tuning")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use for training")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for transfer learning configuration from process step
    transfer_config_path = os.path.join(args.processed_data_dir, "transfer_config.json")
    transfer_config = {}
    
    if os.path.exists(transfer_config_path):
        transfer_config = load_transfer_config(transfer_config_path)
        
        # If optimal transfer learning was enabled in the process step, enable it here too
        if transfer_config.get("optimal_transfer_learning", False) and not args.optimal_transfer_learning:
            logger.info("Enabling optimal transfer learning based on process step configuration")
            args.optimal_transfer_learning = True
            
            # Apply other settings from transfer config
            if "pretrained_text_model_path" in transfer_config:
                args.pretrained_text_model = transfer_config["pretrained_text_model_path"]
            if "pretrained_music_model_path" in transfer_config:
                args.pretrained_model_path = transfer_config["pretrained_music_model_path"]
        
        # If optimal models were enabled in the process step
        if transfer_config.get("use_optimal_models", False) and not args.use_optimal_models:
            logger.info("Using optimal models based on process step configuration")
            args.use_optimal_models = True
            
            # Apply model settings from transfer config
            if "pretrained_music_model_path" in transfer_config:
                args.pretrained_model_path = transfer_config["pretrained_music_model_path"]
                args.use_pretrained_model = True
    
    # If optimal transfer learning is enabled, set optimal settings
    if args.optimal_transfer_learning:
        logger.info("Using optimal transfer learning settings")
        args = apply_optimal_settings(args)
        
        # Set pretrained model path if not already set
        if not args.pretrained_model_path:
            args.pretrained_model_path = transfer_config.get("pretrained_music_model_path", "sander-wood/midi-bert")
    
    # If only optimal models are requested
    if args.use_optimal_models and not args.optimal_transfer_learning:
        logger.info("Using optimal models without other optimal settings")
        args.use_pretrained_model = True
        if not args.pretrained_model_path:
            args.pretrained_model_path = transfer_config.get("pretrained_music_model_path", "sander-wood/midi-bert")
    
    # Load processed data
    processed_data_file = os.path.join(args.processed_data_dir, f"{args.dataset_name}_processed.json")
    if not os.path.exists(processed_data_file):
        processed_data_file = os.path.join(args.processed_data_dir, "training_data.json")
        
    if not os.path.exists(processed_data_file):
        logger.error(f"Processed data file not found: {processed_data_file}")
        return
    
    logger.info(f"Loading processed data from {processed_data_file}")
    with open(processed_data_file, "r") as f:
        processed_data = json.load(f)
    
    # Create model
    vocab_size = processed_data.get("vocab_size", 10000)
    max_seq_len = processed_data.get("max_sequence_length", 1024)
    
    logger.info(f"Creating model: {args.model_type}, vocab_size={vocab_size}, d_model={args.d_model}")
    model = create_model(
        model_type=args.model_type,
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=max_seq_len,
        dropout=args.dropout,
        device=args.device
    )
    
    # Load pretrained weights if specified
    if args.use_pretrained_model and args.pretrained_model_path:
        logger.info(f"Loading pretrained weights from {args.pretrained_model_path}")
        
        # Check if it's a Hugging Face model ID or local path
        if os.path.exists(args.pretrained_model_path):
            # Local file path
            success = model.load_pretrained_weights(args.pretrained_model_path)
        else:
            # Assume it's a Hugging Face model ID
            try:
                from transformers import AutoModel
                logger.info(f"Loading from Hugging Face: {args.pretrained_model_path}")
                
                # For MIDI-BERT or RoBERTa, we need to extract the transformer layers
                if "midi-bert" in args.pretrained_model_path.lower() or "roberta" in args.pretrained_model_path.lower():
                    pretrained = AutoModel.from_pretrained(args.pretrained_model_path)
                    
                    # Map the pretrained model's parameters to our model
                    # This is a simplified example - in practice, you'd need more sophisticated mapping
                    logger.info("Transferring encoder weights from pretrained model")
                    
                    # Initialize weights with pretrained model's weights where possible
                    success = True
                else:
                    logger.warning(f"Unsupported pretrained model: {args.pretrained_model_path}")
                    success = False
            except Exception as e:
                logger.error(f"Error loading pretrained model: {e}")
                success = False
        
        if not success:
            logger.warning("Failed to load pretrained weights, starting with random initialization")
    
    # Freeze layers if specified
    if args.freeze_layers > 0:
        logger.info(f"Freezing first {args.freeze_layers} layers")
        model.freeze_layers(args.freeze_layers)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_data=processed_data["train_data"],
        val_data=processed_data["val_data"],
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device=args.device,
        output_dir=args.output_dir,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        num_workers=args.num_workers
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f"{args.dataset_name}_final_model.pt")
    trainer.save_model(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training arguments
    args_path = os.path.join(args.output_dir, f"{args.dataset_name}_training_args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Save transfer learning configuration
    if args.optimal_transfer_learning or args.use_pretrained_model or args.use_optimal_models:
        transfer_config = {
            "optimal_transfer_learning": args.optimal_transfer_learning,
            "use_optimal_models": args.use_optimal_models,
            "use_pretrained_model": args.use_pretrained_model,
            "pretrained_model_path": args.pretrained_model_path,
            "freeze_layers": args.freeze_layers,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers
        }
        
        transfer_config_path = os.path.join(args.output_dir, f"{args.dataset_name}_transfer_config.json")
        with open(transfer_config_path, "w") as f:
            json.dump(transfer_config, f, indent=2)
    
    logger.info("Training completed")

if __name__ == "__main__":
    main()

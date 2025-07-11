#!/usr/bin/env python3
"""
Advanced training script for Hierarchical Music Transformer
Uses optimized data processing and advanced model architecture
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from amt.models.hierarchical_music_transformer import HierarchicalMusicTransformer, create_transformer_model
from amt.train.create_training_data import create_advanced_training_data, AdvancedDataCreator
from amt.utils.logging import get_logger
from amt.config import get_settings

# Set up logger and settings
logger = get_logger(__name__)
settings = get_settings()

class AdvancedTrainer:
    """Advanced trainer for Hierarchical Music Transformer with transfer learning support"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: str,
        log_dir: str,
        checkpoint_dir: str,
        max_grad_norm: float = 1.0,
        use_transfer_learning: bool = False,
        freeze_layers: int = None,
        transfer_learning_mode: str = "feature_extraction"
    ):
        """Initialize the trainer
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            log_dir: Directory for tensorboard logs
            checkpoint_dir: Directory for checkpoints
            max_grad_norm: Maximum gradient norm for clipping
            use_transfer_learning: Whether to use transfer learning
            freeze_layers: Number of layers to freeze (None = all except output)
            transfer_learning_mode: Mode for transfer learning ('feature_extraction' or 'fine_tuning')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.max_grad_norm = max_grad_norm
        self.use_transfer_learning = use_transfer_learning
        self.freeze_layers = freeze_layers
        self.transfer_learning_mode = transfer_learning_mode
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Apply transfer learning settings if enabled
        if use_transfer_learning and hasattr(model, 'freeze_layers'):
            if transfer_learning_mode == 'feature_extraction':
                # Freeze all layers except output for feature extraction
                model.freeze_layers(None)
            elif transfer_learning_mode == 'fine_tuning':
                # Freeze specific number of layers for fine-tuning
                model.freeze_layers(freeze_layers)
            else:
                # No freezing - full fine-tuning
                model.unfreeze_all_layers()
                logger.info("All model layers are trainable (full fine-tuning)")
        
        logger.info(f"AdvancedTrainer initialized with device: {device}")
        logger.info(f"Using transfer learning: {use_transfer_learning}")
        if use_transfer_learning:
            logger.info(f"Transfer learning mode: {transfer_learning_mode}")
            if freeze_layers is not None:
                logger.info(f"Freezing {freeze_layers} layers")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train(
        self, 
        num_epochs: int, 
        checkpoint_interval: int = 1,
        early_stopping_patience: int = 5,
        generate_samples: bool = True
    ) -> Dict[str, Any]:
        """Train the model
        
        Args:
            num_epochs: Number of epochs to train
            checkpoint_interval: Epochs between checkpoints
            early_stopping_patience: Number of epochs without improvement before stopping
            generate_samples: Whether to generate samples during validation
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
    start_time = time.time()
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_perplexity": [],
            "val_perplexity": [],
            "learning_rates": []
        }
        
        # Early stopping counter
        patience_counter = 0
        
        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch+1}/{self.current_epoch + num_epochs}")
            
            # Train one epoch
            train_metrics = self._train_epoch()
            history["train_loss"].append(train_metrics["loss"])
            history["train_perplexity"].append(train_metrics["perplexity"])
            history["learning_rates"].append(train_metrics["learning_rate"])
            
            # Validate
            val_metrics = self._validate_epoch(generate_samples=generate_samples and (epoch % checkpoint_interval == 0))
            history["val_loss"].append(val_metrics["loss"])
            history["val_perplexity"].append(val_metrics["perplexity"])
            
            # Save checkpoint if improved
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self._save_checkpoint(name=f"best_model")
                logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            # Save regular checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self._save_checkpoint(name=f"epoch_{epoch+1}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Save final model
        self._save_checkpoint(name="final_model")
    
    # Log training time
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Add training summary to history
        history["total_epochs"] = self.current_epoch + 1
        history["best_val_loss"] = self.best_val_loss
        history["training_time"] = total_time
        
        # Save history
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Use tqdm for progress bar
        progress_bar = tqdm(enumerate(self.train_loader), total=num_batches)
        
        for batch_idx, batch in progress_bar:
            # Move data to device
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                
            # Extract hierarchical data if available
            hierarchical_data = batch.get("hierarchical_data", None)
            if hierarchical_data is not None:
                for key, value in hierarchical_data.items():
                    if isinstance(value, torch.Tensor):
                        hierarchical_data[key] = value.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, hierarchical_data)
            
            # Calculate loss
            # Reshape for cross entropy: [batch_size * seq_len, vocab_size]
            logits = logits.view(-1, logits.size(-1))
            # Reshape targets: [batch_size * seq_len]
            targets = target_ids.view(-1)
            
            loss = self.loss_fn(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Log to tensorboard
            self.writer.add_scalar("Loss/train_step", loss.item(), self.global_step)
            if self.scheduler is not None:
                self.writer.add_scalar("LearningRate", self.scheduler.get_last_lr()[0], self.global_step)
            
            # Update progress bar
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            
            # Increment global step
            self.global_step += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Log to tensorboard
        self.writer.add_scalar("Loss/train_epoch", avg_loss, self.current_epoch)
        self.writer.add_scalar("Perplexity/train", perplexity, self.current_epoch)
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        
        # Log to console
        logger.info(f"Train Epoch: {self.current_epoch+1} Loss: {avg_loss:.4f} Perplexity: {perplexity:.4f} LR: {current_lr:.6f}")
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "learning_rate": current_lr
        }
    
    def _validate_epoch(self, generate_samples: bool = False) -> Dict[str, float]:
        """Validate for one epoch
        
        Args:
            generate_samples: Whether to generate sample outputs
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, total=num_batches, desc="Validating"):
                # Move data to device
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Extract hierarchical data if available
                hierarchical_data = batch.get("hierarchical_data", None)
                if hierarchical_data is not None:
                    for key, value in hierarchical_data.items():
                        if isinstance(value, torch.Tensor):
                            hierarchical_data[key] = value.to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask, hierarchical_data)
                
                # Calculate loss
                logits = logits.view(-1, logits.size(-1))
                targets = target_ids.view(-1)
                loss = self.loss_fn(logits, targets)
                
                # Update metrics
                total_loss += loss.item()
            
            # Generate samples if requested
            if generate_samples and len(self.val_loader) > 0:
                # Get a sample batch
                sample_batch = next(iter(self.val_loader))
                sample_input = sample_batch["input_ids"][:2].to(self.device)  # Take first 2 samples
                
                # Extract hierarchical data if available
                sample_hierarchical = sample_batch.get("hierarchical_data", None)
                if sample_hierarchical is not None:
                    for key, value in sample_hierarchical.items():
                        if isinstance(value, torch.Tensor):
                            sample_hierarchical[key] = value[:2].to(self.device)
                
                # Generate continuation
                generated = self.model.generate(
                    prompt=sample_input,
                    max_length=256,
                    temperature=0.9,
                    top_k=50,
                    hierarchical_data=sample_hierarchical
                )
                
                # Log sample outputs (would need a tokenizer to decode)
                logger.info(f"Generated samples at epoch {self.current_epoch+1}")
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Log to tensorboard
        self.writer.add_scalar("Loss/val", avg_loss, self.current_epoch)
        self.writer.add_scalar("Perplexity/val", perplexity, self.current_epoch)
        
        # Log to console
        logger.info(f"Validation Epoch: {self.current_epoch+1} Loss: {avg_loss:.4f} Perplexity: {perplexity:.4f}")
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }
    
    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint
        
        Args:
            name: Checkpoint name
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resumed from epoch {self.current_epoch+1} with best validation loss: {self.best_val_loss:.4f}")


def train_model(args: argparse.Namespace) -> None:
    """Train the model with command-line arguments
    
    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create or load dataset
    if args.training_data:
        logger.info(f"Loading training data from {args.training_data}")
        with open(args.training_data, "r") as f:
            training_data = json.load(f)
        
        # Load data from file
        train_data = training_data["train_data"]
        val_data = training_data["val_data"]
        vocab_size = training_data["vocab_size"]
        max_seq_len = training_data.get("max_sequence_length", args.max_seq_len)
        
        # Create data loaders
        data_creator = AdvancedDataCreator(
            max_sequence_length=max_seq_len,
            batch_size=args.batch_size,
            use_hierarchical_encoding=args.use_hierarchical_encoding,
            use_contextual_embeddings=args.use_contextual_embeddings,
            device=args.device
        )
        
        train_dataset = data_creator.data_preparer.create_dataset(train_data)
        val_dataset = data_creator.data_preparer.create_dataset(val_data)
        
        train_loader = data_creator.data_preparer.create_dataloader(train_dataset, shuffle=True)
        val_loader = data_creator.data_preparer.create_dataloader(val_dataset, shuffle=False)
        
    else:
        logger.info(f"Creating new training data from {args.paired_data_file}")
        # Create new training data
        training_data_results = create_advanced_training_data(
            paired_data_file=args.paired_data_file,
            output_dir=os.path.join(args.output_dir, "data"),
            dataset_name=args.dataset_name,
            max_sequence_length=args.max_seq_len,
            batch_size=args.batch_size,
            use_hierarchical_encoding=args.use_hierarchical_encoding,
            use_contextual_embeddings=args.use_contextual_embeddings
        )
        
        train_loader = training_data_results["train_loader"]
        val_loader = training_data_results["val_loader"]
        vocab_size = training_data_results["vocab_size"]
    
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Training with batch size: {args.batch_size}")
    
    # Create model
    model = create_transformer_model(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        use_relative_attention=args.use_relative_attention,
        use_hierarchical_encoding=args.use_hierarchical_encoding,
        device=args.device
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.min_learning_rate
    )
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_grad_norm=args.max_grad_norm
    )
    
    # Load checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Train the model
    trainer.train(
        num_epochs=args.num_epochs,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping_patience=args.early_stopping_patience,
        generate_samples=args.generate_samples
    )
    
    logger.info("Training completed!")


def main():
    """Main entry point for the training script"""
    parser = argparse.ArgumentParser(
        description="Train hierarchical music transformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument("--paired-data-file", type=str, help="Path to JSON file with paired MIDI and text data")
    parser.add_argument("--training-data", type=str, help="Path to pre-processed training data")
    parser.add_argument("--dataset-name", type=str, default="music_dataset", help="Name of the dataset")
    
    # Model arguments
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--no-hierarchical-encoding", dest="use_hierarchical_encoding", action="store_false", 
                      help="Disable hierarchical token encoding")
    parser.add_argument("--no-relative-attention", dest="use_relative_attention", action="store_false",
                      help="Disable relative position attention")
    parser.add_argument("--no-contextual-embeddings", dest="use_contextual_embeddings", action="store_false",
                      help="Disable contextual embeddings")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6, help="Minimum learning rate for scheduler")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Epochs between checkpoints")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--resume-from", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--device", type=str, help="Device to use (cuda, cuda:0, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--generate-samples", action="store_true", help="Generate samples during validation")
    
    # Transfer learning arguments
    transfer_group = parser.add_argument_group("Transfer learning")
    transfer_group.add_argument("--pretrained-model", type=str, help="Path to pretrained model for transfer learning")
    transfer_group.add_argument("--transfer-learning-mode", type=str, default="fine_tuning", 
                             choices=["feature_extraction", "fine_tuning", "full_fine_tuning"],
                             help="Transfer learning mode")
    transfer_group.add_argument("--freeze-layers", type=int, help="Number of layers to freeze (for fine_tuning mode)")
    
    # Parse arguments
    args = parser.parse_args()

    if args.paired_data_file is None and args.training_data is None:
        parser.error("Either --paired-data-file or --training-data must be provided")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create run name if not provided
    if args.run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.run_name = f"transformer_{timestamp}"
    
    # Create output directories
    log_dir = os.path.join(args.output_dir, "logs", args.run_name)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints", args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Determine if we're using transfer learning
    use_transfer_learning = args.pretrained_model is not None
    
    # Load training data or create it
    if args.training_data and os.path.exists(args.training_data):
        logger.info(f"Loading training data from {args.training_data}")
        # Load existing training data
        with open(args.training_data, "r") as f:
            training_data = json.load(f)
        
        train_loader = training_data["train_loader"]
        val_loader = training_data["val_loader"]
        vocab_size = training_data["vocab_size"]
        
    else:
        logger.info(f"Creating new training data from {args.paired_data_file}")
        # Create new training data
        training_data_results = create_advanced_training_data(
            paired_data_file=args.paired_data_file,
            output_dir=os.path.join(args.output_dir, "data"),
            dataset_name=args.dataset_name,
            max_sequence_length=args.max_seq_len,
            batch_size=args.batch_size,
            use_hierarchical_encoding=args.use_hierarchical_encoding,
            use_contextual_embeddings=args.use_contextual_embeddings
        )
        
        train_loader = training_data_results["train_loader"]
        val_loader = training_data_results["val_loader"]
        vocab_size = training_data_results["vocab_size"]
    
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Training with batch size: {args.batch_size}")
    
    # Create model
    if use_transfer_learning:
        # Load pretrained model for transfer learning
        logger.info(f"Loading pretrained model from {args.pretrained_model} for transfer learning")
        try:
            # Load model with configuration from saved file
            state_dict = torch.load(args.pretrained_model, map_location='cpu')
            if 'config' in state_dict:
                config = state_dict['config']
                # Override vocab size with current vocab size
                config['vocab_size'] = vocab_size
                # Create model with loaded config
                model = HierarchicalMusicTransformer(**config)
                # Load weights
                if 'model_state_dict' in state_dict:
                    model.load_pretrained_weights(args.pretrained_model)
                else:
                    model.load_state_dict(state_dict)
            else:
                # Create model with current arguments
                model = create_transformer_model(
                    vocab_size=vocab_size,
                    d_model=args.d_model,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    max_seq_len=args.max_seq_len,
                    use_relative_attention=args.use_relative_attention,
                    use_hierarchical_encoding=args.use_hierarchical_encoding,
                    device=args.device
                )
                # Load weights
                model.load_pretrained_weights(args.pretrained_model)
                
            logger.info(f"Successfully loaded pretrained model for transfer learning")
        except Exception as e:
            logger.error(f"Error loading pretrained model: {e}")
            logger.info("Creating new model instead")
            model = create_transformer_model(
                vocab_size=vocab_size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                max_seq_len=args.max_seq_len,
                use_relative_attention=args.use_relative_attention,
                use_hierarchical_encoding=args.use_hierarchical_encoding,
                device=args.device
            )
    else:
        # Create new model
        model = create_transformer_model(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            use_relative_attention=args.use_relative_attention,
            use_hierarchical_encoding=args.use_hierarchical_encoding,
            device=args.device
        )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.min_learning_rate
    )
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_grad_norm=args.max_grad_norm,
        use_transfer_learning=use_transfer_learning,
        freeze_layers=args.freeze_layers,
        transfer_learning_mode=args.transfer_learning_mode
    )
    
    # Train the model
    trainer.train(args.num_epochs, checkpoint_interval=args.checkpoint_interval)
    
    # Save final model
    trainer.save_checkpoint("final")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

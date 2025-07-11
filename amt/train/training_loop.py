"""
Training Loop - Training loop functionality for music generation models
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from tqdm import tqdm

from amt.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingLoop:
    """Training loop for music generation models.
    
    This class handles the training loop logic, including epoch iteration,
    batch processing, evaluation, early stopping, and checkpoint management.
    """
    
    def __init__(self,
                 train_step_fn: Callable[[Dict[str, torch.Tensor]], Dict[str, float]],
                 evaluate_fn: Callable[[DataLoader], Dict[str, float]],
                 save_checkpoint_fn: Callable[[str, Optional[Dict[str, float]]], None],
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 checkpoint_dir: str = "models/checkpoints",
                 log_interval: int = 10,
                 eval_interval: int = 100,
                 save_interval: int = 1000,
                 plot_history_fn: Optional[Callable[[Optional[str]], None]] = None):
        """Initialize training loop.
        
        Args:
            train_step_fn: Function to perform a single training step
            evaluate_fn: Function to evaluate model on a dataset
            save_checkpoint_fn: Function to save model checkpoint
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            checkpoint_dir: Directory to save checkpoints
            log_interval: Interval to log training progress
            eval_interval: Interval to evaluate model
            save_interval: Interval to save model
            plot_history_fn: Function to plot training history
        """
        self.train_step_fn = train_step_fn
        self.evaluate_fn = evaluate_fn
        self.save_checkpoint_fn = save_checkpoint_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.plot_history_fn = plot_history_fn
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def run(self,
            num_epochs: int,
            resume_from_epoch: int = 0,
            resume_from_step: int = 0,
            early_stopping_patience: int = 5) -> Dict[str, Any]:
        """Run the training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from_epoch: Epoch to resume from
            resume_from_step: Step to resume from
            early_stopping_patience: Number of epochs to wait for improvement
            
        Returns:
            Dictionary containing training results and history
        """
        # Initialize training state for resuming
        self.epoch = resume_from_epoch
        self.global_step = resume_from_step
        
        # Initialize early stopping variables
        patience_counter = 0
        
        # Start training
        start_time = time.time()
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training loop
            train_losses = []
            
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                # Training step
                metrics = self.train_step_fn(batch)
                train_losses.append(metrics['loss'])
                
                # Update global step
                self.global_step += 1
                
                # Log progress
                if self.global_step % self.log_interval == 0:
                    avg_loss = np.mean(train_losses[-self.log_interval:])
                    self.history['train_loss'].append((self.global_step, avg_loss))
                    self.history['learning_rate'].append((self.global_step, metrics['learning_rate']))
                    
                    logger.info(f"Step {self.global_step}, Loss: {avg_loss:.4f}, LR: {metrics['learning_rate']:.6f}")
                
                # Evaluate model
                if self.val_dataloader is not None and self.global_step % self.eval_interval == 0:
                    val_metrics = self.evaluate_fn(self.val_dataloader)
                    self.history['val_loss'].append((self.global_step, val_metrics['loss']))
                    
                    logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")
                    
                    # Check for improvement
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        patience_counter = 0
                        
                        # Save best model
                        self.save_checkpoint_fn(
                            os.path.join(self.checkpoint_dir, "best_model.pt"),
                            val_metrics
                        )
                    else:
                        patience_counter += 1
                    
                    # Early stopping
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {patience_counter} evaluations without improvement")
                        return self._get_training_results(start_time)
                
                # Save checkpoint
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint_fn(
                        os.path.join(self.checkpoint_dir, f"checkpoint_{self.global_step}.pt")
                    )
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = np.mean(train_losses)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint_fn(
                os.path.join(self.checkpoint_dir, f"epoch_{epoch+1}.pt")
            )
        
        # End of training
        return self._get_training_results(start_time)
    
    def _get_training_results(self, start_time: float) -> Dict[str, Any]:
        """Get training results.
        
        Args:
            start_time: Time when training started
            
        Returns:
            Dictionary containing training results and history
        """
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Save final model
        self.save_checkpoint_fn(
            os.path.join(self.checkpoint_dir, "final_model.pt")
        )
        
        # Plot training history
        if self.plot_history_fn is not None:
            self.plot_history_fn()
        
        return {
            'epochs_completed': self.epoch + 1,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'training_time': total_time,
            'history': self.history
        } 
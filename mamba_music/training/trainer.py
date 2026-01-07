"""
Trainer for Mamba Music models.
"""

import os
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from ..utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Trainer for Mamba Music models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        output_dir: str = "experiments",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        
        # Config defaults
        cfg = cfg or {}
        self.epochs = cfg.get("epochs", 100)
        self.learning_rate = cfg.get("learning_rate", 1e-4)
        self.weight_decay = cfg.get("weight_decay", 0.01)
        self.warmup_steps = cfg.get("warmup_steps", 1000)
        self.gradient_clip = cfg.get("gradient_clip", 1.0)
        self.save_every_n_epochs = cfg.get("save_every_n_epochs", 5)
        
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # State
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.global_step = 0
        
        # Logging
        os.makedirs(output_dir, exist_ok=True)
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(os.path.join(output_dir, "logs"))
        else:
            self.writer = None
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in progress:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            logits = self.model(input_ids)
            
            # Handle multi-head output
            if isinstance(logits, dict):
                loss = 0.0
                for i, (name, head_logits) in enumerate(logits.items()):
                    target = labels[:, :, i]
                    loss += nn.functional.cross_entropy(
                        head_logits.reshape(-1, head_logits.size(-1)),
                        target.reshape(-1),
                        ignore_index=0,
                    )
            else:
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
            
            # Backward
            loss.backward()
            
            # Clip gradients
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            
            self.optimizer.step()
            self.global_step += 1
            
            total_loss += loss.item()
            num_batches += 1
            
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / max(num_batches, 1)

    def validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits = self.model(input_ids)
                
                if isinstance(logits, dict):
                    loss = 0.0
                    for i, (name, head_logits) in enumerate(logits.items()):
                        target = labels[:, :, i]
                        loss += nn.functional.cross_entropy(
                            head_logits.reshape(-1, head_logits.size(-1)),
                            target.reshape(-1),
                            ignore_index=0,
                        )
                else:
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                    )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        path = os.path.join(self.output_dir, "checkpoints", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.output_dir, "checkpoints", "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: val_loss={self.best_val_loss:.4f}")

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            
            # Logging
            if self.writer:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("LR", lr, epoch)
            
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}"
            )
            
            # Save
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(f"epoch_{epoch+1}.pt", is_best)
        
        # Final save
        self.save_checkpoint("final.pt")
        
        if self.writer:
            self.writer.close()
        
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.1f}s")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

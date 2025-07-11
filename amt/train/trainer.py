"""
Model Trainer - Trains music generation models
"""

import os
import time
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from amt.train.model import MusicTransformer
from amt.process.data_preparer import DataPreparer
from amt.utils.logging import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Trainer for Music Transformer model."""

    def __init__(
        self,
        model: MusicTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        save_dir: str = "models/checkpoints",
        device: str = "auto",
        loss_type: str = "ce",
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        self.device = self._get_device(device)

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)

        # --------------------------------------------------
        # Loss function selection
        # --------------------------------------------------
        loss_type = loss_type.lower()
        if loss_type == "label_smooth" or loss_type == "label_smoothing":
            self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
            print(f"Using CrossEntropyLoss with label_smoothing={label_smoothing}")
        elif loss_type == "focal":
            self.criterion = FocalLoss(gamma=focal_gamma, ignore_index=0)
            print(f"Using FocalLoss with gamma={focal_gamma}")
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)
            print("Using standard CrossEntropyLoss")

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_history = {"train_loss": [], "val_loss": [], "learning_rate": []}

        # Setup logging
        os.makedirs(save_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(save_dir, "logs"))

        print("Model Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Save directory: {save_dir}")

    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in progress_bar:
            # Move batch to device
            midi_tokens = batch["midi_tokens"].to(self.device)
            bert_embedding = batch["bert_embedding"].to(self.device)
            tfidf_features = batch["tfidf_features"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Prepare text embeddings: expand to [batch, text_len, 768]
            text_len = getattr(self.model, "max_text_len", 512)

            if torch.all(bert_embedding == 0):
                # Fall back to TF-IDF features when BERT embedding is zero
                text_embeddings = tfidf_features.unsqueeze(1).expand(-1, text_len, -1)
            else:
                text_embeddings = bert_embedding.unsqueeze(1).expand(-1, text_len, -1)

            # Forward pass
            logits = self.model(midi_tokens[:, :-1], text_embeddings)  # Input sequence

            # Calculate loss
            targets = midi_tokens[:, 1:]  # Target sequence
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss / num_batches:.4f}"}
            )

        return total_loss / num_batches

    def validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                midi_tokens = batch["midi_tokens"].to(self.device)
                bert_embedding = batch["bert_embedding"].to(self.device)
                tfidf_features = batch["tfidf_features"].to(self.device)

                # Prepare text embeddings
                text_len = getattr(self.model, "max_text_len", 512)

                if torch.all(bert_embedding == 0):
                    text_embeddings = tfidf_features.unsqueeze(1).expand(-1, text_len, -1)
                else:
                    text_embeddings = bert_embedding.unsqueeze(1).expand(-1, text_len, -1)

                # Forward pass
                logits = self.model(midi_tokens[:, :-1], text_embeddings)

                # Calculate loss
                targets = midi_tokens[:, 1:]
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "model_config": self.model.get_model_info(),
        }

        # Save checkpoint
        checkpoint_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]

        print(f"Checkpoint loaded from epoch {self.current_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

    def train(self, resume_from: Optional[str] = None):
        """Train the model."""
        if resume_from:
            self.load_checkpoint(resume_from)

        print("Starting training...")
        start_time = time.time()

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate_epoch()

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log metrics
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["learning_rate"].append(current_lr)

            # TensorBoard logging
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)

            # Print progress
            print(f"Epoch {epoch + 1}/{self.max_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Time: {time.time() - start_time:.2f}s")
            print("-" * 50)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Save every 10 epochs or if it's the best model
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", is_best)

        # Save final model
        self.save_checkpoint("final_model.pt")

        # Close TensorBoard writer
        self.writer.close()

        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total training time: {time.time() - start_time:.2f}s")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "model_info": self.model.get_model_info(),
        }


def enhanced_train_model(
    data_file: str, output_dir: str = "models/checkpoints", **kwargs
) -> ModelTrainer:
    """Enhanced training function with automatic data loading."""
    # Load and prepare data
    preparer = DataPreparer()
    training_data = preparer.prepare_training_data(
        data_file, output_dir.replace("/checkpoints", "/processed")
    )

    # Create model
    model = MusicTransformer(vocab_size=training_data["vocab_size"], **kwargs)

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=training_data["train_loader"],
        val_loader=training_data["val_loader"],
        save_dir=output_dir,
    )

    # Start training
    trainer.train()

    return trainer


# --------------------------------------------------
#  Custom Losses
# --------------------------------------------------


class FocalLoss(nn.Module):
    """Focal Loss for token classification (per-token). Useful when token frequency is highly imbalanced.

    Args:
        gamma (float): focusing parameter γ. Higher → focus more on hard examples.
        ignore_index (int): token id to ignore (padding).
    """

    def __init__(self, gamma: float = 2.0, ignore_index: int = 0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self._ce = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # logits: [N, vocab] after flatten, targets: [N]
        ce_loss = self._ce(logits, targets)  # [N]
        pt = torch.exp(-ce_loss)  # p_t = e^{-CE}
        focal = (1.0 - pt) ** self.gamma * ce_loss
        return focal.mean()

#!/usr/bin/env python3
"""
Training script for Mamba Music Transformer.

Uses Hydra for configuration management.
"""

import os
import sys
import logging
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from mamba_music.utils.logging import get_logger, setup_logging
from mamba_music.data import OctupleTokenizer, get_dataloaders
from mamba_music.training import Trainer

logger = get_logger(__name__)


def get_model(cfg: DictConfig):
    """Create model based on configuration."""
    model_name = cfg.model.name
    
    if model_name == "transformer":
        from mamba_music.models import HierarchicalMusicTransformer
        model = HierarchicalMusicTransformer(
            d_model=cfg.model.d_model,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            d_ff=cfg.model.d_ff,
            max_seq_len=cfg.model.max_seq_len,
            dropout=cfg.model.dropout,
        )
    elif model_name == "hybrid":
        from mamba_music.models import HybridMambaTransformer
        model = HybridMambaTransformer(
            d_model=cfg.model.d_model,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            num_mamba_per_block=cfg.model.get("num_mamba_per_block", 7),
            d_ff=cfg.model.d_ff,
            max_seq_len=cfg.model.max_seq_len,
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


@hydra.main(config_path="../configs", config_name="train/default", version_base="1.3")
def main(cfg: DictConfig):
    """Main training function."""
    # Setup logging
    setup_logging(cfg.get("log_level", "INFO"))
    
    # Print config
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    
    # Create tokenizer
    logger.info("Creating tokenizer...")
    tokenizer = OctupleTokenizer()
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        train_paths=[],  # TODO: Load from cfg.data.train_path
        val_paths=[],    # TODO: Load from cfg.data.val_path
        tokenizer=tokenizer,
        batch_size=cfg.training.batch_size,
        max_seq_len=cfg.data.max_seq_len,
        num_workers=cfg.data.num_workers,
    )
    
    # Create model
    logger.info(f"Creating model: {cfg.model.name}")
    model = get_model(cfg)
    model = model.to(device)
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg.training,
        device=device,
        output_dir=cfg.paths.output_dir,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

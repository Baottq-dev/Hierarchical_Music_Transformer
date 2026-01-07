"""Dataset classes for Octuple MIDI data."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

from .tokenizer import OctupleTokenizer


class OctupleMusicDataset(Dataset):
    """Dataset for Octuple-tokenized MIDI files."""
    
    def __init__(
        self,
        midi_paths: List[str],
        tokenizer: OctupleTokenizer,
        max_seq_len: int = 2048,
        text_data: Optional[Dict[str, str]] = None,
    ):
        self.midi_paths = midi_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_data = text_data or {}
        self._cache = {}
    
    def __len__(self) -> int:
        return len(self.midi_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        midi_path = self.midi_paths[idx]
        
        # Get tokens (with caching)
        if midi_path not in self._cache:
            try:
                tokens = self.tokenizer.tokenize(midi_path)
                self._cache[midi_path] = tokens
            except Exception as e:
                tokens = [[0] * 8]
        else:
            tokens = self._cache[midi_path]
        
        # Truncate
        tokens = tokens[:self.max_seq_len]
        
        # Convert to tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Create input/target pairs
        input_tokens = tokens_tensor[:-1]
        target_tokens = tokens_tensor[1:]
        
        return {
            "input_ids": input_tokens,
            "labels": target_tokens,
            "attention_mask": torch.ones(len(input_tokens), dtype=torch.bool),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for variable-length sequences."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    batch_size = len(batch)
    input_ids = torch.zeros(batch_size, max_len, 8, dtype=torch.long)
    labels = torch.zeros(batch_size, max_len, 8, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = item["attention_mask"]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def get_dataloaders(
    train_paths: List[str],
    val_paths: List[str],
    tokenizer: OctupleTokenizer,
    batch_size: int = 32,
    max_seq_len: int = 2048,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    train_dataset = OctupleMusicDataset(train_paths, tokenizer, max_seq_len)
    val_dataset = OctupleMusicDataset(val_paths, tokenizer, max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader

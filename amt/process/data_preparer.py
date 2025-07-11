"""
Data Preparer - Prepares data for training and evaluation
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

from amt.process.midi_processor import MIDIProcessor
from amt.process.text_processor import TextProcessor
from amt.utils.logging import get_logger

logger = get_logger(__name__)


class MusicTextDataset(Dataset):
    """Dataset for music and text data."""

    def __init__(self, data: List[Dict[str, Any]], max_sequence_length: int, max_text_length: int):
        self.data = data
        self.max_sequence_length = max_sequence_length
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Get tokens
        tokens = item["midi_tokens"]
        if isinstance(tokens, list):
            tokens = np.array(tokens)

        # Pad or truncate sequence
        if len(tokens) > self.max_sequence_length:
            tokens = tokens[: self.max_sequence_length]
        else:
            tokens = np.pad(tokens, (0, self.max_sequence_length - len(tokens)), "constant")

        # Extract BERT and TF-IDF embeddings from the stored text_features dict
        text_data = item["text_features"]

        # Default placeholders
        bert_emb = np.zeros(768, dtype=np.float32)
        tfidf_feat = np.zeros(768, dtype=np.float32)

        if isinstance(text_data, dict):
            # BERT embedding
            if text_data.get("bert_embedding") is not None:
                bert_arr = np.array(text_data["bert_embedding"], dtype=np.float32)
                if bert_arr.shape[0] >= 768:
                    bert_emb = bert_arr[:768]
                else:
                    bert_emb[: bert_arr.shape[0]] = bert_arr

            # TF-IDF features (original length = 1000) – down-sample / pad to 768
            if text_data.get("tfidf_features") is not None:
                tfidf_arr = np.array(text_data["tfidf_features"], dtype=np.float32)
                if tfidf_arr.shape[0] >= 768:
                    tfidf_feat = tfidf_arr[:768]
                else:
                    tfidf_feat[: tfidf_arr.shape[0]] = tfidf_arr
        elif isinstance(text_data, list):
            # Fallback: treat list as generic embedding, truncate / pad to 768
            generic_arr = np.array(text_data, dtype=np.float32)
            if generic_arr.shape[0] >= 768:
                bert_emb = generic_arr[:768]
            else:
                bert_emb[: generic_arr.shape[0]] = generic_arr

        return {
            "midi_tokens": torch.tensor(tokens, dtype=torch.long),
            "bert_embedding": torch.tensor(bert_emb, dtype=torch.float),  # [768]
            "tfidf_features": torch.tensor(tfidf_feat, dtype=torch.float),  # [768]
            "sequence_length": item["sequence_length"],
        }


class DataPreparer:
    """Prepares data for training."""

    def __init__(
        self,
        max_sequence_length: int = 1024,
        max_text_length: int = 512,
        batch_size: int = 32,
        text_processor_use_gpu: bool = False,
    ):
        """
        Args:
            max_sequence_length: Maximum MIDI token length.
            max_text_length: Maximum text embedding length.
            batch_size: DataLoader batch size.
            text_processor_use_gpu: Whether to load BERT/spaCy models on GPU.
                For most training-time usages we only need pre-computed embeddings,
                so keeping this `False` avoids occupying precious GPU VRAM.
        """
        self.max_sequence_length = max_sequence_length
        self.max_text_length = max_text_length
        self.batch_size = batch_size

        self.midi_processor = MIDIProcessor(max_sequence_length=max_sequence_length)
        self.text_processor = TextProcessor(
            max_length=max_text_length, use_gpu=text_processor_use_gpu
        )

    def load_paired_data(self, paired_data_file: str) -> List[Dict[str, Any]]:
        """Load paired data from file."""
        with open(paired_data_file, encoding="utf-8") as f:
            paired_data = json.load(f)
        return paired_data

    def process_paired_data(self, paired_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process paired data into training format."""
        processed_data = []

        for item in paired_data:
            midi_file = item.get("midi_file")
            text_description = item.get("text_description")

            if not midi_file or not text_description:
                continue

            # Process MIDI
            midi_processed = self.midi_processor.process_midi_file(midi_file)
            if midi_processed is None:
                continue

            # Process text
            text_processed = self.text_processor.process_text(text_description)

            # Combine into training item
            training_item = {
                "midi_file": midi_file,
                "text_description": text_description,
                "midi_tokens": midi_processed["tokens"],
                "midi_metadata": midi_processed["metadata"],
                "text_features": text_processed,
                "sequence_length": midi_processed["sequence_length"],
            }

            processed_data.append(training_item)

        return processed_data

    def create_dataset(self, data: List[Dict[str, Any]]) -> Dataset:
        """Create a PyTorch dataset from processed data."""
        return MusicTextDataset(data, self.max_sequence_length, self.max_text_length)

    def create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create a PyTorch dataloader from a dataset."""
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=2, pin_memory=True
        )

    def split_data(
        self, data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train, validation, and test sets."""
        # Shuffle data
        indices = np.random.permutation(len(data))

        # Calculate split indices
        train_idx = int(len(data) * train_ratio)
        val_idx = train_idx + int(len(data) * val_ratio)

        # Split data
        train_data = [data[i] for i in indices[:train_idx]]
        val_data = [data[i] for i in indices[train_idx:val_idx]]
        test_data = [data[i] for i in indices[val_idx:]]

        return train_data, val_data, test_data

    def prepare_training_data(
        self, paired_data_file: str, output_dir: str = "data/processed"
    ) -> Dict[str, Any]:
        """Prepare complete training data."""
        print("Loading paired data...")
        paired_data = self.load_paired_data(paired_data_file)

        print("Processing paired data...")
        processed_data = self.process_paired_data(paired_data)

        print("Splitting data...")
        train_data, val_data, test_data = self.split_data(processed_data)

        # Create datasets
        train_dataset = self.create_dataset(train_data)
        val_dataset = self.create_dataset(val_data)
        test_dataset = self.create_dataset(test_data)

        # Create dataloaders
        train_loader = self.create_dataloader(train_dataset, shuffle=True)
        val_loader = self.create_dataloader(val_dataset, shuffle=False)
        test_loader = self.create_dataloader(test_dataset, shuffle=False)

        # Save processed data
        os.makedirs(output_dir, exist_ok=True)

        training_data = {
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
            "vocab_size": self.midi_processor.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "max_text_length": self.max_text_length,
        }

        with open(os.path.join(output_dir, "training_data.json"), "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        print("Training data prepared:")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Validation: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")
        print(f"  Vocabulary size: {self.midi_processor.vocab_size}")

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "vocab_size": self.midi_processor.vocab_size,
            "training_data": training_data,
        }

    def get_data_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the processed data."""
        if not data:
            return {"error": "No data available"}

        sequence_lengths = [item["sequence_length"] for item in data]

        # Get metadata statistics
        durations = [
            item["midi_metadata"]["duration"]
            for item in data
            if "duration" in item["midi_metadata"]
        ]
        tempos = [
            item["midi_metadata"]["tempo"] for item in data if "tempo" in item["midi_metadata"]
        ]

        # Count instruments
        instrument_counts = {}
        for item in data:
            if "instruments" in item["midi_metadata"]:
                for instrument in item["midi_metadata"]["instruments"]:
                    if instrument in instrument_counts:
                        instrument_counts[instrument] += 1
                    else:
                        instrument_counts[instrument] = 1

        # Sort instruments by count
        top_instruments = sorted(instrument_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_samples": len(data),
            "sequence_length": {
                "min": min(sequence_lengths) if sequence_lengths else 0,
                "max": max(sequence_lengths) if sequence_lengths else 0,
                "mean": sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0,
                "median": sorted(sequence_lengths)[len(sequence_lengths) // 2]
                if sequence_lengths
                else 0,
            },
            "duration": {
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "mean": sum(durations) / len(durations) if durations else 0,
            },
            "tempo": {
                "min": min(tempos) if tempos else 0,
                "max": max(tempos) if tempos else 0,
                "mean": sum(tempos) / len(tempos) if tempos else 0,
            },
            "top_instruments": dict(top_instruments),
        }


def prepare_training_data(
    paired_data_file: str, output_dir: str = "data/processed"
) -> Dict[str, Any]:
    """Convenience function to prepare training data."""
    preparer = DataPreparer()
    return preparer.prepare_training_data(paired_data_file, output_dir)

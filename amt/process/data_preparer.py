"""
Data Preparer - Prepares data for training and evaluation
"""

import os
import json
import random
import traceback
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from amt.process.midi_processor import MidiProcessor
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

        # Include fused features if available
        fused_features = None
        if "fused_features" in item:
            fused_features = torch.tensor(item["fused_features"], dtype=torch.float)

        result = {
            "midi_tokens": torch.tensor(tokens, dtype=torch.long),
            "bert_embedding": torch.tensor(bert_emb, dtype=torch.float),  # [768]
            "tfidf_features": torch.tensor(tfidf_feat, dtype=torch.float),  # [768]
            "sequence_length": item["sequence_length"],
        }
        
        if fused_features is not None:
            result["fused_features"] = fused_features
            
        return result


class DataPreparer:
    """Prepares data for training."""

    def __init__(
        self,
        midi_processor: 'MidiProcessor',
        text_processor: 'TextProcessor',
        feature_fusion_method: str = "attention",
        output_dir: str = "data/processed",
        is_kaggle: bool = False
    ):
        """Initialize DataPreparer
        
        Args:
            midi_processor: MIDI processor
            text_processor: Text processor
            feature_fusion_method: Method to combine features
            output_dir: Output directory
            is_kaggle: Whether running in Kaggle environment
        """
        self.midi_processor = midi_processor
        self.text_processor = text_processor
        self.feature_fusion_method = feature_fusion_method
        self.output_dir = output_dir
        self.is_kaggle = is_kaggle
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def combine_features(self, text_features, midi_features, method="concat"):
        """Combine features from text and MIDI"""
        # Extract sentence embeddings
        if isinstance(text_features, dict) and "sentence_embedding" in text_features:
            text_embedding = text_features["sentence_embedding"]
        elif isinstance(text_features, dict) and "bert_embedding" in text_features:
            text_embedding = text_features["bert_embedding"]
        else:
            # Fallback
            text_embedding = np.zeros(768)
        
        if isinstance(midi_features, dict) and "sequence_embedding" in midi_features:
            midi_embedding = midi_features["sequence_embedding"]
        else:
            # Fallback
            midi_embedding = np.zeros(512)
        
        # Convert to numpy arrays if they're not already
        if not isinstance(text_embedding, np.ndarray):
            text_embedding = np.array(text_embedding)
        if not isinstance(midi_embedding, np.ndarray):
            midi_embedding = np.array(midi_embedding)
            
        # Ensure we have 2D arrays for consistency
        if len(text_embedding.shape) == 1:
            text_embedding = text_embedding.reshape(1, -1)
        if len(midi_embedding.shape) == 1:
            midi_embedding = midi_embedding.reshape(1, -1)
        
        if method == "concat":
            # Simple concatenation
            # Pad the smaller embedding if dimensions don't match
            if text_embedding.shape[1] != midi_embedding.shape[1]:
                max_dim = max(text_embedding.shape[1], midi_embedding.shape[1])
                if text_embedding.shape[1] < max_dim:
                    padding = np.zeros((text_embedding.shape[0], max_dim - text_embedding.shape[1]))
                    text_embedding = np.hstack([text_embedding, padding])
                if midi_embedding.shape[1] < max_dim:
                    padding = np.zeros((midi_embedding.shape[0], max_dim - midi_embedding.shape[1]))
                    midi_embedding = np.hstack([midi_embedding, padding])
            
            return np.hstack([text_embedding, midi_embedding]).flatten()
        
        elif method == "attention":
            # Attention-based fusion
            try:
                # Convert to torch tensors for matrix operations
                text_tensor = torch.tensor(text_embedding, dtype=torch.float)
                midi_tensor = torch.tensor(midi_embedding, dtype=torch.float)
                
                # Ensure dimensions match for attention
                if text_tensor.shape[1] != midi_tensor.shape[1]:
                    # Project to common dimension
                    common_dim = 512
                    text_tensor = torch.nn.functional.linear(
                        text_tensor, 
                        torch.randn(common_dim, text_tensor.shape[1]) / np.sqrt(text_tensor.shape[1])
                    )
                    midi_tensor = torch.nn.functional.linear(
                        midi_tensor, 
                        torch.randn(common_dim, midi_tensor.shape[1]) / np.sqrt(midi_tensor.shape[1])
                    )
                
                # Compute attention weights
                attn_weights = torch.softmax(torch.matmul(text_tensor, midi_tensor.transpose(0, 1)), dim=1)
                
                # Weighted sum
                fused = text_tensor + torch.matmul(attn_weights, midi_tensor)
                return fused.detach().numpy().flatten()
            except Exception as e:
                logger.error(f"Error in attention fusion: {e}")
                # Fallback to concat
                return self.combine_features(text_features, midi_features, "concat")
        
        elif method == "gated":
            # Gated fusion
            try:
                # Convert to torch tensors
                text_tensor = torch.tensor(text_embedding, dtype=torch.float)
                midi_tensor = torch.tensor(midi_embedding, dtype=torch.float)
                
                # Ensure dimensions match
                if text_tensor.shape[1] != midi_tensor.shape[1]:
                    # Project to common dimension
                    common_dim = 512
                    text_tensor = torch.nn.functional.linear(
                        text_tensor, 
                        torch.randn(common_dim, text_tensor.shape[1]) / np.sqrt(text_tensor.shape[1])
                    )
                    midi_tensor = torch.nn.functional.linear(
                        midi_tensor, 
                        torch.randn(common_dim, midi_tensor.shape[1]) / np.sqrt(midi_tensor.shape[1])
                    )
                
                # Compute gate
                gate = torch.sigmoid(text_tensor + midi_tensor)
                
                # Gated combination
                fused = gate * text_tensor + (1 - gate) * midi_tensor
                return fused.detach().numpy().flatten()
            except Exception as e:
                logger.error(f"Error in gated fusion: {e}")
                # Fallback to concat
                return self.combine_features(text_features, midi_features, "concat")
        
        else:
            # Default: simple average
            if text_embedding.shape[1] != midi_embedding.shape[1]:
                # Project to common dimension
                common_dim = 512
                text_embedding_resized = np.random.randn(common_dim, text_embedding.shape[1]) @ text_embedding.T
                midi_embedding_resized = np.random.randn(common_dim, midi_embedding.shape[1]) @ midi_embedding.T
                return ((text_embedding_resized + midi_embedding_resized) / 2).T.flatten()
            else:
                return ((text_embedding + midi_embedding) / 2).flatten()

    def load_paired_data(self, paired_data_file: str) -> List[Dict[str, Any]]:
        """Load paired data from file."""
        with open(paired_data_file, encoding="utf-8") as f:
            paired_data = json.load(f)
            
        # Check if we need to adjust paths for Kaggle
        if os.path.exists("/kaggle/input"):
            logger.info("Kaggle environment detected. Adjusting MIDI paths...")
            
            # Define possible Kaggle paths
            possible_paths = [
                "/kaggle/input/midi-dataset/midi",
                "/kaggle/input/your-dataset/midi",
                "/kaggle/input/lakh-midi-dataset/midi",
                "/kaggle/input/midi-files/midi",
                "/kaggle/working/data/midi"
            ]
            
            # Find the first path that exists
            kaggle_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    kaggle_path = path
                    logger.info(f"Found Kaggle MIDI path: {kaggle_path}")
                    break
                    
            # Update paths if we found a valid Kaggle path
            if kaggle_path:
                midi_path_keys = ["midi_path", "midi_file", "file_path"]
                paths_updated = 0
                
                # Process each item in the data
                for item in paired_data:
                    # Check for each possible key that might contain a MIDI path
                    for key in list(item.keys()):
                        if (key in midi_path_keys or 
                            ("midi" in key.lower() and "path" in key.lower()) or
                            key.lower().endswith("path")):
                            
                            if isinstance(item[key], str) and item[key].startswith("data/midi/"):
                                relative_path = item[key][len("data/midi/"):]
                                new_path = os.path.join(kaggle_path, relative_path)
                                
                                # Only update if the new path exists or we have no better option
                                if os.path.exists(new_path):
                                    old_path = item[key]
                                    item[key] = new_path
                                    paths_updated += 1
                                    
                    # Check nested dictionaries in metadata
                    if "metadata" in item and isinstance(item["metadata"], dict):
                        for key in list(item["metadata"].keys()):
                            if (key in midi_path_keys or 
                                ("midi" in key.lower() and "path" in key.lower()) or 
                                key.lower().endswith("path")):
                                
                                if isinstance(item["metadata"][key], str) and item["metadata"][key].startswith("data/midi/"):
                                    relative_path = item["metadata"][key][len("data/midi/"):]
                                    new_path = os.path.join(kaggle_path, relative_path)
                                    
                                    # Only update if the new path exists or we have no better option
                                    if os.path.exists(new_path):
                                        old_path = item["metadata"][key]
                                        item["metadata"][key] = new_path
                                        paths_updated += 1
                
                logger.info(f"Updated {paths_updated} MIDI paths for Kaggle compatibility")
            
        return paired_data

    def process_paired_data(self, paired_data: List[Dict[str, Any]], batch_size: int = 16) -> List[Dict[str, Any]]:
        """Process paired MIDI and text data with batch processing.
        
        Args:
            paired_data: List of dictionaries or a dictionary with 'pairs' key
            batch_size: Number of items to process in each batch
            
        Returns:
            List of processed data items
        """
        processed_data = []

        # Handle both formats: list of pairs or dictionary with 'pairs' key
        if isinstance(paired_data, dict) and 'pairs' in paired_data:
            pairs_to_process = paired_data['pairs']
        else:
            pairs_to_process = paired_data
        
        # Create cache directory
        cache_dir = os.path.join(self.output_dir, "midi_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Process data in batches
        total_items = len(pairs_to_process)
        for batch_start in tqdm(range(0, total_items, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, total_items)
            batch = pairs_to_process[batch_start:batch_end]
            
            # Prepare batch data
            batch_midi_paths = []
            batch_texts = []
            valid_indices = []
            
            for idx, item in enumerate(batch):
                try:
                    # Check if required fields are present
                    midi_field = None
                    text_field = None
                    
                    # Check for common field names
                    for field in ['midi_file', 'midi_path']:
                        if field in item:
                            midi_field = field
                            break
                    
                    for field in ['text', 'text_file', 'text_description']:
                        if field in item:
                            text_field = field
                            break
                    
                    if not midi_field or not text_field:
                        logger.warning(f"Skipping item: missing required fields. Item: {item}")
                        continue
                    
                    midi_path = item[midi_field]
                    text = item[text_field]
                    
                    # Handle text file paths
                    if text_field == 'text_file' and os.path.exists(text):
                        with open(text, 'r', encoding='utf-8') as f:
                            text = f.read()
                    
                    # Check for Kaggle environment and adjust paths
                    if os.path.exists("/kaggle/input"):
                        # Try different common paths in Kaggle
                        kaggle_paths = [
                            midi_path,
                            os.path.join('/kaggle/input', midi_path),
                            os.path.join('/kaggle/working', midi_path),
                            os.path.join('/kaggle/input/midi-dataset', os.path.basename(midi_path))
                        ]
                        
                        midi_path = None
                        for path in kaggle_paths:
                            if os.path.exists(path):
                                midi_path = path
                                break
                        
                        if not midi_path:
                            logger.warning(f"Could not find MIDI file in Kaggle environment: {item[midi_field]}")
                            continue
                    
                    # Process MIDI file
                    if not os.path.exists(midi_path):
                        logger.warning(f"MIDI file not found: {midi_path}")
                        continue
                    
                    # Check cache first
                    cache_path = os.path.join(cache_dir, f"{os.path.basename(midi_path)}_processed.json")
                    if os.path.exists(cache_path):
                        try:
                            with open(cache_path, 'r') as f:
                                cached_item = json.load(f)
                                processed_data.append(cached_item)
                                continue
                        except Exception as e:
                            logger.warning(f"Cache read error for {midi_path}: {str(e)}")
                    
                    # Add to batch for processing
                    batch_midi_paths.append(midi_path)
                    batch_texts.append(text)
                    valid_indices.append(idx)
                    
                except Exception as e:
                    logger.error(f"Error preparing item for batch: {str(e)}")
            
            # Process batch if not empty
            if not batch_midi_paths:
                continue

            try:
                # Load MIDI files in batch
                batch_midi_data = []
                for midi_path in batch_midi_paths:
                    try:
                        midi_data = self.midi_processor.load_midi(midi_path)
                        batch_midi_data.append(midi_data)
                    except Exception as e:
                        logger.error(f"Error loading MIDI {midi_path}: {str(e)}")
                        batch_midi_data.append(None)
                
                # Process MIDI features in batch (if possible)
                batch_midi_features = []
                for i, midi_data in enumerate(batch_midi_data):
                    if midi_data is None:
                        batch_midi_features.append(None)
                continue

                    cache_path = os.path.join(cache_dir, f"{os.path.basename(batch_midi_paths[i])}.json")
                    try:
                        # Set a timeout for processing each MIDI file to avoid hanging
                        features = self.midi_processor.extract_features(midi_data, cache_path=cache_path)
                        batch_midi_features.append(features)
                    except Exception as e:
                        logger.error(f"Error extracting MIDI features for {batch_midi_paths[i]}: {str(e)}")
                        # Create default features on error to continue processing
                        default_features = {
                            'sequence_embedding': [0.0] * 512,  # Default embedding size
                            'model_name': 'error_processing'
                        }
                        batch_midi_features.append(default_features)
                        
                        # Cache the default features to avoid reprocessing
                        try:
                            with open(cache_path, 'w') as f:
                                json.dump(default_features, f)
                        except Exception as cache_err:
                            logger.warning(f"Error caching default features: {str(cache_err)}")
                
                # Process text features in batch (if possible)
                batch_text_features = []
                for text in batch_texts:
                    try:
                        features = self.text_processor.extract_features(text)
                        batch_text_features.append(features)
                    except Exception as e:
                        logger.error(f"Error extracting text features: {str(e)}")
                        # Create default text features on error
                        default_features = {
                            'bert_embedding': [0.0] * 768,  # Default embedding size for BERT
                            'sentence_embedding': [0.0] * 768
                        }
                        batch_text_features.append(default_features)
                
                # Combine features and save results
                for i in range(len(valid_indices)):
                    # Continue processing even with default features
                    # We've already replaced None values with default features
                    original_item = batch[valid_indices[i]]
                    midi_path = batch_midi_paths[i]
                    
                    processed_item = {
                        'midi_path': midi_path,
                        'text': batch_texts[i],
                        'midi_features': batch_midi_features[i],
                        'text_features': batch_text_features[i]
                    }
                    
                    # Add metadata if available
                    if 'metadata' in original_item:
                        processed_item['metadata'] = self._make_json_serializable(original_item['metadata'])
                    
                    processed_data.append(processed_item)
                    
                    # Cache processed item
                    try:
                        item_cache_path = os.path.join(cache_dir, f"{os.path.basename(midi_path)}_processed.json")
                        with open(item_cache_path, 'w') as f:
                            # Ensure all values are JSON serializable
                            serializable_item = self._make_json_serializable(processed_item)
                            json.dump(serializable_item, f)
                    except Exception as e:
                        logger.error(f"Error saving cache for {midi_path}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                logger.error(traceback.format_exc())

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

    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._make_json_serializable(obj.tolist())
        else:
            return obj


def prepare_training_data(
    paired_data_file: str, output_dir: str = "data/processed"
) -> Dict[str, Any]:
    """Convenience function to prepare training data."""
    preparer = DataPreparer()
    return preparer.prepare_training_data(paired_data_file, output_dir)

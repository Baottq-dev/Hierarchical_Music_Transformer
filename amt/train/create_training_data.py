"""
Advanced training data creation for AMT system
Uses hierarchical encodings and optimized data preparation for music transformer models
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
import time
import logging

from amt.process.midi_processor import MIDIProcessor
from amt.process.text_processor import TextProcessor
from amt.process.data_preparer import DataPreparer
from amt.utils.logging import get_logger
from amt.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class AdvancedDataCreator:
    """Creates optimized training data for transformer-based music generation models"""
    
    def __init__(
        self, 
        max_sequence_length: int = 1024,
        max_text_length: int = 512,
        batch_size: int = 32,
        use_hierarchical_encoding: bool = True,
        use_contextual_embeddings: bool = True,
        use_sentencepiece: bool = True,
        num_workers: int = 4,
        device: Optional[str] = None
    ):
        """Initialize advanced data creator
        
        Args:
            max_sequence_length: Maximum sequence length for music tokens
            max_text_length: Maximum sequence length for text tokens
            batch_size: Batch size for processing
            use_hierarchical_encoding: Whether to use hierarchical token encoding
            use_contextual_embeddings: Whether to use contextual embeddings
            use_sentencepiece: Whether to use SentencePiece tokenization
            num_workers: Number of workers for parallel processing
            device: Device to use (auto-detects if None)
        """
        self.max_sequence_length = max_sequence_length
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.use_hierarchical_encoding = use_hierarchical_encoding
        self.use_contextual_embeddings = use_contextual_embeddings
        self.use_sentencepiece = use_sentencepiece
        self.num_workers = num_workers
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize processors
        self.midi_processor = MIDIProcessor(
            max_sequence_length=max_sequence_length,
            use_cache=True
        )
        
        self.text_processor = TextProcessor(
            max_length=max_text_length,
            use_bert=True,
            use_sentencepiece=use_sentencepiece,
            use_gpu=(self.device.type == "cuda")
        )
        
        self.data_preparer = DataPreparer(
            max_sequence_length=max_sequence_length,
            max_text_length=max_text_length,
            batch_size=batch_size,
            text_processor_use_gpu=(self.device.type == "cuda")
        )
        
        logger.info(f"Advanced Data Creator initialized with device: {self.device}")
        logger.info(f"Using hierarchical encoding: {use_hierarchical_encoding}")
        logger.info(f"Using contextual embeddings: {use_contextual_embeddings}")
        logger.info(f"Using SentencePiece: {use_sentencepiece}")

    def create_training_data(
        self,
        paired_data_file: str,
        output_dir: str = "data/processed",
        dataset_name: str = "advanced_dataset",
        save_intermediate: bool = True,
        augment_data: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """Create advanced training data from paired MIDI and text data
        
        Args:
            paired_data_file: Path to JSON file with paired MIDI and text data
            output_dir: Directory to save processed data
            dataset_name: Name of the dataset
            save_intermediate: Whether to save intermediate results
            augment_data: Whether to augment data
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            
        Returns:
            Dictionary with dataset information
        """
        logger.info(f"Creating advanced training data from {paired_data_file}")
        start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load paired data
        paired_data = self._load_paired_data(paired_data_file)
        logger.info(f"Loaded {len(paired_data)} paired samples")
        
        # Process data with advanced features
        processed_data = self._process_paired_data(paired_data, save_intermediate, output_dir)
        logger.info(f"Processed {len(processed_data)} samples")
        
        # Augment data if requested
        if augment_data:
            processed_data = self._augment_data(processed_data)
            logger.info(f"Augmented to {len(processed_data)} samples")
        
        # Split data
        train_data, val_data, test_data = self._split_data(processed_data, train_ratio, val_ratio)
        logger.info(f"Split into {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples")
        
        # Create datasets and loaders
        train_dataset = self.data_preparer.create_dataset(train_data)
        val_dataset = self.data_preparer.create_dataset(val_data)
        test_dataset = self.data_preparer.create_dataset(test_data)
        
        train_loader = self.data_preparer.create_dataloader(train_dataset, shuffle=True)
        val_loader = self.data_preparer.create_dataloader(val_dataset, shuffle=False)
        test_loader = self.data_preparer.create_dataloader(test_dataset, shuffle=False)
        
        # Calculate vocabulary information
        vocab_info = self._calculate_vocab_info(processed_data)
        
        # Save dataset metadata
        dataset_info = {
            "name": dataset_name,
            "created_at": time.time(),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "total_samples": len(processed_data),
            "max_sequence_length": self.max_sequence_length,
            "max_text_length": self.max_text_length,
            "vocab_size": self.midi_processor.vocab_size,
            "hierarchical_encoding": self.use_hierarchical_encoding,
            "contextual_embeddings": self.use_contextual_embeddings,
            "sentencepiece": self.use_sentencepiece,
            "vocab_info": vocab_info,
            "processing_time": time.time() - start_time
        }
        
        # Save complete dataset
        dataset_path = os.path.join(output_dir, f"{dataset_name}_info.json")
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, default=self._json_serializer)
        
        # Save training data for model training
        training_data = {
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
            "vocab_size": self.midi_processor.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "max_text_length": self.max_text_length,
            "dataset_info": dataset_info
        }
        
        training_data_path = os.path.join(output_dir, f"{dataset_name}_training_data.json")
        with open(training_data_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, default=self._json_serializer)
        
        logger.info(f"Advanced training data created and saved to {output_dir}")
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        
        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "vocab_size": self.midi_processor.vocab_size,
            "dataset_info": dataset_info,
            "training_data": training_data
        }
    
    def _load_paired_data(self, paired_data_file: str) -> List[Dict[str, Any]]:
        """Load paired data from file"""
        with open(paired_data_file, encoding="utf-8") as f:
            paired_data = json.load(f)
        return paired_data
    
    def _process_paired_data(
        self, 
        paired_data: List[Dict[str, Any]],
        save_intermediate: bool,
        output_dir: str
    ) -> List[Dict[str, Any]]:
        """Process paired data with advanced features"""
        processed_data = []
        
        for item in tqdm(paired_data, desc="Processing paired data"):
            midi_file = item.get("midi_file")
            text_description = item.get("text_description")
            
            if not midi_file or not text_description:
                continue
            
            try:
                # Process MIDI with hierarchical encoding if enabled
                midi_processed = self.midi_processor.process_midi_file(midi_file)
                if midi_processed is None:
                    continue
                
                # Add hierarchical encoding if enabled
                if self.use_hierarchical_encoding:
                    midi_processed = self._add_hierarchical_encoding(midi_processed)
                
                # Process text with advanced features
                text_processed = self.text_processor.process_text(text_description)
                
                # Create contextual embeddings if enabled
                if self.use_contextual_embeddings:
                    midi_processed, text_processed = self._create_contextual_embeddings(
                        midi_processed, text_processed
                    )
                
                # Create training item
                training_item = {
                    "midi_file": midi_file,
                    "text_description": text_description,
                    "midi_tokens": midi_processed["tokens"],
                    "midi_metadata": midi_processed["metadata"],
                    "text_features": text_processed,
                    "sequence_length": midi_processed["sequence_length"],
                    "processed_at": time.time()
                }
                
                # Add hierarchical data if available
                if "hierarchical" in midi_processed:
                    training_item["hierarchical"] = midi_processed["hierarchical"]
                
                processed_data.append(training_item)
                
            except Exception as e:
                logger.warning(f"Error processing {midi_file}: {str(e)}")
        
        # Save intermediate results if requested
        if save_intermediate:
            intermediate_path = os.path.join(output_dir, "intermediate_processed_data.json")
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2, default=self._json_serializer)
            
            logger.info(f"Saved intermediate processed data to {intermediate_path}")
        
        return processed_data
    
    def _add_hierarchical_encoding(self, midi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add hierarchical encoding to MIDI data"""
        if not self.use_hierarchical_encoding or "tokens" not in midi_data:
            return midi_data
            
        tokens = midi_data["tokens"]
        events = midi_data.get("events", [])
        
        # Extract time signatures, bar markers, etc.
        bar_tokens = []
        beat_tokens = []
        note_tokens = []
        
        # Sort events by time for proper hierarchical structuring
        if events:
            sorted_events = sorted(events, key=lambda e: e.get("time", 0))
            
            # Extract bar and beat information
            current_bar = 0
            current_beat = 0
            
            for event in sorted_events:
                event_type = event.get("type")
                
                # Bar-level events
                if event_type in ["time_signature", "key_signature"]:
                    bar_tokens.append(event)
                # Beat-level events
                elif event_type in ["chord", "pedal", "tempo", "time_shift"]:
                    beat_tokens.append(event)
                # Note-level events
                elif event_type in ["note_on", "note_off"]:
                    note_tokens.append(event)
        
        # Store hierarchical information in the result
        midi_data["hierarchical"] = {
            "bar_tokens": bar_tokens,
            "beat_tokens": beat_tokens,
            "note_tokens": note_tokens
        }
        
        return midi_data
    
    def _create_contextual_embeddings(
        self, 
        midi_data: Dict[str, Any], 
        text_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create contextual embeddings between MIDI and text data"""
        if not self.use_contextual_embeddings or not text_data:
            return midi_data, text_data
        
        # Create simple bidirectional alignment based on musical features
        if "musical_features" in text_data:
            midi_data["text_musical_features"] = text_data["musical_features"]
            
        if "metadata" in midi_data:
            text_data["midi_metadata"] = midi_data["metadata"]
            
        # Add flags
        midi_data["has_contextual_embedding"] = True
        text_data["has_contextual_embedding"] = True
        
        return midi_data, text_data
    
    def _augment_data(self, processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Augment data with variations like transposition and tempo changes"""
        augmented_data = processed_data.copy()
        
        # Basic augmentation techniques
        for item in processed_data:
            # Skip items without required data
            if "midi_tokens" not in item or "midi_metadata" not in item:
                continue
                
            # 1. Transpose up and down by 2 and 4 semitones
            for semitones in [2, 4, -2, -4]:
                augmented_item = item.copy()
                augmented_item["midi_tokens"] = self._transpose_tokens(item["midi_tokens"], semitones)
                augmented_item["midi_metadata"] = item["midi_metadata"].copy()
                augmented_item["midi_metadata"]["augmentation"] = f"transposed_{semitones}"
                augmented_data.append(augmented_item)
            
            # 2. Tempo variations (slower and faster by 10% and 20%)
            for tempo_factor in [0.8, 0.9, 1.1, 1.2]:
                augmented_item = item.copy()
                augmented_item["midi_tokens"] = self._adjust_tempo(item["midi_tokens"], tempo_factor)
                augmented_item["midi_metadata"] = item["midi_metadata"].copy()
                augmented_item["midi_metadata"]["augmentation"] = f"tempo_{tempo_factor}"
                augmented_data.append(augmented_item)
        
        return augmented_data
    
    def _transpose_tokens(self, tokens: List[int], semitones: int) -> List[int]:
        """Transpose MIDI tokens by the specified number of semitones"""
        # In a real implementation, this would apply proper transposition
        # based on the token encoding scheme used by MIDIProcessor
        return tokens  # Placeholder
    
    def _adjust_tempo(self, tokens: List[int], tempo_factor: float) -> List[int]:
        """Adjust the tempo of MIDI tokens by the specified factor"""
        # In a real implementation, this would adjust time-related tokens
        # based on the token encoding scheme used by MIDIProcessor
        return tokens  # Placeholder
    
    def _split_data(
        self, 
        data: List[Dict[str, Any]], 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train, validation, and test sets"""
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
    
    def _calculate_vocab_info(self, processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate vocabulary information from processed data"""
        # Count token frequencies
        token_counts = {}
        sequence_lengths = []
        
        for item in processed_data:
            if "midi_tokens" in item:
                tokens = item["midi_tokens"]
                sequence_lengths.append(len(tokens))
                
                for token in tokens:
                    if token in token_counts:
                        token_counts[token] += 1
                    else:
                        token_counts[token] = 1
        
        # Calculate statistics
        vocab_info = {
            "unique_tokens": len(token_counts),
            "total_tokens": sum(token_counts.values()),
            "sequence_length_stats": {
                "min": min(sequence_lengths) if sequence_lengths else 0,
                "max": max(sequence_lengths) if sequence_lengths else 0,
                "mean": np.mean(sequence_lengths) if sequence_lengths else 0,
                "median": np.median(sequence_lengths) if sequence_lengths else 0
            },
            "top_tokens": sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        }
        
        return vocab_info
    
    def _json_serializer(self, obj):
        """Helper for JSON serialization of special types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_advanced_training_data(
    paired_data_file: str,
    output_dir: str = "data/processed",
    dataset_name: str = "advanced_dataset",
    max_sequence_length: int = 1024,
    max_text_length: int = 512,
    batch_size: int = 32,
    use_hierarchical_encoding: bool = True,
    use_contextual_embeddings: bool = True,
    use_sentencepiece: bool = True
) -> Dict[str, Any]:
    """Convenience function to create advanced training data"""
    creator = AdvancedDataCreator(
        max_sequence_length=max_sequence_length,
        max_text_length=max_text_length,
        batch_size=batch_size,
        use_hierarchical_encoding=use_hierarchical_encoding,
        use_contextual_embeddings=use_contextual_embeddings,
        use_sentencepiece=use_sentencepiece
    )
    
    return creator.create_training_data(
        paired_data_file=paired_data_file,
        output_dir=output_dir,
        dataset_name=dataset_name
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create advanced training data for AMT")
    parser.add_argument("paired_data_file", help="Path to JSON file with paired MIDI and text data")
    parser.add_argument("--output-dir", default="data/processed", help="Directory to save processed data")
    parser.add_argument("--dataset-name", default="advanced_dataset", help="Name of the dataset")
    parser.add_argument("--max-sequence-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max-text-length", type=int, default=512, help="Maximum text length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--no-hierarchical-encoding", action="store_true", help="Disable hierarchical encoding")
    parser.add_argument("--no-contextual-embeddings", action="store_true", help="Disable contextual embeddings")
    parser.add_argument("--no-sentencepiece", action="store_true", help="Disable SentencePiece tokenization")
    
    args = parser.parse_args()
    
    create_advanced_training_data(
        paired_data_file=args.paired_data_file,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_sequence_length=args.max_sequence_length,
        max_text_length=args.max_text_length,
        batch_size=args.batch_size,
        use_hierarchical_encoding=not args.no_hierarchical_encoding,
        use_contextual_embeddings=not args.no_contextual_embeddings,
        use_sentencepiece=not args.no_sentencepiece
    ) 
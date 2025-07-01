"""
Data Preparer - Prepares training data from processed MIDI and text
"""

import json
import os
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from .midi_processor import MIDIProcessor
from .text_processor import TextProcessor

class MusicTextDataset(Dataset):
    """Dataset for music-text pairs."""
    
    def __init__(self, data: List[Dict[str, Any]], 
                 max_sequence_length: int = 1024,
                 max_text_length: int = 512):
        self.data = data
        self.max_sequence_length = max_sequence_length
        self.max_text_length = max_text_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get MIDI tokens
        midi_tokens = item.get('midi_tokens', [])
        if len(midi_tokens) > self.max_sequence_length:
            midi_tokens = midi_tokens[:self.max_sequence_length]
        
        # Pad MIDI tokens
        midi_tokens = midi_tokens + [0] * (self.max_sequence_length - len(midi_tokens))
        
        # Get text features
        text_features = item.get('text_features', {})
        bert_embedding = text_features.get('bert_embedding', [])
        tfidf_features = text_features.get('tfidf_features', [])
        
        # Pad text features
        if len(bert_embedding) > self.max_text_length:
            bert_embedding = bert_embedding[:self.max_text_length]
        else:
            bert_embedding = bert_embedding + [0] * (self.max_text_length - len(bert_embedding))
        
        if len(tfidf_features) > 1000:
            tfidf_features = tfidf_features[:1000]
        else:
            tfidf_features = tfidf_features + [0] * (1000 - len(tfidf_features))
        
        return {
            'midi_tokens': torch.tensor(midi_tokens, dtype=torch.long),
            'bert_embedding': torch.tensor(bert_embedding, dtype=torch.float),
            'tfidf_features': torch.tensor(tfidf_features, dtype=torch.float),
            'musical_features': text_features.get('musical_features', {}),
            'sequence_length': min(len(item.get('midi_tokens', [])), self.max_sequence_length)
        }

class DataPreparer:
    """Prepares training data from processed MIDI and text data."""
    
    def __init__(self, 
                 max_sequence_length: int = 1024,
                 max_text_length: int = 512,
                 batch_size: int = 32):
        self.max_sequence_length = max_sequence_length
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        
        self.midi_processor = MIDIProcessor(max_sequence_length=max_sequence_length)
        self.text_processor = TextProcessor(max_length=max_text_length)
    
    def load_paired_data(self, paired_data_file: str) -> List[Dict[str, Any]]:
        """Load paired data from file."""
        with open(paired_data_file, 'r', encoding='utf-8') as f:
            paired_data = json.load(f)
        return paired_data
    
    def process_paired_data(self, paired_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process paired data into training format."""
        processed_data = []
        
        for item in paired_data:
            midi_file = item.get('midi_file')
            text_description = item.get('text_description')
            
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
                'midi_file': midi_file,
                'text_description': text_description,
                'midi_tokens': midi_processed['tokens'],
                'midi_metadata': midi_processed['metadata'],
                'text_features': text_processed,
                'sequence_length': midi_processed['sequence_length']
            }
            
            processed_data.append(training_item)
        
        return processed_data
    
    def create_dataset(self, processed_data: List[Dict[str, Any]]) -> MusicTextDataset:
        """Create dataset from processed data."""
        return MusicTextDataset(
            processed_data,
            max_sequence_length=self.max_sequence_length,
            max_text_length=self.max_text_length
        )
    
    def create_dataloader(self, dataset: MusicTextDataset, shuffle: bool = True) -> DataLoader:
        """Create dataloader from dataset."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for Windows compatibility
            drop_last=True
        )
    
    def split_data(self, processed_data: List[Dict[str, Any]], 
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], 
                                                   List[Dict[str, Any]], 
                                                   List[Dict[str, Any]]]:
        """Split data into train/validation/test sets."""
        total_size = len(processed_data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # Shuffle data
        np.random.shuffle(processed_data)
        
        train_data = processed_data[:train_size]
        val_data = processed_data[train_size:train_size + val_size]
        test_data = processed_data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def prepare_training_data(self, paired_data_file: str, 
                             output_dir: str = "data/processed") -> Dict[str, Any]:
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
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'vocab_size': self.midi_processor.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'max_text_length': self.max_text_length
        }
        
        with open(os.path.join(output_dir, 'training_data.json'), 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training data prepared:")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Validation: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")
        print(f"  Vocabulary size: {self.midi_processor.vocab_size}")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'vocab_size': self.midi_processor.vocab_size,
            'training_data': training_data
        }
    
    def get_data_statistics(self, processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the processed data."""
        stats = {
            'total_samples': len(processed_data),
            'avg_sequence_length': 0,
            'avg_text_length': 0,
            'vocab_size': self.midi_processor.vocab_size,
            'text_features': {
                'has_bert': 0,
                'has_tfidf': 0,
                'has_musical_features': 0
            }
        }
        
        if processed_data:
            sequence_lengths = [item.get('sequence_length', 0) for item in processed_data]
            text_lengths = [item.get('text_features', {}).get('text_length', 0) for item in processed_data]
            
            stats['avg_sequence_length'] = np.mean(sequence_lengths)
            stats['avg_text_length'] = np.mean(text_lengths)
            
            # Count feature availability
            for item in processed_data:
                text_features = item.get('text_features', {})
                if text_features.get('bert_embedding'):
                    stats['text_features']['has_bert'] += 1
                if text_features.get('tfidf_features'):
                    stats['text_features']['has_tfidf'] += 1
                if text_features.get('musical_features'):
                    stats['text_features']['has_musical_features'] += 1
        
        return stats

def prepare_training_data(paired_data_file: str, output_dir: str = "data/processed") -> Dict[str, Any]:
    """Convenience function to prepare training data."""
    preparer = DataPreparer()
    return preparer.prepare_training_data(paired_data_file, output_dir) 
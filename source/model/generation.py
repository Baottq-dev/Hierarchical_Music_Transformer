"""
Music generation module for AMT.
Contains functions for generating music from text descriptions using trained model.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import json
import os
import numpy as np
from typing import List, Dict, Any
from ..data_processing.midi_processor import event_sequence_to_midi
from ..data_processing.text_processor import get_bert_embedding

class MusicGenerator:
    """
    Generator for AMT model.
    """
    def __init__(self, model_path: str):
        """
        Initialize generator.
        Args:
            model_path: Path to trained model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self, model_path: str) -> nn.Module:
        """
        Load trained model.
        Args:
            model_path: Path to model checkpoint
        Returns:
            Loaded model
        """
        # Initialize model with default config
        config = GPT2Config(
            vocab_size=512,  # MIDI event vocabulary size
            n_positions=1024,  # Maximum sequence length
            n_embd=1024,  # Hidden dimension
            n_layer=6,  # Number of layers
            n_head=8  # Number of attention heads
        )
        
        model = GPT2LMHeadModel(config)
        
        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        return model
    
    def generate_music(
        self,
        text_description: str,
        output_file: str,
        max_length: int = 512,
        temperature: float = 1.0
    ) -> bool:
        """
        Generate music from text description.
        Args:
            text_description: Text description
            output_file: Path to output MIDI file
            max_length: Maximum sequence length
            temperature: Sampling temperature
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate text embedding
            text_embedding = get_bert_embedding(text_description)
            text_embedding = torch.tensor(text_embedding, device=self.device).unsqueeze(0)
            
            # Generate music sequence
            with torch.no_grad():
                # For now, generate a simple sequence
                # In a full implementation, you would use the model to generate
                # based on the text embedding
                generated_sequence = self._generate_sequence(text_embedding, max_length, temperature)
            
            # Convert to MIDI
            if generated_sequence:
                success = event_sequence_to_midi(generated_sequence, output_file)
                return success
            
            return False
            
        except Exception as e:
            print(f"Error generating music: {e}")
            return False
    
    def _generate_sequence(self, text_embedding: torch.Tensor, max_length: int, temperature: float) -> List:
        """
        Generate music sequence from text embedding.
        Args:
            text_embedding: Text embedding tensor
            max_length: Maximum sequence length
            temperature: Sampling temperature
        Returns:
            Generated event sequence
        """
        # This is a simplified implementation
        # In practice, you would use the trained model to generate sequences
        
        # For demonstration, create a simple sequence
        # (time_on, note, duration) triplets
        sequence = []
        current_time = 0
        
        for i in range(min(max_length // 3, 50)):  # Generate some notes
            time_on = current_time
            note = 60 + (i % 12)  # C4 to B4
            duration = 120  # 2 beats
            
            sequence.append((time_on, note, duration))
            current_time += 240  # 4 beats between notes
        
        return sequence

def generate_from_text(
    model_path: str,
    text_description: str,
    output_file: str,
    temperature: float = 1.0,
    max_length: int = 512
) -> bool:
    """
    Generate music from text description.
    Args:
        model_path: Path to model checkpoint
        text_description: Text description
        output_file: Path to output MIDI file
        temperature: Sampling temperature
        max_length: Maximum sequence length
    Returns:
        True if successful, False otherwise
    """
    try:
        generator = MusicGenerator(model_path)
        return generator.generate_music(
            text_description=text_description,
            output_file=output_file,
            max_length=max_length,
            temperature=temperature
        )
    except Exception as e:
        print(f"Error in generate_from_text: {e}")
        return False

if __name__ == "__main__":
    # Set paths
    model_path = "models/checkpoints/checkpoint_epoch_10.pt"
    output_dir = "output/generated"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Example text descriptions
    text_descriptions = [
        "A happy and energetic pop song with piano and drums",
        "A sad and melancholic jazz piece with saxophone",
        "An intense rock song with electric guitar and drums"
    ]
    
    # Generate music for each description
    for i, text in enumerate(text_descriptions):
        output_file = f"{output_dir}/generated_{i+1}.mid"
        result = generate_from_text(model_path, text, output_file)
        print(f"Generated music for: {text}")
        print(f"Output file: {output_file}") 
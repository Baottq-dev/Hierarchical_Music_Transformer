"""
Music generation module for AMT.
Contains functions for generating music from semantic tokens using the fine-tuned GPT-2 model.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os
from typing import List, Dict, Any
import numpy as np
import torch.nn as nn
from ..config import MODEL_CONFIG, GENERATION_CONFIG
from ..data_processing.midi_processor import event_sequence_to_midi
from ..data_processing.text_processor import get_bert_embedding

class MusicGenerator:
    """Class for generating music from semantic tokens."""
    
    def __init__(self, model_dir: str):
        """
        Initialize the music generator.
        Args:
            model_dir: Directory containing the fine-tuned model and tokenizer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def generate_sequence(self, 
                         prompt: List[str], 
                         max_length: int = 512,
                         temperature: float = 0.7,
                         top_k: int = 50,
                         top_p: float = 0.9,
                         num_return_sequences: int = 1) -> List[List[str]]:
        """
        Generate music sequences from a prompt.
        Args:
            prompt: List of tokens to use as prompt
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            num_return_sequences: Number of sequences to generate
        Returns:
            List of generated token sequences
        """
        # Convert prompt to input_ids
        input_ids = [self.tokenizer.bos_token_id] + \
                   [self.tokenizer.convert_tokens_to_ids(tok) for tok in prompt]
        input_ids = torch.tensor([input_ids], device=self.device)

        # Generate sequences
        with torch.no_grad():
            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Convert output sequences to tokens
        generated_sequences = []
        for sequence in output_sequences:
            tokens = self.tokenizer.convert_ids_to_tokens(sequence)
            # Remove special tokens
            tokens = [t for t in tokens if t not in 
                     [self.tokenizer.bos_token, self.tokenizer.eos_token, 
                      self.tokenizer.pad_token]]
            generated_sequences.append(tokens)

        return generated_sequences

    def generate_from_cluster(self, 
                            cluster_data: Dict[str, Any],
                            num_sequences: int = 1) -> List[List[str]]:
        """
        Generate music sequences from a cluster's representative sequence.
        Args:
            cluster_data: Dictionary containing cluster information
            num_sequences: Number of sequences to generate
        Returns:
            List of generated token sequences
        """
        if "representative_sequence" not in cluster_data:
            raise ValueError("Cluster data must contain 'representative_sequence'")

        prompt = cluster_data["representative_sequence"]
        return self.generate_sequence(
            prompt=prompt,
            num_return_sequences=num_sequences
        )

    def generate_from_style(self, 
                          style_tokens: List[str],
                          num_sequences: int = 1) -> List[List[str]]:
        """
        Generate music sequences from style tokens.
        Args:
            style_tokens: List of style tokens to use as prompt
            num_sequences: Number of sequences to generate
        Returns:
            List of generated token sequences
        """
        return self.generate_sequence(
            prompt=style_tokens,
            num_return_sequences=num_sequences
        )

def load_generator(model_dir: str) -> MusicGenerator:
    """
    Load a pre-trained music generator.
    Args:
        model_dir: Directory containing the fine-tuned model and tokenizer
    Returns:
        MusicGenerator instance
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    return MusicGenerator(model_dir)

def save_generated_sequences(sequences: List[List[str]], 
                           output_file: str,
                           metadata: Dict[str, Any] = None):
    """
    Save generated sequences to a JSON file.
    Args:
        sequences: List of generated token sequences
        output_file: Path to output JSON file
        metadata: Additional metadata to save with sequences
    """
    output_data = {
        "generated_sequences": sequences,
        "metadata": metadata or {}
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

class AMTGenerator:
    """
    Generator for AMT model.
    """
    def __init__(self, model_path: str):
        """
        Initialize generator.
        Args:
            model_path: Path to trained model checkpoint
        """
        # Load model configuration
        self.config = MODEL_CONFIG
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path: str) -> nn.Module:
        """
        Load trained model.
        Args:
            model_path: Path to model checkpoint
        Returns:
            Loaded model
        """
        # Initialize model
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=self.config["vocab_size"],
            n_positions=self.config["max_seq_length"],
            n_embd=self.config["hidden_dim"],
            n_layer=self.config["num_layers"],
            n_head=self.config["num_heads"]
        ))
        
        # Load checkpoint
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def generate_music(
        self,
        text_description: str,
        output_file: str,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None
    ) -> Dict[str, Any]:
        """
        Generate music from text description.
        Args:
            text_description: Text description
            output_file: Path to output MIDI file
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
        Returns:
            Dictionary containing generation results
        """
        # Get generation parameters
        temperature = temperature or GENERATION_CONFIG["temperature"]
        top_k = top_k or GENERATION_CONFIG["top_k"]
        top_p = top_p or GENERATION_CONFIG["top_p"]
        
        # Generate BERT embedding
        text_embedding = get_bert_embedding(text_description)
        text_embedding = torch.tensor(text_embedding).unsqueeze(0)
        
        # Generate event sequence
        with torch.no_grad():
            # Project BERT embedding
            projected_embedding = self.model.projection(text_embedding)
            
            # Generate sequence
            outputs = self.model.generate(
                inputs_embeds=projected_embedding,
                max_length=GENERATION_CONFIG["max_length"],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=GENERATION_CONFIG["num_return_sequences"],
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id
            )
        
        # Convert to MIDI
        event_sequence = outputs[0].numpy().tolist()
        midi_file = event_sequence_to_midi(event_sequence, output_file)
        
        return {
            "text_description": text_description,
            "event_sequence": event_sequence,
            "midi_file": midi_file
        }

def generate_from_text(
    model_path: str,
    text_description: str,
    output_file: str,
    temperature: float = None,
    top_k: int = None,
    top_p: float = None
) -> Dict[str, Any]:
    """
    Generate music from text description.
    Args:
        model_path: Path to trained model checkpoint
        text_description: Text description
        output_file: Path to output MIDI file
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
    Returns:
        Dictionary containing generation results
    """
    # Initialize generator
    generator = AMTGenerator(model_path)
    
    # Generate music
    result = generator.generate_music(
        text_description=text_description,
        output_file=output_file,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    return result

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
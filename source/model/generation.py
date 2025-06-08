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
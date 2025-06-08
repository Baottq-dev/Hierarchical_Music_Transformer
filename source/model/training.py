"""
Training module for AMT model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import List, Dict, Any
from transformers import GPT2LMHeadModel, GPT2Config
from tqdm import tqdm

class AMTDataset(Dataset):
    """
    Dataset for AMT model.
    """
    def __init__(self, data_file: str):
        """
        Initialize dataset.
        Args:
            data_file: Path to training data JSON file
        """
        with open(data_file, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'text_embedding': torch.tensor(item['text_embedding']),
            'event_sequence': torch.tensor(item['event_sequence'])
        }

class AMTModel(nn.Module):
    """
    AMT model combining BERT and GPT-2.
    """
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 1024):
        """
        Initialize model.
        Args:
            embedding_dim: Dimension of BERT embeddings
            hidden_dim: Hidden dimension for GPT-2
        """
        super().__init__()
        
        # GPT-2 configuration
        config = GPT2Config(
            vocab_size=512,  # MIDI event vocabulary size
            n_positions=1024,  # Maximum sequence length
            n_embd=hidden_dim,
            n_layer=6,
            n_head=8
        )
        
        # GPT-2 model
        self.gpt2 = GPT2LMHeadModel(config)
        
        # Projection layer for BERT embeddings
        self.projection = nn.Linear(embedding_dim, hidden_dim)
    
    def forward(self, text_embedding: torch.Tensor, event_sequence: torch.Tensor):
        """
        Forward pass.
        Args:
            text_embedding: BERT embedding tensor
            event_sequence: MIDI event sequence tensor
        Returns:
            Model output
        """
        # Project BERT embedding
        projected_embedding = self.projection(text_embedding)
        
        # Combine with event sequence
        combined_input = torch.cat([projected_embedding, event_sequence], dim=1)
        
        # Generate output
        outputs = self.gpt2(inputs_embeds=combined_input)
        
        return outputs

def train_model(
    data_file: str,
    output_dir: str,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4
):
    """
    Train AMT model.
    Args:
        data_file: Path to training data JSON file
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    """
    # Create dataset and dataloader
    dataset = AMTDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = AMTModel()
    model.train()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Get batch data
            text_embeddings = batch['text_embedding']
            event_sequences = batch['event_sequence']
            
            # Forward pass
            outputs = model(text_embeddings, event_sequences)
            
            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"{output_dir}/checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)

if __name__ == "__main__":
    # Set paths
    data_file = "data/processed/training_data.json"
    output_dir = "models/checkpoints"
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    train_model(data_file, output_dir) 
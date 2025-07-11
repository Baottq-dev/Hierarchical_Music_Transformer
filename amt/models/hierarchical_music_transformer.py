"""
Hierarchical Music Transformer model for AMT system
This model leverages hierarchical structure of music (bar, beat, note levels)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import os

class PositionalEncoding(nn.Module):
    """
    Positional encoding with support for hierarchical structure
    Combines absolute and relative position information
    """
    def __init__(
        self, 
        d_model: int, 
        max_seq_len: int = 5000, 
        dropout: float = 0.1,
        use_relative: bool = True
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.use_relative = use_relative
        
        # Standard positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # Additional projection for relative position
        if use_relative:
            self.rel_pos_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, hierarchical_info=None):
        """
        Apply positional encoding with hierarchical information
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            hierarchical_info: Optional dict with hierarchical position information
            
        Returns:
            Encoded tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        
        # Apply hierarchical position information if available
        if self.use_relative and hierarchical_info is not None:
            # Get bar and beat positions
            bar_positions = hierarchical_info.get('bar_positions')
            beat_positions = hierarchical_info.get('beat_positions')
            
            if bar_positions is not None:
                # Apply bar-level position information
                bar_pos_emb = self.rel_pos_proj(self.pe[:, bar_positions])
                x = x + 0.5 * bar_pos_emb
                
            if beat_positions is not None:
                # Apply beat-level position information
                beat_pos_emb = self.rel_pos_proj(self.pe[:, beat_positions])
                x = x + 0.3 * beat_pos_emb
        
        return self.dropout(x)


class HierarchicalAttention(nn.Module):
    """
    Multi-head attention that considers hierarchical structure
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
        # Bar and beat level attention projections
        self.bar_query = nn.Linear(d_model, d_model)
        self.bar_key = nn.Linear(d_model, d_model)
        
        self.beat_query = nn.Linear(d_model, d_model)
        self.beat_key = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        hierarchical_info: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with hierarchical attention
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            hierarchical_info: Optional hierarchical structure information
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # Standard attention
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply hierarchical attention if provided
        if hierarchical_info is not None:
            bar_indices = hierarchical_info.get('bar_indices')
            beat_indices = hierarchical_info.get('beat_indices')
            
            if bar_indices is not None:
                # Extract bar-level representations
                bar_x = x[:, bar_indices]
                bar_q = self.bar_query(bar_x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                bar_k = self.bar_key(bar_x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                
                # Compute bar-level attention bias
                bar_bias = torch.matmul(q, bar_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                
                # Add bias to main attention scores
                scores = scores + 0.2 * bar_bias
            
            if beat_indices is not None:
                # Extract beat-level representations
                beat_x = x[:, beat_indices]
                beat_q = self.beat_query(beat_x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                beat_k = self.beat_key(beat_x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                
                # Compute beat-level attention bias
                beat_bias = torch.matmul(q, beat_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                
                # Add bias to main attention scores
                scores = scores + 0.1 * beat_bias
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.output(output)


class HierarchicalTransformerBlock(nn.Module):
    """
    Transformer block with hierarchical attention
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        d_ff: int = 2048, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attn = HierarchicalAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        hierarchical_info: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block with hierarchical attention
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            hierarchical_info: Optional hierarchical structure information
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Multi-head attention with hierarchical structure
        attn_output = self.attn(self.norm1(x), mask, hierarchical_info)
        x = x + attn_output
        
        # Feed forward
        ff_output = self.ff(self.norm2(x))
        x = x + ff_output
        
        return x


class HierarchicalMusicTransformer(nn.Module):
    """
    Advanced transformer model for music generation with hierarchical structure
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        use_relative_attention: bool = True,
        use_hierarchical_encoding: bool = True,
        pretrained_model_path: str = None
    ):
        """
        Initialize hierarchical music transformer
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_relative_attention: Whether to use relative attention
            use_hierarchical_encoding: Whether to use hierarchical encoding
            pretrained_model_path: Path to pretrained model weights
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.use_relative_attention = use_relative_attention
        self.use_hierarchical_encoding = use_hierarchical_encoding
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Hierarchical embeddings
        if use_hierarchical_encoding:
            self.bar_embedding = nn.Embedding(10, d_model)  # For different bar positions
            self.beat_embedding = nn.Embedding(16, d_model)  # For different beat positions
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model, max_seq_len, dropout, use_relative=use_relative_attention
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            HierarchicalTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
        # Load pretrained weights if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self.load_pretrained_weights(pretrained_model_path)
            
    def _init_parameters(self):
        """Initialize parameters using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _extract_hierarchical_info(
        self, 
        hierarchical_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract hierarchical information for attention layers
        
        Args:
            hierarchical_data: Dictionary with hierarchical data
            
        Returns:
            Dictionary with processed hierarchical information
        """
        if hierarchical_data is None or not self.use_hierarchical_encoding:
            return None
            
        # Extract indices and positions
        bar_indices = hierarchical_data.get('bar_indices')
        beat_indices = hierarchical_data.get('beat_indices')
        
        if bar_indices is None and beat_indices is None:
            return None
            
        # Convert to tensors if they're lists
        if bar_indices is not None and not isinstance(bar_indices, torch.Tensor):
            bar_indices = torch.tensor(bar_indices, device=self.token_embedding.weight.device)
            
        if beat_indices is not None and not isinstance(beat_indices, torch.Tensor):
            beat_indices = torch.tensor(beat_indices, device=self.token_embedding.weight.device)
        
        # Create positions for positional encoding
        bar_positions = hierarchical_data.get('bar_positions')
        beat_positions = hierarchical_data.get('beat_positions')
        
        if bar_positions is not None and not isinstance(bar_positions, torch.Tensor):
            bar_positions = torch.tensor(bar_positions, device=self.token_embedding.weight.device)
            
        if beat_positions is not None and not isinstance(beat_positions, torch.Tensor):
            beat_positions = torch.tensor(beat_positions, device=self.token_embedding.weight.device)
        
        return {
            'bar_indices': bar_indices,
            'beat_indices': beat_indices,
            'bar_positions': bar_positions,
            'beat_positions': beat_positions
        }
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        hierarchical_data: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical music transformer
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            mask: Optional attention mask
            hierarchical_data: Optional hierarchical structure data
            
        Returns:
            Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.size()
        
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Add hierarchical embeddings if enabled
        hierarchical_info = None
        if self.use_hierarchical_encoding and hierarchical_data is not None:
            hierarchical_info = self._extract_hierarchical_info(hierarchical_data)
            
            bar_positions = hierarchical_data.get('bar_positions')
            if bar_positions is not None:
                bar_pos = torch.tensor(bar_positions, device=x.device).long() % 10
                x = x + self.bar_embedding(bar_pos)
                
            beat_positions = hierarchical_data.get('beat_positions')
            if beat_positions is not None:
                beat_pos = torch.tensor(beat_positions, device=x.device).long() % 16
                x = x + self.beat_embedding(beat_pos)
        
        # Apply positional encoding
        x = self.pos_encoding(x, hierarchical_info)
        
        # Apply transformer layers
        for layer in self.transformer_blocks:
            x = layer(x, mask, hierarchical_info)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        hierarchical_data: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Generate music sequence from a prompt
        
        Args:
            prompt: Input tensor of token indices [batch_size, prompt_len]
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: If specified, limit sampling to top k logits
            top_p: If specified, limit sampling to top p probability mass
            hierarchical_data: Optional hierarchical structure data
            
        Returns:
            Generated sequence [batch_size, max_length]
        """
        self.eval()
        device = prompt.device
        batch_size, prompt_len = prompt.size()
        
        # Initialize sequence with prompt
        generated = prompt.clone()
        
        # Generate tokens one by one
        for i in range(max_length - prompt_len):
            # Create attention mask to prevent attention to future tokens
            seq_len = generated.size(1)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                logits = self(generated, mask, hierarchical_data)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next token to sequence
            generated = torch.cat((generated, next_token), dim=1)
            
            # Update hierarchical data if needed
            if hierarchical_data is not None:
                # In a real implementation, update hierarchical data based on the generated token
                pass
        
        return generated

    def load_pretrained_weights(self, model_path: str, strict: bool = False):
        """
        Load pretrained weights from a saved model
        
        Args:
            model_path: Path to the pretrained model weights
            strict: Whether to require all keys to match
        """
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
                
            # Check for vocabulary size mismatch
            if 'token_embedding.weight' in state_dict:
                pretrained_vocab_size = state_dict['token_embedding.weight'].shape[0]
                
                if pretrained_vocab_size != self.vocab_size:
                    print(f"Warning: Vocabulary size mismatch. Pretrained: {pretrained_vocab_size}, Current: {self.vocab_size}")
                    
                    if pretrained_vocab_size > self.vocab_size:
                        # Truncate the embedding matrix
                        state_dict['token_embedding.weight'] = state_dict['token_embedding.weight'][:self.vocab_size, :]
                        state_dict['output_layer.weight'] = state_dict['output_layer.weight'][:self.vocab_size, :]
                        if 'output_layer.bias' in state_dict:
                            state_dict['output_layer.bias'] = state_dict['output_layer.bias'][:self.vocab_size]
                    else:
                        # Expand the embedding matrix (initialize new tokens randomly)
                        old_embeddings = state_dict['token_embedding.weight']
                        new_embeddings = torch.randn(self.vocab_size, self.d_model) * 0.02  # Small random values
                        new_embeddings[:pretrained_vocab_size, :] = old_embeddings
                        state_dict['token_embedding.weight'] = new_embeddings
                        
                        old_output_weights = state_dict['output_layer.weight']
                        new_output_weights = torch.randn(self.vocab_size, self.d_model) * 0.02
                        new_output_weights[:pretrained_vocab_size, :] = old_output_weights
                        state_dict['output_layer.weight'] = new_output_weights
                        
                        if 'output_layer.bias' in state_dict:
                            old_bias = state_dict['output_layer.bias']
                            new_bias = torch.zeros(self.vocab_size)
                            new_bias[:pretrained_vocab_size] = old_bias
                            state_dict['output_layer.bias'] = new_bias
            
            # Load weights
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
                
            print(f"Successfully loaded pretrained weights from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            return False
            
    def save_for_transfer_learning(self, save_path: str):
        """
        Save model weights in a format suitable for transfer learning
        
        Args:
            save_path: Path to save the model weights
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model configuration along with weights
        config = {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'use_relative_attention': self.use_relative_attention,
            'use_hierarchical_encoding': self.use_hierarchical_encoding
        }
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': config
        }, save_path)
        
        print(f"Model saved for transfer learning at {save_path}")
        
    def freeze_layers(self, num_layers_to_freeze: int = None):
        """
        Freeze layers for fine-tuning
        
        Args:
            num_layers_to_freeze: Number of layers to freeze from the bottom
                                 If None, freezes all except the output layer
        """
        if num_layers_to_freeze is None:
            # Freeze all except output layer
            for name, param in self.named_parameters():
                if 'output_layer' not in name:
                    param.requires_grad = False
        else:
            # Freeze embedding
            for param in self.token_embedding.parameters():
                param.requires_grad = False
                
            # Freeze positional encoding
            for param in self.pos_encoding.parameters():
                param.requires_grad = False
                
            # Freeze specified transformer blocks
            for i in range(min(num_layers_to_freeze, len(self.transformer_blocks))):
                for param in self.transformer_blocks[i].parameters():
                    param.requires_grad = False
                    
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%} of total)")
        
    def unfreeze_all_layers(self):
        """Unfreeze all layers for full fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
            
        print("All layers unfrozen for fine-tuning")


def create_transformer_model(
    vocab_size: int,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    max_seq_len: int = 1024,
    use_relative_attention: bool = True,
    use_hierarchical_encoding: bool = True,
    device: Optional[str] = None
) -> HierarchicalMusicTransformer:
    """
    Create a hierarchical music transformer model
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        use_relative_attention: Whether to use relative attention
        use_hierarchical_encoding: Whether to use hierarchical encoding
        device: Device to use (auto-detects if None)
        
    Returns:
        HierarchicalMusicTransformer model
    """
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Create model
    model = HierarchicalMusicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        use_relative_attention=use_relative_attention,
        use_hierarchical_encoding=use_hierarchical_encoding
    )
    
    # Move to device
    model.to(device)
    
    return model 
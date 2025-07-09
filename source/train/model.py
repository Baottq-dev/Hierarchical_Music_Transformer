"""
Music Transformer Model - Neural network for music generation
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for text-music interaction."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, music_seq, text_seq, mask=None):
        batch_size = music_seq.size(0)

        # Linear transformations
        Q = self.w_q(music_seq).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(text_seq).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(text_seq).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Output projection
        output = self.w_o(context)
        output = self.layer_norm(output + music_seq)

        return output


class MusicTransformer(nn.Module):
    """Enhanced Music Transformer with cross-attention."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        max_text_len: int = 512,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.use_cross_attention = use_cross_attention

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Text processing
        self.text_projection = nn.Linear(768, d_model)  # BERT embedding size
        self.text_pos_encoding = PositionalEncoding(d_model, max_text_len)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # Cross-attention layers
        if use_cross_attention:
            self.cross_attention_layers = nn.ModuleList(
                [CrossAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
            )

        # Output layers
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Musical constraints
        self.tempo_embedding = nn.Embedding(200, d_model)  # Tempo range
        self.key_embedding = nn.Embedding(24, d_model)  # 12 major + 12 minor keys
        self.style_embedding = nn.Embedding(10, d_model)  # Style categories

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        midi_tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        tempo_ids: Optional[torch.Tensor] = None,
        key_ids: Optional[torch.Tensor] = None,
        style_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            midi_tokens: MIDI token sequence [batch_size, seq_len]
            text_embeddings: Text embeddings [batch_size, text_len, 768]
            text_mask: Text attention mask [batch_size, text_len]
            tempo_ids: Tempo category IDs [batch_size]
            key_ids: Key category IDs [batch_size]
            style_ids: Style category IDs [batch_size]

        Returns:
            Logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = midi_tokens.shape

        # MIDI token embeddings
        midi_embeddings = self.token_embedding(midi_tokens)
        midi_embeddings = self.pos_encoding(midi_embeddings.transpose(0, 1)).transpose(0, 1)

        # Add musical constraint embeddings
        if tempo_ids is not None:
            tempo_emb = self.tempo_embedding(tempo_ids).unsqueeze(1).expand(-1, seq_len, -1)
            midi_embeddings = midi_embeddings + tempo_emb

        if key_ids is not None:
            key_emb = self.key_embedding(key_ids).unsqueeze(1).expand(-1, seq_len, -1)
            midi_embeddings = midi_embeddings + key_emb

        if style_ids is not None:
            style_emb = self.style_embedding(style_ids).unsqueeze(1).expand(-1, seq_len, -1)
            midi_embeddings = midi_embeddings + style_emb

        midi_embeddings = self.dropout(midi_embeddings)

        # Text processing
        if text_embeddings is not None:
            text_embeddings = self.text_projection(text_embeddings)
            text_embeddings = self.text_pos_encoding(text_embeddings.transpose(0, 1)).transpose(
                0, 1
            )
            text_embeddings = self.dropout(text_embeddings)

        # Transformer encoding
        if self.use_cross_attention and text_embeddings is not None:
            # Apply cross-attention
            for cross_attn in self.cross_attention_layers:
                midi_embeddings = cross_attn(midi_embeddings, text_embeddings, text_mask)

        # Self-attention
        output = self.transformer_encoder(midi_embeddings)

        # Output projection
        logits = self.output_projection(output)

        return logits

    def generate(
        self,
        text_embeddings: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        tempo_id: Optional[int] = None,
        key_id: Optional[int] = None,
        style_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate music sequence from text embeddings.

        Args:
            text_embeddings: Text embeddings [batch_size, text_len, 768]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            tempo_id: Tempo category ID
            key_id: Key category ID
            style_id: Style category ID

        Returns:
            Generated token sequence [batch_size, max_length]
        """
        batch_size = text_embeddings.size(0)
        device = text_embeddings.device

        # Initialize with start token
        generated = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        generated[:, 0] = 1  # Start token

        # Prepare musical constraint IDs
        if tempo_id is not None:
            tempo_ids = torch.full((batch_size,), tempo_id, dtype=torch.long, device=device)
        else:
            tempo_ids = None

        if key_id is not None:
            key_ids = torch.full((batch_size,), key_id, dtype=torch.long, device=device)
        else:
            key_ids = None

        if style_id is not None:
            style_ids = torch.full((batch_size,), style_id, dtype=torch.long, device=device)
        else:
            style_ids = None

        # Generate tokens
        for i in range(1, max_length):
            # Get current sequence
            current_seq = generated[:, :i]

            # Forward pass
            with torch.no_grad():
                logits = self.forward(
                    current_seq,
                    text_embeddings,
                    tempo_ids=tempo_ids,
                    key_ids=key_ids,
                    style_ids=style_ids,
                )

                # Get next token logits
                next_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                    next_logits = torch.full_like(next_logits, float("-inf"))
                    next_logits.scatter_(1, top_k_indices, top_k_logits)

                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float("-inf")

                # Sample next token
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated[:, i] = next_token.squeeze(-1)

        return generated

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_seq_len": self.max_seq_len,
            "max_text_len": self.max_text_len,
            "use_cross_attention": self.use_cross_attention,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

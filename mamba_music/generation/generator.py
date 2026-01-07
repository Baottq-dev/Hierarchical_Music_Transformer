"""
Music Generator for Mamba Music models.
"""

import os
from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F

from ..data.tokenizer import OctupleTokenizer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MusicGenerator:
    """Generates music from prompts using trained models."""

    def __init__(
        self,
        model=None,
        tokenizer: OctupleTokenizer = None,
        device: str = "auto",
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model
        self.tokenizer = tokenizer or OctupleTokenizer()
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # TODO: Recreate model from config
        # model_config = checkpoint.get("model_config", {})
        # self.model = HybridMambaTransformer(**model_config)
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
        
        logger.info("Checkpoint loaded successfully")

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate music tokens.
        
        Args:
            prompt_tokens: Optional starting tokens [batch, seq_len, 8]
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Generated tokens [batch, seq_len, 8]
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        top_k = top_k or self.top_k
        top_p = top_p or self.top_p
        
        # Initialize with prompt or BOS token
        if prompt_tokens is None:
            # Start with zeros (BOS)
            generated = torch.zeros(1, 1, 8, dtype=torch.long, device=self.device)
        else:
            generated = prompt_tokens.to(self.device)
        
        for _ in range(max_length - generated.size(1)):
            # Get logits
            logits = self.model(generated)
            
            # Handle multi-head output
            if isinstance(logits, dict):
                next_tokens = []
                for name in ["time_sig", "tempo", "bar", "position", 
                             "instrument", "pitch", "duration", "velocity"]:
                    head_logits = logits[name][:, -1, :] / temperature
                    
                    # Top-k
                    if top_k > 0:
                        indices_to_remove = head_logits < torch.topk(head_logits, top_k)[0][..., -1, None]
                        head_logits[indices_to_remove] = float('-inf')
                    
                    # Sample
                    probs = F.softmax(head_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    next_tokens.append(next_token)
                
                next_token = torch.cat(next_tokens, dim=-1).unsqueeze(1)
            else:
                # Single head - not typical for Octuple
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated

    def generate_to_midi(
        self,
        output_path: str,
        prompt_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> str:
        """Generate music and save as MIDI file."""
        tokens = self.generate(prompt_tokens, **kwargs)
        tokens_list = tokens[0].cpu().numpy().tolist()
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.tokenizer.detokenize(tokens_list, output_path)
        
        logger.info(f"Generated MIDI saved to: {output_path}")
        return output_path

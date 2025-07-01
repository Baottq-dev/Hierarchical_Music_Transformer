"""
Music Generator - Generates music from text descriptions
"""

import os
import json
import torch
import numpy as np
import pretty_midi
from typing import Dict, Any, Optional, List
from .model import MusicTransformer
from ..process.midi_processor import MIDIProcessor
from ..process.text_processor import TextProcessor

class MusicGenerator:
    """Generates music from text descriptions."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 max_length: int = 512,
                 temperature: float = 1.0):
        self.device = self._get_device(device)
        self.max_length = max_length
        self.temperature = temperature
        
        # Initialize processors
        self.midi_processor = MIDIProcessor()
        self.text_processor = TextProcessor()
        
        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        print(f"Loading model from {model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model_config = checkpoint.get('model_config', {})
        self.model = MusicTransformer(
            vocab_size=model_config.get('vocab_size', 1000),
            d_model=model_config.get('d_model', 512),
            n_heads=model_config.get('n_heads', 8),
            n_layers=model_config.get('n_layers', 6)
        )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def process_text(self, text_description: str) -> torch.Tensor:
        """Process text description into embeddings."""
        # Process text
        text_features = self.text_processor.process_text(text_description)
        
        # Get BERT embedding
        bert_embedding = text_features.get('bert_embedding')
        if bert_embedding is not None:
            embedding = torch.tensor(bert_embedding, dtype=torch.float32)
        else:
            # Fallback to TF-IDF
            tfidf_features = text_features.get('tfidf_features', [])
            embedding = torch.tensor(tfidf_features, dtype=torch.float32)
        
        # Reshape to [1, seq_len, embedding_dim]
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0).unsqueeze(0)
        elif embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)
        
        return embedding.to(self.device)
    
    def generate_music(self, 
                      text_description: str,
                      output_file: str,
                      style_id: Optional[int] = None,
                      tempo_id: Optional[int] = None,
                      key_id: Optional[int] = None,
                      **generation_kwargs) -> Dict[str, Any]:
        """Generate music from text description."""
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        print(f"Generating music for: {text_description}")
        
        # Process text
        text_embeddings = self.process_text(text_description)
        
        # Generate tokens
        with torch.no_grad():
            generated_tokens = self.model.generate(
                text_embeddings=text_embeddings,
                max_length=self.max_length,
                temperature=self.temperature,
                style_id=style_id,
                tempo_id=tempo_id,
                key_id=key_id,
                **generation_kwargs
            )
        
        # Convert tokens to MIDI
        midi_data = self.tokens_to_midi(generated_tokens[0].cpu().numpy())
        
        # Save MIDI file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        midi_data.write(output_file)
        
        result = {
            'text_description': text_description,
            'output_file': output_file,
            'generated_tokens': generated_tokens[0].cpu().numpy().tolist(),
            'sequence_length': len(generated_tokens[0]),
            'style_id': style_id,
            'tempo_id': tempo_id,
            'key_id': key_id
        }
        
        print(f"Music generated and saved to: {output_file}")
        return result
    
    def tokens_to_midi(self, tokens: List[int]) -> pretty_midi.PrettyMIDI:
        """Convert token sequence to MIDI data."""
        # Convert tokens to events
        events = self.midi_processor.tokens_to_events(tokens)
        
        # Create MIDI data
        midi_data = pretty_midi.PrettyMIDI()
        
        # Create piano instrument
        piano = pretty_midi.Instrument(program=0)  # Piano
        
        current_time = 0.0
        active_notes = {}  # pitch -> start_time
        
        for event in events:
            if event['type'] == 'note_on':
                pitch = event['pitch']
                velocity = event.get('velocity', 64)
                active_notes[pitch] = current_time
                
            elif event['type'] == 'note_off':
                pitch = event['pitch']
                if pitch in active_notes:
                    start_time = active_notes[pitch]
                    end_time = current_time
                    
                    # Create note
                    note = pretty_midi.Note(
                        velocity=64,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    piano.notes.append(note)
                    
                    del active_notes[pitch]
                    
            elif event['type'] == 'time_shift':
                current_time += event.get('duration', 0.125)
        
        # Add instrument to MIDI data
        midi_data.instruments.append(piano)
        
        return midi_data
    
    def generate_batch(self, 
                      text_descriptions: List[str],
                      output_dir: str = "output/generated",
                      **generation_kwargs) -> List[Dict[str, Any]]:
        """Generate music for multiple text descriptions."""
        results = []
        
        for i, text in enumerate(text_descriptions):
            output_file = os.path.join(output_dir, f"generated_{i+1}.mid")
            
            result = self.generate_music(
                text_description=text,
                output_file=output_file,
                style_id=i % 5,  # Cycle through styles
                **generation_kwargs
            )
            
            results.append(result)
        
        return results
    
    def generate_with_controls(self, 
                              text_description: str,
                              output_file: str,
                              controls: Dict[str, Any]) -> Dict[str, Any]:
        """Generate music with specific controls."""
        # Extract controls
        style_id = controls.get('style_id')
        tempo_id = controls.get('tempo_id')
        key_id = controls.get('key_id')
        temperature = controls.get('temperature', self.temperature)
        max_length = controls.get('max_length', self.max_length)
        
        return self.generate_music(
            text_description=text_description,
            output_file=output_file,
            style_id=style_id,
            tempo_id=tempo_id,
            key_id=key_id,
            temperature=temperature,
            max_length=max_length
        )

def generate_from_text(model_path: str,
                      text_description: str,
                      output_file: str,
                      style_id: Optional[int] = None,
                      **kwargs) -> Dict[str, Any]:
    """Convenience function to generate music from text."""
    generator = MusicGenerator(model_path=model_path)
    return generator.generate_music(
        text_description=text_description,
        output_file=output_file,
        style_id=style_id,
        **kwargs
    ) 
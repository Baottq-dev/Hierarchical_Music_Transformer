"""OctupleMIDI tokenizer wrapper using MidiTok."""

from pathlib import Path
from typing import List, Union, Optional, Dict
import json

try:
    from miditok import Octuple, TokenizerConfig
    MIDITOK_AVAILABLE = True
except ImportError:
    MIDITOK_AVAILABLE = False


class OctupleTokenizer:
    """Wrapper for MidiTok Octuple tokenizer.
    
    Octuple encodes each note as a tuple of 8 components:
    (TimeSig, Tempo, Bar, Position, Instrument, Pitch, Duration, Velocity)
    """
    
    COMPONENTS = [
        "time_sig", "tempo", "bar", "position",
        "instrument", "pitch", "duration", "velocity"
    ]
    
    def __init__(
        self,
        pitch_range: tuple = (21, 108),
        beat_res: dict = None,
        nb_velocities: int = 32,
        use_chords: bool = True,
        use_rests: bool = True,
        use_tempos: bool = True,
        use_time_signatures: bool = True,
        use_programs: bool = False,
    ):
        if not MIDITOK_AVAILABLE:
            raise ImportError(
                "miditok is not installed. Run: pip install miditok>=3.0.0"
            )
        
        if beat_res is None:
            beat_res = {(0, 4): 8, (4, 12): 4}
        
        config = TokenizerConfig(
            pitch_range=pitch_range,
            beat_res=beat_res,
            nb_velocities=nb_velocities,
            additional_tokens={
                "Chord": use_chords,
                "Rest": use_rests,
                "Tempo": use_tempos,
                "TimeSignature": use_time_signatures,
                "Program": use_programs,
            },
        )
        self._tokenizer = Octuple(config)
        self._vocab_size = len(self._tokenizer.vocab)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def vocab(self) -> dict:
        return self._tokenizer.vocab
    
    def tokenize(self, midi_path: Union[str, Path]) -> List[List[int]]:
        """Tokenize a MIDI file.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            List of Octuple tokens, each token is [8 integers]
        """
        tokens = self._tokenizer(midi_path)
        return tokens.ids
    
    def detokenize(self, tokens: List[List[int]], output_path: str) -> None:
        """Convert tokens back to MIDI file.
        
        Args:
            tokens: List of Octuple tokens
            output_path: Path to save MIDI file
        """
        midi = self._tokenizer.tokens_to_midi(tokens)
        midi.dump(output_path)
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary size for each component."""
        # Default sizes
        return {
            "time_sig": 16,
            "tempo": 64,
            "bar": 512,
            "position": 128,
            "instrument": 128,
            "pitch": 128,
            "duration": 128,
            "velocity": 32,
        }
    
    def save(self, path: str) -> None:
        """Save tokenizer config."""
        self._tokenizer.save_params(path)
    
    @classmethod
    def load(cls, path: str) -> "OctupleTokenizer":
        """Load tokenizer from saved config."""
        if not MIDITOK_AVAILABLE:
            raise ImportError("miditok is not installed")
        
        tokenizer = cls.__new__(cls)
        tokenizer._tokenizer = Octuple(params=path)
        tokenizer._vocab_size = len(tokenizer._tokenizer.vocab)
        return tokenizer

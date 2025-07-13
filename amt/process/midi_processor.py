"""
MIDI Processor - Processes MIDI files for training
"""

import concurrent.futures
import hashlib
import inspect
import json
import numpy as np
import os
import tempfile
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple

import mido
import torch
import pretty_midi
from tqdm import tqdm

import torch

from amt.utils.logging import get_logger

logger = get_logger(__name__)


class MidiProcessor:
    """Processes MIDI files for training and generation."""

    def __init__(
        self,
        max_sequence_length: int = 1024,
        use_pretrained_model: bool = False,
        pretrained_model_path: Optional[str] = None,
        use_hierarchical_encoding: bool = False,
        device: str = "cpu",
        use_mixed_precision: bool = False,
        use_cache: bool = True,
        cache_dir: str = "data/processed/midi_cache"
    ):
        """Initialize MidiProcessor
        
        Args:
            max_sequence_length: Maximum sequence length
            use_pretrained_model: Whether to use pretrained model
            pretrained_model_path: Path to pretrained model
            use_hierarchical_encoding: Whether to use hierarchical encoding
            device: Device to use (cpu, cuda)
            use_mixed_precision: Whether to use mixed precision (FP16) for faster processing
            use_cache: Whether to use cache for processed MIDI files
            cache_dir: Directory to store cached files
        """
        self.max_sequence_length = max_sequence_length
        self.use_pretrained_model = use_pretrained_model
        self.pretrained_model_path = pretrained_model_path
        self.use_hierarchical_encoding = use_hierarchical_encoding
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device == "cuda"
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Create cache directory if needed
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize vocabulary
        self.vocab_size = 128 + 128 + 100  # 128 note-on, 128 note-off, 100 time-shift
        
        # Load pretrained model if specified
        self.pretrained_model = None
        self.feature_extractor = None
        if use_pretrained_model and pretrained_model_path:
            self.load_pretrained_model()

    def load_pretrained_model(self):
        """Load pretrained model"""
        if not self.use_pretrained_model or not self.pretrained_model_path:
            logger.warning("No pretrained model specified. Skipping.")
            return
        
        try:
            logger.info(f"Loading pretrained music model from {self.pretrained_model_path}")
            
            # Try to load as a Hugging Face model
            try:
                from transformers import AutoModel, AutoConfig, AutoProcessor, AutoFeatureExtractor
                import torch
                
                # Try to load as a Hugging Face model
                logger.info(f"Attempting to load as a Hugging Face model: {self.pretrained_model_path}")
                
                # List of valid music models to try
                valid_models = [
                    self.pretrained_model_path,  # Try the specified path first
                    'm-a-p/MERT-v1-95M',         # MERT model (95M parameters)
                    'm-a-p/MERT-v1-330M',        # MERT model (330M parameters)
                    'wazenmai/MIDI-BERT'         # MIDI-BERT model
                ]
                
                # Try each model in order
                for model_path in valid_models:
                    logger.info(f"Trying to load model: {model_path}")
                    try:
                        # Try to load the model with trust_remote_code=True for custom models
                        self.pretrained_model = AutoModel.from_pretrained(
                            model_path, 
                            trust_remote_code=True
                        )
                        
                        # Move model to device
                        self.pretrained_model = self.pretrained_model.to(self.device)
                        
                        # Enable mixed precision if requested and on CUDA
                        if self.use_mixed_precision and self.device == "cuda":
                            logger.info("Enabling mixed precision (FP16) for faster processing")
                            # Set model to use half precision
                            self.pretrained_model = self.pretrained_model.half()
                        
                        # Try to load the feature extractor or processor
                        try:
                            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                                model_path,
                                trust_remote_code=True
                            )
                            logger.info(f"Successfully loaded feature extractor for {model_path}")
                        except Exception as e:
                            logger.warning(f"Could not load feature extractor: {str(e)}")
                            try:
                                self.feature_extractor = AutoProcessor.from_pretrained(
                                    model_path,
                                    trust_remote_code=True
                                )
                                logger.info(f"Successfully loaded processor for {model_path}")
                            except Exception as e:
                                logger.warning(f"Could not load processor: {str(e)}")
                        
                        logger.info(f"Successfully loaded model {model_path} on {self.device}")
                        self.pretrained_model_path = model_path
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_path}: {str(e)}")
                        continue
                
                if self.pretrained_model is None:
                    logger.error("Failed to load any model. Using default features.")
            
            except ImportError as e:
                logger.error(f"Could not import transformers: {str(e)}")
                logger.error("Please install transformers: pip install transformers")
        
        except Exception as e:
            logger.error(f"Error loading pretrained model: {str(e)}")
            logger.error(traceback.format_exc())

    def _load_pretrained_model(self):
        """Load pretrained music model for feature extraction"""
        try:
            logger.info(f"Loading pretrained music model from {self.pretrained_model_path}")
            
            # Check if it's a Hugging Face model path
            if '/' in self.pretrained_model_path:
                try:
                    from transformers import AutoModel, AutoConfig, AutoProcessor
                    
                    # Try to load as a Hugging Face model
                    logger.info(f"Attempting to load as a Hugging Face model: {self.pretrained_model_path}")
                    
                    # List of valid music models to try
                    valid_models = [
                        self.pretrained_model_path,  # Try the specified path first
                        'm-a-p/MERT-v1-95M',         # MERT model (95M parameters)
                        'm-a-p/MERT-v1-330M',        # MERT model (330M parameters)
                        'wazenmai/MIDI-BERT'         # MIDI-BERT model
                    ]
                    
                    # Try each model until one works
                    for model_path in valid_models:
                        try:
                            logger.info(f"Trying to load model: {model_path}")
                            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                            self.feature_extractor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True) if hasattr(AutoProcessor, 'from_pretrained') else None
                            self.pretrained_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
                            logger.info(f"Successfully loaded model {model_path}")
                            return
                        except Exception as e:
                            logger.warning(f"Failed to load model {model_path}: {str(e)}")
                    
                    # If we get here, none of the models worked
                    raise ValueError("Could not load any of the specified music models")
                    
                except Exception as e:
                    logger.warning(f"Failed to load as Hugging Face model: {str(e)}")
                    logger.warning("Falling back to torch.load")
                    
                    # Try to load as a local path
                    if os.path.exists(self.pretrained_model_path):
                        self.pretrained_model = torch.load(self.pretrained_model_path)
                        return
                    else:
                        raise FileNotFoundError(f"Pretrained model path does not exist: {self.pretrained_model_path}")
            
            # If not a Hugging Face model, try to load as a local path
            elif os.path.exists(self.pretrained_model_path):
                self.pretrained_model = torch.load(self.pretrained_model_path)
                return
            else:
                raise FileNotFoundError(f"Pretrained model path does not exist: {self.pretrained_model_path}")
                
        except Exception as e:
            logger.error(f"Error loading pretrained model: {str(e)}")
            logger.error(traceback.format_exc())
            self.pretrained_model = None

    def _get_cache_path(self, midi_file: str) -> str:
        """Get cache file path for a MIDI file."""
        if not self.use_cache:
            return None

        # Create a hash of the file path to avoid issues with special characters
        file_hash = hashlib.md5(midi_file.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{file_hash}.json")

    def load_midi(self, midi_file: str) -> Optional[pretty_midi.PrettyMIDI]:
        """Load MIDI file with multiple fallback methods for error tolerance."""
        # Try the fastest method first (with skip_validation)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                return pretty_midi.PrettyMIDI(midi_file, skip_validation=True)
        except Exception:
            pass

        # Method 2: Direct PrettyMIDI loading (without skip_validation)
        try:
            return pretty_midi.PrettyMIDI(midi_file)
        except Exception as e:
            # Suppress warning output for known errors
            if (
                "data byte must be in range" in str(e)
                or "tick" in str(e)
                or "Could not decode key" in str(e)
            ):
                pass
            else:
                print(f"Error loading MIDI file {midi_file}: {e}")

        # Method 3: Try loading with mido (most tolerant but slowest)
        try:
            # Suppress specific warnings during loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                # Load with mido (more tolerant)
                mido_obj = mido.MidiFile(midi_file, clip=True)

                # Create a temporary clean MIDI file
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                    mido_obj.save(tmp.name)
                    tmp_path = tmp.name

                # Try loading the cleaned file with PrettyMIDI
                try:
                    midi_data = pretty_midi.PrettyMIDI(tmp_path)
                    os.unlink(tmp_path)  # Clean up temp file
                    return midi_data
                except Exception:
                    os.unlink(tmp_path)  # Clean up temp file
                    raise
        except Exception as e:
            # Still failed with mido
            print(f"Failed to load {midi_file} with all methods: {e}")

        # All methods failed
        return None

    def extract_features(self, midi_data, cache_path=None):
        """Extract features using pretrained model
        
        Args:
            midi_data: MIDI data
            cache_path: Path to cache extracted features
            
        Returns:
            Dictionary of extracted features
        """
        if self.pretrained_model is None:
            logger.warning("No pretrained model loaded. Returning empty features.")
            return {}
        
        # Check if cached features exist
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cached features: {str(e)}")
        
        try:
            # Convert MIDI to audio using pretty_midi
            if hasattr(midi_data, 'synthesize'):
                try:
                    # Limit MIDI duration to avoid memory issues
                    max_duration = 30.0  # seconds
                    if hasattr(midi_data, 'get_end_time'):
                        original_duration = midi_data.get_end_time()
                        if original_duration > max_duration:
                            logger.warning(f"Limiting MIDI duration from {original_duration:.2f}s to {max_duration}s")
                            # Truncate note events
                            for instrument in midi_data.instruments:
                                instrument.notes = [note for note in instrument.notes if note.start < max_duration]
                    
                    # Use 24000Hz sampling rate as required by the MERT model
                    audio = midi_data.synthesize(fs=24000)
                    
                    # Limit audio length to prevent memory issues (max 30 seconds)
                    max_samples = 30 * 24000
                    if len(audio) > max_samples:
                        logger.warning(f"Truncating audio from {len(audio)} to {max_samples} samples")
                        audio = audio[:max_samples]
                    
                    # Reshape audio to match expected input format - MERT expects [batch_size, sequence_length]
                    # Not [batch_size, channels, channels, sequence_length]
                    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, samples]
                    
                    # Move to device
                    audio_tensor = audio_tensor.to(self.device)
                    
                    # Process with feature extractor if available
                    if self.feature_extractor:
                        inputs = self.feature_extractor(
                            audio_tensor, 
                            sampling_rate=24000, 
                            return_tensors="pt"
                        )
                        
                        # Move inputs to device
                        for key, val in inputs.items():
                            if isinstance(val, torch.Tensor):
                                inputs[key] = val.to(self.device)
                        
                        # Fix input shape if needed - critical fix for MERT model
                        if 'input_values' in inputs:
                            # Check tensor shape and fix if necessary
                            input_shape = inputs['input_values'].shape
                            if len(input_shape) == 4:  # [1, 1, 1, sequence_length]
                                # Reshape to expected format [batch_size, sequence_length]
                                inputs['input_values'] = inputs['input_values'].squeeze(1).squeeze(1)
                            elif len(input_shape) == 3 and input_shape[1] == 1:  # [1, 1, sequence_length]
                                # Reshape to [batch_size, sequence_length]
                                inputs['input_values'] = inputs['input_values'].squeeze(1)
                        
                        # Use mixed precision if enabled
                        if self.use_mixed_precision:
                            with torch.amp.autocast(device_type='cuda'):  # Updated syntax
                                with torch.no_grad():
                                    outputs = self.pretrained_model(**inputs)
                        else:
                            with torch.no_grad():
                                outputs = self.pretrained_model(**inputs)
                            
                        # Extract embeddings (last hidden state)
                        if hasattr(outputs, 'last_hidden_state'):
                            embeddings = outputs.last_hidden_state
                        else:
                            embeddings = outputs[0]  # Fallback
                        
                        # Average pooling over time dimension
                        pooled_embedding = torch.mean(embeddings, dim=1)
                        
                        features = {
                            'sequence_embedding': pooled_embedding.squeeze().cpu().numpy().tolist(),
                            'model_name': self.pretrained_model_path
                        }
                    else:
                        # Fallback if no feature extractor
                        features = {
                            'sequence_embedding': [0.0] * 512,  # Default embedding size
                            'model_name': 'none'
                        }
                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Return default features on error
                    features = {
                        'sequence_embedding': [0.0] * 512,
                        'model_name': 'error_processing'
                    }
            else:
                logger.warning("MIDI data does not have synthesize method")
                features = {
                    'sequence_embedding': [0.0] * 512,  # Default embedding size
                    'model_name': 'none'
                }
                
            # Cache features
            if cache_path:
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(features, f)
                except Exception as e:
                    logger.warning(f"Error caching features: {str(e)}")
                    
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'sequence_embedding': [0.0] * 512,  # Default embedding size
                'model_name': 'error'
            }
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._make_json_serializable(obj.tolist())
        else:
            return obj
    
    def get_vocab_size(self):
        """Get the vocabulary size"""
        return self.vocab_size
    
    def extract_hierarchical_info(self, midi_data):
        """Extract hierarchical information from MIDI data if enabled"""
        if not self.use_hierarchical_encoding:
            return None
            
        # Extract time signature
        time_sig = midi_data.time_signature_changes[0] if midi_data.time_signature_changes else None
        numerator = time_sig.numerator if time_sig else 4
        denominator = time_sig.denominator if time_sig else 4
        
        # Calculate ticks per bar and beat
        ticks_per_beat = midi_data.resolution
        ticks_per_bar = ticks_per_beat * numerator * (4 / denominator)
        
        # Extract bar and beat positions
        bar_positions = []
        beat_positions = []
        bar_indices = []
        beat_indices = []
        
        end_time_ticks = int(midi_data.get_end_time() * ticks_per_beat)
        
        for tick in range(0, end_time_ticks, ticks_per_beat // 4):  # Quarter beat resolution
            time = tick / ticks_per_beat
            
            # Check if this is a bar position
            if tick % int(ticks_per_bar) == 0:
                bar_positions.append(time)
                bar_indices.append(len(beat_positions))
            
            # Check if this is a beat position
            if tick % ticks_per_beat == 0:
                beat_positions.append(time)
                beat_indices.append(len(beat_positions) - 1)
        
        return {
            "bar_positions": bar_positions,
            "beat_positions": beat_positions,
            "bar_indices": bar_indices,
            "beat_indices": beat_indices,
            "time_signature": (numerator, denominator)
        }
        
    def extract_events(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
        """Extract events from MIDI data."""
        events = []

        # Collect all notes from all instruments
        all_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_notes.append(
                    {
                        "pitch": note.pitch,
                        "velocity": note.velocity,
                        "start": note.start,
                        "end": note.end,
                        "instrument": instrument.program,
                    }
                )

        # Sort by start time
        all_notes.sort(key=lambda x: x["start"])

        # Convert to events
        current_time = 0.0
        for note in all_notes:
            # Time shift event
            time_diff = note["start"] - current_time
            if time_diff > 0:
                time_shift_events = self._create_time_shift_events(time_diff)
                events.extend(time_shift_events)
                current_time = note["start"]

            # Note on event
            events.append(
                {
                    "type": "note_on",
                    "pitch": note["pitch"],
                    "velocity": note["velocity"],
                    "time": note["start"],
                }
            )

            # Note off event
            events.append({"type": "note_off", "pitch": note["pitch"], "time": note["end"]})

        return events

    def _create_time_shift_events(self, time_diff: float) -> List[Dict[str, Any]]:
        """Create time shift events for a given time difference."""
        events = []
        remaining_time = time_diff

        while remaining_time > 0:
            shift_time = min(remaining_time, self.time_resolution)
            events.append({"type": "time_shift", "duration": shift_time, "time": shift_time})
            remaining_time -= shift_time

        return events

    def events_to_tokens(self, events: List[Dict[str, Any]]) -> List[int]:
        """Convert events to token sequence."""
        tokens = []

        for event in events:
            if event["type"] == "note_on":
                # Note on token: pitch + offset
                pitch = event["pitch"]
                if self.min_pitch <= pitch <= self.max_pitch:
                    token = len(self.event_types) + (pitch - self.min_pitch)
                    tokens.append(token)

                # Velocity token
                velocity = event["velocity"]
                velocity_token = (
                    len(self.event_types)
                    + (self.max_pitch - self.min_pitch + 1)
                    + min(velocity // 4, self.velocity_bins - 1)
                )
                tokens.append(velocity_token)

            elif event["type"] == "note_off":
                # Note off token: pitch + offset + special marker
                pitch = event["pitch"]
                if self.min_pitch <= pitch <= self.max_pitch:
                    token = len(self.event_types) + (pitch - self.min_pitch)
                    tokens.append(token)

            elif event["type"] == "time_shift":
                # Time shift token
                time_token = (
                    len(self.event_types)
                    + (self.max_pitch - self.min_pitch + 1)
                    + self.velocity_bins
                )
                tokens.append(time_token)

        return tokens

    def tokens_to_events(self, tokens: List[int]) -> List[Dict[str, Any]]:
        """Convert token sequence to events."""
        events = []
        current_time = 0.0
        current_velocity = 64  # Default velocity

        for token in tokens:
            if token < len(self.event_types):
                # Event type token
                event_type = self.event_types[token]
                if event_type == "time_shift":
                    events.append(
                        {"type": "time_shift", "duration": self.time_resolution, "time": current_time}
                    )
                    current_time += self.time_resolution
            elif token < len(self.event_types) + (self.max_pitch - self.min_pitch + 1):
                # Note on token
                pitch = token - len(self.event_types) + self.min_pitch
                events.append(
                    {
                        "type": "note_on",
                        "pitch": pitch,
                        "velocity": current_velocity,
                        "time": current_time,
                    }
                )
            elif token < len(self.event_types) + (self.max_pitch - self.min_pitch + 1) + self.velocity_bins:
                # Velocity token
                velocity_idx = (
                    token - len(self.event_types) - (self.max_pitch - self.min_pitch + 1)
                )
                current_velocity = min((velocity_idx + 1) * 4, 127)
            else:
                # Note off token
                pitch = (
                    token
                    - len(self.event_types)
                    - (self.max_pitch - self.min_pitch + 1)
                    - self.velocity_bins
                    + self.min_pitch
                )
                events.append({"type": "note_off", "pitch": pitch, "time": current_time})

        return events
        
    def events_to_midi(self, events: List[Dict[str, Any]]) -> pretty_midi.PrettyMIDI:
        """Convert events to MIDI data."""
        midi_data = pretty_midi.PrettyMIDI()
        
        # Create piano instrument
        piano = pretty_midi.Instrument(program=0)  # Piano
        
        current_time = 0.0
        active_notes = {}  # pitch -> (start_time, velocity)
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x.get("time", 0))
        
        for event in sorted_events:
            if event["type"] == "note_on":
                pitch = event["pitch"]
                velocity = event.get("velocity", 64)
                active_notes[pitch] = (current_time, velocity)
                
            elif event["type"] == "note_off":
                pitch = event["pitch"]
                if pitch in active_notes:
                    start_time, velocity = active_notes[pitch]
                    end_time = current_time
                    
                    # Create note
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    piano.notes.append(note)
                    
                    del active_notes[pitch]
                    
            elif event["type"] == "time_shift":
                current_time += event.get("duration", self.time_resolution)
        
        # Add any notes that haven't been turned off
        for pitch, (start_time, velocity) in active_notes.items():
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=current_time + 0.5  # Add a small duration
            )
            piano.notes.append(note)
        
        # Add instrument to MIDI data
        midi_data.instruments.append(piano)
        
        return midi_data

    def process_midi_file(self, midi_file: str) -> Optional[Dict[str, Any]]:
        """Process a single MIDI file."""
        # Handle Kaggle environment path adjustment
        original_midi_file = midi_file
        if midi_file.startswith("data/midi/"):
            # Check if we're running in Kaggle environment
            if os.path.exists("/kaggle/input"):
                # Try different Kaggle dataset paths
                possible_paths = [
                    "/kaggle/input/midi-dataset/midi",
                    "/kaggle/input/your-dataset/midi",
                    "/kaggle/input/lakh-midi-dataset/midi",
                    "/kaggle/input/midi-files/midi",
                    "/kaggle/working/data/midi"
                ]
                
                relative_path = midi_file[len("data/midi/"):]
                for kaggle_path in possible_paths:
                    potential_path = os.path.join(kaggle_path, relative_path)
                    if os.path.exists(potential_path):
                        midi_file = potential_path
                        logger.info(f"Using Kaggle path: {midi_file}")
                        break
                
                # If we didn't find the file but we're in Kaggle, try harder with a general search
                if midi_file == original_midi_file:
                    for root_path in ["/kaggle/input", "/kaggle/working"]:
                        if os.path.exists(root_path):
                            for root, dirs, files in os.walk(root_path):
                                file_name = os.path.basename(original_midi_file)
                                for file in files:
                                    if file == file_name:
                                        midi_file = os.path.join(root, file)
                                        logger.info(f"Found file via search: {midi_file}")
                                        break

        # Check cache first if enabled
        cache_path = self._get_cache_path(original_midi_file)  # Use original path for cache key
        if self.use_cache and cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache for {midi_file}: {e}")

        # Process the file if not in cache
        midi_data = self.load_midi(midi_file)
        if midi_data is None:
            logger.warning(f"Failed to load MIDI file: {midi_file}")
            return None

        # Extract events
        events = self.extract_events(midi_data)

        # Convert to tokens
        tokens = self.events_to_tokens(events)

        # Truncate if too long
        if len(tokens) > self.max_sequence_length:
            tokens = tokens[: self.max_sequence_length]

        # Get metadata safely
        try:
            end_time = midi_data.get_end_time()
        except Exception:
            end_time = 0.0

        try:
            tempo = midi_data.estimate_tempo()
        except Exception:
            tempo = 120.0

        # Collect instrument programs
        instruments = []
        try:
            for instrument in midi_data.instruments:
                if instrument.program not in instruments:
                    instruments.append(instrument.program)
        except Exception:
            pass

        # Create result
        result = {
            "tokens": tokens,
            "sequence_length": len(tokens),
            "metadata": {
                "file_path": original_midi_file,  # Store original path for consistency
                "kaggle_path": midi_file if midi_file != original_midi_file else None,  # Store Kaggle path if different
                "file_name": os.path.basename(original_midi_file),
                "duration": end_time, 
                "tempo": tempo, 
                "instruments": instruments
            },
        }

        # Save to cache if enabled
        if self.use_cache and cache_path:
            try:
                with open(cache_path, "w") as f:
                    json.dump(result, f)
            except Exception as e:
                print(f"Error saving cache for {midi_file}: {e}")

        # After processing tokens, extract features if pretrained model is available
        if self.use_pretrained_model and self.pretrained_model is not None:
            hierarchical_info = self.extract_hierarchical_info(midi_data) if self.use_hierarchical_encoding else None
            features = self.extract_features(midi_data) # Pass midi_data directly
            result["features"] = features
            if hierarchical_info:
                result["hierarchical_info"] = hierarchical_info
                
        return result

    def process_midi_files_parallel(
        self,
        midi_files: List[str],
        max_workers: int = 4,
        batch_size: int = 100,
        checkpoint_interval: int = 10,
        checkpoint_file: str = "data/processed/midi_checkpoint.json",
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process multiple MIDI files in parallel with batching and checkpointing."""

        # Kiểm tra xem có checkpoint trước đó không
        processed_data = []
        last_processed_idx = -1

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file) as f:
                    checkpoint = json.load(f)
                    processed_data = checkpoint.get("processed_data", [])
                    last_processed_idx = checkpoint.get("last_processed_idx", -1)
                    print(
                        f"Resuming from checkpoint: {last_processed_idx + 1}/{len(midi_files)} files processed"
                    )
            except Exception as e:
                print(f"Error loading checkpoint: {e}")

        # Nếu đã xử lý hết, trả về kết quả
        if last_processed_idx >= len(midi_files) - 1:
            print("All files already processed.")
            return processed_data

        # Chia các file thành các batch
        remaining_files = midi_files[last_processed_idx + 1 :]
        batches = [
            remaining_files[i : i + batch_size] for i in range(0, len(remaining_files), batch_size)
        ]

        print(f"Processing {len(remaining_files)} files in {len(batches)} batches")

        # Xử lý từng batch
        for batch_idx, batch_files in enumerate(batches):
            batch_start_time = time.time()
            print(f"Processing batch {batch_idx + 1}/{len(batches)}...")

            # Sử dụng ProcessPoolExecutor thay vì ThreadPoolExecutor để tận dụng nhiều CPU cores
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                if show_progress:
                    # Với progress bar
                    futures = {
                        executor.submit(self.process_midi_file, midi_file): midi_file
                        for midi_file in batch_files
                    }
                    batch_results = []

                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc=f"Batch {batch_idx + 1}",
                    ):
                        midi_file = futures[future]
                        try:
                            result = future.result()
                            if result:
                                batch_results.append(result)
                        except Exception as e:
                            print(f"Error processing {midi_file}: {e}")
                else:
                    # Không có progress bar
                    batch_results = list(
                        filter(None, executor.map(self.process_midi_file, batch_files))
                    )

            # Thêm kết quả batch vào dữ liệu đã xử lý
            processed_data.extend(batch_results)
            current_idx = last_processed_idx + len(processed_data)

            # Tính toán thời gian và hiệu suất
            batch_time = time.time() - batch_start_time
            files_per_second = len(batch_results) / batch_time if batch_time > 0 else 0
            success_rate = len(batch_results) / len(batch_files) * 100

            print(
                f"Batch {batch_idx + 1} completed in {batch_time:.2f}s ({files_per_second:.2f} files/s, {success_rate:.1f}% success)"
            )

            # Lưu checkpoint sau mỗi batch
            checkpoint = {
                "processed_data": processed_data,
                "last_processed_idx": current_idx,
                "total_files": len(midi_files),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "batch_info": {
                    "current_batch": batch_idx + 1,
                    "total_batches": len(batches),
                    "batch_size": batch_size,
                    "processing_time": batch_time,
                    "files_per_second": files_per_second,
                    "success_rate": success_rate,
                },
            }

            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

            # Lưu checkpoint
            with open(checkpoint_file, "w") as f:

                def convert(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

                json.dump(checkpoint, f, default=convert)

            print(f"Checkpoint saved: {current_idx + 1}/{len(midi_files)} files processed")

            # Dừng xử lý sau checkpoint_interval batch nếu được chỉ định
            if checkpoint_interval > 0 and (batch_idx + 1) >= checkpoint_interval:
                print(f"Reached {checkpoint_interval} batches, stopping as requested")
                break

            # Ước tính thời gian còn lại
            if batch_idx < len(batches) - 1:
                remaining_batches = len(batches) - batch_idx - 1
                est_time_remaining = remaining_batches * batch_time
                print(
                    f"Estimated time remaining: {est_time_remaining/3600:.1f} hours ({est_time_remaining/60:.1f} minutes)"
                )

        return processed_data

    def save_processed_data(self, processed_data: List[Dict[str, Any]], output_file: str):
        """Save processed data to file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        def convert_numpy_types(obj):
            """Convert numpy types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            elif isinstance(obj, tuple):
                return [convert_numpy_types(i) for i in obj]
            return obj
        
        # Convert numpy types before saving
        serializable_data = convert_numpy_types(processed_data)
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Processed data saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            # Try to save with more aggressive error handling
            try:
                # Remove potentially problematic fields
                for item in serializable_data:
                    if "features" in item:
                        del item["features"]
                    if "hierarchical_info" in item:
                        del item["hierarchical_info"]
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(serializable_data, f, indent=2, ensure_ascii=False)
                logger.warning(f"Saved processed data with reduced features to {output_file}")
            except Exception as e2:
                logger.error(f"Failed to save even with reduced features: {e2}")
                raise

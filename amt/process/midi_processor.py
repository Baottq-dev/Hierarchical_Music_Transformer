"""
MIDI Processor - Processes MIDI files for training
"""

import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import mido
import numpy as np
import pretty_midi
from tqdm import tqdm

import torch

from amt.utils.logging import get_logger

logger = get_logger(__name__)


class MIDIProcessor:
    """Processes MIDI files for training and generation."""

    def __init__(
        self,
        max_sequence_length: int = 1024,
        time_resolution: float = 0.125,
        velocity_bins: int = 32,
        pitch_range: Tuple[int, int] = (21, 108),
        use_cache: bool = True,
        cache_dir: str = "data/processed/cache",
    ):
        self.max_sequence_length = max_sequence_length
        self.time_resolution = time_resolution
        self.velocity_bins = velocity_bins
        self.pitch_range = pitch_range
        self.min_pitch, self.max_pitch = pitch_range
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        # Create cache directory if needed
        if use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        # Event vocabulary
        self.event_types = ["note_on", "note_off", "time_shift", "velocity"]
        self.vocab_size = (
            len(self.event_types) + self.max_pitch - self.min_pitch + 1 + velocity_bins + 1
        )

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
        # Check cache first if enabled
        cache_path = self._get_cache_path(midi_file)
        if self.use_cache and cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache for {midi_file}: {e}")

        # Process the file if not in cache
        midi_data = self.load_midi(midi_file)
        if midi_data is None:
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
                "file_path": midi_file,
                "file_name": os.path.basename(midi_file),
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
        """Save processed data to a file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_data = convert_numpy_types(processed_data)
        
        with open(output_file, "w") as f:
            json.dump(serializable_data, f, indent=2)

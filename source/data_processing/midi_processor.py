"""
MIDI processing module for AMT.
Contains functions for processing MIDI files from Lakh MIDI Clean dataset.
"""

import mido
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Constants
TIME_RESOLUTION = 480  # ticks per quarter note
MAX_TIME_SHIFT = 512   # maximum time shift in ticks
VELOCITY_BINS = 32     # number of velocity bins
MIDI_PROGRAMS = {
    'piano': 0,
    'guitar': 24,
    'violin': 40,
    'drums': 0,  # Channel 10
    'bass': 33,
    'saxophone': 66,
    'trumpet': 56,
    'flute': 73,
    'clarinet': 71,
    'cello': 42,
    'viola': 41,
    'trombone': 57,
    'organ': 19,
    'synth': 80
}

def quantize_time_shift(time_shift: int) -> int:
    """
    Quantize time shift to nearest tick.
    Args:
        time_shift: Time shift in ticks
    Returns:
        Quantized time shift
    """
    return min(max(0, time_shift), MAX_TIME_SHIFT)

def quantize_velocity(velocity: int) -> int:
    """
    Quantize velocity to nearest bin.
    Args:
        velocity: MIDI velocity (0-127)
    Returns:
        Quantized velocity bin
    """
    bin_size = 128 // VELOCITY_BINS
    return min(velocity // bin_size, VELOCITY_BINS - 1)

def get_midi_metadata(midi_file: str) -> Dict[str, Any]:
    """
    Extract metadata from MIDI file.
    Args:
        midi_file: Path to MIDI file
    Returns:
        Dictionary containing MIDI metadata
    """
    try:
        midi = mido.MidiFile(midi_file)
        
        # Get basic metadata
        metadata = {
            "ticks_per_beat": midi.ticks_per_beat,
            "num_tracks": len(midi.tracks),
            "duration": midi.length,
            "tempo": [],
            "time_signatures": [],
            "key_signatures": [],
            "tracks": []
        }
        
        # Process each track
        for i, track in enumerate(midi.tracks):
            track_info = {
                "name": track.name,
                "program": 0,  # Default program
                "notes": [],
                "control_changes": []
            }
            
            current_time = 0
            active_notes = defaultdict(list)
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'program_change':
                    track_info["program"] = msg.program
                elif msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note].append((current_time, msg.velocity))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time, velocity = active_notes[msg.note].pop()
                        duration = current_time - start_time
                        track_info["notes"].append({
                            "note": msg.note,
                            "start_time": start_time,
                            "duration": duration,
                            "velocity": velocity
                        })
                elif msg.type == 'control_change':
                    track_info["control_changes"].append({
                        "control": msg.control,
                        "value": msg.value,
                        "time": current_time
                    })
                elif msg.type == 'set_tempo':
                    metadata["tempo"].append({
                        "tempo": msg.tempo,
                        "time": current_time
                    })
                elif msg.type == 'time_signature':
                    metadata["time_signatures"].append({
                        "numerator": msg.numerator,
                        "denominator": msg.denominator,
                        "time": current_time
                    })
                elif msg.type == 'key_signature':
                    metadata["key_signatures"].append({
                        "key": msg.key,
                        "time": current_time
                    })
            
            metadata["tracks"].append(track_info)
        
        return metadata
    except Exception as e:
        print(f"Error processing MIDI file {midi_file}: {e}")
        return None

def midi_to_event_sequence(midi_file: str) -> List[Tuple[int, int, int]]:
    """
    Convert MIDI file to sequence of (TIME_ON, NOTE, DURATION) triplets.
    Args:
        midi_file: Path to MIDI file
    Returns:
        List of (TIME_ON, NOTE, DURATION) triplets
    """
    try:
        midi = mido.MidiFile(midi_file)
        events = []
        
        # Process each track
        for track in midi.tracks:
            current_time = 0
            active_notes = defaultdict(list)
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note].append((current_time, msg.velocity))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time, velocity = active_notes[msg.note].pop()
                        duration = current_time - start_time
                        
                        # Quantize values
                        time_on = quantize_time_shift(start_time)
                        duration = quantize_time_shift(duration)
                        velocity = quantize_velocity(velocity)
                        
                        events.append((time_on, msg.note, duration))
        
        return events
    except Exception as e:
        print(f"Error converting MIDI to event sequence: {e}")
        return []

def event_sequence_to_midi(events: List[Tuple[int, int, int]], output_file: str) -> bool:
    """
    Convert event sequence to MIDI file.
    Args:
        events: List of (TIME_ON, NOTE, DURATION) triplets
        output_file: Path to output MIDI file
    Returns:
        True if successful, False otherwise
    """
    try:
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        
        current_time = 0
        for time_on, note, duration in events:
            # Add note on
            track.append(mido.Message('note_on', note=note, velocity=64, time=time_on - current_time))
            current_time = time_on
            
            # Add note off
            track.append(mido.Message('note_off', note=note, velocity=0, time=duration))
            current_time += duration
        
        midi.save(output_file)
        return True
    except Exception as e:
        print(f"Error converting event sequence to MIDI: {e}")
        return False

def analyze_midi_file(midi_file: str) -> Dict[str, Any]:
    """
    Analyze MIDI file and extract features.
    Args:
        midi_file: Path to MIDI file
    Returns:
        Dictionary containing MIDI features
    """
    try:
        midi = mido.MidiFile(midi_file)
        features = {
            "note_density": 0,
            "velocity_mean": 0,
            "velocity_std": 0,
            "note_range": {"min": 127, "max": 0},
            "time_signatures": [],
            "tempo_mean": 0,
            "tempo_std": 0
        }
        
        # Process each track
        all_notes = []
        all_velocities = []
        all_tempos = []
        
        for track in midi.tracks:
            current_time = 0
            active_notes = defaultdict(list)
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note].append((current_time, msg.velocity))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time, velocity = active_notes[msg.note].pop()
                        duration = current_time - start_time
                        all_notes.append(msg.note)
                        all_velocities.append(velocity)
                elif msg.type == 'set_tempo':
                    all_tempos.append(mido.tempo2bpm(msg.tempo))
                elif msg.type == 'time_signature':
                    features["time_signatures"].append(f"{msg.numerator}/{msg.denominator}")
        
        if all_notes:
            features["note_density"] = len(all_notes) / midi.length
            features["note_range"]["min"] = min(all_notes)
            features["note_range"]["max"] = max(all_notes)
        
        if all_velocities:
            features["velocity_mean"] = np.mean(all_velocities)
            features["velocity_std"] = np.std(all_velocities)
        
        if all_tempos:
            features["tempo_mean"] = np.mean(all_tempos)
            features["tempo_std"] = np.std(all_tempos)
        
        return features
    except Exception as e:
        print(f"Error analyzing MIDI file: {e}")
        return None 

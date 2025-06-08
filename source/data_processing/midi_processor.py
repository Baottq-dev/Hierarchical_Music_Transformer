"""
MIDI processing module for AMT.
Contains functions for processing MIDI files and converting them to event sequences.
"""

import mido
import math
import numpy as np
from typing import List, Dict, Any, Tuple
import os
from collections import defaultdict

# Constants from the paper
TIME_RESOLUTION = 0.01  # 10ms steps for time shifts
MAX_TIME_SHIFT = 1.0    # Max time shift of 1 second
VELOCITY_BINS = 32
MAX_VELOCITY = 127
NOTE_RANGE = 128  # MIDI note range (0-127)

# MIDI program numbers for common instruments
INSTRUMENT_PROGRAMS = {
    'piano': 0,
    'acoustic_grand_piano': 0,
    'bright_acoustic_piano': 1,
    'electric_grand_piano': 2,
    'honky_tonk_piano': 3,
    'electric_piano_1': 4,
    'electric_piano_2': 5,
    'harpsichord': 6,
    'clavinet': 7,
    'celesta': 8,
    'glockenspiel': 9,
    'music_box': 10,
    'vibraphone': 11,
    'marimba': 12,
    'xylophone': 13,
    'tubular_bells': 14,
    'dulcimer': 15,
    'acoustic_guitar_nylon': 24,
    'acoustic_guitar_steel': 25,
    'electric_guitar_jazz': 26,
    'electric_guitar_clean': 27,
    'electric_guitar_muted': 28,
    'overdriven_guitar': 29,
    'distortion_guitar': 30,
    'guitar_harmonics': 31,
    'acoustic_bass': 32,
    'electric_bass_finger': 33,
    'electric_bass_pick': 34,
    'fretless_bass': 35,
    'slap_bass_1': 36,
    'slap_bass_2': 37,
    'synth_bass_1': 38,
    'synth_bass_2': 39,
    'violin': 40,
    'viola': 41,
    'cello': 42,
    'contrabass': 43,
    'tremolo_strings': 44,
    'pizzicato_strings': 45,
    'orchestral_harp': 46,
    'timpani': 47,
    'string_ensemble_1': 48,
    'string_ensemble_2': 49,
    'synth_strings_1': 50,
    'synth_strings_2': 51,
    'choir_aahs': 52,
    'voice_oohs': 53,
    'synth_voice': 54,
    'orchestra_hit': 55,
    'trumpet': 56,
    'trombone': 57,
    'tuba': 58,
    'muted_trumpet': 59,
    'french_horn': 60,
    'brass_section': 61,
    'synth_brass_1': 62,
    'synth_brass_2': 63,
    'soprano_sax': 64,
    'alto_sax': 65,
    'tenor_sax': 66,
    'baritone_sax': 67,
    'oboe': 68,
    'english_horn': 69,
    'bassoon': 70,
    'clarinet': 71,
    'piccolo': 72,
    'flute': 73,
    'recorder': 74,
    'pan_flute': 75,
    'blown_bottle': 76,
    'shakuhachi': 77,
    'whistle': 78,
    'ocarina': 79,
    'lead_1_square': 80,
    'lead_2_sawtooth': 81,
    'lead_3_calliope': 82,
    'lead_4_chiff': 83,
    'lead_5_charang': 84,
    'lead_6_voice': 85,
    'lead_7_fifths': 86,
    'lead_8_bass_lead': 87,
    'pad_1_new_age': 88,
    'pad_2_warm': 89,
    'pad_3_polysynth': 90,
    'pad_4_choir': 91,
    'pad_5_bowed': 92,
    'pad_6_metallic': 93,
    'pad_7_halo': 94,
    'pad_8_sweep': 95,
    'fx_1_rain': 96,
    'fx_2_soundtrack': 97,
    'fx_3_crystal': 98,
    'fx_4_atmosphere': 99,
    'fx_5_brightness': 100,
    'fx_6_goblins': 101,
    'fx_7_echoes': 102,
    'fx_8_sci_fi': 103,
    'sitar': 104,
    'banjo': 105,
    'shamisen': 106,
    'koto': 107,
    'kalimba': 108,
    'bag_pipe': 109,
    'fiddle': 110,
    'shanai': 111,
    'tinkle_bell': 112,
    'agogo': 113,
    'steel_drums': 114,
    'woodblock': 115,
    'taiko_drum': 116,
    'melodic_tom': 117,
    'synth_drum': 118,
    'reverse_cymbal': 119,
    'guitar_fret_noise': 120,
    'breath_noise': 121,
    'seashore': 122,
    'bird_tweet': 123,
    'telephone_ring': 124,
    'helicopter': 125,
    'applause': 126,
    'gunshot': 127
}

def quantize_time_shift(dt: float) -> int:
    """
    Quantize time shift into discrete steps.
    Args:
        dt: Time shift in seconds
    Returns:
        Quantized time shift index
    """
    if dt > MAX_TIME_SHIFT:
        dt = MAX_TIME_SHIFT
    elif dt < 0:
        dt = 0
    return min(int(round(dt / TIME_RESOLUTION)), int(MAX_TIME_SHIFT / TIME_RESOLUTION) - 1)

def quantize_velocity(velocity: int) -> int:
    """
    Quantize MIDI velocity into bins.
    Args:
        velocity: MIDI velocity (0-127)
    Returns:
        Quantized velocity index
    """
    if velocity > MAX_VELOCITY:
        velocity = MAX_VELOCITY
    elif velocity < 0:
        velocity = 0
    return min(int(round(velocity * (VELOCITY_BINS - 1) / MAX_VELOCITY)), VELOCITY_BINS - 1)

def get_midi_metadata(midi_file: str) -> Dict[str, Any]:
    """
    Extract metadata from MIDI file.
    Args:
        midi_file: Path to MIDI file
    Returns:
        Dictionary containing MIDI metadata
    """
    try:
        mid = mido.MidiFile(midi_file)
        metadata = {
            "ticks_per_beat": mid.ticks_per_beat,
            "num_tracks": len(mid.tracks),
            "duration": mid.length,
            "tempo": None,
            "time_signature": None,
            "key_signature": None,
            "tracks": []
        }
        
        # Process each track
        for i, track in enumerate(mid.tracks):
            track_info = {
                "name": track.name if hasattr(track, 'name') else f"Track {i}",
                "program": None,
                "notes": [],
                "control_changes": []
            }
            
            for msg in track:
                if msg.type == 'set_tempo':
                    metadata["tempo"] = msg.tempo
                elif msg.type == 'time_signature':
                    metadata["time_signature"] = f"{msg.numerator}/{msg.denominator}"
                elif msg.type == 'key_signature':
                    metadata["key_signature"] = msg.key
                elif msg.type == 'program_change':
                    track_info["program"] = msg.program
                elif msg.type == 'note_on' and msg.velocity > 0:
                    track_info["notes"].append({
                        "note": msg.note,
                        "velocity": msg.velocity,
                        "time": msg.time
                    })
                elif msg.type == 'control_change':
                    track_info["control_changes"].append({
                        "control": msg.control,
                        "value": msg.value,
                        "time": msg.time
                    })
            
            metadata["tracks"].append(track_info)
        
        return metadata
    except Exception as e:
        print(f"Error extracting metadata from {midi_file}: {e}")
        return None

def midi_to_event_sequence(midi_file: str) -> List[str]:
    """
    Convert MIDI file to event sequence.
    Args:
        midi_file: Path to MIDI file
    Returns:
        List of event tokens
    """
    try:
        mid = mido.MidiFile(midi_file)
        events = []
        current_time = 0
        active_notes = defaultdict(dict)  # Track active notes per track
        
        for track_idx, track in enumerate(mid.tracks):
            track_time = 0
            for msg in track:
                track_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Note on event
                    time_shift = quantize_time_shift(track_time)
                    velocity = quantize_velocity(msg.velocity)
                    events.append(f"TIME_SHIFT_{time_shift}")
                    events.append(f"TRACK_{track_idx}")
                    events.append(f"NOTE_ON_{msg.note}")
                    events.append(f"VELOCITY_{velocity}")
                    active_notes[track_idx][msg.note] = track_time
                    
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Note off event
                    if msg.note in active_notes[track_idx]:
                        time_shift = quantize_time_shift(track_time - active_notes[track_idx][msg.note])
                        events.append(f"TIME_SHIFT_{time_shift}")
                        events.append(f"TRACK_{track_idx}")
                        events.append(f"NOTE_OFF_{msg.note}")
                        del active_notes[track_idx][msg.note]
        
        return events
    except Exception as e:
        print(f"Error converting MIDI to event sequence: {e}")
        return None

def event_sequence_to_midi(events: List[str], output_file: str, tempo: int = 500000):
    """
    Convert event sequence back to MIDI file.
    Args:
        events: List of event tokens
        output_file: Path to save MIDI file
        tempo: Tempo in microseconds per beat
    """
    try:
        mid = mido.MidiFile()
        tracks = defaultdict(mido.MidiTrack)
        mid.tracks = list(tracks.values())
        
        # Add tempo to first track
        tracks[0].append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        current_times = defaultdict(int)
        active_notes = defaultdict(dict)
        
        for event in events:
            if event.startswith('TIME_SHIFT_'):
                time_shift = int(event.split('_')[2])
                for track_idx in tracks:
                    current_times[track_idx] += time_shift * TIME_RESOLUTION
                    
            elif event.startswith('TRACK_'):
                current_track = int(event.split('_')[1])
                
            elif event.startswith('NOTE_ON_'):
                note = int(event.split('_')[2])
                active_notes[current_track][note] = current_times[current_track]
                
            elif event.startswith('VELOCITY_'):
                velocity = int(event.split('_')[1])
                velocity = int(round(velocity * MAX_VELOCITY / (VELOCITY_BINS - 1)))
                
                # Add note on message
                tracks[current_track].append(mido.Message('note_on', 
                    note=note, velocity=velocity, time=int(current_times[current_track])))
                
            elif event.startswith('NOTE_OFF_'):
                note = int(event.split('_')[2])
                if note in active_notes[current_track]:
                    # Add note off message
                    tracks[current_track].append(mido.Message('note_off', 
                        note=note, velocity=0, 
                        time=int(current_times[current_track] - active_notes[current_track][note])))
                    del active_notes[current_track][note]
        
        # Save MIDI file
        mid.save(output_file)
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
        mid = mido.MidiFile(midi_file)
        features = {
            "num_tracks": len(mid.tracks),
            "duration": mid.length,
            "tracks": [],
            "global_features": {
                "tempo": None,
                "time_signatures": set(),
                "key_signatures": set()
            }
        }
        
        for track_idx, track in enumerate(mid.tracks):
            track_features = {
                "name": track.name if hasattr(track, 'name') else f"Track {track_idx}",
                "program": None,
                "notes": [],
                "note_density": 0,
                "velocity_mean": 0,
                "velocity_std": 0,
                "note_range": {"min": 127, "max": 0},
                "control_changes": defaultdict(list)
            }
            
            track_time = 0
            for msg in track:
                track_time += msg.time
                
                if msg.type == 'set_tempo':
                    features["global_features"]["tempo"] = msg.tempo
                elif msg.type == 'time_signature':
                    features["global_features"]["time_signatures"].add(
                        f"{msg.numerator}/{msg.denominator}")
                elif msg.type == 'key_signature':
                    features["global_features"]["key_signatures"].add(msg.key)
                elif msg.type == 'program_change':
                    track_features["program"] = msg.program
                elif msg.type == 'note_on' and msg.velocity > 0:
                    track_features["notes"].append({
                        "note": msg.note,
                        "velocity": msg.velocity,
                        "time": track_time
                    })
                    track_features["note_range"]["min"] = min(
                        track_features["note_range"]["min"], msg.note)
                    track_features["note_range"]["max"] = max(
                        track_features["note_range"]["max"], msg.note)
                elif msg.type == 'control_change':
                    track_features["control_changes"][msg.control].append({
                        "value": msg.value,
                        "time": track_time
                    })
            
            # Calculate track statistics
            if track_features["notes"]:
                velocities = [note["velocity"] for note in track_features["notes"]]
                track_features["note_density"] = len(track_features["notes"]) / mid.length
                track_features["velocity_mean"] = np.mean(velocities)
                track_features["velocity_std"] = np.std(velocities)
            
            features["tracks"].append(track_features)
        
        # Convert sets to lists for JSON serialization
        features["global_features"]["time_signatures"] = list(
            features["global_features"]["time_signatures"])
        features["global_features"]["key_signatures"] = list(
            features["global_features"]["key_signatures"])
        
        return features
    except Exception as e:
        print(f"Error analyzing MIDI file: {e}")
        return None 
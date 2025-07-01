"""
MIDI Processor - Processes MIDI files for training
"""

import os
import json
import numpy as np
import pretty_midi
from typing import List, Dict, Any, Tuple, Optional
import torch

class MIDIProcessor:
    """Processes MIDI files for training and generation."""
    
    def __init__(self, 
                 max_sequence_length: int = 1024,
                 time_resolution: float = 0.125,
                 velocity_bins: int = 32,
                 pitch_range: Tuple[int, int] = (21, 108)):
        self.max_sequence_length = max_sequence_length
        self.time_resolution = time_resolution
        self.velocity_bins = velocity_bins
        self.pitch_range = pitch_range
        self.min_pitch, self.max_pitch = pitch_range
        
        # Event vocabulary
        self.event_types = ['note_on', 'note_off', 'time_shift', 'velocity']
        self.vocab_size = len(self.event_types) + self.max_pitch - self.min_pitch + 1 + velocity_bins + 1
    
    def load_midi(self, midi_file: str) -> Optional[pretty_midi.PrettyMIDI]:
        """Load MIDI file."""
        try:
            return pretty_midi.PrettyMIDI(midi_file)
        except Exception as e:
            print(f"Error loading MIDI file {midi_file}: {e}")
            return None
    
    def extract_events(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
        """Extract events from MIDI data."""
        events = []
        
        # Collect all notes from all instruments
        all_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_notes.append({
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'start': note.start,
                    'end': note.end,
                    'instrument': instrument.program
                })
        
        # Sort by start time
        all_notes.sort(key=lambda x: x['start'])
        
        # Convert to events
        current_time = 0.0
        for note in all_notes:
            # Time shift event
            time_diff = note['start'] - current_time
            if time_diff > 0:
                time_shift_events = self._create_time_shift_events(time_diff)
                events.extend(time_shift_events)
                current_time = note['start']
            
            # Note on event
            events.append({
                'type': 'note_on',
                'pitch': note['pitch'],
                'velocity': note['velocity'],
                'time': note['start']
            })
            
            # Note off event
            events.append({
                'type': 'note_off',
                'pitch': note['pitch'],
                'time': note['end']
            })
        
        return events
    
    def _create_time_shift_events(self, time_diff: float) -> List[Dict[str, Any]]:
        """Create time shift events for a given time difference."""
        events = []
        remaining_time = time_diff
        
        while remaining_time > 0:
            shift_time = min(remaining_time, self.time_resolution)
            events.append({
                'type': 'time_shift',
                'duration': shift_time,
                'time': shift_time
            })
            remaining_time -= shift_time
        
        return events
    
    def events_to_tokens(self, events: List[Dict[str, Any]]) -> List[int]:
        """Convert events to token sequence."""
        tokens = []
        
        for event in events:
            if event['type'] == 'note_on':
                # Note on token: pitch + offset
                pitch = event['pitch']
                if self.min_pitch <= pitch <= self.max_pitch:
                    token = len(self.event_types) + (pitch - self.min_pitch)
                    tokens.append(token)
                
                # Velocity token
                velocity = event['velocity']
                velocity_token = len(self.event_types) + (self.max_pitch - self.min_pitch + 1) + min(velocity // 4, self.velocity_bins - 1)
                tokens.append(velocity_token)
                
            elif event['type'] == 'note_off':
                # Note off token: pitch + offset + special marker
                pitch = event['pitch']
                if self.min_pitch <= pitch <= self.max_pitch:
                    token = len(self.event_types) + (pitch - self.min_pitch)
                    tokens.append(token)
                
            elif event['type'] == 'time_shift':
                # Time shift token
                time_token = len(self.event_types) + (self.max_pitch - self.min_pitch + 1) + self.velocity_bins
                tokens.append(time_token)
        
        return tokens
    
    def tokens_to_events(self, tokens: List[int]) -> List[Dict[str, Any]]:
        """Convert token sequence back to events."""
        events = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token < len(self.event_types):
                # Event type token
                event_type = self.event_types[token]
                if event_type == 'time_shift':
                    events.append({'type': 'time_shift', 'duration': self.time_resolution})
                i += 1
                
            else:
                # Pitch token
                pitch = token - len(self.event_types) + self.min_pitch
                if self.min_pitch <= pitch <= self.max_pitch:
                    # Check if next token is velocity
                    if i + 1 < len(tokens):
                        next_token = tokens[i + 1]
                        if next_token >= len(self.event_types) + (self.max_pitch - self.min_pitch + 1):
                            # This is a note_on with velocity
                            velocity = (next_token - len(self.event_types) - (self.max_pitch - self.min_pitch + 1)) * 4
                            events.append({
                                'type': 'note_on',
                                'pitch': pitch,
                                'velocity': velocity
                            })
                            i += 2
                        else:
                            # This is a note_off
                            events.append({
                                'type': 'note_off',
                                'pitch': pitch
                            })
                            i += 1
                    else:
                        # Assume note_off if no next token
                        events.append({
                            'type': 'note_off',
                            'pitch': pitch
                        })
                        i += 1
                else:
                    i += 1
        
        return events
    
    def process_midi_file(self, midi_file: str) -> Optional[Dict[str, Any]]:
        """Process a single MIDI file."""
        midi_data = self.load_midi(midi_file)
        if midi_data is None:
            return None
        
        # Extract events
        events = self.extract_events(midi_data)
        
        # Convert to tokens
        tokens = self.events_to_tokens(events)
        
        # Truncate if too long
        if len(tokens) > self.max_sequence_length:
            tokens = tokens[:self.max_sequence_length]
        
        return {
            'file_path': midi_file,
            'events': events,
            'tokens': tokens,
            'sequence_length': len(tokens),
            'metadata': {
                'duration': midi_data.get_end_time(),
                'tempo': midi_data.estimate_tempo(),
                'instruments': [inst.program for inst in midi_data.instruments]
            }
        }
    
    def save_processed_data(self, processed_data: List[Dict[str, Any]], output_file: str):
        """Save processed data to file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        for item in processed_data:
            serializable_item = item.copy()
            if 'tokens' in serializable_item:
                serializable_item['tokens'] = serializable_item['tokens'].tolist() if isinstance(serializable_item['tokens'], np.ndarray) else serializable_item['tokens']
            serializable_data.append(serializable_item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed data saved to {output_file}")
        print(f"Total processed files: {len(processed_data)}") 
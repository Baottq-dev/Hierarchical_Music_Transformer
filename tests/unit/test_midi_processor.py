"""
Unit tests for the MIDI Processor module.
"""

import os
import pytest
import numpy as np
import pretty_midi
from typing import List, Dict, Any

from amt.process.midi_processor import MidiProcessor


class TestMidiProcessor:
    """Test suite for the MidiProcessor class."""
    
    def test_init(self):
        """Test initialization of the MIDIProcessor."""
        processor = MidiProcessor(
            max_sequence_length=512,
            time_resolution=0.25,
            velocity_bins=16,
            pitch_range=(24, 96),
            use_cache=False
        )
        
        assert processor.max_sequence_length == 512
        assert processor.time_resolution == 0.25
        assert processor.velocity_bins == 16
        assert processor.pitch_range == (24, 96)
        assert processor.min_pitch == 24
        assert processor.max_pitch == 96
        assert processor.use_cache is False
    
    def test_load_midi(self, simple_midi_file):
        """Test loading a MIDI file."""
        processor = MidiProcessor(use_cache=False)
        midi_data = processor.load_midi(simple_midi_file)
        
        assert midi_data is not None
        assert isinstance(midi_data, pretty_midi.PrettyMIDI)
        assert len(midi_data.instruments) == 1
        assert len(midi_data.instruments[0].notes) == 2
    
    def test_extract_events(self, simple_midi_file):
        """Test extracting events from a MIDI file."""
        processor = MidiProcessor(use_cache=False)
        midi_data = processor.load_midi(simple_midi_file)
        events = processor.extract_events(midi_data)
        
        # Check that events were extracted
        assert len(events) > 0
        
        # Check that we have note_on and note_off events
        note_on_events = [e for e in events if e['type'] == 'note_on']
        note_off_events = [e for e in events if e['type'] == 'note_off']
        
        assert len(note_on_events) == 2  # Two notes in the simple MIDI file
        assert len(note_off_events) == 2
        
        # Check specific note properties
        assert note_on_events[0]['pitch'] == 60  # C4
        assert note_on_events[1]['pitch'] == 64  # E4
        
        # Check velocities
        assert note_on_events[0]['velocity'] == 64
        assert note_on_events[1]['velocity'] == 80
    
    def test_events_to_tokens(self, simple_midi_file):
        """Test converting events to tokens."""
        processor = MidiProcessor(use_cache=False)
        midi_data = processor.load_midi(simple_midi_file)
        events = processor.extract_events(midi_data)
        tokens = processor.events_to_tokens(events)
        
        # Check that tokens were created
        assert len(tokens) > 0
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
    
    def test_tokens_to_events(self, simple_midi_file):
        """Test converting tokens back to events."""
        processor = MidiProcessor(use_cache=False)
        midi_data = processor.load_midi(simple_midi_file)
        original_events = processor.extract_events(midi_data)
        tokens = processor.events_to_tokens(original_events)
        reconstructed_events = processor.tokens_to_events(tokens)
        
        # Check that events were reconstructed
        assert len(reconstructed_events) > 0
        
        # Check that we have note_on and note_off events
        note_on_events = [e for e in reconstructed_events if e['type'] == 'note_on']
        note_off_events = [e for e in reconstructed_events if e['type'] == 'note_off']
        
        assert len(note_on_events) > 0
        assert len(note_off_events) > 0
    
    def test_events_to_midi(self, simple_midi_file):
        """Test converting events back to MIDI."""
        processor = MidiProcessor(use_cache=False)
        original_midi = processor.load_midi(simple_midi_file)
        events = processor.extract_events(original_midi)
        
        # Convert events back to MIDI
        reconstructed_midi = processor.events_to_midi(events)
        
        # Check that the MIDI was reconstructed
        assert isinstance(reconstructed_midi, pretty_midi.PrettyMIDI)
        assert len(reconstructed_midi.instruments) > 0
        
        # Check that notes were preserved
        original_notes = original_midi.instruments[0].notes
        reconstructed_notes = reconstructed_midi.instruments[0].notes
        
        assert len(reconstructed_notes) == len(original_notes)
    
    def test_process_midi_file(self, simple_midi_file):
        """Test processing a complete MIDI file."""
        processor = MidiProcessor(use_cache=False)
        result = processor.process_midi_file(simple_midi_file)
        
        # Check the result structure
        assert result is not None
        assert 'metadata' in result
        assert 'tokens' in result
        assert 'sequence_length' in result
        
        # Check metadata
        assert result['metadata']['file_path'] == simple_midi_file
        assert result['metadata']['file_name'] == os.path.basename(simple_midi_file)
        
        # Check tokens
        assert len(result['tokens']) > 0
        assert result['sequence_length'] == len(result['tokens'])
    
    def test_roundtrip_conversion(self, simple_midi_file, temp_dir):
        """Test a complete roundtrip conversion: MIDI -> tokens -> MIDI."""
        processor = MidiProcessor(use_cache=False)
        
        # Process the original MIDI file
        result = processor.process_midi_file(simple_midi_file)
        tokens = result['tokens']
        
        # Convert tokens back to events
        events = processor.tokens_to_events(tokens)
        
        # Convert events back to MIDI
        reconstructed_midi = processor.events_to_midi(events)
        
        # Save the reconstructed MIDI
        output_path = os.path.join(temp_dir, "reconstructed.mid")
        reconstructed_midi.write(output_path)
        
        # Verify the file exists
        assert os.path.exists(output_path)
        
        # Load it back and check basic properties
        reloaded_midi = processor.load_midi(output_path)
        assert reloaded_midi is not None
        assert len(reloaded_midi.instruments) > 0 
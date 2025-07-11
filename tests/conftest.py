"""
Common fixtures for testing the AMT package.
"""

import os
import sys
import pytest
import tempfile
import shutil
import numpy as np
import pretty_midi

# Add the project root to the path so we can import the amt package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_midi_file(temp_dir):
    """Create a simple MIDI file for testing."""
    midi_path = os.path.join(temp_dir, "test.mid")
    
    # Create a simple MIDI file with one note
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Piano
    
    # Add a single note (middle C for 1 second)
    note = pretty_midi.Note(
        velocity=64,
        pitch=60,  # C4
        start=0.0,
        end=1.0
    )
    instrument.notes.append(note)
    
    # Add another note (E4 for 1 second starting at 1.0)
    note = pretty_midi.Note(
        velocity=80,
        pitch=64,  # E4
        start=1.0,
        end=2.0
    )
    instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    midi.write(midi_path)
    
    return midi_path


@pytest.fixture
def complex_midi_file(temp_dir):
    """Create a more complex MIDI file for testing."""
    midi_path = os.path.join(temp_dir, "complex.mid")
    
    # Create a MIDI file with multiple notes and instruments
    midi = pretty_midi.PrettyMIDI()
    
    # Piano instrument
    piano = pretty_midi.Instrument(program=0)
    
    # Add a C major chord
    for pitch in [60, 64, 67]:  # C4, E4, G4
        note = pretty_midi.Note(
            velocity=64,
            pitch=pitch,
            start=0.0,
            end=1.0
        )
        piano.notes.append(note)
    
    # Add a melody
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=1.0 + i * 0.5,
            end=1.0 + (i + 1) * 0.5
        )
        piano.notes.append(note)
    
    midi.instruments.append(piano)
    
    # Bass instrument
    bass = pretty_midi.Instrument(program=32)  # Acoustic bass
    
    # Add bass notes
    for i, pitch in enumerate([36, 38, 40, 41]):
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=i * 1.0,
            end=(i + 1) * 1.0
        )
        bass.notes.append(note)
    
    midi.instruments.append(bass)
    midi.write(midi_path)
    
    return midi_path 
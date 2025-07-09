"""
Collect Module - Data Collection Components
Handles MIDI file collection, metadata extraction, and text pairing
"""

from .data_pairing import DataPairing
from .midi_collector import MIDICollector
from .text_collector import TextCollector

__all__ = ["MIDICollector", "TextCollector", "DataPairing"]

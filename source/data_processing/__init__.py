"""
Data processing module for AMT.
"""

from .midi_processor import (
    midi_to_event_sequence,
    event_sequence_to_midi,
    analyze_midi_file
)
from .text_processor import (
    clean_text,
    extract_music_keywords,
    get_text_features,
    get_bert_embedding,
    process_text_descriptions
)
from .collect_text import collect_text_descriptions
from .process_midi import process_midi_files
from .process_text import process_text_data
from .prepare_training import prepare_training_data

__all__ = [
    'midi_to_event_sequence',
    'event_sequence_to_midi',
    'analyze_midi_file',
    'clean_text',
    'extract_music_keywords',
    'get_text_features',
    'get_bert_embedding',
    'process_text_descriptions',
    'collect_text_descriptions',
    'process_midi_files',
    'process_text_data',
    'prepare_training_data'
] 
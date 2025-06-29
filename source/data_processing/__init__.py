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
    get_bert_embeddings,
    process_text_descriptions
)

__all__ = [
    'midi_to_event_sequence',
    'event_sequence_to_midi',
    'analyze_midi_file',
    'clean_text',
    'extract_music_keywords',
    'get_text_features',
    'get_bert_embedding',
    'get_bert_embeddings',
    'process_text_descriptions'
] 
"""
Configuration module for AMT.
"""

import os
from typing import Dict, Any

# Data paths
DATA_DIR = "data"
MIDI_DIR = os.path.join(DATA_DIR, "midi")
TEXT_DIR = os.path.join(DATA_DIR, "text")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
REFERENCE_DIR = os.path.join(DATA_DIR, "reference")
EVALUATION_DIR = os.path.join(DATA_DIR, "evaluation")

# Output files
PAIRED_DATA_FILE = os.path.join(OUTPUT_DIR, "automated_paired_data.json")
MIDI_METADATA_FILE = os.path.join(OUTPUT_DIR, "midi_metadata_list.json")
TEXT_EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "text_embeddings.json")
CLUSTERED_DATA_FILE = os.path.join(OUTPUT_DIR, "clustered_text_data.json")
TRAINING_DATA_FILE = os.path.join(OUTPUT_DIR, "amt_training_data.json")

# Model paths
MODEL_DIR = "models"
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Data processing
MIDI_CONFIG = {
    "time_resolution": 480,  # ticks per quarter note
    "max_time_shift": 512,   # maximum time shift in ticks
    "velocity_bins": 32,     # number of velocity bins
    "midi_programs": {
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
}

TEXT_CONFIG = {
    "max_length": 512,
    "music_genres": {
        'rock', 'pop', 'jazz', 'classical', 'electronic', 'hip hop', 'r&b', 'blues',
        'country', 'folk', 'metal', 'punk', 'reggae', 'soul', 'funk', 'disco'
    },
    "music_instruments": {
        'piano', 'guitar', 'drums', 'bass', 'violin', 'saxophone', 'trumpet',
        'flute', 'clarinet', 'cello', 'viola', 'trombone', 'organ', 'synth'
    },
    "music_emotions": {
        'happy', 'sad', 'energetic', 'calm', 'angry', 'peaceful', 'melancholic',
        'joyful', 'dark', 'bright', 'intense', 'soft', 'loud', 'gentle'
    }
}

# Model configuration
MODEL_CONFIG = {
    "embedding_dim": 768,    # BERT embedding dimension
    "hidden_dim": 1024,      # GPT-2 hidden dimension
    "vocab_size": 512,       # MIDI event vocabulary size
    "max_seq_length": 1024,  # Maximum sequence length
    "num_layers": 6,         # Number of transformer layers
    "num_heads": 8,          # Number of attention heads
    "dropout": 0.1           # Dropout rate
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "warmup_steps": 1000,
    "max_grad_norm": 1.0,
    "save_steps": 1000,
    "eval_steps": 1000
}

# Generation configuration
GENERATION_CONFIG = {
    "max_length": 1024,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "num_return_sequences": 1
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": [
        "note_density_ratio",
        "velocity_similarity",
        "note_range_similarity",
        "time_signature_match",
        "tempo_similarity"
    ],
    "reference_dir": REFERENCE_DIR,
    "output_dir": EVALUATION_DIR
}

def get_config() -> Dict[str, Any]:
    """
    Get complete configuration.
    Returns:
        Dictionary containing all configuration
    """
    return {
        "data": {
            "midi": MIDI_CONFIG,
            "text": TEXT_CONFIG
        },
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "generation": GENERATION_CONFIG,
        "evaluation": EVALUATION_CONFIG
    } 
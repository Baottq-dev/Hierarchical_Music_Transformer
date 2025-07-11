"""
Integration test for music generation.
"""

import os
import pytest
from amt.generate.generator import MusicGenerator

@pytest.mark.skip(reason="Requires trained model and is slow")
def test_music_generation():
    """Test music generation with a trained model."""
    model_path = "models/checkpoints/best_model.pt"
    if not os.path.exists(model_path):
        pytest.skip(f"Model file {model_path} not found")
    
    gen = MusicGenerator(model_path, device="cpu")

    result1 = gen.generate_music(
        "A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle.",
        "o1.mid",
    )
    result2 = gen.generate_music(
        "Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach",
        "o2.mid",
    )
    
    assert len(result1["generated_tokens"]) > 0
    assert len(result2["generated_tokens"]) > 0
    
    # Clean up generated files
    if os.path.exists("o1.mid"):
        os.remove("o1.mid")
    if os.path.exists("o2.mid"):
        os.remove("o2.mid") 
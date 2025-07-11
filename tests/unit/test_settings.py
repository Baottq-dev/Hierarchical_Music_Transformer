#!/usr/bin/env python3
"""
Test script for AMT settings.
"""

import pytest
from amt.config import get_settings

def test_settings_default_values():
    """Test that default settings are loaded correctly."""
    settings = get_settings()
    
    # Check that required attributes exist
    assert hasattr(settings, "base_dir"), "base_dir not found in settings"
    assert hasattr(settings, "data_dir"), "data_dir not found in settings"
    assert hasattr(settings, "midi_dir"), "midi_dir not found in settings"
    assert hasattr(settings, "text_dir"), "text_dir not found in settings"
    assert hasattr(settings, "processed_dir"), "processed_dir not found in settings"
    assert hasattr(settings, "output_dir"), "output_dir not found in settings"
    assert hasattr(settings, "model_dir"), "model_dir not found in settings"
    assert hasattr(settings, "checkpoint_dir"), "checkpoint_dir not found in settings"
    assert hasattr(settings, "evaluation_dir"), "evaluation_dir not found in settings"
    assert hasattr(settings, "batch_size"), "batch_size not found in settings"
    assert hasattr(settings, "learning_rate"), "learning_rate not found in settings"
    assert hasattr(settings, "log_level"), "log_level not found in settings"
    
    # Check default values
    assert settings.batch_size == 32, "Default batch_size should be 32"
    assert settings.learning_rate == 0.0001, "Default learning_rate should be 0.0001"
    assert settings.log_level == "info", "Default log_level should be 'info'" 
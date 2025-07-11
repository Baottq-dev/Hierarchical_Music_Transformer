#!/usr/bin/env python3
"""
Test script for AMT configuration with environment variables.
"""

import os
import sys
import pytest
from pathlib import Path

@pytest.fixture(scope="function")
def setup_env_vars():
    """Set up environment variables for testing."""
    # Save original environment variables
    original_env = {}
    env_vars = [
        "AMT_DATA_DIR", "AMT_MIDI_DIR", "AMT_TEXT_DIR", 
        "AMT_PROCESSED_DIR", "AMT_OUTPUT_DIR", 
        "AMT_BATCH_SIZE", "AMT_LOG_LEVEL"
    ]
    
    for var in env_vars:
        if var in os.environ:
            original_env[var] = os.environ[var]
    
    # Set environment variables for testing
    os.environ["AMT_DATA_DIR"] = "./custom_data"
    os.environ["AMT_MIDI_DIR"] = "./custom_data/midi"
    os.environ["AMT_TEXT_DIR"] = "./custom_data/text"
    os.environ["AMT_PROCESSED_DIR"] = "./custom_data/processed"
    os.environ["AMT_OUTPUT_DIR"] = "./custom_data/output"
    os.environ["AMT_BATCH_SIZE"] = "64"
    os.environ["AMT_LOG_LEVEL"] = "debug"
    
    yield
    
    # Restore original environment variables
    for var in env_vars:
        if var in original_env:
            os.environ[var] = original_env[var]
        elif var in os.environ:
            del os.environ[var]

def test_settings_with_env_vars(setup_env_vars):
    """Test that settings are correctly loaded from environment variables."""
    from amt.config import AMTSettings
    
    # Create a new instance to pick up the environment variables
    settings = AMTSettings()
    
    # Verify that environment variables were applied
    assert "custom_data" in str(settings.data_dir), "Data directory not set correctly"
    assert settings.batch_size == 64, "Batch size not set correctly"
    assert settings.log_level == "debug", "Log level not set correctly"
    
    # Verify that path handling works correctly
    assert "custom_data/midi" in str(settings.midi_dir).replace("\\", "/"), "MIDI directory not set correctly"
    assert "custom_data/text" in str(settings.text_dir).replace("\\", "/"), "Text directory not set correctly"
    assert "custom_data/processed" in str(settings.processed_dir).replace("\\", "/"), "Processed directory not set correctly"
    assert "custom_data/output" in str(settings.output_dir).replace("\\", "/"), "Output directory not set correctly"

def test_environment_variables(setup_env_vars):
    """Test that environment variables are correctly set."""
    assert os.environ.get("AMT_DATA_DIR") == "./custom_data"
    assert os.environ.get("AMT_MIDI_DIR") == "./custom_data/midi"
    assert os.environ.get("AMT_TEXT_DIR") == "./custom_data/text"
    assert os.environ.get("AMT_PROCESSED_DIR") == "./custom_data/processed"
    assert os.environ.get("AMT_OUTPUT_DIR") == "./custom_data/output"
    assert os.environ.get("AMT_BATCH_SIZE") == "64"
    assert os.environ.get("AMT_LOG_LEVEL") == "debug" 
#!/usr/bin/env python3
"""
Simple test script for AMT configuration.
"""

import os
import pytest
from pathlib import Path

@pytest.fixture(scope="function")
def setup_custom_env():
    """Set up custom environment variables for testing."""
    # Save original environment variables
    original_env = {}
    env_vars = [
        "AMT_DATA_DIR", "AMT_MIDI_DIR", "AMT_TEXT_DIR", 
        "AMT_PROCESSED_DIR", "AMT_OUTPUT_DIR", "AMT_MODEL_DIR",
        "AMT_CHECKPOINT_DIR", "AMT_EVALUATION_DIR",
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
    os.environ["AMT_MODEL_DIR"] = "./custom_models"
    os.environ["AMT_CHECKPOINT_DIR"] = "./custom_models/checkpoints"
    os.environ["AMT_EVALUATION_DIR"] = "./custom_models/evaluation"
    os.environ["AMT_BATCH_SIZE"] = "64"
    os.environ["AMT_LOG_LEVEL"] = "debug"
    
    yield
    
    # Restore original environment variables
    for var in env_vars:
        if var in original_env:
            os.environ[var] = original_env[var]
        elif var in os.environ:
            del os.environ[var]

def test_amt_settings_direct(setup_custom_env):
    """Test creating AMTSettings directly."""
    # Import directly from the file to avoid circular imports
    from amt.config import AMTSettings
    
    # Create settings directly
    settings = AMTSettings()
    
    # Verify that environment variables were applied
    assert "custom_data" in str(settings.data_dir), "Data directory not set correctly"
    assert "custom_data/midi" in str(settings.midi_dir).replace("\\", "/"), "MIDI directory not set correctly"
    assert "custom_data/text" in str(settings.text_dir).replace("\\", "/"), "Text directory not set correctly"
    assert "custom_data/processed" in str(settings.processed_dir).replace("\\", "/"), "Processed directory not set correctly"
    assert "custom_data/output" in str(settings.output_dir).replace("\\", "/"), "Output directory not set correctly"
    assert "custom_models" in str(settings.model_dir).replace("\\", "/"), "Model directory not set correctly"
    assert "custom_models/checkpoints" in str(settings.checkpoint_dir).replace("\\", "/"), "Checkpoint directory not set correctly"
    assert "custom_models/evaluation" in str(settings.evaluation_dir).replace("\\", "/"), "Evaluation directory not set correctly"
    assert settings.batch_size == 64, "Batch size not set correctly"
    assert settings.log_level == "debug", "Log level not set correctly" 
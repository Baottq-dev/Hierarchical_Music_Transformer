"""
Configuration module for AMT using pydantic BaseSettings.
This allows for environment variable overrides and validation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, ClassVar, Type

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AMTSettings(BaseSettings):
    """
    AMT configuration settings using pydantic BaseSettings.
    
    This allows configuration to be loaded from environment variables, 
    with validation and type conversion.
    
    Environment variables are prefixed with AMT_, e.g. AMT_DATA_DIR
    """
    
    # Base directories
    base_dir: Path = Field(
        default=Path("."), 
        description="Base directory for the project"
    )
    
    # Data directories
    data_dir: Path = Field(
        default=Path("data"), 
        description="Directory for all data"
    )
    midi_dir: Path = Field(
        default=Path(""), 
        description="Directory for MIDI files"
    )
    text_dir: Path = Field(
        default=Path(""), 
        description="Directory for text files"
    )
    processed_dir: Path = Field(
        default=Path(""), 
        description="Directory for processed data"
    )
    output_dir: Path = Field(
        default=Path(""), 
        description="Directory for output files"
    )
    
    # Model directories
    model_dir: Path = Field(
        default=Path("models"), 
        description="Directory for model files"
    )
    checkpoint_dir: Path = Field(
        default=Path(""), 
        description="Directory for model checkpoints"
    )
    evaluation_dir: Path = Field(
        default=Path(""), 
        description="Directory for evaluation results"
    )
    
    # Processing parameters
    max_sequence_length: int = Field(
        default=1024, 
        description="Maximum sequence length for MIDI tokens"
    )
    max_text_length: int = Field(
        default=512, 
        description="Maximum sequence length for text tokens"
    )
    
    # Training parameters
    batch_size: int = Field(
        default=32, 
        description="Batch size for training"
    )
    learning_rate: float = Field(
        default=1e-4, 
        description="Learning rate for training"
    )
    weight_decay: float = Field(
        default=1e-5, 
        description="Weight decay for training"
    )
    max_epochs: int = Field(
        default=100, 
        description="Maximum number of epochs for training"
    )
    
    # Model parameters
    vocab_size: int = Field(
        default=1000, 
        description="Vocabulary size for the model"
    )
    d_model: int = Field(
        default=512, 
        description="Model dimension"
    )
    nhead: int = Field(
        default=8, 
        description="Number of attention heads"
    )
    num_encoder_layers: int = Field(
        default=6, 
        description="Number of encoder layers"
    )
    num_decoder_layers: int = Field(
        default=6, 
        description="Number of decoder layers"
    )
    dim_feedforward: int = Field(
        default=2048, 
        description="Dimension of feedforward network"
    )
    dropout: float = Field(
        default=0.1, 
        description="Dropout rate"
    )
    
    # Generation parameters
    temperature: float = Field(
        default=1.0, 
        description="Temperature for sampling"
    )
    top_k: int = Field(
        default=50, 
        description="Top-k sampling parameter"
    )
    top_p: float = Field(
        default=0.95, 
        description="Top-p sampling parameter"
    )
    repetition_penalty: float = Field(
        default=1.2, 
        description="Repetition penalty for sampling"
    )
    max_generate_length: int = Field(
        default=1024, 
        description="Maximum length for generation"
    )
    
    # Logging
    log_level: str = Field(
        default="info", 
        description="Logging level"
    )
    log_file: Optional[str] = Field(
        default=None, 
        description="Log file path"
    )
    
    # Use SettingsConfigDict instead of Config class
    model_config = SettingsConfigDict(
        env_prefix="AMT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    @model_validator(mode='after')
    def validate_paths(self) -> 'AMTSettings':
        """Ensure paths are absolute and derive subdirectories."""
        # Make base_dir absolute
        if not self.base_dir.is_absolute():
            self.base_dir = self.base_dir.absolute()
        
        # Make data_dir absolute if it's not already
        if not self.data_dir.is_absolute():
            self.data_dir = self.base_dir / self.data_dir
        
        # Set subdirectories if not explicitly provided
        if str(self.midi_dir) == "":
            self.midi_dir = self.data_dir / "midi"
        elif not self.midi_dir.is_absolute():
            self.midi_dir = self.base_dir / self.midi_dir
            
        if str(self.text_dir) == "":
            self.text_dir = self.data_dir / "text"
        elif not self.text_dir.is_absolute():
            self.text_dir = self.base_dir / self.text_dir
            
        if str(self.processed_dir) == "":
            self.processed_dir = self.data_dir / "processed"
        elif not self.processed_dir.is_absolute():
            self.processed_dir = self.base_dir / self.processed_dir
            
        if str(self.output_dir) == "":
            self.output_dir = self.data_dir / "output"
        elif not self.output_dir.is_absolute():
            self.output_dir = self.base_dir / self.output_dir
            
        # Make model_dir absolute if it's not already
        if not self.model_dir.is_absolute():
            self.model_dir = self.base_dir / self.model_dir
            
        # Set model subdirectories if not explicitly provided
        if str(self.checkpoint_dir) == "":
            self.checkpoint_dir = self.model_dir / "checkpoints"
        elif not self.checkpoint_dir.is_absolute():
            self.checkpoint_dir = self.base_dir / self.checkpoint_dir
            
        if str(self.evaluation_dir) == "":
            self.evaluation_dir = self.model_dir / "evaluation"
        elif not self.evaluation_dir.is_absolute():
            self.evaluation_dir = self.base_dir / self.evaluation_dir
            
        return self
    
    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary with string paths."""
        result = super().model_dump(*args, **kwargs)
        # Convert Path objects to strings for serialization
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
        return result


# Create a global instance of settings
settings = AMTSettings()


def get_settings() -> AMTSettings:
    """Get the global settings instance."""
    return settings 
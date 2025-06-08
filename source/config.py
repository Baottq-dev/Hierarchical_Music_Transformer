"""
Configuration module for AMT.
Contains configuration settings and path management for the project.
"""

import os
import json
from typing import Dict, Any
from pathlib import Path

class Config:
    """Configuration manager for AMT."""
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration.
        Args:
            config_file: Path to configuration file (optional)
        """
        self.base_dir = Path(__file__).parent.parent
        self.config = self._load_default_config()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "paths": {
                "data_dir": str(self.base_dir / "data"),
                "raw_data_dir": str(self.base_dir / "data" / "raw"),
                "processed_data_dir": str(self.base_dir / "data" / "processed"),
                "model_dir": str(self.base_dir / "models"),
                "output_dir": str(self.base_dir / "output"),
                "logs_dir": str(self.base_dir / "logs")
            },
            "model": {
                "name": "gpt2",
                "max_length": 512,
                "batch_size": 1,
                "learning_rate": 5e-5,
                "num_epochs": 3
            },
            "generation": {
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9,
                "max_length": 512
            },
            "clustering": {
                "n_clusters": 10,
                "random_state": 42
            }
        }

    def load_config(self, config_file: str):
        """
        Load configuration from file.
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, "r") as f:
                user_config = json.load(f)
            
            # Update default config with user config
            self._update_dict(self.config, user_config)
        except Exception as e:
            print(f"Error loading config file: {e}")

    def _update_dict(self, d: Dict, u: Dict):
        """Recursively update dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_dict(d[k], v)
            else:
                d[k] = v

    def save_config(self, config_file: str):
        """
        Save current configuration to file.
        Args:
            config_file: Path to save configuration file
        """
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config file: {e}")

    def get_path(self, key: str) -> str:
        """
        Get path from configuration.
        Args:
            key: Path key (e.g., "data_dir", "model_dir")
        Returns:
            Path string
        """
        return self.config["paths"].get(key)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config["model"]

    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration."""
        return self.config["generation"]

    def get_clustering_config(self) -> Dict[str, Any]:
        """Get clustering configuration."""
        return self.config["clustering"]

    def create_directories(self):
        """Create all necessary directories."""
        for path in self.config["paths"].values():
            os.makedirs(path, exist_ok=True)

def load_config(config_file: str = None) -> Config:
    """
    Load configuration from file.
    Args:
        config_file: Path to configuration file (optional)
    Returns:
        Config instance
    """
    return Config(config_file) 
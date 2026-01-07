"""
Logging utilities for Mamba Music.
"""

import logging
import sys
from typing import Dict, Optional

# Define log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s - %(message)s"

# Global logger cache
_loggers: Dict[str, logging.Logger] = {}
_default_level = "info"


def setup_logging(level: str = "info") -> None:
    """Setup default logging level."""
    global _default_level
    _default_level = level.lower()


def get_logger(
    name: str, 
    level: Optional[str] = None, 
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Get a logger with the specified name and configuration.
    
    Args:
        name: Name of the logger
        level: Log level (debug, info, warning, error, critical)
        format_str: Format string for log messages
        
    Returns:
        Configured logger
    """
    # Check if logger already exists
    if name in _loggers:
        return _loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    logger_level = level or _default_level
    logger.setLevel(LOG_LEVELS.get(logger_level.lower(), logging.INFO))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_str or DEFAULT_FORMAT)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Cache logger
    _loggers[name] = logger
    
    return logger
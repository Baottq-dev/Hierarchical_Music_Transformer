"""
Logging utilities for AMT.
"""

import logging
import os
import sys
from typing import Dict, Optional, Any

from amt.config import get_settings

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

# Global logger cache to avoid creating multiple loggers for the same name
_loggers: Dict[str, logging.Logger] = {}


def get_logger(
    name: str, 
    level: Optional[str] = None, 
    format_str: Optional[str] = None,
    log_file: Optional[str] = None,
    use_console: bool = True
) -> logging.Logger:
    """
    Get a logger with the specified name and configuration.
    
    Args:
        name: Name of the logger
        level: Log level (debug, info, warning, error, critical)
        format_str: Format string for log messages
        log_file: Path to log file
        use_console: Whether to log to console
        
    Returns:
        Configured logger
    """
    # Check if logger already exists
    if name in _loggers:
        return _loggers[name]
    
    # Get settings
    settings = get_settings()
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level (use provided level or default from settings)
    logger_level = level or settings.log_level
    logger.setLevel(LOG_LEVELS.get(logger_level.lower(), logging.INFO))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_str or DEFAULT_FORMAT)
    
    # Add console handler
    if use_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if specified
    log_file_path = log_file or settings.log_file
    if log_file_path:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Cache logger
    _loggers[name] = logger
    
    return logger


def configure_root_logger(
    level: Optional[str] = None,
    format_str: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure the root logger.
    
    Args:
        level: Log level (debug, info, warning, error, critical)
        format_str: Format string for log messages
        log_file: Path to log file
        
    Returns:
        Configured root logger
    """
    settings = get_settings()
    return get_logger(
        "amt", 
        level or settings.log_level, 
        format_str, 
        log_file or settings.log_file
    )


def set_log_level(logger_name: str, level: str) -> None:
    """
    Set the log level for a logger.
    
    Args:
        logger_name: Name of the logger
        level: Log level (debug, info, warning, error, critical)
    """
    if logger_name in _loggers:
        _loggers[logger_name].setLevel(LOG_LEVELS.get(level.lower(), logging.INFO)) 
"""
AMT - Automated Music Transcription

A Python package for automated music transcription and generation using Transformer models.
"""

__version__ = "0.1.0"

# Import main subpackages to make them available at the top level
from . import collect
from . import process
from . import train
from . import generate
from . import evaluate
from . import utils 
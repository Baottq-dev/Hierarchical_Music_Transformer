# AMT: Automated Music Transcription

Welcome to the documentation for AMT, an automated music transcription system using transformer models.

[![CI](https://github.com/username/AMT/actions/workflows/ci.yml/badge.svg)](https://github.com/username/AMT/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is AMT?

AMT (Automated Music Transcription) is a system that converts audio recordings of music into symbolic representations (MIDI) and vice versa. It uses transformer models to understand the relationship between music and its textual descriptions.

## Key Features

- **Bidirectional Conversion**: Convert between audio/MIDI and text descriptions
- **Transformer Architecture**: State-of-the-art transformer models for music understanding
- **Flexible Configuration**: Easy to configure via environment variables or config files
- **Command-line Interface**: Simple CLI for all operations
- **Comprehensive Documentation**: Detailed docs and examples

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/AMT.git
cd AMT

# Install the package
pip install -e .
```

### Basic Usage

```bash
# Collect data
amt collect --midi_dir data/midi --output_dir data/output

# Process data
amt process --input_dir data/midi --output_dir data/processed

# Train model
amt train --data_dir data/processed --model_dir models

# Generate music
amt generate --checkpoint models/checkpoints/model.pt --output_dir output

# Test model
amt test --checkpoint models/checkpoints/model.pt --test_dir data/test
```

## Documentation Structure

- **[Overview](overview/)**: Introduction to AMT, its architecture, and how it works
- **[Usage](usage/)**: Detailed guides on how to use AMT for various tasks
- **[API Reference](api/)**: Complete API documentation for developers
- **[Development](development/)**: Information for contributors and developers

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
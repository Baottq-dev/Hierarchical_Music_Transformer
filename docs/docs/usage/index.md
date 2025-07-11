# Usage Guide

This section provides detailed instructions on how to use the AMT system for various tasks.

## Quick Navigation

- [Installation](installation.md): How to install the AMT package
- [Configuration](configuration.md): How to configure the AMT system
- [Data Collection](data-collection.md): How to collect and prepare data
- [Data Processing](data-processing.md): How to process MIDI and text data
- [Training](training.md): How to train the AMT model
- [Generation](generation.md): How to generate music and text
- [Evaluation](evaluation.md): How to evaluate the model's performance

## Basic Workflow

The typical workflow for using AMT involves the following steps:

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/username/AMT.git
cd AMT

# Install the package
pip install -e .
```

### 2. Data Collection

```bash
# Collect MIDI files and text descriptions
amt collect --midi_dir data/midi --text_dir data/text --output_dir data/collected
```

### 3. Data Processing

```bash
# Process the collected data
amt process --input_dir data/collected --output_dir data/processed
```

### 4. Training

```bash
# Train the model
amt train --data_dir data/processed --model_dir models
```

### 5. Generation

```bash
# Generate MIDI from text
amt generate --checkpoint models/checkpoints/model.pt --text "A cheerful piano melody in C major" --output_dir output

# Generate text from MIDI
amt generate --checkpoint models/checkpoints/model.pt --midi input.mid --output_dir output
```

### 6. Evaluation

```bash
# Evaluate the model
amt test --checkpoint models/checkpoints/model.pt --test_dir data/test
```

## Command-Line Interface

AMT provides a comprehensive command-line interface (CLI) for all operations. The CLI follows a consistent pattern:

```bash
amt <command> [options]
```

Where `<command>` is one of:
- `collect`: Collect data
- `process`: Process data
- `train`: Train the model
- `generate`: Generate music or text
- `test`: Test the model

For detailed information about each command, use the `--help` option:

```bash
amt collect --help
amt process --help
amt train --help
amt generate --help
amt test --help
```

## Python API

In addition to the CLI, AMT provides a Python API that can be used in your own scripts:

```python
from amt.collect import collect_data
from amt.process import process_data
from amt.train import train_model
from amt.generate import generate_music
from amt.evaluate import evaluate_model

# Collect data
collect_data(midi_dir="data/midi", text_dir="data/text", output_dir="data/collected")

# Process data
process_data(input_dir="data/collected", output_dir="data/processed")

# Train model
train_model(data_dir="data/processed", model_dir="models")

# Generate music
generate_music(checkpoint="models/checkpoints/model.pt", text="A cheerful piano melody in C major", output_dir="output")

# Evaluate model
evaluate_model(checkpoint="models/checkpoints/model.pt", test_dir="data/test")
```

For more detailed information about the Python API, see the [API Reference](../api/index.md). 
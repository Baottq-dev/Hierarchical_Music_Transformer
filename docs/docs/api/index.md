# API Reference

This section provides detailed documentation for the AMT API. It's organized by module, with each page covering a specific aspect of the system.

## Modules

- [Configuration](config.md): Configuration settings and utilities
- [Data Collection](collect.md): Tools for collecting and pairing MIDI and text data
- [Data Processing](process.md): Tools for processing MIDI and text data
- [Training](train.md): Model definition and training utilities
- [Generation](generate.md): Tools for generating music and text
- [Evaluation](evaluate.md): Metrics and evaluation utilities
- [Utilities](utils.md): Common utilities used across the system

## Core Classes

### Configuration

```python
from amt.config import get_settings

settings = get_settings()
print(settings.midi_dir)
```

Key classes:
- `AMTSettings`: Central configuration class

### Data Collection

```python
from amt.collect.midi_collector import MIDICollector
from amt.collect.text_collector import TextCollector
from amt.collect.data_pairing import DataPairer

# Collect MIDI files
midi_collector = MIDICollector(midi_dir="data/midi")
midi_data = midi_collector.collect()

# Collect text descriptions
text_collector = TextCollector(text_dir="data/text")
text_data = text_collector.collect()

# Pair MIDI and text data
data_pairer = DataPairer()
paired_data = data_pairer.pair(midi_data, text_data)
```

Key classes:
- `MIDICollector`: Collects MIDI files
- `TextCollector`: Collects text descriptions
- `DataPairer`: Pairs MIDI and text data

### Data Processing

```python
from amt.process.midi_processor import MIDIProcessor
from amt.process.text_processor import TextProcessor
from amt.process.data_preparer import DataPreparer

# Process MIDI files
midi_processor = MIDIProcessor()
processed_midi = midi_processor.process_midi_file("path/to/midi.mid")

# Process text descriptions
text_processor = TextProcessor()
processed_text = text_processor.process_text_file("path/to/text.txt")

# Prepare data for training
data_preparer = DataPreparer(output_dir="data/processed")
data_preparer.prepare_training_data(dataset=[...])
```

Key classes:
- `MIDIProcessor`: Processes MIDI files
- `TextProcessor`: Processes text descriptions
- `DataPreparer`: Prepares data for training

### Training

```python
from amt.train.model import MusicTransformer
from amt.train.trainer import Trainer, create_trainer
from amt.train.training_loop import TrainingLoop

# Create model
model = MusicTransformer(vocab_size=512)

# Create trainer
trainer = create_trainer(
    model=model,
    train_data_path="data/processed/train.json",
    val_data_path="data/processed/val.json"
)

# Train model
trainer.train(num_epochs=100)
```

Key classes:
- `MusicTransformer`: Transformer model for music generation
- `MusicTextTransformer`: Bidirectional transformer model
- `Trainer`: Manages the training process
- `TrainingLoop`: Handles the training loop logic

### Generation

```python
from amt.generate.generator import Generator

# Create generator
generator = Generator(checkpoint="models/checkpoints/model.pt")

# Generate MIDI from text
midi = generator.generate_from_text("A cheerful piano melody in C major")

# Generate text from MIDI
text = generator.generate_from_midi("path/to/midi.mid")
```

Key classes:
- `Generator`: Generates music and text

### Evaluation

```python
from amt.evaluate.evaluator import Evaluator
from amt.evaluate.metrics import calculate_metrics

# Create evaluator
evaluator = Evaluator(checkpoint="models/checkpoints/model.pt")

# Evaluate model
results = evaluator.evaluate(test_dir="data/test")

# Calculate metrics
metrics = calculate_metrics(predictions, targets)
```

Key classes:
- `Evaluator`: Evaluates model performance
- `Tester`: Runs comprehensive tests

## Using the API

For more detailed information about how to use each component of the API, see the individual module pages linked above. Each page includes:

- Class and function documentation
- Examples
- Parameter descriptions
- Return value descriptions 
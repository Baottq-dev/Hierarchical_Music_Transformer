# AMT - Automated Music Transcription

AMT is a project that focuses on automated music transcription and generation using transformer-based models. It can convert between MIDI music files and text descriptions, enabling both music-to-text and text-to-music generation.

## Features

- Convert MIDI music to text descriptions
- Generate MIDI music from text descriptions
- Hierarchical music representation
- Advanced transformer architecture
- Transfer learning support for improved performance

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AMT.git
cd AMT

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

AMT follows a pipeline approach:

1. **Collect data**: Gather MIDI files and text descriptions
2. **Process data**: Convert MIDI and text into model-compatible formats
3. **Train models**: Train the transformer models
4. **Generate music**: Generate MIDI from text or text from MIDI
5. **Evaluate**: Evaluate model performance

### Data Collection

```bash
python collect.py --midi-dir path/to/midi/files --output-file data/collected_data.json
```

### Data Processing

```bash
python process.py single --midi-file path/to/midi/file.mid --text-file path/to/description.txt --output-dir data/processed
```

Or for batch processing:

```bash
python process.py paired --paired-data-file data/collected_data.json --output-dir data/processed
```

### Training

```bash
python train.py --paired-data-file data/processed/paired_data.json --output-dir models
```

### Transfer Learning

AMT now supports transfer learning to improve model performance:

#### Text Model Transfer Learning

```bash
# Process data with a pre-trained text model
python process.py paired --paired-data-file data/collected_data.json --output-dir data/processed --use-pretrained-text-model --pretrained-text-model-path models/pretrained_bert

# Fine-tune the text model on music descriptions
python process.py paired --paired-data-file data/collected_data.json --output-dir data/processed --use-pretrained-text-model --pretrained-text-model-path models/pretrained_bert --enable-text-fine-tuning
```

#### Music Model Transfer Learning

```bash
# Train with a pre-trained music model
python train.py --paired-data-file data/processed/paired_data.json --output-dir models --pretrained-model models/checkpoints/pretrained_model.pt --transfer-learning-mode fine_tuning --freeze-layers 3
```

Transfer learning modes:
- `feature_extraction`: Freezes all layers except output layer
- `fine_tuning`: Freezes a specified number of layers
- `full_fine_tuning`: All layers are trainable

### Generation

```bash
python generate.py --model-path models/checkpoints/model.pt --text "A cheerful piano melody with jazz influences"
```

## Project Structure

- `amt/`: Main package directory
  - `collect/`: Data collection modules
  - `process/`: Data processing modules
  - `train/`: Model training modules
  - `generate/`: Music generation modules
  - `evaluate/`: Evaluation modules
  - `models/`: Model architecture definitions
  - `utils/`: Utility functions

## Documentation

For more detailed documentation, see the `docs/` directory:

- [Project Overview](docs/project_overview.md)
- [Running Guide](docs/running_guide.md)
- [Model Deep Dive](docs/05_model_deep_dive.md)
- [Configuration](docs/configuration.md)
- [Checkpoint System](docs/checkpoint_system.md)

## License

[MIT License](LICENSE) 
# AMT - Automated Music Transcription

A Python project for automated music transcription and generation using Transformer models.

## Features

- MIDI file processing and tokenization
- Text-to-music generation
- Music Transformer model
- Evaluation metrics for music generation

## Installation

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/amt.git
   cd amt
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Regular Installation

```bash
pip install -e .
```

## Usage

### Data Collection

```bash
python run_collect.py --midi-dir data/midi --output-dir data/output
```

### Data Processing

```bash
python run_process.py --input-file data/output/automated_paired_data.json --output-dir data/processed
```

### Model Training

```bash
python run_train.py --data-file data/processed/training_data.json --output-dir models/checkpoints
```

### Music Generation

```bash
python run_generate.py --checkpoint models/checkpoints/model_best.pt --prompt "A classical piano piece in C major"
```

## Project Structure

- `source/`: Core library modules
  - `collect/`: Data collection utilities
  - `process/`: Data processing and preparation
  - `train/`: Model training and generation
  - `test/`: Evaluation and metrics
- `data/`: Data directory
- `models/`: Model checkpoints
- `docs/`: Documentation

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run pre-commit hooks (`pre-commit run --all-files`)
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request 
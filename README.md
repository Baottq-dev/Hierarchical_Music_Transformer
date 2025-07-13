# Automatic Music Transcription (AMT) Project

This project provides tools for automatic music transcription and processing of MIDI files, with emphasis on using pretrained models for music understanding.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.27+
- Additional dependencies in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download language model for spaCy: `python -m spacy download en_core_web_sm`

## Project Structure

- `amt/`: Core package
  - `collect/`: Tools for collecting MIDI data
  - `process/`: Tools for processing MIDI and text
  - `evaluate/`: Evaluation tools
  - `generate/`: Generation tools
  - `models/`: Model definitions
  - `train/`: Training code
- `data/`: Data storage
  - `midi/`: MIDI files
  - `processed/`: Processed data
  - `output/`: Output files
- `examples/`: Example files and notebooks

## Paired Data Format

The paired data JSON file should have a structure like:

```json
{
  "pairs": [
    {
      "midi_file": "data/midi/artist/song.mid",
      "text_file": "path/to/description.txt",
      "metadata": {
        "composer": "Example Composer",
        "title": "Piano Sonata No. 1",
        "year": 2023
      }
    },
    ...
  ],
  "processing_options": {
    "use_hierarchical_encoding": true,
    "use_contextual_embeddings": true,
    "max_sequence_length": 1024
  }
}
```

The system also accepts a direct array of pairs or objects with direct `text` field instead of `text_file`.

## Usage

### Processing MIDI Files

To process paired data using the default settings:

```bash
python process.py --paired-data-file examples/paired_data_sample.json --output-dir data/processed
```

### Using with MERT

This project now uses MERT (Music undERstanding model with large-scale self-supervised Training) from m-a-p for enhanced music understanding:

```bash
# Enable MERT explicitly
python process.py --paired-data-file examples/paired_data_sample.json --use-pretrained-music-model --pretrained-music-model-path m-a-p/MERT-v1-95M
```

Or use the optimal transfer learning settings:

```bash
python process.py --paired-data-file examples/paired_data_sample.json --optimal-transfer-learning
```

### Kaggle Support

The project automatically detects when running in a Kaggle environment and adjusts paths accordingly.

## Models

The project supports several pretrained models:

- **MERT** (m-a-p/MERT-v1-95M): A self-supervised learning model for music audio that uses a combination of RVQ-VAE and CQT-based teacher models
- **RoBERTa** (roberta-base): For text processing and understanding

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
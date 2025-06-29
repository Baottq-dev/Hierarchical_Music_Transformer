# AMT (Audio Music Transformer)

A symbolic music generation system that creates music from text descriptions using BERT embeddings and GPT-2 architecture, based on the paper "The Beat Goes On: Symbolic Music Generation with Text Controls".

## 🎵 Features

- **Text-to-Music Generation**: Generate MIDI music from natural language descriptions
- **BERT Text Embeddings**: Extract semantic meaning from text descriptions
- **GPT-2 Music Generation**: Generate musical sequences using transformer architecture
- **MIDI Processing**: Comprehensive MIDI file analysis and conversion
- **Wikipedia Integration**: Automatic text description collection from Wikipedia
- **Clustering**: Semantic clustering of music styles and genres
- **Evaluation Metrics**: Quality assessment of generated music
- **Unified Pipeline**: Single command to run complete pipeline

## 📁 Project Structure

```
AMT/
├── data/
│   ├── midi/                    # Input MIDI files (Lakh MIDI dataset)
│   ├── output/                  # Generated data files
│   ├── processed/               # Processed data
│   ├── reference/               # Reference MIDI files for evaluation
│   └── evaluation/              # Evaluation results
├── models/
│   └── checkpoints/             # Trained model checkpoints
├── source/                      # Core modules
│   ├── data_collection/         # Data collection modules
│   │   ├── midi_metadata.py     # MIDI file scanning and metadata extraction
│   │   └── wikipedia_collector.py # Wikipedia text collection
│   ├── data_processing/         # Data processing modules
│   │   ├── midi_processor.py    # MIDI to event sequence conversion
│   │   ├── text_processor.py    # Text cleaning and BERT embeddings
│   │   ├── process_data.py      # Main data processing pipeline
│   │   └── prepare_training_data.py # Training data preparation
│   ├── model/                   # Model modules
│   │   ├── training.py          # GPT-2 model training
│   │   ├── generation.py        # Music generation
│   │   └── clustering.py        # K-means clustering
│   ├── evaluation/              # Evaluation modules
│   │   └── metrics.py           # Music quality evaluation metrics
│   ├── utils/                   # Utility modules
│   │   ├── data_preparation.py  # Training data preparation utilities
│   │   └── environment.py       # Environment verification
│   └── config.py                # Configuration parameters
├── main.py                      # Main pipeline controller
├── collect_data.py              # Data collection script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Prepare MIDI Data

Place your MIDI files in the `data/midi/` directory with the following structure:
```
data/midi/
├── Artist_Name_1/
│   ├── song1.mid
│   └── song2.mid
└── Artist_Name_2/
    ├── song3.mid
    └── song4.mid
```

### 3. Run the Complete Pipeline

```bash
# Run all steps at once
python main.py

# Or run step by step
python main.py collect
python main.py process
python main.py prepare
python main.py train
python main.py generate
python main.py evaluate
```

## 📋 Pipeline Usage

### Main Pipeline Controller (`main.py`)

The `main.py` file provides a unified interface to control the entire AMT pipeline:

```bash
# Run complete pipeline
python main.py

# Run specific steps
python main.py collect process prepare
python main.py train generate
python main.py evaluate

# List all available steps
python main.py --list-steps

# Check dependencies
python main.py --check-deps

# Get help
python main.py --help
```

### Pipeline Steps

| Step | Description | Output Files |
|------|-------------|--------------|
| **collect** | Collect MIDI metadata and Wikipedia descriptions | `data/output/midi_metadata_list.json`, `data/output/automated_paired_data.json` |
| **process** | Process text embeddings and perform clustering | `data/output/text_embeddings.json`, `data/output/clustered_text_data.json` |
| **prepare** | Prepare training data from MIDI and embeddings | `data/output/amt_training_data.json` |
| **train** | Train GPT-2 model with MIDI sequences | `models/checkpoints/checkpoint_epoch_N.pt` |
| **generate** | Generate music from text descriptions | `output/generated_music.mid` |
| **evaluate** | Evaluate generated music quality | `data/evaluation/results.json` |

## 📋 Step-by-Step Usage

### Step 1: Data Collection (`collect`)

```bash
# Using main.py
python main.py collect

# Or directly
python collect_data.py --midi_dir "./data/midi" --delay 2.0
```

**Parameters:**
- `--midi_dir`: MIDI files directory (default: `./data/midi`)
- `--output_dir`: Output directory (default: `./data/output`)
- `--delay`: Wikipedia request delay in seconds (default: 1.0)

### Step 2: Data Processing (`process`)

```bash
# Using main.py
python main.py process

# Or directly
python source/data_processing/process_data.py
```

**Output:**
- `data/output/text_embeddings.json` - BERT text embeddings
- `data/output/clustered_text_data.json` - Clustered embeddings

### Step 3: Training Data Preparation (`prepare`)

```bash
# Using main.py
python main.py prepare

# Or directly
python source/data_processing/prepare_training_data.py
```

**Output:**
- `data/output/amt_training_data.json` - Training data for model

### Step 4: Model Training (`train`)

```bash
# Using main.py
python main.py train

# Or directly
python source/model/training.py
```

**Parameters:**
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)

**Output:**
- `models/checkpoints/checkpoint_epoch_N.pt` - Model checkpoints

### Step 5: Music Generation (`generate`)

```bash
# Using main.py
python main.py generate

# Or directly
python source/model/generation.py
```

**Parameters:**
- `--text_description`: Text description for generation
- `--output_file`: Output MIDI file path
- `--temperature`: Sampling temperature (default: 1.0)
- `--max_length`: Maximum sequence length (default: 512)

### Step 6: Evaluation (`evaluate`)

```bash
# Using main.py
python main.py evaluate

# Or directly
python source/evaluation/metrics.py
```

## 📊 Output Files

After running the pipeline, you'll find these files:

### Data Collection Output:
- `data/output/midi_metadata_list.json` - MIDI file metadata
- `data/output/automated_paired_data.json` - MIDI + Wikipedia descriptions

### Data Processing Output:
- `data/output/text_embeddings.json` - BERT text embeddings
- `data/output/clustered_text_data.json` - Clustered embeddings
- `data/output/amt_training_data.json` - Training data

### Model Output:
- `models/checkpoints/checkpoint_epoch_N.pt` - Model checkpoints
- `output/generated_music.mid` - Generated MIDI files

### Evaluation Output:
- `data/evaluation/results.json` - Evaluation metrics and scores

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.5+
- Mido 1.2+
- scikit-learn 0.24+
- spaCy 3.0+
- Other dependencies in `requirements.txt`

## 🎯 Evaluation Metrics

The system includes evaluation metrics for assessing generated music quality:

- **Note Density Ratio**: Similarity in note density
- **Velocity Similarity**: Similarity in velocity distributions
- **Note Range Similarity**: Similarity in note ranges
- **Time Signature Match**: Match in time signatures
- **Tempo Similarity**: Similarity in tempo
- **Overall Score**: Weighted combination of all metrics

## 🚨 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure virtual environment is activated
2. **No MIDI files found**: Check `data/midi/` directory structure
3. **Wikipedia API errors**: Increase `--delay` parameter
4. **Out of memory**: Reduce `--batch_size`
5. **Model not found**: Train model first or check checkpoint path

### Performance Tips

- Use `--skip_wikipedia` for faster testing
- Reduce `--batch_size` if you have limited RAM
- Increase `--delay` to avoid Wikipedia rate limiting
- Use `python main.py --check-deps` to verify dependencies

### Quick Commands

```bash
# Check if everything is set up correctly
python main.py --check-deps

# Run a quick test with just data collection
python main.py collect

# Run processing and training only
python main.py process prepare train

# Generate music with custom parameters
python main.py generate --text_description "A happy jazz piece"
```

## 📚 Technical Details

### Architecture

1. **Text Processing**: BERT embeddings for semantic understanding
2. **MIDI Processing**: Event-based representation (TIME_ON, NOTE, DURATION)
3. **Clustering**: K-means clustering of text embeddings
4. **Generation**: GPT-2 with BERT conditioning
5. **Evaluation**: Multi-metric quality assessment

### Data Flow

```
MIDI Files → Metadata → Wikipedia → BERT Embeddings → Clustering → Training Data → Model Training → Music Generation → Evaluation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on "The Beat Goes On: Symbolic Music Generation with Text Controls"
- Uses Lakh MIDI dataset for training
- Built with PyTorch and Transformers libraries
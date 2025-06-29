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

## 📁 Project Structure

```
AMT/
├── data/
│   ├── midi/                    # Input MIDI files (Lakh MIDI dataset)
│   ├── output/                  # Generated data files
│   └── processed/               # Processed data
├── models/
│   └── checkpoints/             # Trained model checkpoints
├── output/                      # Generated music files
├── source/                      # Core modules
│   ├── data_collection/         # Data collection modules
│   ├── data_processing/         # Data processing modules
│   ├── model/                   # Model modules
│   ├── evaluation/              # Evaluation modules
│   └── utils/                   # Utility modules
├── collect_data.py              # Data collection script
├── train.py                     # Training script
├── test.py                      # Testing script
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
# Step 1: Collect data
python collect_data.py

# Step 2: Train model
python train.py

# Step 3: Test generation
python test.py
```

## 📋 Step-by-Step Usage

### Step 1: Data Collection (`collect_data.py`)

```bash
# Basic usage
python collect_data.py

# Skip Wikipedia collection (faster for testing)
python collect_data.py --skip_wikipedia

# Custom parameters
python collect_data.py --midi_dir "./data/midi" --delay 2.0
```

**Parameters:**
- `--midi_dir`: MIDI files directory (default: `./data/midi`)
- `--output_dir`: Output directory (default: `./data/output`)
- `--skip_wikipedia`: Skip Wikipedia collection
- `--delay`: Wikipedia request delay in seconds (default: 1.0)

### Step 2: Training (`train.py`)

```bash
# Basic usage
python train.py

# Custom training parameters
python train.py --batch_size 16 --epochs 20 --lr 5e-5

# Skip data processing (if already done)
python train.py --skip_processing

# Skip model training (data processing only)
python train.py --skip_training
```

**Parameters:**
- `--paired_file`: Paired data JSON file (default: `./data/output/automated_paired_data.json`)
- `--output_dir`: Output directory (default: `./data/output`)
- `--model_dir`: Model checkpoint directory (default: `./models/checkpoints`)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--skip_processing`: Skip data processing
- `--skip_training`: Skip model training

### Step 3: Testing (`test.py`)

```bash
# Basic usage
python test.py

# Custom generation
python test.py --text_description "A melancholic jazz piece with saxophone"

# With evaluation
python test.py --original_file "data/midi/Artist/song.mid"

# Custom parameters
python test.py --temperature 0.8 --max_length 1024
```

**Parameters:**
- `--model_path`: Model checkpoint path (default: `./models/checkpoints/checkpoint_epoch_10.pt`)
- `--output_dir`: Output directory (default: `./output`)
- `--text_description`: Text description for generation (default: "A happy pop song with piano and drums")
- `--max_length`: Maximum sequence length (default: 512)
- `--temperature`: Sampling temperature (default: 1.0)
- `--original_file`: Original MIDI file for evaluation
- `--skip_generation`: Skip music generation
- `--skip_evaluation`: Skip music evaluation

## 📊 Output Files

After running the pipeline, you'll find these files:

### Data Collection Output:
- `data/output/midi_metadata_list.json` - MIDI file metadata
- `data/output/automated_paired_data.json` - MIDI + Wikipedia descriptions

### Training Output:
- `data/output/text_embeddings.json` - BERT text embeddings
- `data/output/clustered_text_data.json` - Clustered embeddings
- `data/output/amt_training_data.json` - Training data
- `models/checkpoints/checkpoint_epoch_N.pt` - Model checkpoints

### Testing Output:
- `output/generated_music.mid` - Generated MIDI files
- `output/evaluation_results.json` - Evaluation metrics (if evaluation performed)

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
- Use `--skip_training` to test data pipeline only
- Increase `--delay` to avoid Wikipedia rate limiting

## 📚 Technical Details

### Architecture

1. **Text Processing**: BERT embeddings for semantic understanding
2. **MIDI Processing**: Event-based representation (TIME_ON, NOTE, DURATION)
3. **Clustering**: K-means clustering of text embeddings
4. **Generation**: GPT-2 with BERT conditioning
5. **Evaluation**: Multi-metric quality assessment

### Data Flow

```
MIDI Files → Metadata → Wikipedia → BERT Embeddings → Clustering → Training Data → Model Training → Music Generation
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
# AMT (Audio Music Transformer) - Clean Modular Structure

🎵 **Symbolic Music Generation with Text Controls** - Clean modular pipeline with collect, process, train, test structure.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Usage](#module-usage)
- [Pipeline](#pipeline)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

AMT is a clean, modular implementation of text-to-music generation using transformer architecture. The project is organized into four main modules:

- **COLLECT**: Data collection and pairing
- **PROCESS**: Data processing and preparation
- **TRAIN**: Model training and generation
- **TEST**: Evaluation and testing

## 🏗️ Architecture

```
AMT/
├── source/
│   ├── collect/           # Data collection module
│   │   ├── midi_collector.py
│   │   ├── text_collector.py
│   │   └── data_pairing.py
│   ├── process/           # Data processing module
│   │   ├── midi_processor.py
│   │   ├── text_processor.py
│   │   └── data_preparer.py
│   ├── train/             # Training module
│   │   ├── model.py
│   │   ├── trainer.py
│   │   └── generator.py
│   └── test/              # Testing module
│       ├── evaluator.py
│       ├── metrics.py
│       └── tester.py
├── run_collect.py         # Collect module runner
├── run_process.py         # Process module runner
├── run_train.py           # Train module runner
├── run_test.py            # Test module runner
└── main.py                # Main pipeline runner
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd AMT
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## ⚡ Quick Start

### Option 1: Run Complete Pipeline

```bash
python main.py --steps collect,process,train,test
```

### Option 2: Run Individual Modules

```bash
# 1. Collect data
python run_collect.py --midi_dir data/midi --output_dir data/output

# 2. Process data
python run_process.py --input_file data/output/complete_dataset.json

# 3. Train model
python run_train.py --data_file data/processed/training_data.json

# 4. Test model
python run_test.py --model_path models/checkpoints/best_model.pt --comprehensive
```

## 📦 Module Usage

### 1. COLLECT Module

**Purpose**: Collect MIDI files and pair them with text descriptions.

```python
from source.collect import MIDICollector, TextCollector, DataPairing

# Collect MIDI metadata
collector = MIDICollector("data/midi")
metadata = collector.collect_all_metadata()

# Collect text descriptions
text_collector = TextCollector()
paired_data = text_collector.collect_text_for_all_midi(metadata)

# Create complete dataset
pairing = DataPairing("data/midi")
dataset = pairing.create_paired_dataset("data/output/complete_dataset.json")
```

**Command Line**:
```bash
python run_collect.py --midi_dir data/midi --filter_quality --min_text_length 20
```

### 2. PROCESS Module

**Purpose**: Process MIDI and text data for training.

```python
from source.process import MIDIProcessor, TextProcessor, DataPreparer

# Process MIDI files
midi_processor = MIDIProcessor(max_sequence_length=1024)
processed_midi = midi_processor.process_midi_file("song.mid")

# Process text descriptions
text_processor = TextProcessor(max_length=512)
processed_text = text_processor.process_text("A happy pop song")

# Prepare training data
preparer = DataPreparer()
training_data = preparer.prepare_training_data("paired_data.json")
```

**Command Line**:
```bash
python run_process.py --input_file data/output/complete_dataset.json --max_sequence_length 1024
```

### 3. TRAIN Module

**Purpose**: Train the Music Transformer model.

```python
from source.train import MusicTransformer, ModelTrainer, MusicGenerator

# Create model
model = MusicTransformer(
    vocab_size=1000,
    d_model=512,
    n_heads=8,
    n_layers=6
)

# Train model
trainer = ModelTrainer(model, train_loader, val_loader)
trainer.train()

# Generate music
generator = MusicGenerator("models/checkpoints/best_model.pt")
result = generator.generate_music(
    text_description="A happy pop song",
    output_file="generated_song.mid"
)
```

**Command Line**:
```bash
python run_train.py --data_file data/processed/training_data.json --max_epochs 100
```

### 4. TEST Module

**Purpose**: Evaluate model performance and generated music.

```python
from source.test import ModelEvaluator, EvaluationMetrics, ModelTester

# Evaluate generated files
evaluator = ModelEvaluator()
results = evaluator.evaluate_batch(["generated_1.mid", "generated_2.mid"])

# Calculate detailed metrics
metrics = EvaluationMetrics()
detailed_metrics = metrics.calculate_all_metrics(midi_data)

# Run comprehensive testing
tester = ModelTester("models/checkpoints/best_model.pt")
test_results = tester.run_comprehensive_test("test_results")
```

**Command Line**:
```bash
python run_test.py --model_path models/checkpoints/best_model.pt --comprehensive --benchmark
```

## 🔄 Pipeline

### Complete Pipeline Flow

```
1. COLLECT
   ├── Scan MIDI files
   ├── Extract metadata
   ├── Collect text descriptions
   └── Create paired dataset

2. PROCESS
   ├── Process MIDI files
   ├── Process text descriptions
   ├── Prepare training data
   └── Split into train/val/test

3. TRAIN
   ├── Create model
   ├── Train with cross-attention
   ├── Save checkpoints
   └── Generate sample music

4. TEST
   ├── Evaluate model performance
   ├── Calculate metrics
   ├── Run benchmarks
   └── Generate reports
```

### Custom Pipeline

```bash
# Run specific steps
python main.py --steps collect,process

# Run with custom parameters
python main.py --steps train --max_epochs 50 --learning_rate 5e-5

# Resume training
python main.py --steps train --resume_from models/checkpoints/checkpoint_epoch_50.pt
```

## 📚 API Reference

### Core Classes

#### MIDICollector
- `collect_midi_files()`: Collect all MIDI files
- `extract_metadata(midi_file)`: Extract metadata from MIDI file
- `collect_all_metadata()`: Collect metadata from all files

#### TextCollector
- `search_wikipedia(query)`: Search Wikipedia for descriptions
- `collect_text_for_midi(metadata)`: Collect text for MIDI file
- `collect_text_for_all_midi(metadata_list)`: Collect text for all files

#### MIDIProcessor
- `process_midi_file(midi_file)`: Process single MIDI file
- `extract_events(midi_data)`: Extract events from MIDI
- `events_to_tokens(events)`: Convert events to tokens

#### TextProcessor
- `process_text(text)`: Process text description
- `extract_musical_features(text)`: Extract musical features
- `get_bert_embedding(text)`: Get BERT embeddings

#### MusicTransformer
- `forward(midi_tokens, text_embeddings)`: Forward pass
- `generate(text_embeddings, **kwargs)`: Generate music
- `get_model_info()`: Get model information

#### ModelTrainer
- `train_epoch()`: Train for one epoch
- `validate_epoch()`: Validate for one epoch
- `train()`: Complete training
- `save_checkpoint(filename)`: Save checkpoint

#### ModelEvaluator
- `evaluate_single_file(midi_file)`: Evaluate single file
- `evaluate_batch(files)`: Evaluate batch of files
- `evaluate_generated_vs_reference(gen_file, ref_file)`: Compare files

## 💡 Examples

### Generate Music from Text

```python
from source.train import MusicGenerator

# Load trained model
generator = MusicGenerator("models/checkpoints/best_model.pt")

# Generate music
text_descriptions = [
    "A happy and energetic pop song with piano and drums",
    "A sad and melancholic jazz piece with saxophone",
    "An intense rock song with electric guitar and drums"
]

for i, text in enumerate(text_descriptions):
    result = generator.generate_music(
        text_description=text,
        output_file=f"generated_{i+1}.mid",
        style_id=i
    )
    print(f"Generated: {result['output_file']}")
```

### Evaluate Generated Music

```python
from source.test import ModelEvaluator, EvaluationMetrics

# Initialize evaluators
evaluator = ModelEvaluator()
metrics = EvaluationMetrics()

# Evaluate generated files
generated_files = ["generated_1.mid", "generated_2.mid", "generated_3.mid"]
results = evaluator.evaluate_batch(generated_files)

# Calculate detailed metrics
for file in generated_files:
    import pretty_midi
    midi_data = pretty_midi.PrettyMIDI(file)
    detailed_metrics = metrics.calculate_all_metrics(midi_data)
    print(f"Metrics for {file}: {detailed_metrics}")
```

### Custom Training Configuration

```python
from source.train import MusicTransformer, ModelTrainer

# Create custom model
model = MusicTransformer(
    vocab_size=2000,
    d_model=768,
    n_heads=12,
    n_layers=8,
    d_ff=3072,
    use_cross_attention=True
)

# Custom training
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=5e-5,
    weight_decay=1e-4,
    max_epochs=200
)

# Start training
trainer.train()
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install transformers spacy pretty_midi
   python -m spacy download en_core_web_sm
   ```

2. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python run_train.py --batch_size 16
   
   # Use CPU
   python run_train.py --device cpu
   ```

3. **MIDI Processing Errors**
   ```bash
   # Check MIDI file format
   # Ensure files are valid MIDI format
   ```

4. **Text Processing Errors**
   ```bash
   # Install spaCy model
   python -m spacy download en_core_web_sm
   
   # Check internet connection for BERT downloads
   ```

### Performance Tips

1. **GPU Acceleration**
   ```bash
   # Use CUDA if available
   python run_train.py --device cuda
   ```

2. **Memory Optimization**
   ```bash
   # Reduce sequence length
   python run_process.py --max_sequence_length 512
   
   # Reduce batch size
   python run_train.py --batch_size 16
   ```

3. **Faster Training**
   ```bash
   # Use smaller model
   python run_train.py --d_model 256 --n_layers 4
   ```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the examples

---

**Happy Music Generation! 🎵** 
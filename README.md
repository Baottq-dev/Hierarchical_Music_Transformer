# Mamba Music - Hybrid Mamba-Transformer for Symbolic Music Generation

A PyTorch library for symbolic music generation using OctupleMIDI tokenization and hybrid Mamba-Transformer architecture.

## 🎯 Features

- **OctupleMIDI Tokenization**: 8-dimensional token representation for rich musical information
- **Hybrid Architecture**: Combines Mamba's efficiency with Transformer's expressiveness
- **Hierarchical Attention**: Musical structure-aware attention mechanism
- **Multi-head Output**: Separate prediction heads for each MIDI component
- **FMD Evaluation**: Fréchet Music Distance for quality assessment

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for Mamba support)

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/Baottq-dev/Hierarchical_Music_Transformer.git
cd Hierarchical_Music_Transformer

# Install package
pip install -e .

# For development
pip install -e ".[dev]"

# For Mamba support (requires CUDA)
pip install -e ".[mamba]"
```

## 📁 Project Structure

```
mamba_music/
├── configs/                # Hydra configuration files
│   ├── model/              # Model configs (transformer, hybrid)
│   ├── data/               # Data configs (octuple)
│   └── train/              # Training configs
├── docker/                 # Docker support
├── experiments/            # Experiment outputs
├── scripts/                # Entry point scripts
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── generate.py         # Generation script
├── src/mamba_music/        # Main package
│   ├── data/               # Tokenizer, dataset
│   ├── models/             # Model architectures
│   │   └── components/     # Embeddings, attention, output
│   ├── training/           # Trainer, losses
│   ├── evaluation/         # Metrics (FMD, OA)
│   └── generation/         # Generator
└── Makefile                # Automation commands
```

## 🎹 Quick Start

### Training

```bash
# Default training
make train

# Train with specific config
python scripts/train.py model=hybrid training.epochs=50
```

### Generation

```bash
python scripts/generate.py --checkpoint experiments/checkpoints/best_model.pt
```

### Docker

```bash
# Build image
make docker-build

# Run training in Docker
make docker-train
```

## 🔬 Architecture

### OctupleMIDI Token Format

Each note is represented as an 8-tuple:
```
(TimeSig, Tempo, Bar, Position, Instrument, Pitch, Duration, Velocity)
```

### Model Components

- **OctupleEmbeddingLayer**: Separate embeddings for each token component
- **HierarchicalAttention**: Structure-aware multi-head attention
- **HybridBlock**: Mamba layers + Attention layer (Phase 2)
- **MultiHeadOutput**: 8 prediction heads for each component

## 📊 Evaluation Metrics

- **FMD (Fréchet Music Distance)**: Distribution-based quality metric
- **Overlapped Area**: Distribution overlap for pitch/duration/velocity

## 🛠️ Development

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test
```

## 📝 License

MIT License

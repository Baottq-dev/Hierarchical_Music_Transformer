# Running Guide for AMT

This guide explains how to run the AMT (Automated Music Transcription) system.

## Pipeline Overview

The AMT system follows a pipeline approach:

1. **Collect**: Gather MIDI files and text descriptions
2. **Process**: Convert MIDI and text into model-compatible formats
3. **Train**: Train the transformer models
4. **Generate**: Generate MIDI from text or text from MIDI
5. **Evaluate**: Evaluate model performance

## 1. Data Collection

The collection step gathers MIDI files and pairs them with text descriptions.

```bash
python collect.py --midi-dir path/to/midi/files --output-file data/collected_data.json
```

Options:
- `--midi-dir`: Directory containing MIDI files
- `--text-dir`: Directory containing text descriptions (optional)
- `--output-file`: Path to save collected data
- `--batch-size`: Batch size for processing
- `--num-workers`: Number of worker processes

## 2. Data Processing

The processing step converts MIDI files and text descriptions into formats suitable for model training.

### Single file processing:

```bash
python process.py single --midi-file path/to/midi/file.mid --text-file path/to/description.txt --output-dir data/processed
```

### Batch processing:

```bash
python process.py paired --paired-data-file data/collected_data.json --output-dir data/processed
```

Options:
- `--use-hierarchical-encoding`: Enable hierarchical token encoding
- `--use-relative-attention`: Enable relative position attention
- `--use-contextual-embeddings`: Enable contextual embeddings
- `--batch-size`: Batch size for processing
- `--num-workers`: Number of worker processes

### Transfer Learning Options for Processing:

You can now use pre-trained language models and fine-tune them for music descriptions:

```bash
# Use a pre-trained text model
python process.py paired --paired-data-file data/collected_data.json --output-dir data/processed --use-pretrained-text-model --pretrained-text-model-path models/pretrained_bert

# Fine-tune the text model on music descriptions
python process.py paired --paired-data-file data/collected_data.json --output-dir data/processed --use-pretrained-text-model --pretrained-text-model-path models/pretrained_bert --enable-text-fine-tuning
```

## 3. Training

The training step trains the transformer model on processed data.

```bash
python train.py --paired-data-file data/processed/paired_data.json --output-dir models
```

Options:
- `--d-model`: Model dimension
- `--num-heads`: Number of attention heads
- `--num-layers`: Number of transformer layers
- `--max-seq-len`: Maximum sequence length
- `--batch-size`: Batch size for training
- `--learning-rate`: Learning rate
- `--num-epochs`: Number of training epochs
- `--device`: Device to train on (auto, cuda, cpu)

### Transfer Learning Options for Training:

You can now leverage pre-trained models to improve training:

```bash
# Train with a pre-trained music model using feature extraction (frozen layers)
python train.py --paired-data-file data/processed/paired_data.json --output-dir models --pretrained-model models/checkpoints/pretrained_model.pt --transfer-learning-mode feature_extraction

# Train with a pre-trained music model using fine-tuning (partially frozen)
python train.py --paired-data-file data/processed/paired_data.json --output-dir models --pretrained-model models/checkpoints/pretrained_model.pt --transfer-learning-mode fine_tuning --freeze-layers 3

# Train with a pre-trained music model using full fine-tuning (all layers trainable)
python train.py --paired-data-file data/processed/paired_data.json --output-dir models --pretrained-model models/checkpoints/pretrained_model.pt --transfer-learning-mode full_fine_tuning
```

Transfer learning modes:
- `feature_extraction`: Freezes all layers except output layer
- `fine_tuning`: Freezes a specified number of layers
- `full_fine_tuning`: All layers are trainable

## 4. Generation

The generation step creates MIDI from text descriptions or text from MIDI files.

```bash
# Generate MIDI from text
python generate.py text-to-midi --model-path models/checkpoints/model.pt --text "A cheerful piano melody with jazz influences" --output-file output.mid

# Generate text from MIDI
python generate.py midi-to-text --model-path models/checkpoints/model.pt --midi-file input.mid
```

Options:
- `--model-path`: Path to trained model
- `--temperature`: Sampling temperature (higher = more random)
- `--top-k`: Top-k sampling parameter
- `--top-p`: Top-p (nucleus) sampling parameter

## 5. Evaluation

The evaluation step assesses model performance.

```bash
python test.py --model-path models/checkpoints/model.pt --test-data data/evaluation/test_data.json
```

Options:
- `--model-path`: Path to trained model
- `--test-data`: Path to test data
- `--output-dir`: Directory to save evaluation results

## Advanced Usage

### Continuing from Checkpoints

If processing is interrupted, you can continue from where you left off:

```bash
python process.py paired --paired-data-file data/collected_data.json --output-dir data/processed --continue-from-checkpoint
```

### Using Pre-trained Models

You can use pre-trained models for better performance:

```bash
# Using a pre-trained model for generation
python generate.py text-to-midi --model-path pretrained_models/music_transformer.pt --text "A gentle piano ballad"
```

### Transfer Learning Benefits

Using transfer learning can provide several benefits:

1. **Faster convergence**: Models train more quickly when starting from pre-trained weights
2. **Better performance**: Pre-trained models often achieve higher quality results
3. **Less data required**: Transfer learning works well with smaller datasets
4. **Domain adaptation**: Adapt general models to music-specific tasks

## Troubleshooting

- **Out of memory errors**: Reduce batch size or sequence length
- **Slow processing**: Increase number of workers or use GPU
- **Poor generation quality**: Try different sampling parameters or fine-tune the model more 
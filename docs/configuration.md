# AMT Configuration System

AMT uses a centralized configuration system based on Pydantic's BaseSettings. This allows for:

- Default configuration values
- Environment variable overrides
- Type validation
- Path handling

## Basic Usage

To use the configuration in your code:

```python
from amt.config import get_settings

# Get the settings
settings = get_settings()

# Use settings
print(f"Using MIDI directory: {settings.midi_dir}")
print(f"Batch size: {settings.batch_size}")
```

## Available Settings

### Directories

- `base_dir`: Base directory for the project (default: ".")
- `data_dir`: Directory for all data (default: "data")
- `midi_dir`: Directory for MIDI files (default: "data/midi")
- `text_dir`: Directory for text files (default: "data/text")
- `processed_dir`: Directory for processed data (default: "data/processed")
- `output_dir`: Directory for output files (default: "data/output")
- `model_dir`: Directory for model files (default: "models")
- `checkpoint_dir`: Directory for model checkpoints (default: "models/checkpoints")
- `evaluation_dir`: Directory for evaluation results (default: "models/evaluation")

### Processing Parameters

- `max_sequence_length`: Maximum sequence length for MIDI tokens (default: 1024)
- `max_text_length`: Maximum sequence length for text tokens (default: 512)

### Training Parameters

- `batch_size`: Batch size for training (default: 32)
- `learning_rate`: Learning rate for training (default: 1e-4)
- `weight_decay`: Weight decay for training (default: 1e-5)
- `max_epochs`: Maximum number of epochs for training (default: 100)

### Model Parameters

- `vocab_size`: Vocabulary size for the model (default: 1000)
- `d_model`: Model dimension (default: 512)
- `nhead`: Number of attention heads (default: 8)
- `num_encoder_layers`: Number of encoder layers (default: 6)
- `num_decoder_layers`: Number of decoder layers (default: 6)
- `dim_feedforward`: Dimension of feedforward network (default: 2048)
- `dropout`: Dropout rate (default: 0.1)

### Generation Parameters

- `temperature`: Temperature for sampling (default: 1.0)
- `top_k`: Top-k sampling parameter (default: 50)
- `top_p`: Top-p sampling parameter (default: 0.95)
- `repetition_penalty`: Repetition penalty for sampling (default: 1.2)
- `max_generate_length`: Maximum length for generation (default: 1024)

### Logging

- `log_level`: Logging level (default: "info")
- `log_file`: Log file path (default: None)

## Environment Variables

All settings can be overridden using environment variables with the `AMT_` prefix. For example:

```bash
# Set the MIDI directory
export AMT_MIDI_DIR=/path/to/midi/files

# Set the batch size
export AMT_BATCH_SIZE=64

# Set the log level
export AMT_LOG_LEVEL=debug
```

## .env File

You can also create a `.env` file in the project root to set environment variables:

```
AMT_MIDI_DIR=/path/to/midi/files
AMT_BATCH_SIZE=64
AMT_LOG_LEVEL=debug
```

## Path Handling

All paths are automatically converted to absolute paths using the `base_dir` setting. For example, if `base_dir` is `/home/user/amt` and `midi_dir` is `data/midi`, the actual path used will be `/home/user/amt/data/midi`. 
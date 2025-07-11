#!/usr/bin/env python3
"""
Command Line Interface for AMT (Automated Music Transcription)
"""

import click
from pathlib import Path

from amt.utils.logging import get_logger
from amt.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@click.group()
@click.version_option()
@click.option("--log-level", default=settings.log_level, 
              type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
              help="Set the logging level")
def cli(log_level):
    """Automated Music Transcription (AMT) CLI tool."""
    # Set log level
    logger.setLevel(log_level.upper())


@cli.command()
@click.option("--midi_dir", default=str(settings.midi_dir), help="Directory containing MIDI files")
@click.option("--output_dir", default=str(settings.output_dir), help="Output directory")
@click.option("--filter_quality", is_flag=True, help="Filter data by quality")
@click.option("--min_text_length", type=int, default=20, help="Minimum text length")
@click.option("--min_duration", type=float, default=10.0, help="Minimum MIDI duration")
def collect(midi_dir, output_dir, filter_quality, min_text_length, min_duration):
    """Collect MIDI files and text descriptions."""
    from amt.collect.collector import collect_data
    
    logger.info("🎵 Starting Data Collection...")
    collect_data(
        midi_dir=midi_dir,
        output_dir=output_dir,
        filter_quality=filter_quality,
        min_text_length=min_text_length,
        min_duration=min_duration
    )
    logger.info("🎉 Data collection completed!")


@cli.command()
@click.option("--input_dir", default=str(settings.midi_dir), help="Directory containing MIDI files")
@click.option("--output_dir", default=str(settings.processed_dir), help="Output directory for processed data")
@click.option("--max_files", type=int, default=None, help="Maximum number of files to process")
@click.option("--batch_size", type=int, default=settings.batch_size, help="Batch size for processing")
@click.option("--num_workers", type=int, default=4, help="Number of workers for parallel processing")
def process(input_dir, output_dir, max_files, batch_size, num_workers):
    """Process MIDI files into training data."""
    from amt.process.data_preparer import process_data
    
    logger.info("🎹 Starting MIDI Processing...")
    process_data(
        input_dir=input_dir,
        output_dir=output_dir,
        max_files=max_files,
        batch_size=batch_size,
        num_workers=num_workers
    )
    logger.info("✅ Processing completed!")


@cli.command()
@click.option("--data_dir", default=str(settings.processed_dir), help="Directory containing processed data")
@click.option("--model_dir", default=str(settings.model_dir), help="Directory to save model checkpoints")
@click.option("--batch_size", type=int, default=settings.batch_size, help="Training batch size")
@click.option("--epochs", type=int, default=settings.max_epochs, help="Number of training epochs")
@click.option("--lr", type=float, default=settings.learning_rate, help="Learning rate")
@click.option("--resume", is_flag=True, help="Resume training from latest checkpoint")
def train(data_dir, model_dir, batch_size, epochs, lr, resume):
    """Train the AMT model."""
    from amt.train.trainer import train_model
    
    logger.info("🧠 Starting Model Training...")
    train_model(
        data_dir=data_dir,
        model_dir=model_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=lr,
        resume=resume
    )
    logger.info("🎉 Training completed!")


@cli.command()
@click.option("--checkpoint", required=True, help="Path to model checkpoint")
@click.option("--output_dir", default=str(settings.output_dir), help="Directory to save generated outputs")
@click.option("--num_samples", type=int, default=5, help="Number of samples to generate")
@click.option("--temperature", type=float, default=settings.temperature, help="Sampling temperature")
@click.option("--top_k", type=int, default=settings.top_k, help="Top-k sampling parameter")
@click.option("--top_p", type=float, default=settings.top_p, help="Top-p sampling parameter")
@click.option("--seed_text", default=None, help="Seed text for conditional generation")
def generate(checkpoint, output_dir, num_samples, temperature, top_k, top_p, seed_text):
    """Generate music using the trained model."""
    from amt.generate.generator import generate_music
    
    logger.info("🎵 Starting Music Generation...")
    generate_music(
        checkpoint=checkpoint,
        output_dir=output_dir,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed_text=seed_text
    )
    logger.info("🎉 Generation completed!")


@cli.command()
@click.option("--checkpoint", required=True, help="Path to model checkpoint")
@click.option("--test_dir", default=str(settings.processed_dir), help="Directory containing test data")
@click.option("--output_dir", default=str(settings.evaluation_dir), help="Directory to save test results")
@click.option("--batch_size", type=int, default=settings.batch_size, help="Test batch size")
def test(checkpoint, test_dir, output_dir, batch_size):
    """Test and evaluate the trained model."""
    from amt.evaluate.evaluator import evaluate_model
    
    logger.info("🔍 Starting Model Evaluation...")
    evaluate_model(
        checkpoint=checkpoint,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=batch_size
    )
    logger.info("✅ Evaluation completed!")


if __name__ == "__main__":
    cli() 
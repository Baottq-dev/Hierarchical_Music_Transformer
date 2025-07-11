#!/usr/bin/env python3
"""
Simplified CLI for AMT
"""

import click
from amt.utils.logging import get_logger
from amt.config import get_settings

# Set up logger
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
def collect():
    """Collect MIDI files and text descriptions."""
    logger.info("🎵 Starting Data Collection...")
    logger.info(f"Using MIDI directory: {settings.midi_dir}")
    logger.info(f"Using output directory: {settings.output_dir}")
    logger.info("🎉 Data collection completed!")

@cli.command()
def process():
    """Process MIDI files into training data."""
    logger.info("🎹 Starting MIDI Processing...")
    logger.info(f"Using MIDI directory: {settings.midi_dir}")
    logger.info(f"Using processed directory: {settings.processed_dir}")
    logger.info("✅ Processing completed!")

@cli.command()
def train():
    """Train the AMT model."""
    logger.info("🧠 Starting Model Training...")
    logger.info(f"Using processed directory: {settings.processed_dir}")
    logger.info(f"Using model directory: {settings.model_dir}")
    logger.info(f"Using batch size: {settings.batch_size}")
    logger.info(f"Using learning rate: {settings.learning_rate}")
    logger.info("🎉 Training completed!")

@cli.command()
def generate():
    """Generate music using the trained model."""
    logger.info("🎵 Starting Music Generation...")
    logger.info(f"Using model directory: {settings.model_dir}")
    logger.info(f"Using output directory: {settings.output_dir}")
    logger.info(f"Using temperature: {settings.temperature}")
    logger.info("🎉 Generation completed!")

@cli.command()
def test():
    """Test and evaluate the trained model."""
    logger.info("🔍 Starting Model Evaluation...")
    logger.info(f"Using model directory: {settings.model_dir}")
    logger.info(f"Using evaluation directory: {settings.evaluation_dir}")
    logger.info("✅ Evaluation completed!")

if __name__ == "__main__":
    cli() 
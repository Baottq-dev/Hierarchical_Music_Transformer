#!/usr/bin/env python
"""AMT Command-Line Interface.

Usage examples:
  python run.py pipeline            # execute end-to-end data pipeline (metadata → training data)
  python run.py train               # train model using default paths
  python run.py generate -t "Calm piano" -o output/generated/calm.mid -c models/checkpoints/checkpoint_epoch_10.pt
  python run.py evaluate -r data/reference/ref.mid -g output/generated/calm.mid

This wrapper simply gọi vào các module bên trong AMT/source để bạn không cần nhớ đường dẫn dài.
"""

import argparse
import sys
from pathlib import Path

# Import modules from package
from AMT.source.scripts.main import main as pipeline_main
from AMT.source.model.training import train_model
from AMT.source.model.generation import AMTGenerator
from AMT.source.evaluation.metrics import evaluate_generated_music

DEFAULT_TRAIN_DATA = "data/output/amt_training_data.json"
DEFAULT_CHECKPOINT_DIR = "models/checkpoints"


def cmd_pipeline(_args: argparse.Namespace):
    """Run the data-preparation pipeline."""
    pipeline_main()


def cmd_train(args: argparse.Namespace):
    """Train AMT model."""
    output_dir = args.output_dir or DEFAULT_CHECKPOINT_DIR
    train_model(
        data_file=DEFAULT_TRAIN_DATA,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )


def cmd_generate(args: argparse.Namespace):
    """Generate music from text prompt."""
    if not Path(args.checkpoint).exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    generator = AMTGenerator(args.checkpoint)
    generator.generate_music(
        text_description=args.text,
        output_file=args.output,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(f"✅ Generated MIDI saved to {args.output}")


def cmd_evaluate(args: argparse.Namespace):
    """Evaluate generated MIDI against reference."""
    scores = evaluate_generated_music(args.reference, args.generated)
    print("Evaluation metrics:")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="AMT CLI", description="Convenient entry-point for AMT project")
    sub = parser.add_subparsers(dest="command", required=True)

    # pipeline
    sub.add_parser("pipeline", help="Run data-preparation pipeline")

    # train
    p_train = sub.add_parser("train", help="Train AMT model")
    p_train.add_argument("--output-dir", default=DEFAULT_CHECKPOINT_DIR, help="Directory to save checkpoints")
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=1e-4)

    # generate
    p_gen = sub.add_parser("generate", help="Generate music from prompt")
    p_gen.add_argument("-t", "--text", required=True, help="Text prompt")
    p_gen.add_argument("-o", "--output", required=True, help="Output MIDI path")
    p_gen.add_argument("-c", "--checkpoint", required=True, help="Path to model checkpoint")
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument("--top-k", type=int, default=50)
    p_gen.add_argument("--top-p", type=float, default=0.9)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate generated MIDI")
    p_eval.add_argument("-r", "--reference", required=True, help="Reference MIDI file")
    p_eval.add_argument("-g", "--generated", required=True, help="Generated MIDI file")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main() 
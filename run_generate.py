#!/usr/bin/env python3
"""
CLI: Generate MIDI from text prompt using a trained checkpoint.
Usage example:
    python run_generate.py \
        --checkpoint models/checkpoints/best_model.pt \
        --prompt "A slow, emotional piano ballad in C minor" \
        --out_file outputs/piano_ballad.mid \
        --max_length 512 --temperature 1.0 --top_k 50 --top_p 0.9
"""

import argparse
import os

from source.train.generator import MusicGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MIDI from text prompt")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to trained model checkpoint (.pt)"
    )
    parser.add_argument("--prompt", required=True, help="Text prompt / description")
    parser.add_argument("--out_file", required=True, help="Destination .mid path")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--device", default="auto", help="Device cpu / cuda / auto")

    args = parser.parse_args()

    # Create generator and load model
    generator = MusicGenerator(
        model_path=args.checkpoint,
        device=args.device,
        max_length=args.max_length,
        temperature=args.temperature,
    )

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out_file)
    if out_dir == "":
        out_dir = "."  # current directory
    os.makedirs(out_dir, exist_ok=True)

    # Generate
    generator.generate_music(
        text_description=args.prompt,
        output_file=args.out_file,
        top_k=args.top_k,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()

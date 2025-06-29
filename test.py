#!/usr/bin/env python3
"""
Test Script for AMT
Generates music and evaluates quality
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add source to path
sys.path.append('source')

from source.model.generation import MusicGenerator
from source.evaluation.metrics import evaluate_generated_music

def generate_music(model_path, text_description, output_file, max_length=512, temperature=1.0):
    """Generate music from text description"""
    print(f"🎼 Generating music...")
    print(f"📝 Description: {text_description}")
    print(f"🎯 Output: {output_file}")
    
    try:
        # Initialize generator
        generator = MusicGenerator(model_path)
        
        # Generate music
        success = generator.generate_music(
            text_description=text_description,
            output_file=output_file,
            max_length=max_length,
            temperature=temperature
        )
        
        if success:
            print("✅ Music generated successfully!")
            return True
        else:
            print("❌ Failed to generate music!")
            return False
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def evaluate_music(original_file, generated_file):
    """Evaluate generated music quality"""
    print(f"📊 Evaluating music quality...")
    print(f"🎵 Original: {original_file}")
    print(f"🎵 Generated: {generated_file}")
    
    try:
        # Evaluate music
        metrics = evaluate_generated_music(original_file, generated_file)
        
        print("\n📈 Evaluation Results:")
        print(f"   Note Density Ratio: {metrics['note_density_ratio']:.3f}")
        print(f"   Velocity Similarity: {metrics['velocity_similarity']:.3f}")
        print(f"   Note Range Similarity: {metrics['note_range_similarity']:.3f}")
        print(f"   Time Signature Match: {metrics['time_signature_match']:.3f}")
        print(f"   Tempo Similarity: {metrics['tempo_similarity']:.3f}")
        print(f"   Overall Score: {metrics['overall_score']:.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test AMT model")
    parser.add_argument("--model_path", default="./models/checkpoints/checkpoint_epoch_10.pt",
                       help="Model checkpoint path")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--text_description", default="A happy pop song with piano and drums",
                       help="Text description for generation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--original_file", help="Original MIDI file for evaluation")
    parser.add_argument("--skip_generation", action="store_true", help="Skip music generation")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip music evaluation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎵 AMT Testing")
    print("=" * 50)
    print(f"Model Path: {args.model_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Text Description: {args.text_description}")
    print(f"Max Length: {args.max_length}")
    print(f"Temperature: {args.temperature}")
    
    # Check model file
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file {args.model_path} does not exist!")
        print("Please run train.py first")
        return
    
    generated_file = os.path.join(args.output_dir, "generated_music.mid")
    
    # Step 1: Generate music
    if not args.skip_generation:
        print("\n🎼 Step 1: Generating music...")
        if not generate_music(
            model_path=args.model_path,
            text_description=args.text_description,
            output_file=generated_file,
            max_length=args.max_length,
            temperature=args.temperature
        ):
            print("❌ Music generation failed!")
            return
    else:
        print("\n⏭️ Skipping music generation")
        if not os.path.exists(generated_file):
            print(f"❌ Error: Generated file {generated_file} does not exist!")
            return
    
    # Step 2: Evaluate music (if original file provided)
    if args.original_file and not args.skip_evaluation:
        print("\n📊 Step 2: Evaluating music...")
        if not os.path.exists(args.original_file):
            print(f"❌ Error: Original file {args.original_file} does not exist!")
            return
        
        metrics = evaluate_music(args.original_file, generated_file)
        if metrics:
            # Save evaluation results
            eval_file = os.path.join(args.output_dir, "evaluation_results.json")
            with open(eval_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"📄 Evaluation results saved to: {eval_file}")
    else:
        print("\n⏭️ Skipping music evaluation")
    
    print("\n🎉 Testing completed!")
    print(f"🎵 Generated music: {generated_file}")
    if args.original_file and not args.skip_evaluation:
        print(f"📊 Evaluation results: {os.path.join(args.output_dir, 'evaluation_results.json')}")

if __name__ == "__main__":
    main() 
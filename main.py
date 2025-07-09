import argparse
import json
import os
import sys
from typing import List


def print_banner():
    """Print AMT pipeline banner."""
    print("=" * 60)
    print("🎵 AMT (Audio Music Transformer) Pipeline")
    print("   Symbolic Music Generation with Text Controls")
    print("   🚀 Clean Modular Structure: collect, process, train, test")
    print("=" * 60)


def run_collect_step(args):
    """Run data collection step."""
    print(f"\n{'='*20} STEP: COLLECT {'='*20}")
    print("📊 Collect MIDI metadata and text descriptions")
    print("-" * 60)

    try:
        from source.collect import DataPairing

        # Create directories
        os.makedirs("data/midi", exist_ok=True)
        os.makedirs("data/output", exist_ok=True)

        # Create paired dataset
        pairing = DataPairing(args.midi_dir)
        dataset = pairing.create_paired_dataset(args.output_file)

        # Filter if requested
        if args.filter_quality:
            dataset = pairing.filter_paired_data(
                dataset, min_text_length=args.min_text_length, min_duration=args.min_duration
            )

        # Validate
        stats = pairing.validate_paired_data(dataset)
        print("Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("✅ Data collection completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        return False


def run_process_step(args):
    """Run data processing step."""
    print(f"\n{'='*20} STEP: PROCESS {'='*20}")
    print("🚀 Process MIDI and text data for training")
    print("-" * 60)

    try:
        from source.process import DataPreparer, MIDIProcessor, TextProcessor

        # Load paired data
        with open(args.input_file) as f:
            paired_data = json.load(f)

        # Process MIDI data
        midi_processor = MIDIProcessor(max_sequence_length=args.max_sequence_length)
        processed_midi = []

        for item in paired_data:
            midi_file = item.get("midi_file")
            if midi_file and os.path.exists(midi_file):
                processed = midi_processor.process_midi_file(midi_file)
                if processed:
                    processed_midi.append(processed)

        # Process text data
        text_processor = TextProcessor(max_length=args.max_text_length)
        processed_texts = []

        for item in paired_data:
            text = item.get("text_description", "")
            if text:
                processed = text_processor.process_text(text)
                processed_texts.append(processed)

        # Prepare training data
        data_preparer = DataPreparer(
            max_sequence_length=args.max_sequence_length,
            max_text_length=args.max_text_length,
            batch_size=args.batch_size,
        )

        # Combine processed data
        processed_data = []
        for i, midi_item in enumerate(processed_midi):
            if i < len(processed_texts):
                combined_item = {
                    "midi_file": midi_item["file_path"],
                    "text_description": paired_data[i].get("text_description", ""),
                    "midi_tokens": midi_item["tokens"],
                    "midi_metadata": midi_item["metadata"],
                    "text_features": processed_texts[i],
                    "sequence_length": midi_item["sequence_length"],
                }
                processed_data.append(combined_item)

        # Split data
        train_data, val_data, test_data = data_preparer.split_data(
            processed_data, train_ratio=args.train_ratio, val_ratio=args.val_ratio
        )

        # Save training data
        training_data = {
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
            "vocab_size": midi_processor.vocab_size,
            "max_sequence_length": args.max_sequence_length,
            "max_text_length": args.max_text_length,
            "total_samples": len(processed_data),
        }

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "training_data.json"), "w") as f:
            json.dump(training_data, f, indent=2)

        print(f"✅ Data processing completed! Processed {len(processed_data)} items")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Validation: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")
        print(f"  Vocabulary size: {midi_processor.vocab_size}")
        return True
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return False


def run_train_step(args):
    """Run model training step."""
    print(f"\n{'='*20} STEP: TRAIN {'='*20}")
    print("🚀 Train Music Transformer model")
    print("-" * 60)

    try:
        from source.train import ModelTrainer, MusicTransformer

        # Load training data
        with open(args.data_file) as f:
            training_data = json.load(f)

        # Create model
        model = MusicTransformer(
            vocab_size=training_data["vocab_size"],
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            max_text_len=args.max_text_len,
            use_cross_attention=True,
        )

        # Create data loaders
        from source.process import DataPreparer

        data_preparer = DataPreparer(
            max_sequence_length=args.max_seq_len,
            max_text_length=args.max_text_len,
            batch_size=args.batch_size,
        )

        train_dataset = data_preparer.create_dataset(training_data["train_data"])
        val_dataset = data_preparer.create_dataset(training_data["val_data"])

        train_loader = data_preparer.create_dataloader(train_dataset, shuffle=True)
        val_loader = data_preparer.create_dataloader(val_dataset, shuffle=False)

        # Create trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            save_dir=args.output_dir,
            device=args.device,
        )

        # Start training
        if args.resume_from:
            trainer.train(resume_from=args.resume_from)
        else:
            trainer.train()

        print("✅ Model training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return False


def run_test_step(args):
    """Run testing and evaluation step."""
    print(f"\n{'='*20} STEP: TEST {'='*20}")
    print("🧪 Test model performance and evaluate generated music")
    print("-" * 60)

    try:
        from source.test import ModelEvaluator, ModelTester

        # Initialize tester
        tester = ModelTester(args.model_path)

        # Test model loading
        load_result = tester.test_model_loading()
        if not load_result["success"]:
            print(f"❌ Model loading failed: {load_result['error']}")
            return False

        # Run comprehensive testing
        if args.comprehensive:
            comprehensive_results = tester.run_comprehensive_test(args.output_dir)
            if not comprehensive_results["pipeline_integration"]["success"]:
                print("❌ Comprehensive testing failed")
                return False

        # Run performance benchmark
        if args.benchmark:
            benchmark_results = tester.benchmark_performance(num_samples=args.num_samples)
            if benchmark_results["success"]:
                print("✅ Performance benchmark completed")
                print(
                    f"  Average generation time: {benchmark_results.get('avg_generation_time', 0):.2f}s"
                )
            else:
                print("❌ Performance benchmark failed")

        # Evaluate generated files if provided
        if args.generated_files:
            evaluator = ModelEvaluator()
            evaluation_results = evaluator.evaluate_batch(
                generated_files=args.generated_files, reference_files=args.reference_files
            )

            # Save evaluation report
            evaluator.generate_evaluation_report(
                evaluation_results, os.path.join(args.output_dir, "evaluation_report.json")
            )
            evaluator.plot_metrics(evaluation_results, args.output_dir)

        print("✅ Testing completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        return False


def run_pipeline_steps(steps: List[str], args) -> bool:
    """Run specified pipeline steps."""
    step_functions = {
        "collect": run_collect_step,
        "process": run_process_step,
        "train": run_train_step,
        "test": run_test_step,
    }

    for step in steps:
        if step in step_functions:
            success = step_functions[step](args)
            if not success:
                print(f"❌ Pipeline failed at step: {step}")
                return False
        else:
            print(f"❌ Unknown step: {step}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="AMT Pipeline - Music Generation with Text Controls"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["collect", "process", "train", "test"],
        default=["collect", "process", "train", "test"],
        help="Pipeline steps to run",
    )

    # Collect step arguments
    parser.add_argument("--midi_dir", default="data/midi", help="MIDI files directory")
    parser.add_argument(
        "--output_file",
        default="data/output/complete_dataset.json",
        help="Output file for collected data",
    )
    parser.add_argument("--filter_quality", action="store_true", help="Filter data by quality")
    parser.add_argument("--min_text_length", type=int, default=20, help="Minimum text length")
    parser.add_argument("--min_duration", type=float, default=10.0, help="Minimum MIDI duration")

    # Process step arguments
    parser.add_argument(
        "--input_file", default="data/output/complete_dataset.json", help="Input paired data file"
    )
    parser.add_argument("--max_sequence_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max_text_length", type=int, default=512, help="Max text length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data ratio")

    # Train step arguments
    parser.add_argument(
        "--data_file", default="data/processed/training_data.json", help="Training data file"
    )
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward dimension")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max_text_len", type=int, default=512, help="Max text length")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--resume_from", help="Resume training from checkpoint")

    # Test step arguments
    parser.add_argument("--model_path", help="Path to trained model")
    parser.add_argument("--generated_files", nargs="+", help="Generated MIDI files to evaluate")
    parser.add_argument("--reference_files", nargs="+", help="Reference MIDI files for comparison")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive testing")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples for benchmark"
    )

    # Output directories
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument("--model_dir", default="models/checkpoints", help="Model output directory")
    parser.add_argument("--test_dir", default="test_results", help="Test output directory")

    args = parser.parse_args()

    print_banner()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    # Run pipeline steps
    success = run_pipeline_steps(args.steps, args)

    if success:
        print("\n🎉 Pipeline completed successfully!")
        print("📁 Output directories:")
        print(f"  - Data: {args.output_dir}")
        print(f"  - Models: {args.model_dir}")
        print(f"  - Tests: {args.test_dir}")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

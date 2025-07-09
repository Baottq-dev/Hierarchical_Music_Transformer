"""
Train Module Runner - Trains the Music Transformer model
"""

import argparse
import os

from source.train import ModelTrainer, MusicTransformer


def main():
    parser = argparse.ArgumentParser(description="Train Music Transformer model")
    parser.add_argument("--data_file", required=True, help="Training data file")
    parser.add_argument("--output_dir", default="models/checkpoints", help="Output directory")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward dimension")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max_text_len", type=int, default=512, help="Maximum text length")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--resume_from", help="Resume training from checkpoint")
    parser.add_argument(
        "--loss", choices=["ce", "label_smooth", "focal"], default="ce", help="Loss function type"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (only if loss=label_smooth)",
    )
    parser.add_argument(
        "--focal_gamma", type=float, default=2.0, help="Focusing parameter gamma for Focal Loss"
    )

    args = parser.parse_args()

    print("🚀 Starting Model Training...")
    print("=" * 50)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load training data
    print(f"\nStep 1: Loading training data from {args.data_file}...")
    with open(args.data_file, encoding="utf-8") as f:
        import json

        training_data = json.load(f)

    vocab_size = training_data.get("vocab_size", 1000)
    print(f"✅ Loaded training data with vocabulary size: {vocab_size}")

    # Step 2: Create model
    print("\nStep 2: Creating model...")
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        max_text_len=args.max_text_len,
        use_cross_attention=True,
    )

    model_info = model.get_model_info()
    print("Model created:")
    print(f"  Parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable: {model_info['trainable_parameters']:,}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Attention heads: {args.n_heads}")
    print(f"  Transformer layers: {args.n_layers}")

    # Step 3: Prepare data loaders
    print("\n📊 Step 3: Preparing data loaders...")
    from source.process.data_preparer import DataPreparer

    data_preparer = DataPreparer(
        max_sequence_length=args.max_seq_len,
        max_text_length=args.max_text_len,
        batch_size=args.batch_size,
        text_processor_use_gpu=False,
    )

    # Create datasets
    train_dataset = data_preparer.create_dataset(training_data["train_data"])
    val_dataset = data_preparer.create_dataset(training_data["val_data"])

    # Create dataloaders
    train_loader = data_preparer.create_dataloader(train_dataset, shuffle=True)
    val_loader = data_preparer.create_dataloader(val_dataset, shuffle=False)

    print("✅ Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Batch size: {args.batch_size}")

    # Step 4: Create trainer
    print("\n🎯 Step 4: Creating trainer...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        save_dir=args.output_dir,
        device=args.device,
        loss_type=args.loss,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
    )

    print("✅ Trainer created:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Device: {trainer.device}")

    # Step 5: Start training
    print("\n🎵 Step 5: Starting training...")
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.train(resume_from=args.resume_from)
    else:
        trainer.train()

    # Step 6: Training completed
    print("\n🎉 Training completed!")
    print(f"📁 Model saved to: {args.output_dir}")

    # Print final statistics
    stats = trainer.get_training_stats()
    print("\n📈 Final Statistics:")
    print(f"  Best validation loss: {stats['best_val_loss']:.4f}")
    print(f"  Training epochs: {stats['current_epoch']}")

    # Save training configuration
    config = {
        "model_config": model_info,
        "training_config": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "max_text_len": args.max_text_len,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "loss": args.loss,
            "label_smoothing": args.label_smoothing,
            "focal_gamma": args.focal_gamma,
        },
        "training_stats": stats,
    }

    config_file = os.path.join(args.output_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"📄 Training configuration saved to: {config_file}")


if __name__ == "__main__":
    main()

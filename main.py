"""
Training entry point for prompted segmentation models.

Usage:
    uv run python main.py --config src/configs/experiment.yaml
"""

import argparse

from src.configs.config import ExperimentConfig
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train prompted segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/experiment.yaml",
        help="Path to experiment YAML config",
    )
    args = parser.parse_args()

    # Load config
    config = ExperimentConfig.from_yaml(args.config)

    # Seed all RNGs before anything else
    set_seed(config.training.seed)

    print("=" * 70)
    print("PROMPTED SEGMENTATION TRAINING")
    print("=" * 70)
    print(f"Model: {config.model.arch} (encoder: {config.model.encoder})")
    print(f"Loss: {config.loss.type}")
    print(f"Augmentation: {config.data.augmentation_tier}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Epochs: {config.training.epochs}")
    print(f"LR: {config.training.lr}")
    print("=" * 70)

    # Create data pipeline
    train_loader, val_loader = create_dataloaders(config)
    print(f"\nTrain: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    # Create trainer and run
    trainer = Trainer(config)
    trainer.fit(train_loader, val_loader)

    # Save predictions on validation set (output_dir defaults to experiment predictions/)
    trainer.save_predictions(val_loader)


if __name__ == "__main__":
    main()

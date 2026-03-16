"""
Smoke test — runs the full training pipeline on a tiny synthetic dataset.

Usage:
    uv run python smoke_test.py

Creates a few dummy samples (random images + masks) in a temp directory,
then runs 2 training epochs and saves predictions to verify the end-to-end
pipeline works without requiring real data.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np

from src.configs.config import (
    DataConfig, ExperimentConfig, LossConfig,
    ModelConfig, OutputConfig, TrainingConfig,
)
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.utils.seed import set_seed


NUM_SAMPLES = 8   # per split — enough for 2 batches of 2
IMAGE_SIZE  = 640


def make_dummy_pkl(path: Path, n: int):
    """Write n random (image, mask, prompt, dataset) samples to a pickle."""
    samples = []
    prompts = [("segment crack", "cracks"), ("segment taping area", "drywall")]
    for i in range(n):
        prompt_text, dataset = prompts[i % 2]
        image = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        # mask: mostly zeros with a random rectangle of 255
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        r0, c0 = np.random.randint(0, IMAGE_SIZE // 2, size=2)
        r1, c1 = r0 + np.random.randint(50, IMAGE_SIZE // 2), c0 + np.random.randint(50, IMAGE_SIZE // 2)
        mask[r0:r1, c0:c1] = 255
        samples.append({"image": image, "mask": mask, "prompt": prompt_text, "dataset": dataset})
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def build_smoke_config(tmp_dir: Path) -> ExperimentConfig:
    train_pkl = tmp_dir / "train.pkl"
    val_pkl   = tmp_dir / "val.pkl"
    make_dummy_pkl(train_pkl, NUM_SAMPLES)
    make_dummy_pkl(val_pkl,   NUM_SAMPLES)

    config = ExperimentConfig(
        data=DataConfig(
            train_pkl=str(train_pkl),
            val_pkl=str(val_pkl),
            batch_size=2,
            num_workers=0,          # avoid multiprocessing in smoke test
            augmentation_tier="baseline",
        ),
        model=ModelConfig(
            arch="Unet",
            encoder="resnet34",
            num_prompts=2,
            embed_dim=128,
        ),
        training=TrainingConfig(
            epochs=2,
            lr=1e-4,
            weight_decay=1e-4,
            scheduler="cosine",
            early_stopping_patience=10,
            seed=42,
        ),
        loss=LossConfig(type="dice_bce", dice_weight=0.5, bce_weight=0.5),
        output=OutputConfig(exp_name="smoke_test"),
    )
    return config


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST — synthetic data, 2 epochs")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        config = build_smoke_config(Path(tmp))
        set_seed(config.training.seed)

        train_loader, val_loader = create_dataloaders(config)
        print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

        trainer = Trainer(config)
        trainer.fit(train_loader, val_loader)
        trainer.save_predictions(val_loader)

    print("\nSmoke test passed.")

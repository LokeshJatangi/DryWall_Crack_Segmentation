"""
Visualize batches from the PyTorch DataLoader (post-augmentation).

Shows denormalized images alongside ground-truth masks and overlays,
so you can verify augmentations and mask alignment before training.

Usage:
    uv run python src/visualization/visualize_dataloader.py --split train --n 16
    uv run python src/visualization/visualize_dataloader.py --split valid --n 8 --tier baseline
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.augmentations.transforms import get_train_transform, get_val_transform
from src.data.dataset import SegmentationDataset
from src.visualization.viz_utils import overlay_rgb, save_figure, mask_coverage

# ImageNet mean/std (must match values in transforms.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

DATASET_COLOR = {
    'cracks':  (255,  80,  80),   # red
    'drywall': ( 80, 160, 255),   # blue
}


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization.

    Args:
        tensor: (3, H, W) float32 normalized tensor

    Returns:
        (H, W, 3) uint8 array in [0, 255]
    """
    img = tensor.permute(1, 2, 0).numpy()          # (H, W, 3)
    img = img * IMAGENET_STD + IMAGENET_MEAN       # un-normalize
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def visualize_dataloader_samples(
    pkl_path: str,
    output_dir: str,
    split: str = 'train',
    augmentation_tier: str = 'full',
    n_samples: int = 16,
    seed: int = 0,
) -> None:
    """
    Load n_samples from the dataset, apply transforms, and save visualizations.

    Each row: Original (denormalized) | Mask | Overlay
    Saved as: one grid PNG + individual PNGs per sample.

    Args:
        pkl_path:          Path to merged pickle file.
        output_dir:        Directory to save output images.
        split:             'train' or 'valid' (controls which transforms are used).
        augmentation_tier: Augmentation tier (only used when split='train').
        n_samples:         Number of samples to visualize.
        seed:              Index offset to start sampling from.
    """
    # Build transform matching the split
    if split == 'train':
        transform = get_train_transform(tier=augmentation_tier)
    else:
        transform = get_val_transform()

    dataset = SegmentationDataset(pkl_path=pkl_path, transform=transform)
    total = len(dataset)
    n = min(n_samples, total)

    print(f"Dataset: {total} samples | Visualizing: {n} | Split: {split} | Tier: {augmentation_tier}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    individual_dir = output_path / "individual"
    individual_dir.mkdir(exist_ok=True)

    # Collect samples
    rows = []
    for i in range(seed, seed + n):
        sample = dataset[i % total]
        image_np  = denormalize(sample['image'])
        mask_np   = sample['mask'].numpy()          # (H, W) float32 {0,1}
        prompt    = sample['prompt']
        ds_name   = sample['dataset']
        overlay   = overlay_rgb(image_np, mask_np, color=DATASET_COLOR.get(ds_name, (255, 255, 0)))
        rows.append((image_np, mask_np, overlay, prompt, ds_name))

    # --- Grid figure (all samples) ---
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2D

    col_titles = ['Image (denormalized)', 'Mask', 'Overlay']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold')

    for row_idx, (image_np, mask_np, overlay, prompt, ds_name) in enumerate(rows):
        cov = mask_coverage(mask_np)

        axes[row_idx, 0].imshow(image_np)
        axes[row_idx, 0].set_ylabel(f"[{ds_name}]\n{prompt}", fontsize=8, rotation=0,
                                     labelpad=90, va='center')
        axes[row_idx, 0].axis('off')

        axes[row_idx, 1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f"{cov:.1f}% foreground", fontsize=8)
        axes[row_idx, 1].axis('off')

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].axis('off')

    plt.suptitle(
        f"DataLoader samples — {split} / {augmentation_tier} tier  ({n} of {total})",
        fontsize=13, y=1.002
    )
    plt.tight_layout()

    grid_path = output_path / f"grid_{split}_{augmentation_tier}.png"
    save_figure(fig, grid_path, dpi=120)
    print(f"Saved grid → {grid_path}")

    # --- Individual PNGs ---
    for i, (image_np, mask_np, overlay, prompt, ds_name) in enumerate(rows):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_np)
        axes[0].set_title(f"Image  [{ds_name}]", fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Mask  '{prompt}'", fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=10)
        axes[2].axis('off')

        plt.tight_layout()
        out = individual_dir / f"{i:03d}_{ds_name}_{split}.png"
        save_figure(fig, out, dpi=120)

    print(f"Saved {n} individual images → {individual_dir}/")


if __name__ == "__main__":
    BASE = Path("/mnt/disks/work/lokesh/seg")
    MERGED_DIR = BASE / "processed_data" / "merged"

    parser = argparse.ArgumentParser(description="Visualize DataLoader batches (post-augmentation)")
    parser.add_argument("--split",  default="train", choices=["train", "valid", "test"],
                        help="Which split to visualize (default: train)")
    parser.add_argument("--tier",   default="full",
                        choices=["baseline", "geometric", "photometric", "edge", "full"],
                        help="Augmentation tier — only used when split=train (default: full)")
    parser.add_argument("--n",      type=int, default=16,
                        help="Number of samples to visualize (default: 16)")
    parser.add_argument("--seed",   type=int, default=0,
                        help="Start index for sampling (default: 0)")
    parser.add_argument("--out",    type=str, default=None,
                        help="Output directory (default: processed_data/merged/<split>/viz_loader)")
    args = parser.parse_args()

    pkl_map = {
        'train': MERGED_DIR / "train" / "merged_train.pkl",
        'valid': MERGED_DIR / "valid" / "merged_valid.pkl",
        'test':  MERGED_DIR / "test"  / "merged_test.pkl",
    }
    pkl_path = pkl_map[args.split]

    if not pkl_path.exists():
        print(f"Pickle not found: {pkl_path}")
        print("Run merge_datasets.py first.")
        raise SystemExit(1)

    output_dir = args.out or str(MERGED_DIR / args.split / "viz_loader")

    visualize_dataloader_samples(
        pkl_path=str(pkl_path),
        output_dir=output_dir,
        split=args.split,
        augmentation_tier=args.tier,
        n_samples=args.n,
        seed=args.seed,
    )

    print("\nDone.")

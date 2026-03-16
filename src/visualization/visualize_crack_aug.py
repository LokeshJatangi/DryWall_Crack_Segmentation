"""
Visualize CrackWidthAugmentation effects on failure cases.

Shows original mask vs dilate/erode at multiple kernel sizes so you can
pick the right kernel range and probability.

Usage:
    uv run python src/visualization/visualize_crack_aug.py \
        --csv experiments/<run>/failure_analysis/per_sample_metrics.csv \
        --pkl processed_data/merged/valid/merged_valid.pkl \
        --top-n 10

    # Custom kernel sizes
    uv run python src/visualization/visualize_crack_aug.py \
        --csv experiments/<run>/failure_analysis/per_sample_metrics.csv \
        --pkl processed_data/merged/valid/merged_valid.pkl \
        --kernels 2 3 5 7
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.viz_utils import load_pickle, overlay_rgb


def apply_morph(mask: np.ndarray, kernel_size: int, op: str) -> np.ndarray:
    """Apply morphological dilate or erode with elliptical kernel."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if op == "dilate":
        return cv2.dilate(mask, kernel, iterations=1)
    else:
        return cv2.erode(mask, kernel, iterations=1)


def pixel_diff(original: np.ndarray, modified: np.ndarray) -> tuple[int, int]:
    """Return (pixels_added, pixels_removed) between original and modified masks."""
    orig_fg = (original > 0)
    mod_fg = (modified > 0)
    added = int((mod_fg & ~orig_fg).sum())
    removed = int((orig_fg & ~mod_fg).sum())
    return added, removed


def fg_percent(mask: np.ndarray) -> float:
    """Foreground pixel percentage."""
    return (mask > 0).sum() / mask.size * 100


def load_crack_failures(csv_path: str, top_n: int) -> list[dict]:
    """Load top-N worst crack failures from per_sample_metrics.csv."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["prompt"] == "segment crack":
                rows.append({
                    "index": int(row["index"]),
                    "iou": float(row["iou"]),
                    "dice": float(row["dice"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "gt_coverage": float(row["gt_coverage"]),
                })
    # Already sorted by IoU in CSV, but sort again to be safe
    rows.sort(key=lambda r: r["iou"])
    return rows[:top_n]


def visualize_sample(
    image: np.ndarray,
    mask: np.ndarray,
    kernels: list[int],
    sample_info: dict,
    rank: int,
) -> plt.Figure:
    """
    Create a figure for one sample showing:
      Row 1: Original | Dilate k=2 | Dilate k=3 | ... | Dilate k=N
      Row 2: Original | Erode k=2  | Erode k=3  | ... | Erode k=N

    Each panel shows image+mask overlay with fg% and pixel diff stats.
    """
    n_cols = 1 + len(kernels)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    row_labels = ["Dilate", "Erode"]
    ops = ["dilate", "erode"]

    for row_idx, (op, label) in enumerate(zip(ops, row_labels)):
        # Column 0: original
        ax = axes[row_idx, 0]
        overlay = overlay_rgb(image, mask, color=(0, 255, 0), alpha=0.5)
        ax.imshow(overlay)
        pct = fg_percent(mask)
        ax.set_title(f"Original\nfg={pct:.2f}%", fontsize=9)
        if row_idx == 0:
            ax.set_ylabel("Dilate", fontsize=11, fontweight="bold")
        else:
            ax.set_ylabel("Erode", fontsize=11, fontweight="bold")
        ax.axis("off")

        # Columns 1..N: morphed versions
        for ki, k in enumerate(kernels):
            ax = axes[row_idx, ki + 1]
            morphed = apply_morph(mask, k, op)
            overlay = overlay_rgb(image, morphed, color=(0, 255, 0), alpha=0.5)

            # Diff overlay: added=cyan, removed=red
            orig_fg = mask > 0
            morph_fg = morphed > 0
            added_px = morph_fg & ~orig_fg
            removed_px = orig_fg & ~morph_fg

            if added_px.any():
                overlay = overlay_rgb(overlay, added_px.astype(np.uint8) * 255,
                                      color=(0, 255, 255), alpha=0.7)
            if removed_px.any():
                overlay = overlay_rgb(overlay, removed_px.astype(np.uint8) * 255,
                                      color=(255, 0, 0), alpha=0.7)

            ax.imshow(overlay)
            added, removed = pixel_diff(mask, morphed)
            new_pct = fg_percent(morphed)
            ax.set_title(
                f"k={k} {op}\nfg={new_pct:.2f}% (+{added}/-{removed}px)",
                fontsize=8,
            )
            ax.axis("off")

    iou = sample_info["iou"]
    dice = sample_info["dice"]
    prec = sample_info["precision"]
    rec = sample_info["recall"]
    cov = sample_info["gt_coverage"]
    fig.suptitle(
        f"#{rank+1} — val idx={sample_info['index']}  |  "
        f"IoU={iou:.3f}  Dice={dice:.3f}  P={prec:.3f}  R={rec:.3f}  "
        f"GT coverage={cov:.1f}%\n"
        f"Green=mask  Cyan=added  Red=removed",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CrackWidthAugmentation on failure cases"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to per_sample_metrics.csv from failure analysis",
    )
    parser.add_argument(
        "--pkl", default="processed_data/merged/valid/merged_valid.pkl",
        help="Path to validation pickle",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of worst crack failures")
    parser.add_argument(
        "--kernels", type=int, nargs="+", default=[2, 3, 5, 7],
        help="Kernel sizes to visualize (default: 2 3 5 7)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: next to CSV)",
    )
    args = parser.parse_args()

    # Load failures
    failures = load_crack_failures(args.csv, args.top_n)
    if not failures:
        print("No crack failures found in CSV.")
        return
    print(f"Loaded {len(failures)} worst crack failures")

    # Load pickle
    samples = load_pickle(args.pkl)
    print(f"Loaded {len(samples)} samples from {args.pkl}")

    # Output dir
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.csv).parent / "crack_aug_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate per-sample figures
    for rank, info in enumerate(failures):
        idx = info["index"]
        sample = samples[idx]
        image = sample["image"]   # (H, W, 3) uint8
        mask = sample["mask"]     # (H, W) uint8 {0, 255}

        fig = visualize_sample(image, mask, args.kernels, info, rank)
        fname = f"crack_aug_{rank:02d}_idx{idx}_iou{info['iou']:.3f}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{rank+1}/{len(failures)}] Saved {fname}")

    # Summary grid: just original masks with fg coverage stats
    print(f"\nAll figures saved to: {out_dir}")
    print(f"\nKernel sizes tested: {args.kernels}")
    print("Look at cyan (added) vs red (removed) pixels to decide kernel range.")


if __name__ == "__main__":
    main()

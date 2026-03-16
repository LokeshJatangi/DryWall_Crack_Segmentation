"""
Shared utilities for visualization scripts.

Provides: load_pickle, mask_coverage, save_figure,
          overlay_rgb, overlay_bgr, save_grid.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_pickle(path: str) -> list:
    """Load a dataset pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def mask_coverage(mask: np.ndarray) -> float:
    """Return foreground pixel percentage (0.0–100.0)."""
    return float((mask > 0).sum() / mask.size * 100)


def save_figure(fig, path, dpi: int = 150) -> None:
    """Save and close a matplotlib figure."""
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def overlay_rgb(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a colored mask over an RGB uint8 image.

    Args:
        image: (H, W, 3) uint8 RGB
        mask:  (H, W), foreground where > 0
        color: RGB tuple (default red)
        alpha: mask opacity

    Returns:
        (H, W, 3) uint8 blended image
    """
    out = image.astype(np.float32)
    fg = mask > 0
    out[fg] = alpha * np.array(color, dtype=np.float32) + (1 - alpha) * out[fg]
    return out.clip(0, 255).astype(np.uint8)


def overlay_bgr(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 0, 255),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend a colored mask over a BGR uint8 image (OpenCV).

    Args:
        image: (H, W, 3) uint8 BGR
        mask:  (H, W), foreground where > 0
        color: BGR tuple (default red = (0, 0, 255))
        alpha: mask opacity

    Returns:
        (H, W, 3) uint8 blended image
    """
    out = image.astype(np.float32)
    fg = mask > 0
    out[fg] = alpha * np.array(color, dtype=np.float32) + (1 - alpha) * out[fg]
    return out.clip(0, 255).astype(np.uint8)


def save_grid(
    items: list,
    draw_fn,
    out_dir: Path,
    grid_name: str,
    suptitle: str,
    cols: int = 3,
    cell_size: int = 8,
    dpi: int = 120,
) -> None:
    """
    Build and save a matplotlib grid figure.

    Each item must have an 'image_path' key.
    draw_fn(ax, img_rgb, item) draws into the given axis.

    Args:
        items:     List of data dicts with 'image_path'.
        draw_fn:   Callable(ax, img_rgb, item).
        out_dir:   Output directory.
        grid_name: Output filename.
        suptitle:  Figure super-title.
        cols:      Number of columns.
        cell_size: Inches per cell.
        dpi:       Output DPI.
    """
    n = len(items)
    c = min(cols, n)
    r = (n + c - 1) // c
    fig, axes = plt.subplots(r, c, figsize=(cell_size * c, cell_size * r))
    axes = np.atleast_2d(axes if n > 1 else np.array([axes]))
    for idx, item in enumerate(items):
        img = np.array(Image.open(item["image_path"]).convert("RGB"))
        draw_fn(axes[idx // c, idx % c], img, item)
    for idx in range(n, r * c):
        axes[idx // c, idx % c].axis("off")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    grid_path = out_dir / grid_name
    plt.savefig(grid_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved grid: {grid_path}")

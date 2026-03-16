"""
Visualizer script for preprocessed dataset.

Displays side-by-side visualizations: Original Image | Binary Mask | Overlay
"""

from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.visualization.viz_utils import load_pickle, overlay_rgb, save_figure, mask_coverage


def visualize_samples(
    samples: List[Dict],
    output_dir: str,
    num_samples: int = 10,
    save_individual: bool = True
) -> None:
    """
    Visualize dataset samples.

    Args:
        samples: List of dataset samples
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        save_individual: Whether to save individual visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Limit to requested number
    samples_to_viz = samples[:num_samples]

    print(f"Visualizing {len(samples_to_viz)} samples...")

    # Create grid visualization
    n_samples = len(samples_to_viz)
    fig = plt.figure(figsize=(15, 5 * n_samples))
    gs = GridSpec(n_samples, 3, figure=fig, hspace=0.3, wspace=0.1)

    for idx, sample in enumerate(samples_to_viz):
        image = sample['image']
        mask = sample['mask']
        prompt = sample['prompt']
        filename = sample['filename']

        # Create overlay
        overlay = overlay_rgb(image, mask)

        # Plot original image
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(image)
        ax1.set_title(f"Original\n{filename}", fontsize=10)
        ax1.axis('off')

        # Plot binary mask
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(mask, cmap='gray')
        ax2.set_title(f"Mask\nPrompt: '{prompt}'", fontsize=10)
        ax2.axis('off')

        # Plot overlay
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.imshow(overlay)
        ax3.set_title("Overlay", fontsize=10)
        ax3.axis('off')

        # Save individual visualization
        if save_individual:
            individual_fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image)
            axes[0].set_title(f"Original: {filename}", fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title(f"Mask (Prompt: '{prompt}')", fontsize=12)
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title("Overlay", fontsize=12)
            axes[2].axis('off')

            plt.tight_layout()
            individual_output = output_path / f"sample_{idx:03d}_{Path(filename).stem}.png"
            save_figure(individual_fig, individual_output)

    # Save grid visualization
    grid_output = output_path / "dataset_visualization_grid.png"
    save_figure(fig, grid_output)

    print(f"✓ Saved grid visualization to: {grid_output}")
    if save_individual:
        print(f"✓ Saved {len(samples_to_viz)} individual visualizations to: {output_path}")

    # Print sample statistics
    print("\n" + "="*50)
    print("VISUALIZATION STATISTICS")
    print("="*50)
    for idx, sample in enumerate(samples_to_viz):
        coverage = mask_coverage(sample['mask'])
        print(f"Sample {idx}: {sample['filename']}")
        print(f"  Mask coverage: {coverage:.2f}%")
    print("="*50)


def display_interactive(samples: List[Dict], num_samples: int = 5) -> None:
    """
    Display interactive visualization (for Jupyter notebooks or display).

    Args:
        samples: List of dataset samples
        num_samples: Number of samples to display
    """
    samples_to_show = samples[:num_samples]

    for idx, sample in enumerate(samples_to_show):
        image = sample['image']
        mask = sample['mask']
        prompt = sample['prompt']
        filename = sample['filename']

        overlay = overlay_rgb(image, mask)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title(f"Original: {filename}", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f"Mask (Prompt: '{prompt}')", fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import argparse

    PROCESSED_ROOT = "/mnt/disks/work/lokesh/seg/processed_data"

    parser = argparse.ArgumentParser(description="Visualize preprocessed dataset samples")
    parser.add_argument("dataset", choices=["cracks", "drywall"], help="Dataset to visualize")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"],
                        help="Split to visualize (default: valid)")
    parser.add_argument("--n", type=int, default=10, help="Number of samples (default: 10)")
    parser.add_argument("--no-individual", action="store_true",
                        help="Skip saving individual per-sample images")
    args = parser.parse_args()

    pickle_path = f"{PROCESSED_ROOT}/{args.dataset}/{args.split}/{args.dataset}_{args.split}.pkl"
    output_dir  = f"{PROCESSED_ROOT}/{args.dataset}/{args.split}/viz"

    print(f"Loading {args.dataset} / {args.split} ...")
    samples = load_pickle(pickle_path)
    print(f"✓ Loaded {len(samples)} samples")

    visualize_samples(
        samples=samples,
        output_dir=output_dir,
        num_samples=min(args.n, len(samples)),
        save_individual=not args.no_individual,
    )

    print("\n✓ Visualization complete!")

"""
Interactive OpenCV visualizer for dataset pickle files.

Displays images with masks and allows keyboard navigation.
Perfect for reviewing datasets and model predictions.

Controls:
- 'd' or Right Arrow: Next image
- 'a' or Left Arrow: Previous image
- 'p' or Space: Play/Pause auto-advance
- 'q' or ESC: Quit
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional
import tkinter as tk

import numpy as np
import cv2

from src.visualization.viz_utils import load_pickle, overlay_bgr, mask_coverage


def get_screen_size():
    """Get screen dimensions."""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return screen_width, screen_height
    except:
        # Fallback to common resolution
        return 1920, 1080


def create_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str,
    filename: str,
    index: int,
    total: int,
    pred_mask: Optional[np.ndarray] = None,
    scale_factor: float = 4.0
) -> np.ndarray:
    """
    Create visualization with image, mask, and overlay.

    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W) - ground truth (shown in RED)
        prompt: Text prompt
        filename: Image filename
        index: Current index
        total: Total number of images
        pred_mask: Optional predicted mask for comparison (shown in GREEN)
        scale_factor: Display scale factor (default: 4.0 for 4x larger display)

    Returns:
        Combined visualization image
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create mask visualization (3 channels for color)
    # RED = Ground Truth mask
    mask_vis = np.zeros_like(image_bgr)
    mask_vis[:, :, 2] = (mask > 0).astype(np.uint8) * 255  # Red channel for ground truth

    # Create overlay
    overlay = overlay_bgr(image_bgr, mask)

    # If prediction mask is provided, add it in GREEN (for later model predictions)
    if pred_mask is not None:
        pred_vis = np.zeros_like(image_bgr)
        pred_vis[:, :, 1] = pred_mask * 255  # Green channel for prediction
        pred_bool = pred_mask > 0
        overlay[pred_bool] = cv2.addWeighted(
            overlay[pred_bool], 0.7,
            pred_vis[pred_bool], 0.3,
            0
        )

    # Combine: Original | Mask | Overlay
    combined = np.hstack([image_bgr, mask_vis, overlay])

    # Add text information
    text_height = 60
    info_bar = np.zeros((text_height, combined.shape[1], 3), dtype=np.uint8)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)

    # Line 1: Image info
    text1 = f"Image {index + 1}/{total}: {filename}"
    cv2.putText(info_bar, text1, (10, 20), font, font_scale, color, thickness)

    # Line 2: Prompt
    text2 = f"Prompt: '{prompt}'"
    cv2.putText(info_bar, text2, (10, 45), font, font_scale, color, thickness)

    # Add controls on the right
    controls = "Controls: d Next | a Prev | p Play/Pause | q Quit"
    text_size = cv2.getTextSize(controls, font, 0.5, 1)[0]
    text_x = combined.shape[1] - text_size[0] - 10
    cv2.putText(info_bar, controls, (text_x, 45), font, 0.5, (150, 150, 150), 1)

    # Add mask statistics
    coverage = mask_coverage(mask)

    if coverage == 0.0:
        stats = f"⚠ NO MASK (0% coverage)"
        color_stat = (0, 0, 255)  # Red for warning
    else:
        stats = f"Mask: {coverage:.1f}% coverage"
        color_stat = (150, 150, 150)

    cv2.putText(info_bar, stats, (text_x, 20), font, 0.5, color_stat, 1)

    # Combine info bar with visualization
    result = np.vstack([info_bar, combined])

    # Scale up for better visibility (4x by default)
    if scale_factor != 1.0:
        new_width = int(result.shape[1] * scale_factor)
        new_height = int(result.shape[0] * scale_factor)
        result = cv2.resize(result, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    return result


def visualize_interactive(
    pickle_path: str,
    window_name: str = "Dataset Visualizer",
    start_index: int = 0,
    scale_factor: float = None,
    auto_play_delay: int = 100
) -> None:
    """
    Interactive visualization with keyboard navigation and auto-play.

    Args:
        pickle_path: Path to pickle file
        window_name: Name of OpenCV window
        start_index: Starting image index
        scale_factor: Display scale factor (if None, auto-calculate for 70% screen)
        auto_play_delay: Delay in milliseconds for auto-play mode (default: 100ms)
    """
    # Load data
    print(f"Loading data from {pickle_path}...")
    data = load_pickle(pickle_path)
    print(f"✓ Loaded {len(data)} samples")

    if len(data) == 0:
        print("Error: No data found in pickle file")
        return

    # Get screen size and calculate scale factor
    screen_width, screen_height = get_screen_size()

    # Create a test visualization to get dimensions
    sample = data[0]
    test_vis = create_visualization(
        sample['image'], sample['mask'], sample['prompt'],
        sample.get('filename', 'test'), 0, len(data),
        sample.get('pred_mask', None), scale_factor=1.0
    )

    # Calculate scale to fit screen (target 50% of screen height/width, whichever is smaller)
    if scale_factor is None:
        # Use 50% of single screen dimensions for safety
        target_width = screen_width * 0.5
        target_height = screen_height * 0.5

        width_scale = target_width / test_vis.shape[1]
        height_scale = target_height / test_vis.shape[0]
        scale_factor = min(width_scale, height_scale)

        # Cap maximum scale to avoid gigantic windows
        scale_factor = min(scale_factor, 3.0)

        print(f"Auto-calculated scale factor: {scale_factor:.2f}x (target: 50% screen)")
        print(f"Window size: {int(test_vis.shape[1] * scale_factor)}x{int(test_vis.shape[0] * scale_factor)}")

    # Initialize
    current_idx = start_index
    playing = False  # Auto-play mode
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "="*50)
    print("INTERACTIVE VISUALIZER")
    print("="*50)
    print("Controls:")
    print("  d or → : Next image")
    print("  a or ← : Previous image")
    print("  p or SPACE : Play/Pause auto-advance")
    print("  q or ESC : Quit")
    print("="*50)
    print(f"Auto-play delay: {auto_play_delay}ms")
    print("="*50)

    while True:
        # Get current sample
        sample = data[current_idx]
        image = sample['image']
        mask = sample['mask']
        prompt = sample['prompt']
        filename = sample.get('filename', f"sample_{current_idx}")

        # Check if this is a prediction result (has 'pred_mask' key)
        pred_mask = sample.get('pred_mask', None)

        # Create visualization
        vis = create_visualization(
            image, mask, prompt, filename,
            current_idx, len(data), pred_mask, scale_factor
        )

        # Display
        cv2.imshow(window_name, vis)

        # Wait for key press (different delay for play/pause mode)
        wait_time = auto_play_delay if playing else 0
        key = cv2.waitKey(wait_time) & 0xFF

        # In play mode, auto-advance
        if playing and key == 255:  # No key pressed (timeout)
            current_idx = (current_idx + 1) % len(data)
            continue

        # Handle navigation
        if key == ord('d') or key == 83:  # 'd' or right arrow
            current_idx = (current_idx + 1) % len(data)
            print(f"→ Next: {current_idx + 1}/{len(data)}")

        elif key == ord('a') or key == 81:  # 'a' or left arrow
            current_idx = (current_idx - 1) % len(data)
            print(f"← Previous: {current_idx + 1}/{len(data)}")

        elif key == ord('p') or key == 32:  # 'p' or SPACE
            playing = not playing
            status = "▶ PLAYING" if playing else "⏸ PAUSED"
            print(f"{status}")

        elif key == ord('q') or key == 27:  # 'q' or ESC
            print("\nExiting visualizer...")
            break

    cv2.destroyAllWindows()
    print("✓ Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive visualizer for dataset pickle files"
    )
    parser.add_argument(
        "--pickle_path",
        type=str,
        help="Path to pickle file (e.g., processed_data/drywall/valid/drywall_valid.pkl)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting image index (default: 0)"
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Dataset Visualizer",
        help="OpenCV window name (default: 'Dataset Visualizer')"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display scale factor (default: auto-calculate for 70%% screen)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=100,
        help="Auto-play delay in milliseconds (default: 100ms)"
    )

    args = parser.parse_args()

    # Run interactive visualizer
    visualize_interactive(
        pickle_path=args.pickle_path,
        window_name=args.window_name,
        start_index=args.start,
        scale_factor=args.scale,
        auto_play_delay=args.delay
    )

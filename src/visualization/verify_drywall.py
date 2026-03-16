"""
Verification script for drywall dataset preprocessing.

Visually confirms that bounding box annotations and the generated
binary masks align correctly on the original images.

Shows 4 panels per sample:
  1. Original image
  2. Bounding boxes drawn on image
  3. Binary mask from bbox
  4. Mask overlay on image

Usage:
    python src/visualization/verify_drywall.py [--split train] [--num 5] [--save]
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.visualization.viz_utils import overlay_bgr, mask_coverage


DATASET_DIR = Path("/mnt/disks/work/lokesh/seg/datasets/Drywall-Join-Detect.v2i.coco")
OUTPUT_DIR = Path("/mnt/disks/work/lokesh/seg/processed_data/verification/drywall")


def load_coco(split: str):
    """Load COCO annotations and build lookup maps."""
    ann_path = DATASET_DIR / split / "_annotations.coco.json"
    with open(ann_path) as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}

    id_to_anns = {}
    for ann in coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    return id_to_img, id_to_anns


COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 165, 0),  # orange
    (0, 128, 255),  # light blue
]


def draw_bboxes(image: np.ndarray, annotations: list) -> np.ndarray:
    """Draw numbered bounding boxes with distinct colors."""
    vis = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, ann in enumerate(annotations):
        x, y, w, h = [int(v) for v in ann["bbox"]]
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"#{i + 1} [{w}x{h}]"
        cv2.putText(vis, label, (x, max(y - 5, 15)), font, 0.5, color, 2)
    return vis


def bbox_to_mask(annotations: list, shape: tuple) -> np.ndarray:
    """Convert bbox annotations to combined binary mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for ann in annotations:
        x, y, w, h = [int(v) for v in ann["bbox"]]
        x = max(0, x)
        y = max(0, y)
        w = min(shape[1] - x, w)
        h = min(shape[0] - y, h)
        mask[y : y + h, x : x + w] = 1
    return mask


def build_panel(image_bgr: np.ndarray, annotations: list, idx: int, total: int, filename: str) -> np.ndarray:
    """Build a 4-panel verification image for one sample."""
    mask = bbox_to_mask(annotations, image_bgr.shape)
    bbox_vis = draw_bboxes(image_bgr, annotations)
    mask_vis = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    overlay = overlay_bgr(image_bgr, mask)

    top = np.hstack([image_bgr, bbox_vis])
    bottom = np.hstack([mask_vis, overlay])
    panel = np.vstack([top, bottom])

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = image_bgr.shape[:2]
    labels = [
        ("Original", (10, 25)),
        ("BBoxes", (w + 10, 25)),
        ("Mask", (10, h + 25)),
        ("Overlay", (w + 10, h + 25)),
    ]
    for text, pos in labels:
        cv2.putText(panel, text, pos, font, 0.7, (0, 255, 255), 2)

    # Info bar
    info = f"[{idx + 1}/{total}] {filename}  |  annotations: {len(annotations)}  |  mask coverage: {mask_coverage(mask):.1f}%"
    bar = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, info, (10, 28), font, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, panel])


def run_interactive(split: str, num_samples: int, save: bool, multi_only: bool = False):
    """Run interactive verification viewer."""
    id_to_img, id_to_anns = load_coco(split)
    image_ids = list(id_to_img.keys())

    if multi_only:
        image_ids = [iid for iid in image_ids if len(id_to_anns.get(iid, [])) >= 2]
        print(f"Filtered to {len(image_ids)} images with 2+ annotations")

    image_ids = image_ids[:num_samples]

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(id_to_img)} images from '{split}' split")
    print(f"Showing {len(image_ids)} samples")
    print("\nControls:  d/→ Next  |  a/← Prev  |  s Save  |  q/ESC Quit\n")

    idx = 0
    window = "Drywall BBox vs Mask Verification"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        img_info = id_to_img[image_ids[idx]]
        img_path = DATASET_DIR / split / img_info["file_name"]
        image_bgr = cv2.imread(str(img_path))
        annotations = id_to_anns.get(image_ids[idx], [])

        panel = build_panel(image_bgr, annotations, idx, len(image_ids), img_info["file_name"])

        # Scale for display
        screen_h = 900
        scale = screen_h / panel.shape[0]
        display = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        cv2.imshow(window, display)

        if save:
            out_path = OUTPUT_DIR / f"verify_{split}_{idx:03d}_{Path(img_info['file_name']).stem}.png"
            cv2.imwrite(str(out_path), panel)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("d") or key == 83:  # next
            idx = (idx + 1) % len(image_ids)
        elif key == ord("a") or key == 81:  # prev
            idx = (idx - 1) % len(image_ids)
        elif key == ord("s"):  # save current
            out_path = OUTPUT_DIR / f"verify_{split}_{idx:03d}_{Path(img_info['file_name']).stem}.png"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), panel)
            print(f"Saved: {out_path}")
        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
    if save:
        print(f"\nSaved {len(image_ids)} verification images to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify drywall bbox-to-mask alignment")
    parser.add_argument("--split", default="train", choices=["train", "valid"])
    parser.add_argument("--num", type=int, default=10, help="Number of samples to verify")
    parser.add_argument("--save", action="store_true", help="Save all verification panels to disk")
    parser.add_argument("--multi-only", action="store_true", help="Show only images with 2+ annotations")
    args = parser.parse_args()

    run_interactive(args.split, args.num, args.save, args.multi_only)
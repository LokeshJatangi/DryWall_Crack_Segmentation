"""
Visualize annotation quality issues in the cracks dataset.

Three modes:
  tiny   — RED bboxes for tiny annotations (area < 100px), GREEN for normal
  bleed  — RED bboxes that extend beyond the image boundary, yellow dashes = image border
  disco  — RED overlay on disconnected tiny component islands (< 100px) in the generated mask

Saves individual PNGs + a summary grid per mode.

Run:
  python src/visualization/visualize_tiny_annotations.py            # all modes
  python src/visualization/visualize_tiny_annotations.py tiny
  python src/visualization/visualize_tiny_annotations.py bleed
  python src/visualization/visualize_tiny_annotations.py disco
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src.visualization.viz_utils import save_grid

BASE = Path("datasets/cracks.v1i.coco")
AREA_THRESHOLD = 100
BLEED_TOLERANCE = 5
TINY_COMPONENT_AREA = 100

OUTPUT_DIR       = Path("processed_data/verification/tiny_annotations")
BLEED_OUTPUT_DIR = Path("processed_data/verification/bleed")
DISCO_OUTPUT_DIR = Path("processed_data/verification/disconnected")


def collect_tiny_annotation_images():
    """Find all images that contain at least one tiny annotation."""
    affected_images = []

    for split in ["train", "valid"]:
        ann_path = BASE / split / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        # Find images with at least one tiny annotation
        for img_id, anns in anns_by_image.items():
            has_tiny = any(a["area"] < AREA_THRESHOLD for a in anns)
            if has_tiny:
                affected_images.append({
                    "split": split,
                    "image_info": images[img_id],
                    "annotations": anns,
                    "image_path": BASE / split / images[img_id]["file_name"],
                })

    return affected_images


def draw_annotations(ax, image, annotations, filename, split):
    """Draw image with bboxes colored by annotation size."""
    ax.imshow(image)

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        is_tiny = ann["area"] < AREA_THRESHOLD

        color = "red" if is_tiny else "limegreen"
        linewidth = 2 if is_tiny else 1
        linestyle = "-" if is_tiny else "--"

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=linewidth, edgecolor=color,
            facecolor=color, alpha=0.3 if is_tiny else 0.1,
            linestyle=linestyle,
        )
        ax.add_patch(rect)

        label = f"area={ann['area']}"
        fontsize = 8 if is_tiny else 6
        ax.text(
            x, y - 2, label,
            fontsize=fontsize, color=color,
            fontweight="bold" if is_tiny else "normal",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )

    n_tiny = sum(1 for a in annotations if a["area"] < AREA_THRESHOLD)
    n_normal = len(annotations) - n_tiny
    ax.set_title(
        f"[{split}] {filename}\n"
        f"{n_tiny} tiny (red) | {n_normal} normal (green) | {len(annotations)} total",
        fontsize=9,
    )
    ax.axis("off")


def visualize_tiny_annotations():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    affected = collect_tiny_annotation_images()
    print(f"Found {len(affected)} images with tiny annotations (area < {AREA_THRESHOLD}px)")

    if not affected:
        print("No tiny annotations found.")
        return

    # Save individual visualizations
    for idx, item in enumerate(affected):
        img = np.array(Image.open(item["image_path"]))
        fname = item["image_info"]["file_name"]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        draw_annotations(ax, img, item["annotations"], fname, item["split"])
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"tiny_{idx:03d}_{item['split']}_{Path(fname).stem}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path.name}")

    # Save summary grid
    n = len(affected)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, item in enumerate(affected):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        img = np.array(Image.open(item["image_path"]))
        fname = item["image_info"]["file_name"]
        draw_annotations(ax, img, item["annotations"], fname, item["split"])

    # Hide empty subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"Tiny Annotations Summary (area < {AREA_THRESHOLD}px)\n"
        f"{n} affected images | Red = tiny/noise | Green = normal",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    grid_path = OUTPUT_DIR / "tiny_annotations_grid.png"
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved grid: {grid_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Split':<8} {'File':<65} {'Ann ID':<8} {'Area':<10} {'BBox'}")
    print(f"{'='*70}")
    for item in affected:
        for ann in item["annotations"]:
            if ann["area"] < AREA_THRESHOLD:
                print(
                    f"{item['split']:<8} "
                    f"{item['image_info']['file_name'][:63]:<65} "
                    f"{ann['id']:<8} "
                    f"{ann['area']:<10} "
                    f"{ann['bbox']}"
                )
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Bleed visualisation
# ---------------------------------------------------------------------------

def collect_bleed_annotations():
    """Find annotations whose bbox extends beyond image boundaries."""
    affected = []
    for split in ["train", "valid"]:
        ann_path = BASE / split / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        for img_id, anns in anns_by_image.items():
            img_info = images[img_id]
            iw, ih = img_info["width"], img_info["height"]
            bleed_anns = [
                ann for ann in anns
                if (ann["bbox"][0] < -BLEED_TOLERANCE
                    or ann["bbox"][1] < -BLEED_TOLERANCE
                    or ann["bbox"][0] + ann["bbox"][2] > iw + BLEED_TOLERANCE
                    or ann["bbox"][1] + ann["bbox"][3] > ih + BLEED_TOLERANCE)
            ]
            if bleed_anns:
                affected.append({
                    "split": split,
                    "image_info": img_info,
                    "all_annotations": anns,
                    "bleed_annotations": bleed_anns,
                    "image_path": BASE / split / img_info["file_name"],
                })
    return affected


def draw_bleed(ax, image, all_anns, bleed_anns, img_info, split):
    """Draw image with bleed bboxes in red, normal in green, image border in yellow."""
    ax.imshow(image)
    iw, ih = img_info["width"], img_info["height"]
    bleed_ids = {a["id"] for a in bleed_anns}

    for ann in all_anns:
        bx, by, bw, bh = ann["bbox"]
        is_bleed = ann["id"] in bleed_ids
        color = "red" if is_bleed else "limegreen"
        rect = patches.Rectangle(
            (bx, by), bw, bh,
            linewidth=2 if is_bleed else 1,
            edgecolor=color, facecolor=color,
            alpha=0.35 if is_bleed else 0.1,
        )
        ax.add_patch(rect)
        if is_bleed:
            ox = min(0, bx) + max(0, bx + bw - iw)
            oy = min(0, by) + max(0, by + bh - ih)
            ax.text(
                bx, by - 3, f"bleed Δx={ox:.0f} Δy={oy:.0f}",
                fontsize=7, color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
            )

    # Image boundary marker
    ax.add_patch(patches.Rectangle(
        (0, 0), iw, ih,
        linewidth=2, edgecolor="yellow", facecolor="none", linestyle="--",
    ))
    ax.set_title(
        f"[{split}] {img_info['file_name']}\n"
        f"{len(bleed_anns)} bleed (red) | {len(all_anns) - len(bleed_anns)} normal (green) "
        f"| yellow = image boundary",
        fontsize=9,
    )
    ax.axis("off")


def visualize_bleed():
    BLEED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    affected = collect_bleed_annotations()
    print(f"Found {len(affected)} images with bleed annotations")
    if not affected:
        print("No bleed annotations found.")
        return

    for idx, item in enumerate(affected):
        img = np.array(Image.open(item["image_path"]).convert("RGB"))
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        draw_bleed(ax, img, item["all_annotations"], item["bleed_annotations"],
                   item["image_info"], item["split"])
        plt.tight_layout()
        stem = Path(item["image_info"]["file_name"]).stem
        out_path = BLEED_OUTPUT_DIR / f"bleed_{idx:03d}_{item['split']}_{stem}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path.name}")

    def _draw(ax, img, item):
        draw_bleed(ax, img, item["all_annotations"], item["bleed_annotations"],
                   item["image_info"], item["split"])

    save_grid(
        affected, _draw, BLEED_OUTPUT_DIR, "bleed_grid.png",
        f"Bleed Annotations | {len(affected)} images | Red=bleed | Yellow dashes=image boundary",
    )

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Split':<8} {'File':<50} {'Ann ID':<8} {'BBox'}")
    print(f"{'='*80}")
    for item in affected:
        for ann in item["bleed_annotations"]:
            print(f"{item['split']:<8} {item['image_info']['file_name'][:48]:<50} "
                  f"{ann['id']:<8} {ann['bbox']}")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Disconnected-components visualisation
# ---------------------------------------------------------------------------

def collect_disconnected_images():
    """Find images whose combined mask has tiny disconnected islands (< TINY_COMPONENT_AREA px)."""
    affected = []
    for split in ["train", "valid"]:
        ann_path = BASE / split / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        for img_id, anns in anns_by_image.items():
            img_info = images[img_id]
            ih, iw = img_info["height"], img_info["width"]

            # Build combined binary mask (value 1 = foreground)
            mask = np.zeros((ih, iw), dtype=np.uint8)
            for ann in anns:
                seg = ann.get("segmentation", [])
                if seg:
                    for polygon in seg:
                        if len(polygon) < 6:
                            continue
                        pts = np.array(polygon, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 1)
                else:
                    bx, by, bw, bh = [int(round(v)) for v in ann["bbox"]]
                    mask[max(0, by):min(ih, by + bh), max(0, bx):min(iw, bx + bw)] = 1

            if mask.sum() == 0:
                continue

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            tiny = [
                {
                    "label": lbl,
                    "area": int(stats[lbl, cv2.CC_STAT_AREA]),
                    "x": int(stats[lbl, cv2.CC_STAT_LEFT]),
                    "y": int(stats[lbl, cv2.CC_STAT_TOP]),
                    "w": int(stats[lbl, cv2.CC_STAT_WIDTH]),
                    "h": int(stats[lbl, cv2.CC_STAT_HEIGHT]),
                    "cx": float(centroids[lbl][0]),
                    "cy": float(centroids[lbl][1]),
                }
                for lbl in range(1, num_labels)
                if stats[lbl, cv2.CC_STAT_AREA] < TINY_COMPONENT_AREA
            ]
            if tiny:
                affected.append({
                    "split": split,
                    "image_info": img_info,
                    "mask": mask,
                    "labels": labels,
                    "num_labels": num_labels,
                    "tiny_components": tiny,
                    "image_path": BASE / split / img_info["file_name"],
                })
    return affected


def draw_disconnected(ax_img, ax_mask, image, item):
    """Side-by-side: image overlay + pure mask, tiny islands in red."""
    mask = item["mask"]
    labels = item["labels"]
    tiny = item["tiny_components"]
    img_info = item["image_info"]
    split = item["split"]
    h, w = mask.shape

    # Colour map: green = normal fg, red = tiny island
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[mask > 0] = [0, 200, 0]
    for comp in tiny:
        colored[labels == comp["label"]] = [255, 0, 0]

    # Overlay on image
    overlay = image.copy()
    fg = mask > 0
    overlay[fg] = (0.45 * colored[fg] + 0.55 * image[fg]).astype(np.uint8)

    ax_img.imshow(overlay)
    for comp in tiny:
        ax_img.add_patch(patches.Rectangle(
            (comp["x"], comp["y"]), comp["w"], comp["h"],
            linewidth=2, edgecolor="red", facecolor="none",
        ))
        ax_img.text(
            comp["cx"], comp["cy"], f"{comp['area']}px",
            fontsize=7, color="white", ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="red", alpha=0.8),
        )
    n_normal = item["num_labels"] - 1 - len(tiny)
    ax_img.set_title(
        f"[{split}] {img_info['file_name']}\n"
        f"{len(tiny)} tiny islands (red) | {n_normal} normal components (green)",
        fontsize=8,
    )
    ax_img.axis("off")

    # Pure mask panel
    ax_mask.imshow(colored)
    ax_mask.set_title("Mask  (red = tiny islands)", fontsize=8)
    ax_mask.axis("off")


def visualize_disconnected():
    DISCO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    affected = collect_disconnected_images()
    print(f"Found {len(affected)} images with disconnected tiny components (< {TINY_COMPONENT_AREA}px)")
    if not affected:
        print("No disconnected components found.")
        return

    for idx, item in enumerate(affected):
        img = np.array(Image.open(item["image_path"]).convert("RGB"))
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        draw_disconnected(axes[0], axes[1], img, item)
        plt.tight_layout()
        stem = Path(item["image_info"]["file_name"]).stem
        out_path = DISCO_OUTPUT_DIR / f"disco_{idx:03d}_{item['split']}_{stem}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path.name}")

    # Grid: each row = one image (overlay | mask)
    n = len(affected)
    fig, axes = plt.subplots(n, 2, figsize=(16, 8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for idx, item in enumerate(affected):
        img = np.array(Image.open(item["image_path"]).convert("RGB"))
        draw_disconnected(axes[idx, 0], axes[idx, 1], img, item)
    fig.suptitle(
        f"Disconnected Components (< {TINY_COMPONENT_AREA}px) | {n} images\n"
        "Red = tiny islands | Green = normal foreground",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    grid_path = DISCO_OUTPUT_DIR / "disconnected_grid.png"
    plt.savefig(grid_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved grid: {grid_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Split':<8} {'File':<50} {'Components':<12} {'Areas'}")
    print(f"{'='*70}")
    for item in affected:
        areas = [c["area"] for c in item["tiny_components"]]
        print(f"{item['split']:<8} {item['image_info']['file_name'][:48]:<50} "
              f"{len(areas):<12} {areas}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("tiny", "all"):
        print("=== Tiny Annotations ===")
        visualize_tiny_annotations()

    if mode in ("bleed", "all"):
        print("\n=== Bleed Annotations ===")
        visualize_bleed()

    if mode in ("disco", "disconnected", "all"):
        print("\n=== Disconnected Components ===")
        visualize_disconnected()
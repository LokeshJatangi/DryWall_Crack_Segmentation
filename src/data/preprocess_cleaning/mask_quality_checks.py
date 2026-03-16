"""
Consolidated mask/annotation/image quality checks for segmentation datasets.

Checks:
  Image-level:
    1.  File integrity — corrupted files, missing on disk, extra files on disk
    2.  Wrong image mode — non-RGB images
    3.  Image-annotation size mismatch — actual image vs annotation dimensions
    4.  Resolution stats — unique resolutions, non-640x640 counts
    # 5.  Blurriness — (placeholder, not yet implemented)

  Annotation-level:
    6.  Empty masks — images with no annotations / 0 mask pixels
    7.  Annotations without matching images — orphan annotations
    8.  Missing classes — expected categories not present
    9.  Unused categories — defined but never referenced
    10. Category distribution — annotation counts per category
    11. Annotation bleed — bbox extending beyond image boundaries
    12. Tiny annotations — area < threshold (from JSON area field)
    13. Disconnected components — tiny floating pixel islands in generated mask
    14. Incorrect label IDs — category IDs not in dataset category list
    15. Degenerate polygons — segmentation polygons with < 3 vertices
    16. Multi-polygon annotations — annotations with multiple disjoint polygons
    17. Crowd annotations — iscrowd=1 annotations
    18. Bbox area vs annotation area mismatch — bbox w*h != JSON area
    19. Annotation distribution — min/max/avg annotations per image

  Cross-level:
    20. Corrupted directory check — validate files in corrupted/ folder

Usage:
    uv run python src/data/mask_quality_checks.py
    uv run python src/data/mask_quality_checks.py --dataset drywall
    uv run python src/data/mask_quality_checks.py --output reports/mask_quality_report.txt
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

BASE_DIR = Path("/mnt/disks/work/lokesh/seg")

DATASETS = {
    "drywall": {
        "path": BASE_DIR / "datasets" / "Drywall-Join-Detect.v2i.coco",
        "splits": ["train", "valid"],
        "expected_categories": {1: "drywall-join"},
        "has_segmentation": False,  # bbox only
        "prompt": "segment taping area",
    },
    "cracks": {
        "path": BASE_DIR / "datasets" / "cracks.v1i.coco",
        "splits": ["train", "valid"],
        "expected_categories": {1: "NewCracks - v2 2024-05-18 10-54pm"},
        "has_segmentation": True,
        "prompt": "segment crack",
    },
}

# Thresholds
TINY_COMPONENT_AREA = 100  # pixels — disconnected island threshold
SMALL_ANNOTATION_AREA = 100  # pixels — suspiciously small annotation
BLEED_TOLERANCE = 5  # pixels — how far annotation can extend beyond image
AREA_MISMATCH_TOLERANCE = 1  # bbox area vs annotation area difference threshold


def load_coco(dataset_path: Path, split: str):
    """Load COCO annotations and build lookup maps."""
    ann_path = dataset_path / split / "_annotations.coco.json"
    if not ann_path.exists():
        return None, None, None

    with open(ann_path) as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}
    id_to_anns = {}
    for ann in coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    return coco, id_to_img, id_to_anns


def segmentation_to_mask(segmentation, h, w):
    """Convert COCO segmentation polygons to binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for polygon in segmentation:
        if len(polygon) < 6:  # need at least 3 points
            continue
        pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def bbox_to_mask(bbox, h, w):
    """Convert COCO bbox [x, y, w, h] to binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = [int(round(v)) for v in bbox]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    mask[y1:y2, x1:x2] = 1
    return mask


def check_split(dataset_name: str, dataset_cfg: dict, split: str):
    """Run all quality checks on a single split."""
    coco, id_to_img, id_to_anns = load_coco(dataset_cfg["path"], split)
    if coco is None:
        return None

    split_dir = dataset_cfg["path"] / split
    valid_cat_ids = set(c["id"] for c in coco["categories"])
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    has_seg = dataset_cfg["has_segmentation"]

    findings = {
        "dataset": dataset_name,
        "split": split,
        "total_images": len(id_to_img),
        "total_annotations": len(coco["annotations"]),
        "categories": cat_id_to_name,
        # Image-level checks
        "missing_on_disk": [],              # 1. files referenced but missing
        "extra_on_disk": [],                # 1. files on disk but not in annotations
        "corrupted": [],                    # 1. corrupted/unloadable images
        "wrong_mode": [],                   # 2. non-RGB images
        "size_mismatch": [],                # 3. image vs annotation dimension mismatch
        "unique_resolutions": {},           # 4. resolution distribution
        "non_640x640_count": 0,             # 4. images not at target resolution
        # Annotation-level checks
        "empty_masks": [],                  # 6. images with no annotation pixels
        "orphan_annotations": [],           # 7. annotations without matching images
        "missing_classes": [],              # 8. expected categories not present
        "unused_categories": [],            # 9. defined but unreferenced categories
        "category_distribution": {},        # 10. annotation counts per category
        "annotation_bleed": [],             # 11. bbox beyond image boundaries
        "tiny_annotations": [],             # 12. area < threshold
        "disconnected_components": [],      # 13. tiny floating pixel islands
        "incorrect_labels": [],             # 14. invalid category IDs
        "degenerate_polygons": [],          # 15. polygons with < 3 vertices
        "multi_polygon": [],                # 16. multiple disjoint polygons
        "crowd_annotations": [],            # 17. iscrowd=1
        "area_mismatches": [],              # 18. bbox area vs annotation area
        "annotation_distribution": {},      # 19. min/max/avg annotations per image
        # Segmentation-specific
        "empty_segmentations_count": 0,     # count of empty segmentation fields
    }

    # --- File integrity checks ---
    disk_files = {f for f in os.listdir(split_dir) if not f.endswith(".json")}
    ann_files = {img["file_name"] for img in id_to_img.values()}
    findings["missing_on_disk"] = sorted(ann_files - disk_files)
    findings["extra_on_disk"] = sorted(disk_files - ann_files)

    # --- Per-image file checks (corrupted, wrong mode, size mismatch) ---
    widths = []
    heights = []
    for img_info in id_to_img.values():
        fpath = split_dir / img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]
        widths.append(img_w)
        heights.append(img_h)

        if not fpath.exists():
            findings["corrupted"].append(
                (img_info["file_name"], "missing on disk")
            )
            continue
        try:
            im = Image.open(fpath)
            im.verify()
            im = Image.open(fpath)
            if im.mode != "RGB":
                findings["wrong_mode"].append(
                    (img_info["file_name"], im.mode)
                )
            actual_w, actual_h = im.size
            if actual_w != img_w or actual_h != img_h:
                findings["size_mismatch"].append(
                    f"{img_info['file_name']}: annotation says {img_w}x{img_h}, "
                    f"actual image is {actual_w}x{actual_h}"
                )
        except Exception as e:
            findings["corrupted"].append(
                (img_info["file_name"], str(e))
            )

    # --- Resolution stats ---
    size_counts = Counter(zip(widths, heights))
    findings["unique_resolutions"] = {f"{w}x{h}": cnt for (w, h), cnt in size_counts.items()}
    findings["non_640x640_count"] = sum(
        1 for w, h in zip(widths, heights) if w != 640 or h != 640
    )

    # --- Category checks ---
    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])
    findings["category_distribution"] = {
        cat_id_to_name.get(k, f"unknown_{k}"): v for k, v in cat_counts.items()
    }
    present_cat_ids = set(cat_counts.keys())

    # Missing classes (expected but absent)
    for cat_id, cat_name in dataset_cfg["expected_categories"].items():
        if cat_id not in present_cat_ids:
            findings["missing_classes"].append(
                f"Category {cat_id} ({cat_name}) has no annotations"
            )

    # Unused categories (defined in JSON but no annotations)
    for cat in coco["categories"]:
        if cat["id"] not in cat_counts:
            findings["unused_categories"].append(
                f"Category {cat['id']} ({cat['name']}): defined but unused"
            )

    # Invalid category annotations
    findings["incorrect_labels"] = [
        f"Ann {a['id']} (image {id_to_img[a['image_id']]['file_name']}): "
        f"category_id={a['category_id']} not in {valid_cat_ids}"
        for a in coco["annotations"]
        if a["category_id"] not in valid_cat_ids and a["image_id"] in id_to_img
    ]

    # --- Orphan annotations (annotations referencing non-existent images) ---
    findings["orphan_annotations"] = [
        f"Ann {ann['id']}: references non-existent image_id {ann['image_id']}"
        for ann in coco["annotations"]
        if ann["image_id"] not in id_to_img
    ]

    # --- Per-annotation checks ---
    empty_seg_count = 0
    for ann in coco["annotations"]:
        img_info = id_to_img.get(ann["image_id"])
        if img_info is None:
            continue

        img_h = img_info["height"]
        img_w = img_info["width"]
        fname = img_info["file_name"]
        ann_id = ann["id"]

        # Bbox values
        bx, by, bw, bh = ann["bbox"]

        # Check 11: Annotation bleed (bbox out of bounds)
        if (bx < -BLEED_TOLERANCE or by < -BLEED_TOLERANCE
                or bx + bw > img_w + BLEED_TOLERANCE
                or by + bh > img_h + BLEED_TOLERANCE):
            findings["annotation_bleed"].append(
                f"Ann {ann_id} (image {fname}): bbox [{bx},{by},{bw},{bh}] "
                f"exceeds image {img_w}x{img_h}"
            )

        # Check 12: Tiny annotations
        ann_area = ann.get("area", 0)
        bbox_area = bw * bh
        effective_area = ann_area if ann_area > 0 else bbox_area
        if 0 < effective_area < SMALL_ANNOTATION_AREA:
            findings["tiny_annotations"].append(
                f"Ann {ann_id} (image {fname}): area={ann_area:.1f}, "
                f"bbox_area={bbox_area:.1f}, bbox={ann['bbox']}"
            )

        # Check 15 & 16: Degenerate and multi-polygon
        seg = ann.get("segmentation", [])
        if not seg or (isinstance(seg, list) and all(len(s) == 0 for s in seg)):
            empty_seg_count += 1

        if has_seg and isinstance(seg, list):
            if len(seg) > 1:
                findings["multi_polygon"].append(
                    f"Ann {ann_id} (image {fname}): {len(seg)} disjoint polygons"
                )
            for poly_idx, polygon in enumerate(seg):
                n_vertices = len(polygon) // 2
                if n_vertices < 4:
                    findings["degenerate_polygons"].append(
                        f"Ann {ann_id} (image {fname}): polygon {poly_idx} "
                        f"has {n_vertices} vertices (need >=4 for clean edges)"
                    )

        # Check 17: Crowd annotations
        if ann.get("iscrowd", 0) == 1:
            findings["crowd_annotations"].append(
                f"Ann {ann_id} (image {fname})"
            )

        # Check 18: Bbox area vs annotation area mismatch
        calculated_area = bw * bh
        annotation_area = ann.get("area", 0)
        if abs(calculated_area - annotation_area) > AREA_MISMATCH_TOLERANCE:
            findings["area_mismatches"].append(
                f"Ann {ann_id} (image {fname}): bbox_area={calculated_area:.1f}, "
                f"json_area={annotation_area:.1f}, diff={abs(calculated_area - annotation_area):.1f}"
            )

    findings["empty_segmentations_count"] = empty_seg_count

    # --- Per-image mask checks ---
    ann_counts = []
    for img_id, img_info in id_to_img.items():
        img_h = img_info["height"]
        img_w = img_info["width"]
        fname = img_info["file_name"]
        anns = id_to_anns.get(img_id, [])
        ann_counts.append(len(anns))

        # Check 6: Empty masks
        if not anns:
            findings["empty_masks"].append(
                f"{fname} (id={img_id}): no annotations at all"
            )
            continue

        # Generate combined mask for this image
        combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for ann in anns:
            if has_seg and ann.get("segmentation") and ann["segmentation"]:
                m = segmentation_to_mask(ann["segmentation"], img_h, img_w)
                combined_mask = np.maximum(combined_mask, m)
            else:
                m = bbox_to_mask(ann["bbox"], img_h, img_w)
                combined_mask = np.maximum(combined_mask, m)

        total_pixels = combined_mask.sum()
        if total_pixels == 0:
            findings["empty_masks"].append(
                f"{fname} (id={img_id}): {len(anns)} annotations but 0 mask pixels"
            )

        # Check 13: Disconnected components (tiny islands)
        if total_pixels > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                combined_mask, connectivity=8
            )
            for lbl in range(1, num_labels):  # label 0 is background
                comp_area = stats[lbl, cv2.CC_STAT_AREA]
                if comp_area < TINY_COMPONENT_AREA:
                    findings["disconnected_components"].append(
                        f"{fname} (id={img_id}): component {lbl}/{num_labels-1} "
                        f"has only {comp_area}px (threshold: {TINY_COMPONENT_AREA}px)"
                    )

    # Check 19: Annotation distribution
    if ann_counts:
        findings["annotation_distribution"] = {
            "min": min(ann_counts),
            "max": max(ann_counts),
            "avg": round(sum(ann_counts) / len(ann_counts), 2),
        }

    return findings


def check_corrupted_dir(dataset_path: Path):
    """Check files in the corrupted/ directory if it exists."""
    cdir = dataset_path / "corrupted"
    if not cdir.exists():
        return None

    results = []
    for f in sorted(cdir.iterdir()):
        try:
            im = Image.open(f)
            im.verify()
            im = Image.open(f)
            results.append((f.name, f"OK — {im.size}, {im.mode}"))
        except Exception as e:
            results.append((f.name, f"BROKEN — {e}"))
    return results


def format_report(all_findings: list, corrupted_dirs: dict) -> str:
    """Format all findings into a readable report."""
    lines = []
    lines.append("=" * 96)
    lines.append("MASK / ANNOTATION / IMAGE QUALITY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 96)

    # --- Summary table ---
    lines.append("")
    lines.append("SUMMARY TABLE")
    lines.append("-" * 120)
    header = (
        f"{'Dataset':<12} {'Split':<8} {'Images':>7} {'Anns':>7} "
        f"{'SzMis':>6} {'Empty':>6} {'MsCls':>6} {'Bleed':>6} "
        f"{'Tiny':>6} {'Disco':>6} {'BadID':>6} {'Degen':>6} "
        f"{'Crpt':>6} {'Orph':>6} {'Crowd':>6} {'ArMis':>6}"
    )
    lines.append(header)
    lines.append("-" * 120)

    for f in all_findings:
        if f is None:
            continue
        row = (
            f"{f['dataset']:<12} {f['split']:<8} {f['total_images']:>7} "
            f"{f['total_annotations']:>7} "
            f"{len(f['size_mismatch']):>6} "
            f"{len(f['empty_masks']):>6} "
            f"{len(f['missing_classes']):>6} "
            f"{len(f['annotation_bleed']):>6} "
            f"{len(f['tiny_annotations']):>6} "
            f"{len(f['disconnected_components']):>6} "
            f"{len(f['incorrect_labels']):>6} "
            f"{len(f['degenerate_polygons']):>6} "
            f"{len(f['corrupted']):>6} "
            f"{len(f['orphan_annotations']):>6} "
            f"{len(f['crowd_annotations']):>6} "
            f"{len(f['area_mismatches']):>6}"
        )
        lines.append(row)

    lines.append("-" * 120)
    lines.append("")
    lines.append("Column legend:")
    lines.append("  SzMis = Image-annotation size mismatch")
    lines.append("  Empty = Empty masks (no annotated pixels)")
    lines.append("  MsCls = Missing classes (expected category absent)")
    lines.append("  Bleed = Annotation bleed (extends beyond image boundary)")
    lines.append("  Tiny  = Tiny annotations (area < 100px from JSON area field)")
    lines.append("  Disco = Disconnected components (tiny islands < 100px in generated mask)")
    lines.append("  BadID = Incorrect label IDs (invalid category)")
    lines.append("  Degen = Degenerate polygons (<4 vertices)")
    lines.append("  Crpt  = Corrupted/unloadable images")
    lines.append("  Orph  = Orphan annotations (no matching image)")
    lines.append("  Crowd = Crowd annotations (iscrowd=1)")
    lines.append("  ArMis = Bbox area vs JSON area mismatch")

    # --- Detailed findings per split ---
    for f in all_findings:
        if f is None:
            continue

        lines.append("")
        lines.append("=" * 96)
        lines.append(f"DETAILS: {f['dataset'].upper()} / {f['split'].upper()}")
        lines.append(f"  Images: {f['total_images']}, Annotations: {f['total_annotations']}")
        lines.append(f"  Categories: {f['categories']}")
        lines.append("=" * 96)

        # File integrity
        lines.append("")
        lines.append("  FILE INTEGRITY")
        lines.append(f"    Missing on disk: {len(f['missing_on_disk'])}")
        for item in f["missing_on_disk"][:10]:
            lines.append(f"      - {item}")
        if len(f["missing_on_disk"]) > 10:
            lines.append(f"      ... and {len(f['missing_on_disk']) - 10} more")

        lines.append(f"    Extra on disk (no annotation): {len(f['extra_on_disk'])}")
        for item in f["extra_on_disk"][:10]:
            lines.append(f"      - {item}")
        if len(f["extra_on_disk"]) > 10:
            lines.append(f"      ... and {len(f['extra_on_disk']) - 10} more")

        lines.append(f"    Corrupted/unloadable: {len(f['corrupted'])}")
        for fname, err in f["corrupted"][:10]:
            lines.append(f"      - {fname}: {err}")

        lines.append(f"    Non-RGB mode: {len(f['wrong_mode'])}")
        for fname, mode in f["wrong_mode"][:10]:
            lines.append(f"      - {fname}: {mode}")

        # Resolution
        lines.append("")
        lines.append("  RESOLUTION")
        lines.append(f"    Unique sizes: {len(f['unique_resolutions'])}")
        for sz, cnt in f["unique_resolutions"].items():
            lines.append(f"      {sz}: {cnt}")
        lines.append(f"    Non-640x640: {f['non_640x640_count']}")

        # Category info
        lines.append("")
        lines.append("  CATEGORIES")
        for cat_name, cnt in f["category_distribution"].items():
            lines.append(f"    {cat_name}: {cnt} annotations")
        if f["unused_categories"]:
            lines.append(f"    Unused: {f['unused_categories']}")

        # Annotation distribution
        if f["annotation_distribution"]:
            dist = f["annotation_distribution"]
            lines.append("")
            lines.append("  ANNOTATION DISTRIBUTION")
            lines.append(f"    Min per image: {dist['min']}")
            lines.append(f"    Max per image: {dist['max']}")
            lines.append(f"    Avg per image: {dist['avg']}")

        # Segmentation info
        lines.append("")
        lines.append(f"  EMPTY SEGMENTATION FIELDS: {f['empty_segmentations_count']}/{f['total_annotations']}")

        # All check results
        checks = [
            ("Size Mismatch", f["size_mismatch"]),
            ("Empty Masks", f["empty_masks"]),
            ("Missing Classes", f["missing_classes"]),
            ("Annotation Bleed", f["annotation_bleed"]),
            ("Tiny Annotations (area < 100px)", f["tiny_annotations"]),
            ("Disconnected Components", f["disconnected_components"]),
            ("Incorrect Label IDs", f["incorrect_labels"]),
            ("Degenerate Polygons", f["degenerate_polygons"]),
            ("Multi-Polygon Annotations", f["multi_polygon"]),
            ("Orphan Annotations", f["orphan_annotations"]),
            ("Crowd Annotations", f["crowd_annotations"]),
            ("Bbox vs JSON Area Mismatch", f["area_mismatches"]),
        ]

        for check_name, issues in checks:
            lines.append("")
            status = "PASS" if not issues else f"FAIL ({len(issues)} issues)"
            lines.append(f"  {check_name}: {status}")
            if issues:
                for issue in issues[:20]:
                    lines.append(f"    - {issue}")
                if len(issues) > 20:
                    lines.append(f"    ... and {len(issues) - 20} more")

    # --- Corrupted directory checks ---
    for ds_name, results in corrupted_dirs.items():
        if results is None:
            continue
        lines.append("")
        lines.append("=" * 96)
        lines.append(f"CORRUPTED DIRECTORY: {ds_name}")
        lines.append("=" * 96)
        for fname, status in results:
            lines.append(f"  {fname}: {status}")

    lines.append("")
    lines.append("=" * 96)
    lines.append("END OF REPORT")
    lines.append("=" * 96)

    return "\n".join(lines)


def remove_tiny_annotations(dataset_name: str, dataset_cfg: dict):
    """Remove tiny annotations (area < SMALL_ANNOTATION_AREA) from the train split COCO JSON.

    Only modifies the train split — never touches valid/test.
    Writes the cleaned JSON back in place and prints a summary.
    """
    split = "train"
    ann_path = dataset_cfg["path"] / split / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"  [{dataset_name}] No train annotations file found, skipping.")
        return

    with open(ann_path) as f:
        coco = json.load(f)

    original_count = len(coco["annotations"])
    kept = []
    removed = []

    for ann in coco["annotations"]:
        bw, bh = ann["bbox"][2], ann["bbox"][3]
        ann_area = ann.get("area", 0)
        effective_area = ann_area if ann_area > 0 else bw * bh
        if 0 < effective_area < SMALL_ANNOTATION_AREA:
            removed.append(ann)
        else:
            kept.append(ann)

    if not removed:
        print(f"  [{dataset_name}/train] No tiny annotations found — nothing to remove.")
        return

    coco["annotations"] = kept

    # Remove images that now have zero annotations, keeping unannotated ones already present
    annotated_image_ids = {ann["image_id"] for ann in kept}
    removed_image_ids = {ann["image_id"] for ann in removed} - annotated_image_ids
    if removed_image_ids:
        coco["images"] = [img for img in coco["images"] if img["id"] not in removed_image_ids]

    with open(ann_path, "w") as f:
        json.dump(coco, f)

    print(
        f"  [{dataset_name}/train] Removed {len(removed)} tiny annotations "
        f"({original_count} → {len(kept)})."
    )
    if removed_image_ids:
        print(f"    Also dropped {len(removed_image_ids)} images that became annotation-free.")
    for ann in removed:
        print(
            f"    - Ann {ann['id']} (image_id {ann['image_id']}): "
            f"area={ann.get('area', 0):.1f}, bbox={ann['bbox']}"
        )


def remove_disconnected_annotations(dataset_name: str, dataset_cfg: dict):
    """Remove annotations whose pixels fall entirely inside tiny disconnected components.

    Strategy per image (train split only):
      1. Build the combined mask from all annotations.
      2. Run connected components — label every component < TINY_COMPONENT_AREA as "tiny".
      3. For each annotation, rasterise it individually.
         If ALL of its pixels overlap with tiny-component labels → remove it.
         Annotations that also contribute to large components are kept untouched.
    Writes the cleaned JSON back in place.
    """
    split = "train"
    ann_path = dataset_cfg["path"] / split / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"  [{dataset_name}] No train annotations file found, skipping.")
        return

    with open(ann_path) as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    has_seg = dataset_cfg["has_segmentation"]
    remove_ids: set = set()

    for img_id, anns in anns_by_image.items():
        img_info = id_to_img.get(img_id)
        if img_info is None:
            continue
        ih, iw = img_info["height"], img_info["width"]

        # Step 1: combined mask
        combined = np.zeros((ih, iw), dtype=np.uint8)
        for ann in anns:
            if has_seg and ann.get("segmentation"):
                m = segmentation_to_mask(ann["segmentation"], ih, iw)
            else:
                m = bbox_to_mask(ann["bbox"], ih, iw)
            combined = np.maximum(combined, m)

        if combined.sum() == 0:
            continue

        # Step 2: connected components — collect tiny label ids
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            combined, connectivity=8
        )
        tiny_label_ids = {
            lbl for lbl in range(1, num_labels)
            if stats[lbl, cv2.CC_STAT_AREA] < TINY_COMPONENT_AREA
        }
        if not tiny_label_ids:
            continue

        tiny_pixel_mask = np.isin(labels, list(tiny_label_ids))

        # Step 3: per-annotation overlap check
        for ann in anns:
            if has_seg and ann.get("segmentation"):
                ann_mask = segmentation_to_mask(ann["segmentation"], ih, iw)
            else:
                ann_mask = bbox_to_mask(ann["bbox"], ih, iw)

            ann_pixels = ann_mask > 0
            if ann_pixels.sum() == 0:
                continue

            # Remove only if every pixel of this annotation is inside a tiny component
            if (tiny_pixel_mask & ann_pixels).sum() == ann_pixels.sum():
                remove_ids.add(ann["id"])

    if not remove_ids:
        print(f"  [{dataset_name}/train] No disconnected-only annotations found — nothing to remove.")
        return

    original_count = len(coco["annotations"])
    removed = [a for a in coco["annotations"] if a["id"] in remove_ids]
    kept    = [a for a in coco["annotations"] if a["id"] not in remove_ids]
    coco["annotations"] = kept

    # Drop images that now have zero annotations
    annotated_image_ids = {ann["image_id"] for ann in kept}
    removed_image_ids   = {ann["image_id"] for ann in removed} - annotated_image_ids
    if removed_image_ids:
        coco["images"] = [img for img in coco["images"] if img["id"] not in removed_image_ids]

    with open(ann_path, "w") as f:
        json.dump(coco, f)

    print(
        f"  [{dataset_name}/train] Removed {len(removed)} disconnected-only annotations "
        f"({original_count} → {len(kept)})."
    )
    if removed_image_ids:
        print(f"    Also dropped {len(removed_image_ids)} images that became annotation-free.")
    for ann in removed:
        print(
            f"    - Ann {ann['id']} (image_id {ann['image_id']}): "
            f"area={ann.get('area', 0):.1f}, bbox={ann['bbox']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Consolidated quality checks for segmentation datasets"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(BASE_DIR / "reports" / "mask_quality_report_2.txt"),
        help="Output file for the report (default: reports/mask_quality_report.txt)",
    )
    parser.add_argument(
        "--dataset",
        choices=["drywall", "cracks", "all"],
        default="all",
        help="Which dataset to check (default: all)",
    )
    parser.add_argument(
        "--remove-tiny",
        action="store_true",
        help="Remove tiny annotations (area < SMALL_ANNOTATION_AREA) from train splits in place, then exit.",
    )
    parser.add_argument(
        "--remove-disconnected",
        action="store_true",
        help="Remove annotations whose pixels fall entirely inside tiny disconnected components, then exit.",
    )
    args = parser.parse_args()

    datasets_to_check = (
        list(DATASETS.keys()) if args.dataset == "all"
        else [args.dataset]
    )

    if args.remove_tiny:
        print(f"Removing tiny annotations (area < {SMALL_ANNOTATION_AREA}px) from train splits...")
        for ds_name in datasets_to_check:
            remove_tiny_annotations(ds_name, DATASETS[ds_name])
        print("Done.")
        return

    if args.remove_disconnected:
        print(f"Removing disconnected-only annotations (component < {TINY_COMPONENT_AREA}px) from train splits...")
        for ds_name in datasets_to_check:
            remove_disconnected_annotations(ds_name, DATASETS[ds_name])
        print("Done.")
        return

    all_findings = []
    corrupted_dirs = {}

    for ds_name in datasets_to_check:
        cfg = DATASETS[ds_name]
        print(f"\nChecking {ds_name}...")
        for split in cfg["splits"]:
            print(f"  Split: {split}...", end=" ", flush=True)
            findings = check_split(ds_name, cfg, split)
            if findings:
                all_findings.append(findings)
                total_issues = sum(
                    len(findings[k]) for k in [
                        "size_mismatch", "empty_masks", "missing_classes",
                        "annotation_bleed", "tiny_annotations",
                        "disconnected_components", "incorrect_labels",
                        "degenerate_polygons", "corrupted",
                        "orphan_annotations", "crowd_annotations",
                        "area_mismatches", "multi_polygon",
                    ]
                )
                print(f"{total_issues} issues found")
            else:
                print("skipped (no annotations file)")

        # Check corrupted directory
        corrupted_dirs[ds_name] = check_corrupted_dir(cfg["path"])

    # Generate report
    report = format_report(all_findings, corrupted_dirs)

    # Print to console
    print("\n" + report)

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
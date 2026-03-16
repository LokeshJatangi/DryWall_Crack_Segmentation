"""
Duplicate detection for drywall dataset — annotation-first approach.

Pipeline:
  1. Group by annotation count
  2. Within each group, find pairs with matching bboxes (primary signal)
  3. Run SSIM on bbox-matched pairs to confirm and measure visual diff
  4. Lower SSIM threshold for bbox-confirmed pairs (catches lighting variants)
  5. Report results; optionally remove

Usage:
    python src/data/dedup_drywall_v2.py --split train
    python src/data/dedup_drywall_v2.py --split train --remove
    python src/data/dedup_drywall_v2.py --split valid --bbox-tolerance 10
"""

import json
import argparse
from pathlib import Path
from itertools import combinations

import cv2
from skimage.metrics import structural_similarity as ssim


DATASET_DIR = Path("datasets/Drywall-Join-Detect.v2i.coco")
VIZ_DIR = Path("processed_data/verification/drywall_duplicates")

# Bbox-confirmed pairs: even low SSIM means same scene with diff augmentation
SSIM_THRESHOLD_BBOX_MATCH = 0.70
# No bbox match but visually identical (safety net)
SSIM_THRESHOLD_VISUAL_ONLY = 0.95


def load_coco(split: str):
    """Load COCO annotations and build lookup maps."""
    ann_path = DATASET_DIR / split / "_annotations.coco.json"
    with open(ann_path) as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}

    id_to_anns = {}
    for ann in coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    return coco, id_to_img, id_to_anns


def bboxes_match(anns_a: list, anns_b: list, tolerance: int = 5) -> bool:
    """Check if two annotation lists have matching bboxes within tolerance."""
    if len(anns_a) != len(anns_b):
        return False

    bboxes_a = sorted([tuple(a["bbox"]) for a in anns_a])
    bboxes_b = sorted([tuple(a["bbox"]) for a in anns_b])

    for ba, bb in zip(bboxes_a, bboxes_b):
        if any(abs(a - b) > tolerance for a, b in zip(ba, bb)):
            return False
    return True


def compute_ssim(img_path_a: str, img_path_b: str) -> float:
    """Compute SSIM between two images (grayscale)."""
    img_a = cv2.imread(img_path_a, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)

    if img_a is None or img_b is None:
        return 0.0
    if img_a.shape != img_b.shape:
        return 0.0

    score, _ = ssim(img_a, img_b, full=True)
    return score


def build_bbox_signature(anns: list) -> tuple:
    """
    Create a hashable bbox signature for an image's annotations.
    Sorted list of rounded bboxes for grouping.
    """
    bboxes = sorted([tuple(int(round(v)) for v in a["bbox"]) for a in anns])
    return tuple(bboxes)


def find_duplicates(split: str, bbox_tolerance: int = 5):
    """
    Find duplicates using annotation-first approach.

    Step 1: Group by annotation count
    Step 2: Find bbox-matched pairs (primary signal)
    Step 3: SSIM to confirm and measure visual difference
    """
    coco, id_to_img, id_to_anns = load_coco(split)

    # --- Step 1: Group by annotation count ---
    count_groups = {}
    for img_id in id_to_img:
        n = len(id_to_anns.get(img_id, []))
        count_groups.setdefault(n, []).append(img_id)

    print(f"Split: {split} ({len(id_to_img)} images)")
    for n, ids in sorted(count_groups.items()):
        print(f"  {n} annotation(s): {len(ids)} images")

    # --- Step 2: Find bbox-matched pairs within each count group ---
    bbox_matched_pairs = []
    bbox_unmatched_ids = {}  # for visual-only check later

    total_bbox_comparisons = 0

    for ann_count, img_ids in count_groups.items():
        if len(img_ids) < 2:
            continue

        if ann_count == 0:
            # No annotations — all are candidates, skip bbox matching
            bbox_unmatched_ids[ann_count] = img_ids
            continue

        # Sub-group by bbox signature (exact rounded coords) for fast matching
        sig_groups = {}
        for img_id in img_ids:
            sig = build_bbox_signature(id_to_anns[img_id])
            sig_groups.setdefault(sig, []).append(img_id)

        # Exact signature matches → confirmed bbox match
        for sig, group_ids in sig_groups.items():
            if len(group_ids) >= 2:
                for id_a, id_b in combinations(group_ids, 2):
                    bbox_matched_pairs.append((id_a, id_b))

        # Cross-signature: check with tolerance for near-matches
        sigs = list(sig_groups.keys())
        for i, j in combinations(range(len(sigs)), 2):
            sig_a, sig_b = sigs[i], sigs[j]
            if len(sig_a) != len(sig_b):
                continue
            # Check if all bboxes match within tolerance
            match = True
            for ba, bb in zip(sig_a, sig_b):
                if any(abs(a - b) > bbox_tolerance for a, b in zip(ba, bb)):
                    match = False
                    break
            if match:
                total_bbox_comparisons += 1
                for id_a in sig_groups[sig_a]:
                    for id_b in sig_groups[sig_b]:
                        bbox_matched_pairs.append((id_a, id_b))

    print(f"\nBbox-matched candidate pairs: {len(bbox_matched_pairs)}")

    # --- Step 3: SSIM on bbox-matched pairs ---
    duplicates = []
    ssim_comparisons = 0

    print("\nRunning SSIM on bbox-matched pairs...")
    for id_a, id_b in bbox_matched_pairs:
        path_a = str(DATASET_DIR / split / id_to_img[id_a]["file_name"])
        path_b = str(DATASET_DIR / split / id_to_img[id_b]["file_name"])
        score = compute_ssim(path_a, path_b)
        ssim_comparisons += 1

        if score >= SSIM_THRESHOLD_BBOX_MATCH:
            duplicates.append({
                "id_a": id_a,
                "id_b": id_b,
                "file_a": id_to_img[id_a]["file_name"],
                "file_b": id_to_img[id_b]["file_name"],
                "ssim": score,
                "match_type": "bbox+ssim",
            })

    # --- Step 4: Visual-only check on no-annotation images ---
    for ann_count, img_ids in bbox_unmatched_ids.items():
        if len(img_ids) < 2:
            continue
        for id_a, id_b in combinations(img_ids, 2):
            path_a = str(DATASET_DIR / split / id_to_img[id_a]["file_name"])
            path_b = str(DATASET_DIR / split / id_to_img[id_b]["file_name"])
            score = compute_ssim(path_a, path_b)
            ssim_comparisons += 1
            if score >= SSIM_THRESHOLD_VISUAL_ONLY:
                duplicates.append({
                    "id_a": id_a,
                    "id_b": id_b,
                    "file_a": id_to_img[id_a]["file_name"],
                    "file_b": id_to_img[id_b]["file_name"],
                    "ssim": score,
                    "match_type": "visual-only",
                })

    # --- Report ---
    print(f"Total SSIM comparisons: {ssim_comparisons}")
    print(f"\n{'='*80}")
    print(f"DUPLICATES FOUND: {len(duplicates)}")
    print(f"{'='*80}")

    if duplicates:
        for dup in duplicates:
            print(f"\n  [{dup['match_type']}] SSIM={dup['ssim']:.4f}")
            print(f"    A: id={dup['id_a']} {dup['file_a']}")
            print(f"    B: id={dup['id_b']} {dup['file_b']}")
    else:
        print("  No duplicates found.")

    return coco, id_to_img, id_to_anns, duplicates


def visualize_duplicates(split: str, id_to_img, id_to_anns, duplicates):
    """
    Create stitched comparison images for each duplicate pair.
    Marks each image as KEEP (green) or REMOVE (red) based on duplicate groups.

    Layout per pair:
        [Image A + KEEP/REMOVE] [Image B + KEEP/REMOVE] [Abs Diff (amplified)]
        [A + BBoxes]            [B + BBoxes]             [Info panel]
    """
    import numpy as np

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX

    COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 165, 0), (0, 128, 255),
    ]

    # Build groups to determine keep/remove
    groups = build_duplicate_groups(duplicates)
    keep_ids = set()
    remove_ids = set()
    for _, members in groups.items():
        keep = min(members)
        keep_ids.add(keep)
        remove_ids.update(members - {keep})

    def draw_keep_remove_badge(img, img_id):
        """Draw a KEEP or REMOVE badge on the image."""
        if img_id in keep_ids:
            label, bg_color = "KEEP", (0, 140, 0)
        elif img_id in remove_ids:
            label, bg_color = "REMOVE", (0, 0, 200)
        else:
            return
        text_size = cv2.getTextSize(label, font, 1.0, 3)[0]
        pad = 8
        tw, th = text_size
        # Top-right corner
        x1 = img.shape[1] - tw - pad * 2
        y1 = 0
        cv2.rectangle(img, (x1, y1), (x1 + tw + pad * 2, th + pad * 2), bg_color, -1)
        cv2.putText(img, label, (x1 + pad, th + pad), font, 1.0, (255, 255, 255), 3)

    for idx, dup in enumerate(duplicates):
        path_a = str(DATASET_DIR / split / dup["file_a"])
        path_b = str(DATASET_DIR / split / dup["file_b"])
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)

        if img_a is None or img_b is None:
            continue

        h, w = img_a.shape[:2]

        # --- Row 1: Image A | Image B | Abs Diff ---
        diff = cv2.absdiff(img_a, img_b)
        diff_amplified = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

        # Draw KEEP/REMOVE badges
        draw_keep_remove_badge(img_a, dup["id_a"])
        draw_keep_remove_badge(img_b, dup["id_b"])

        # --- Row 2: A+BBoxes | B+BBoxes | Info panel ---
        bbox_a = img_a.copy()
        anns_a = id_to_anns.get(dup["id_a"], [])
        for i, ann in enumerate(anns_a):
            bx, by, bw, bh = [int(v) for v in ann["bbox"]]
            color = COLORS[i % len(COLORS)]
            cv2.rectangle(bbox_a, (bx, by), (bx + bw, by + bh), color, 2)

        bbox_b = img_b.copy()
        anns_b = id_to_anns.get(dup["id_b"], [])
        for i, ann in enumerate(anns_b):
            bx, by, bw, bh = [int(v) for v in ann["bbox"]]
            color = COLORS[i % len(COLORS)]
            cv2.rectangle(bbox_b, (bx, by), (bx + bw, by + bh), color, 2)

        # Info panel
        info = np.zeros((h, w, 3), dtype=np.uint8)

        # Find which group these belong to
        group_members = set()
        for _, members in groups.items():
            if dup["id_a"] in members:
                group_members = members
                break

        lines = [
            f"Pair #{idx + 1}",
            f"Match: {dup['match_type']}",
            f"SSIM: {dup['ssim']:.4f}",
            f"",
            f"Group: {sorted(group_members)}",
            f"  KEEP:   id={min(group_members)}",
            f"  REMOVE: {sorted(group_members - {min(group_members)})}",
            f"",
            f"Image A (id={dup['id_a']}):",
            f"  {dup['file_a'][:40]}",
            f"",
            f"Image B (id={dup['id_b']}):",
            f"  {dup['file_b'][:40]}",
            f"",
            f"Diff mean: {diff.mean():.1f}",
            f"Diff max:  {diff.max()}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(info, line, (15, 25 + i * 25), font, 0.50, (255, 255, 255), 1)

        # --- Labels ---
        labels_top = [("Image A", img_a), ("Image B", img_b), ("Abs Diff (amplified)", diff_amplified)]
        for label, panel_img in labels_top:
            cv2.putText(panel_img, label, (10, 25), font, 0.7, (0, 255, 255), 2)

        cv2.putText(bbox_a, "A + BBoxes", (10, 25), font, 0.7, (0, 255, 255), 2)
        cv2.putText(bbox_b, "B + BBoxes", (10, 25), font, 0.7, (0, 255, 255), 2)

        # --- Stitch ---
        row1 = np.hstack([img_a, img_b, diff_amplified])
        row2 = np.hstack([bbox_a, bbox_b, info])
        stitched = np.vstack([row1, row2])

        out_path = VIZ_DIR / f"dup_{split}_{idx:03d}_ssim{dup['ssim']:.3f}.png"
        cv2.imwrite(str(out_path), stitched)
        print(f"  Saved: {out_path.name}")

    print(f"\nSaved {len(duplicates)} stitched comparisons to {VIZ_DIR}")
    print(f"  Total groups: {len(groups)}, KEEP: {len(keep_ids)}, REMOVE: {len(remove_ids)}")


def build_duplicate_groups(duplicates):
    """
    Build connected components from duplicate pairs.

    E.g., pairs (12,552), (12,359), (552,359) → one group {12, 552, 359}.
    Keep the lowest id in each group, remove the rest.
    """
    # Union-Find
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the smaller id as root
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    for dup in duplicates:
        union(dup["id_a"], dup["id_b"])

    # Collect groups
    groups = {}
    for node in parent:
        root = find(node)
        groups.setdefault(root, set()).add(node)

    return groups


def remove_duplicates(split: str, coco, id_to_img, duplicates):
    """Remove duplicates using connected components — keep one per group."""
    groups = build_duplicate_groups(duplicates)

    # Keep the smallest id in each group, remove the rest
    ids_to_remove = set()
    for root, members in groups.items():
        keep = min(members)
        remove = members - {keep}
        ids_to_remove.update(remove)

    if not ids_to_remove:
        print("No duplicates to remove.")
        return

    # Report groups
    print(f"\nDuplicate groups: {len(groups)}")
    for root, members in groups.items():
        keep = min(members)
        remove = sorted(members - {keep})
        keep_name = id_to_img[keep]["file_name"]
        print(f"  Group (keep id={keep} {keep_name[:50]}):")
        for rid in remove:
            print(f"    remove id={rid} {id_to_img[rid]['file_name'][:50]}")

    print(f"\nRemoving {len(ids_to_remove)} duplicate images (from {len(groups)} groups)...")

    # Update COCO JSON
    before_imgs = len(coco["images"])
    before_anns = len(coco["annotations"])
    coco["images"] = [img for img in coco["images"] if img["id"] not in ids_to_remove]
    coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] not in ids_to_remove]

    # Delete files
    for img_id in ids_to_remove:
        img_path = DATASET_DIR / split / id_to_img[img_id]["file_name"]
        if img_path.exists():
            img_path.unlink()
            print(f"  Deleted: {img_path.name}")

    # Save updated JSON
    ann_path = DATASET_DIR / split / "_annotations.coco.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f)
    print(f"  Updated: {ann_path}")

    print(f"\n  Before: {before_imgs} images, {before_anns} annotations")
    print(f"  After:  {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"  Removed: {before_imgs - len(coco['images'])} images, {before_anns - len(coco['annotations'])} annotations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotation-first duplicate detection for drywall dataset")
    parser.add_argument("--split", default="train", choices=["train", "valid"])
    parser.add_argument("--bbox-tolerance", type=int, default=5, help="Bbox match tolerance in pixels (default: 5)")
    parser.add_argument("--remove", action="store_true", help="Remove duplicates (default: dry-run)")
    args = parser.parse_args()

    coco, id_to_img, id_to_anns, duplicates = find_duplicates(args.split, args.bbox_tolerance)

    if duplicates:
        print(f"\n{'='*80}")
        print("GENERATING STITCHED VISUALIZATIONS")
        print(f"{'='*80}")
        visualize_duplicates(args.split, id_to_img, id_to_anns, duplicates)

    if args.remove and duplicates:
        print(f"\n{'='*80}")
        print("REMOVING DUPLICATES")
        print(f"{'='*80}")
        remove_duplicates(args.split, coco, id_to_img, duplicates)
    elif duplicates:
        print("\nDry-run mode. Use --remove to delete duplicates.")
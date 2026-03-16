"""
Duplicate detection for cracks dataset — Roboflow augmentation-aware approach.

Roboflow generates multiple augmented variants of each source image, naming them:
    {source_name}.rf.{hash}.jpg

All variants sharing the same {source_name} are augmentations (rotations,
flips, photometric changes) of the same original image.

Pipeline:
  1. Group images by base name (source_name before .rf.)  [primary signal]
  2. Build a graph: nodes = image IDs, edges = same base-name group
  3. Connected components → duplicate groups
  4. Keep one canonical image per group (first alphabetically)
  5. Move duplicates → {split}/duplicates/ subfolder
  6. Update COCO JSON (remove moved images + their annotations)

Why not SSIM?
  Roboflow augmentations include photometric transforms (brightness, contrast,
  HSV jitter) in addition to rotations, which drop SSIM to ~0.23 — below any
  reliable threshold. Base-name grouping is deterministic and 100% accurate.

Usage:
    python src/data/dedup_cracks.py --split train
    python src/data/dedup_cracks.py --split train --move
    python src/data/dedup_cracks.py --split train --move --viz
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


DATASET_DIR = Path("datasets/cracks.v1i.coco")
VIZ_DIR = Path("processed_data/verification/cracks_duplicates")

# Max groups to visualize (groups sorted by size descending)
VIZ_MAX_GROUPS = 50
# Max variants to show per group in the viz grid
VIZ_MAX_PER_GROUP = 9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_name(file_name: str) -> str:
    """Extract source name: everything before the last '.rf.' segment."""
    if ".rf." in file_name:
        return file_name.split(".rf.")[0]
    return file_name  # no .rf. → treat as its own group


def load_coco(split: str):
    ann_path = DATASET_DIR / split / "_annotations.coco.json"
    with open(ann_path) as f:
        coco = json.load(f)
    id_to_img = {img["id"]: img for img in coco["images"]}
    id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        id_to_anns[ann["image_id"]].append(ann)
    return coco, id_to_img, dict(id_to_anns)


# ---------------------------------------------------------------------------
# Union-Find (connected components)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self._parent = {}

    def find(self, x):
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Keep smaller ID as root for determinism
        if ra > rb:
            ra, rb = rb, ra
        self._parent[rb] = ra

    def groups(self):
        """Return {root: set(members)} for all nodes."""
        result = defaultdict(set)
        for node in self._parent:
            result[self.find(node)].add(node)
        return dict(result)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_duplicate_groups(split: str):
    """
    Group images by Roboflow base name, then build connected components.

    Returns:
        coco, id_to_img, id_to_anns, groups
        groups: {root_id: set(all_ids_in_group)}  — only groups with size > 1
    """
    coco, id_to_img, id_to_anns = load_coco(split)

    # Map: base_name → list of image IDs sharing that source
    base_to_ids = defaultdict(list)
    for img in coco["images"]:
        b = base_name(img["file_name"])
        base_to_ids[b].append(img["id"])

    uf = UnionFind()

    # Add all image IDs as nodes first
    for img in coco["images"]:
        uf.find(img["id"])

    # Connect all IDs that share the same base name
    for b, ids in base_to_ids.items():
        if len(ids) < 2:
            continue
        anchor = ids[0]
        for other in ids[1:]:
            uf.union(anchor, other)

    all_groups = uf.groups()
    dup_groups = {root: members for root, members in all_groups.items() if len(members) > 1}

    # Stats
    total = len(coco["images"])
    in_groups = sum(len(m) for m in dup_groups.values())
    will_remove = sum(len(m) - 1 for m in dup_groups.values())

    print(f"\nSplit: {split} ({total} images, {len(coco['annotations'])} annotations)")
    print(f"  Unique source images (base names): {len(base_to_ids)}")
    print(f"  Duplicate groups (size > 1):       {len(dup_groups)}")
    print(f"  Images in duplicate groups:        {in_groups}")
    print(f"  Will KEEP (one per group):          {len(dup_groups)}")
    print(f"  Will MOVE to duplicates/:           {will_remove}")

    return coco, id_to_img, id_to_anns, dup_groups


def select_canonical(group_ids: set, id_to_img: dict) -> int:
    """
    Pick the canonical (kept) image from a duplicate group.
    Strategy: alphabetically first file_name → deterministic, reproducible.
    """
    return min(group_ids, key=lambda img_id: id_to_img[img_id]["file_name"])


def move_duplicates(split: str, coco, id_to_img, id_to_anns, dup_groups):
    """
    For each duplicate group:
      - keep canonical (alphabetically first file_name)
      - move others to {split}/duplicates/
    Then update and overwrite the COCO JSON.
    """
    dup_dir = DATASET_DIR / split / "duplicates"
    dup_dir.mkdir(exist_ok=True)

    ids_to_move = set()
    for root, members in dup_groups.items():
        canonical = select_canonical(members, id_to_img)
        ids_to_move.update(members - {canonical})

    if not ids_to_move:
        print("No duplicates to move.")
        return

    print(f"\nMoving {len(ids_to_move)} images to {dup_dir} ...")

    moved = 0
    skipped = 0
    for img_id in sorted(ids_to_move):
        src = DATASET_DIR / split / id_to_img[img_id]["file_name"]
        dst = dup_dir / id_to_img[img_id]["file_name"]
        if src.exists():
            shutil.move(str(src), str(dst))
            moved += 1
        else:
            print(f"  [WARN] not found: {src.name}")
            skipped += 1

    print(f"  Moved: {moved}  |  Not found: {skipped}")

    # Update COCO JSON
    before_imgs = len(coco["images"])
    before_anns = len(coco["annotations"])
    coco["images"] = [img for img in coco["images"] if img["id"] not in ids_to_move]
    coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] not in ids_to_move]

    ann_path = DATASET_DIR / split / "_annotations.coco.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    print(f"\n  COCO JSON updated: {ann_path.name}")
    print(f"  Images:      {before_imgs} → {len(coco['images'])}  (removed {before_imgs - len(coco['images'])})")
    print(f"  Annotations: {before_anns} → {len(coco['annotations'])}  (removed {before_anns - len(coco['annotations'])})")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_groups(split: str, id_to_img, id_to_anns, dup_groups):
    """
    For the top VIZ_MAX_GROUPS largest groups, save a grid image showing
    all variants side-by-side with their file names and annotation counts.

    Layout per group:
        Row 1: up to VIZ_MAX_PER_GROUP variants (colour-coded KEEP / MOVE)
        Row 2: same images with segmentation polygons drawn
    """
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX

    KEEP_COLOR  = (0, 200, 0)    # green border  = canonical
    MOVE_COLOR  = (0, 0, 220)    # red border    = will be moved
    POLY_COLORS = [
        (0, 255, 255), (255, 0, 255), (255, 165, 0),
        (0, 128, 255), (128, 0, 255), (255, 128, 0),
    ]

    THUMB = 320   # thumbnail size (square)

    # Sort groups largest → smallest
    sorted_groups = sorted(dup_groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    sorted_groups = sorted_groups[:VIZ_MAX_GROUPS]

    print(f"\nGenerating visualizations for {len(sorted_groups)} groups → {VIZ_DIR}")

    for g_idx, (root, members) in enumerate(sorted_groups):
        canonical = select_canonical(members, id_to_img)
        # Sorted so canonical is first
        ordered = sorted(members, key=lambda i: (i != canonical, id_to_img[i]["file_name"]))
        sample   = ordered[:VIZ_MAX_PER_GROUP]
        n        = len(sample)

        row_raw  = []
        row_poly = []

        for img_id in sample:
            img_path = str(DATASET_DIR / split / id_to_img[img_id]["file_name"])
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (THUMB, THUMB))

            is_keep = (img_id == canonical)
            border_color = KEEP_COLOR if is_keep else MOVE_COLOR
            label = "KEEP" if is_keep else "MOVE"

            # --- Raw panel ---
            raw = img.copy()
            cv2.rectangle(raw, (0, 0), (THUMB - 1, THUMB - 1), border_color, 6)
            cv2.putText(raw, label, (8, 28), font, 0.8, border_color, 2)
            n_anns = len(id_to_anns.get(img_id, []))
            cv2.putText(raw, f"id={img_id}", (8, THUMB - 30), font, 0.45, (200, 200, 200), 1)
            cv2.putText(raw, f"anns={n_anns}", (8, THUMB - 12), font, 0.45, (200, 200, 200), 1)
            row_raw.append(raw)

            # --- Polygon panel ---
            poly = img.copy()
            cv2.rectangle(poly, (0, 0), (THUMB - 1, THUMB - 1), border_color, 6)
            orig_h = id_to_img[img_id].get("height", THUMB)
            orig_w = id_to_img[img_id].get("width",  THUMB)
            sx = THUMB / orig_w
            sy = THUMB / orig_h
            for a_idx, ann in enumerate(id_to_anns.get(img_id, [])):
                color = POLY_COLORS[a_idx % len(POLY_COLORS)]
                for seg in ann.get("segmentation", []):
                    pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    pts[:, 0] *= sx
                    pts[:, 1] *= sy
                    pts = pts.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(poly, [pts], isClosed=True, color=color, thickness=2)
                # bbox
                bx, by, bw, bh = ann["bbox"]
                cv2.rectangle(poly,
                               (int(bx * sx), int(by * sy)),
                               (int((bx + bw) * sx), int((by + bh) * sy)),
                               color, 1)
            row_poly.append(poly)

        # Pad to equal widths
        while len(row_raw)  < n: row_raw.append(np.zeros((THUMB, THUMB, 3), np.uint8))
        while len(row_poly) < n: row_poly.append(np.zeros((THUMB, THUMB, 3), np.uint8))

        grid_top = np.hstack(row_raw)
        grid_bot = np.hstack(row_poly)

        # Group header bar
        header_h = 40
        total_w = THUMB * n
        header = np.zeros((header_h, total_w, 3), np.uint8)
        more = f"  (+{len(members) - n} more)" if len(members) > n else ""
        cv2.putText(header,
                    f"Group #{g_idx + 1}  |  {len(members)} variants{more}  |  "
                    f"base: {id_to_img[canonical]['file_name'].split('.rf.')[0][:60]}",
                    (8, 27), font, 0.5, (0, 220, 220), 1)

        grid = np.vstack([header, grid_top, grid_bot])

        out_path = VIZ_DIR / f"group_{g_idx + 1:04d}_n{len(members):03d}.png"
        cv2.imwrite(str(out_path), grid)

    print(f"  Saved {len(sorted_groups)} grid images to {VIZ_DIR}")


# ---------------------------------------------------------------------------
# Post-move visualization (reconstruct groups from filesystem)
# ---------------------------------------------------------------------------

def visualize_from_disk(split: str):
    """
    Reconstruct duplicate groups from the current filesystem state:
      - canonicals live in  {split}/
      - duplicates live in  {split}/duplicates/
    Groups are matched by base name (before .rf.).

    Produces the same KEEP/MOVE grid as visualize_groups(), usable AFTER
    --move has been run (when the COCO JSON only contains canonicals).
    """
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX

    split_dir = DATASET_DIR / split
    dup_dir   = DATASET_DIR / split / "duplicates"

    if not dup_dir.exists():
        print(f"[WARN] duplicates/ not found at {dup_dir}. Run --move first.")
        return

    KEEP_COLOR  = (0, 200, 0)
    MOVE_COLOR  = (0, 0, 220)
    POLY_COLORS = [
        (0, 255, 255), (255, 0, 255), (255, 165, 0),
        (0, 128, 255), (128, 0, 255), (255, 128, 0),
    ]
    THUMB = 320

    # Load COCO for annotation lookup (canonical images only after dedup)
    coco, id_to_img, id_to_anns = load_coco(split)
    fname_to_id = {img["file_name"]: img["id"] for img in coco["images"]}

    # Build groups: base → {"canonical": path, "dups": [path, ...]}
    base_to_canonical: dict[str, Path] = {}
    for f in sorted(split_dir.glob("*.jpg")):
        b = base_name(f.name)
        base_to_canonical[b] = f

    base_to_dups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(dup_dir.glob("*.jpg")):
        b = base_name(f.name)
        base_to_dups[b].append(f)

    # Only groups that have at least one dup
    paired = [(b, base_to_canonical[b], base_to_dups[b])
              for b in base_to_canonical if b in base_to_dups]

    # Sort largest → smallest
    paired.sort(key=lambda t: -len(t[2]))
    sample = paired[:VIZ_MAX_GROUPS]

    print(f"\nGenerating post-move visualizations for {len(sample)} groups → {VIZ_DIR}")

    for g_idx, (b, canon_path, dup_paths) in enumerate(sample):
        all_paths = [canon_path] + dup_paths[:VIZ_MAX_PER_GROUP - 1]
        n = len(all_paths)

        row_raw  = []
        row_poly = []

        for p_idx, img_path in enumerate(all_paths):
            is_keep = (p_idx == 0)
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (THUMB, THUMB))

            border_color = KEEP_COLOR if is_keep else MOVE_COLOR
            label = "KEEP" if is_keep else "MOVED"

            # --- Raw panel ---
            raw = img.copy()
            cv2.rectangle(raw, (0, 0), (THUMB - 1, THUMB - 1), border_color, 6)
            cv2.putText(raw, label, (8, 28), font, 0.8, border_color, 2)
            cv2.putText(raw, img_path.name[:40], (8, THUMB - 30), font, 0.38, (200, 200, 200), 1)

            # annotation count (only available for canonicals in JSON)
            img_id = fname_to_id.get(img_path.name)
            n_anns = len(id_to_anns.get(img_id, [])) if img_id is not None else "–"
            cv2.putText(raw, f"anns={n_anns}", (8, THUMB - 12), font, 0.45, (200, 200, 200), 1)
            row_raw.append(raw)

            # --- Polygon panel ---
            poly = img.copy()
            cv2.rectangle(poly, (0, 0), (THUMB - 1, THUMB - 1), border_color, 6)
            if img_id is not None:
                orig_h = id_to_img[img_id].get("height", THUMB)
                orig_w = id_to_img[img_id].get("width",  THUMB)
                sx = THUMB / orig_w
                sy = THUMB / orig_h
                for a_idx, ann in enumerate(id_to_anns.get(img_id, [])):
                    color = POLY_COLORS[a_idx % len(POLY_COLORS)]
                    for seg in ann.get("segmentation", []):
                        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                        pts[:, 0] *= sx
                        pts[:, 1] *= sy
                        pts = pts.astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(poly, [pts], isClosed=True, color=color, thickness=2)
                    bx, by, bw, bh = ann["bbox"]
                    cv2.rectangle(poly,
                                   (int(bx * sx), int(by * sy)),
                                   (int((bx + bw) * sx), int((by + bh) * sy)),
                                   color, 1)
            else:
                cv2.putText(poly, "no annotation\n(moved)", (8, THUMB // 2),
                            font, 0.6, (100, 100, 255), 1)
            row_poly.append(poly)

        grid_top = np.hstack(row_raw)
        grid_bot = np.hstack(row_poly)

        total_w  = THUMB * n
        header_h = 40
        header   = np.zeros((header_h, total_w, 3), np.uint8)
        more     = f"  (+{len(dup_paths) - (n - 1)} more dups)" if len(dup_paths) >= n else ""
        cv2.putText(header,
                    f"Group #{g_idx + 1}  |  1 canonical + {len(dup_paths)} moved{more}  |  base: {b[:55]}",
                    (8, 27), font, 0.5, (0, 220, 220), 1)

        grid = np.vstack([header, grid_top, grid_bot])
        out_path = VIZ_DIR / f"group_{g_idx + 1:04d}_n{len(dup_paths) + 1:03d}.png"
        cv2.imwrite(str(out_path), grid)

    print(f"  Saved {len(sample)} grid images to {VIZ_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Roboflow augmentation-aware duplicate removal for cracks dataset"
    )
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"],
                        help="Dataset split to process (default: train)")
    parser.add_argument("--move", action="store_true",
                        help="Move duplicates to {split}/duplicates/ (default: dry-run)")
    parser.add_argument("--viz", action="store_true",
                        help="Generate group visualizations (pre-move, from COCO JSON)")
    parser.add_argument("--viz-post", action="store_true",
                        help="Generate group visualizations from disk (post-move)")
    args = parser.parse_args()

    if args.viz_post:
        # Works independently of COCO JSON state — reads from filesystem
        print(f"\n{'='*80}")
        print("GENERATING POST-MOVE VISUALIZATIONS (from filesystem)")
        print(f"{'='*80}")
        visualize_from_disk(args.split)
    else:
        coco, id_to_img, id_to_anns, dup_groups = find_duplicate_groups(args.split)

        if args.viz:
            print(f"\n{'='*80}")
            print("GENERATING VISUALIZATIONS")
            print(f"{'='*80}")
            visualize_groups(args.split, id_to_img, id_to_anns, dup_groups)

        if args.move:
            print(f"\n{'='*80}")
            print("MOVING DUPLICATES")
            print(f"{'='*80}")
            move_duplicates(args.split, coco, id_to_img, id_to_anns, dup_groups)
        else:
            print(f"\nDry-run mode — no files moved. Use --move to move duplicates.")
            print("  Add --viz to generate visualizations (pre-move).")
            print("  Use --viz-post to visualize current filesystem state (post-move).")

            print(f"\nSample groups (top 5 by size):")
            for root, members in sorted(dup_groups.items(), key=lambda kv: -len(kv[1]))[:5]:
                canonical = select_canonical(members, id_to_img)
                print(f"  [{len(members)} variants]  KEEP: {id_to_img[canonical]['file_name']}")
                for img_id in sorted(members - {canonical})[:3]:
                    print(f"    MOVE: {id_to_img[img_id]['file_name']}")
                if len(members) > 4:
                    print(f"    ... +{len(members) - 4} more")

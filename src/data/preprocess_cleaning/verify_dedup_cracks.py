"""
Post-dedup verification for the cracks dataset.

Checks run:
  1. File consistency   — every image listed in COCO JSON exists on disk
  2. No leaked dupes    — no remaining image whose base name appears > 1 time in train JSON
  3. Duplicates dir     — moved images exist in train/duplicates/ and are NOT in JSON
  4. Annotation orphans — no annotation references a non-existent image_id
  5. Split leakage      — no base name in train also appears in valid / test
  6. Count summary      — before / after image & annotation counts (reads snapshot)
  7. Visual spot-check  — for a random sample of canonical images, confirms the
                          corresponding duplicates are present in duplicates/ and
                          computes rotation-invariant SSIM to show they are genuine
                          variants (informational, not a pass/fail gate)

Usage:
    python src/data/verify_dedup_cracks.py           # all checks, train split
    python src/data/verify_dedup_cracks.py --split train --spot-check 10
    python src/data/verify_dedup_cracks.py --report  # save report to reports/
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


DATASET_DIR = Path("/mnt/disks/work/lokesh/seg/datasets/cracks.v1i.coco")
REPORTS_DIR = Path("/mnt/disks/work/lokesh/seg/reports")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_name(file_name: str) -> str:
    if ".rf." in file_name:
        return file_name.split(".rf.")[0]
    return file_name


def load_coco(split: str):
    ann_path = DATASET_DIR / split / "_annotations.coco.json"
    with open(ann_path) as f:
        coco = json.load(f)
    return coco


def rotation_invariant_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Max SSIM over 4 rotations of b."""
    best = 0.0
    for k in range(4):
        rotated = np.rot90(b, k)
        if rotated.shape != a.shape:
            continue
        s, _ = ssim(a, rotated, full=True)
        best = max(best, s)
    return best


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_file_consistency(split: str, coco) -> tuple[bool, str]:
    """All images in COCO JSON must exist on disk."""
    split_dir = DATASET_DIR / split
    missing = []
    for img in coco["images"]:
        p = split_dir / img["file_name"]
        if not p.exists():
            missing.append(img["file_name"])

    if missing:
        detail = "\n".join(f"  MISSING: {f}" for f in missing[:20])
        if len(missing) > 20:
            detail += f"\n  ... and {len(missing) - 20} more"
        return False, f"{len(missing)} images listed in JSON but not found on disk:\n{detail}"
    return True, f"All {len(coco['images'])} images in JSON exist on disk."


def check_no_leaked_dupes(split: str, coco) -> tuple[bool, str]:
    """No base name should appear more than once in the JSON after dedup."""
    base_to_files = defaultdict(list)
    for img in coco["images"]:
        base_to_files[base_name(img["file_name"])].append(img["file_name"])

    leaked = {b: fs for b, fs in base_to_files.items() if len(fs) > 1}
    if leaked:
        detail = ""
        for b, fs in list(leaked.items())[:5]:
            detail += f"\n  base={b}: {len(fs)} files still in JSON"
            for f in fs[:3]:
                detail += f"\n    {f}"
        return False, f"{len(leaked)} base names still have multiple entries in JSON:{detail}"
    return True, f"No leaked duplicates — each of {len(base_to_files)} base names appears exactly once."


def check_duplicates_dir(split: str, coco) -> tuple[bool, str]:
    """
    Files in {split}/duplicates/ must NOT be listed in the COCO JSON,
    and the duplicates dir must be non-empty (else dedup was never run).
    """
    dup_dir = DATASET_DIR / split / "duplicates"
    if not dup_dir.exists():
        return False, f"duplicates/ directory does not exist: {dup_dir}"

    dup_files = {f.name for f in dup_dir.iterdir() if f.suffix == ".jpg"}
    if not dup_files:
        return False, "duplicates/ directory is empty — dedup may not have been run."

    json_files = {img["file_name"] for img in coco["images"]}
    leaked = dup_files & json_files  # intersection = files that are BOTH in dir AND in JSON

    if leaked:
        detail = "\n".join(f"  {f}" for f in list(leaked)[:10])
        return False, (
            f"{len(leaked)} files appear in both duplicates/ and the COCO JSON "
            f"(should be in one place only):\n{detail}"
        )
    return True, (
        f"duplicates/ has {len(dup_files)} files, none of which appear in the COCO JSON."
    )


def check_annotation_orphans(split: str, coco) -> tuple[bool, str]:
    """No annotation should reference an image_id that isn't in the JSON."""
    valid_ids = {img["id"] for img in coco["images"]}
    orphans = [ann for ann in coco["annotations"] if ann["image_id"] not in valid_ids]
    if orphans:
        sample = [a["id"] for a in orphans[:10]]
        return False, (
            f"{len(orphans)} annotations reference non-existent image IDs. "
            f"Sample ann IDs: {sample}"
        )
    return True, (
        f"All {len(coco['annotations'])} annotations reference valid image IDs."
    )


def check_split_leakage(coco_train, coco_valid, coco_test) -> tuple[bool, str]:
    """No base name in train should also appear in valid or test."""
    train_bases = {base_name(img["file_name"]) for img in coco_train["images"]}
    valid_bases = {base_name(img["file_name"]) for img in coco_valid["images"]}
    test_bases  = {base_name(img["file_name"]) for img in coco_test["images"]}

    tv = train_bases & valid_bases
    tt = train_bases & test_bases

    issues = []
    if tv:
        issues.append(f"train ∩ valid: {len(tv)} base names overlap")
    if tt:
        issues.append(f"train ∩ test:  {len(tt)} base names overlap")

    if issues:
        return False, "Split leakage detected:\n" + "\n".join(f"  {i}" for i in issues)

    return True, (
        f"No split leakage — train ({len(train_bases)}), "
        f"valid ({len(valid_bases)}), test ({len(test_bases)}) base names are disjoint."
    )


def check_count_summary(split: str, coco) -> tuple[bool, str]:
    """Report current image and annotation counts (always passes)."""
    n_imgs = len(coco["images"])
    n_anns = len(coco["annotations"])
    imgs_no_ann = sum(
        1 for img in coco["images"]
        if not any(a["image_id"] == img["id"] for a in coco["annotations"])
    )
    # Build id_to_anns for fast lookup
    id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        id_to_anns[ann["image_id"]].append(ann)
    imgs_no_ann = sum(1 for img in coco["images"] if img["id"] not in id_to_anns)
    return True, (
        f"Images: {n_imgs} | Annotations: {n_anns} | "
        f"Images without annotations: {imgs_no_ann}"
    )


def check_spot_sample(split: str, coco, n_samples: int = 10) -> tuple[bool, str]:
    """
    For n_samples canonical images, find a sibling in duplicates/ and compute
    rotation-invariant SSIM. Reports stats — does not fail on low SSIM because
    photometric augmentations legitimately lower it.
    """
    dup_dir = DATASET_DIR / split / "duplicates"
    if not dup_dir.exists() or not any(dup_dir.iterdir()):
        return True, "Skipped — duplicates/ empty or absent."

    split_dir = DATASET_DIR / split

    # Build base → canonical file
    base_to_canonical = {}
    for img in coco["images"]:
        b = base_name(img["file_name"])
        if b not in base_to_canonical:
            base_to_canonical[b] = img["file_name"]

    # Build base → dup files
    dup_files = list(dup_dir.glob("*.jpg"))
    base_to_dups = defaultdict(list)
    for f in dup_files:
        base_to_dups[base_name(f.name)].append(f)

    # Filter to bases that have both canonical AND dups
    paired_bases = [b for b in base_to_canonical if b in base_to_dups]
    if not paired_bases:
        return True, "No paired bases found for spot-check (duplicates may have different base names)."

    sample_bases = random.sample(paired_bases, min(n_samples, len(paired_bases)))
    results = []

    for b in sample_bases:
        canon_path = split_dir / base_to_canonical[b]
        dup_path   = base_to_dups[b][0]

        img_c = cv2.imread(str(canon_path), cv2.IMREAD_GRAYSCALE)
        img_d = cv2.imread(str(dup_path),   cv2.IMREAD_GRAYSCALE)

        if img_c is None or img_d is None:
            results.append((b, None))
            continue

        score = rotation_invariant_ssim(img_c, img_d)
        results.append((b, score))

    valid_scores = [s for _, s in results if s is not None]
    detail_lines = []
    for b, s in results:
        detail_lines.append(
            f"  {b[:55]:<55}  SSIM={s:.3f}" if s is not None else f"  {b:<55}  [read error]"
        )

    summary = (
        f"Spot-checked {len(valid_scores)} canonical/duplicate pairs.\n"
        f"  SSIM range: {min(valid_scores):.3f} – {max(valid_scores):.3f}  "
        f"(mean={sum(valid_scores)/len(valid_scores):.3f})\n"
        f"  Note: low SSIM is expected — Roboflow applies photometric augmentation.\n"
        + "\n".join(detail_lines)
    )
    return True, summary


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(split: str = "train", spot_n: int = 10, save_report: bool = False):
    coco_train = load_coco("train")
    coco_valid = load_coco("valid")
    coco_test  = load_coco("test")
    coco = coco_train if split == "train" else load_coco(split)

    checks = [
        ("File consistency",      lambda: check_file_consistency(split, coco)),
        ("No leaked duplicates",  lambda: check_no_leaked_dupes(split, coco)),
        ("Duplicates directory",  lambda: check_duplicates_dir(split, coco)),
        ("Annotation orphans",    lambda: check_annotation_orphans(split, coco)),
        ("Split leakage",         lambda: check_split_leakage(coco_train, coco_valid, coco_test)),
        ("Count summary",         lambda: check_count_summary(split, coco)),
        ("Spot-check SSIM",       lambda: check_spot_sample(split, coco, spot_n)),
    ]

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"CRACKS DEDUP VERIFICATION  |  split={split}")
    lines.append(f"{'='*80}\n")

    all_passed = True
    for name, fn in checks:
        passed, detail = fn()
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        lines.append(f"[{status}] {name}")
        lines.append(f"       {detail}\n")

    lines.append("─" * 80)
    lines.append(f"OVERALL: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    lines.append("─" * 80)

    report = "\n".join(lines)
    print(report)

    if save_report:
        REPORTS_DIR.mkdir(exist_ok=True)
        out = REPORTS_DIR / f"dedup_cracks_verification_{split}.txt"
        out.write_text(report)
        print(f"\nReport saved to {out}")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-dedup verification for cracks dataset")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--spot-check", type=int, default=10,
                        help="Number of canonical/dup pairs to SSIM-check (default: 10)")
    parser.add_argument("--report", action="store_true",
                        help="Save report to reports/dedup_cracks_verification_{split}.txt")
    args = parser.parse_args()

    ok = run_all(split=args.split, spot_n=args.spot_check, save_report=args.report)
    raise SystemExit(0 if ok else 1)

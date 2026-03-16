"""
Merge drywall and cracks datasets into a unified format.

Loads processed pickles from both datasets, adds dataset identifiers,
merges them, and exports unified pickles + PNG masks.
"""

import pickle
import random
from pathlib import Path

import cv2
import numpy as np


def load_pickle(path: str) -> list:
    """Load a processed dataset pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def add_dataset_field(samples: list, dataset_name: str) -> list:
    """Add 'dataset' field to each sample dict."""
    for sample in samples:
        sample['dataset'] = dataset_name
    return samples


# def stratified_sample(samples: list, n: int, seed: int = 42) -> list:
#     """
#     Sample exactly n items from samples without replacement.
#     If len(samples) <= n, returns all samples unchanged.
#     """
#     if len(samples) <= n:
#         return samples
#     rng = random.Random(seed)
#     return rng.sample(samples, n)


def merge_and_shuffle(drywall_samples: list, cracks_samples: list, seed: int = 42) -> list:
    """Merge two sample lists and shuffle deterministically."""
    combined = drywall_samples + cracks_samples
    rng = random.Random(seed)   # isolated — does not touch global random state
    rng.shuffle(combined)
    return combined


def export_masks_png(samples: list, output_dir: Path, split: str) -> None:
    """Save masks as single-channel PNG files with values {0, 255}."""
    mask_dir = output_dir / "masks_png" / split
    mask_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        dataset = sample['dataset']
        image_id = sample['image_id']
        mask = sample['mask']

        # Ensure mask values are {0, 255}
        mask_binary = (mask > 0).astype(np.uint8) * 255

        filename = f"{dataset}_{image_id}.png"
        cv2.imwrite(str(mask_dir / filename), mask_binary)


def save_pickle(samples: list, path: Path) -> None:
    """Save merged samples as pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(samples, f)


def process_split(
    drywall_pkl: Path | None,
    cracks_pkl: Path | None,
    output_dir: Path,
    split: str,
    # drywall_target: int | None = None,  # Uncomment to enable stratified sampling
    # cracks_target: int | None = None,
    # seed: int = 42,
) -> int:
    """Process and merge a single split (uses all available samples)."""
    drywall_samples = []
    cracks_samples = []

    if drywall_pkl and drywall_pkl.exists():
        drywall_samples = load_pickle(str(drywall_pkl))
        add_dataset_field(drywall_samples, 'drywall')
        # if drywall_target is not None:
        #     drywall_samples = stratified_sample(drywall_samples, drywall_target, seed)
        print(f"  Drywall {split}: {len(drywall_samples)} samples")

    if cracks_pkl and cracks_pkl.exists():
        cracks_samples = load_pickle(str(cracks_pkl))
        add_dataset_field(cracks_samples, 'cracks')
        # if cracks_target is not None:
        #     cracks_samples = stratified_sample(cracks_samples, cracks_target, seed)
        print(f"  Cracks {split}: {len(cracks_samples)} samples")

    if not drywall_samples and not cracks_samples:
        print(f"  No data found for {split} split, skipping")
        return 0

    merged = merge_and_shuffle(drywall_samples, cracks_samples, seed=42)

    # Save merged pickle
    pkl_path = output_dir / split / f"merged_{split}.pkl"
    save_pickle(merged, pkl_path)
    print(f"  Saved merged pickle: {pkl_path} ({len(merged)} samples)")

    # Export PNG masks
    export_masks_png(merged, output_dir, split)
    print(f"  Exported {len(merged)} PNG masks")

    return len(merged)


if __name__ == "__main__":
    BASE_DIR = Path("/mnt/disks/work/lokesh/seg")
    PROCESSED_DIR = BASE_DIR / "processed_data"
    OUTPUT_DIR = PROCESSED_DIR / "merged"

    print("=" * 50)
    print("MERGING DATASETS")
    print("=" * 50)

    # Define paths for each split
    splits_config = {
        'train': {
            'drywall': PROCESSED_DIR / "drywall" / "train" / "drywall_train.pkl",
            'cracks': PROCESSED_DIR / "cracks" / "train" / "cracks_train.pkl",
        },
        'valid': {
            'drywall': PROCESSED_DIR / "drywall" / "valid" / "drywall_valid.pkl",
            'cracks': PROCESSED_DIR / "cracks" / "valid" / "cracks_valid.pkl",
        },
        'test': {
            'drywall': None,  # No test split for drywall
            'cracks': PROCESSED_DIR / "cracks" / "test" / "cracks_test.pkl",
        },
    }

    total = 0
    for split, paths in splits_config.items():
        print(f"\n--- {split.upper()} ---")
        count = process_split(
            drywall_pkl=paths['drywall'],
            cracks_pkl=paths['cracks'],
            output_dir=OUTPUT_DIR,
            split=split,
        )
        total += count

    # Print summary
    print("\n" + "=" * 50)
    print(f"MERGE COMPLETE — {total} total samples")

    # Verify a sample
    train_pkl = OUTPUT_DIR / "train" / "merged_train.pkl"
    if train_pkl.exists():
        samples = load_pickle(str(train_pkl))
        s = samples[0]
        print(f"\nSample verification:")
        print(f"  Image shape: {s['image'].shape}")
        print(f"  Mask shape: {s['mask'].shape}")
        print(f"  Mask values: {set(np.unique(s['mask']))}")
        print(f"  Prompt: '{s['prompt']}'")
        print(f"  Dataset: '{s['dataset']}'")

        # Count per dataset
        datasets = {}
        for s in samples:
            datasets[s['dataset']] = datasets.get(s['dataset'], 0) + 1
        print(f"  Per dataset: {datasets}")

    print("=" * 50)

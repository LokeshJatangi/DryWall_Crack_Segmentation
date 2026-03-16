"""
Dataset preprocessing script for COCO format datasets.

Converts COCO format annotations to (image, prompt, mask) tuples.
Handles both bbox and segmentation polygon formats.
Saves as pickle file for train/val/test splits.
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image


def load_coco_annotations(json_path: str) -> Dict:
    """Load COCO format annotations from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def bbox_to_mask(bbox: List[int], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert COCO bbox to binary mask.

    Args:
        bbox: COCO format bbox [x, y, width, height]
        image_shape: (height, width) of the image

    Returns:
        Binary mask as numpy array (height, width)
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    x, y, w, h = bbox

    # Ensure bbox is within image bounds
    x = max(0, int(x))
    y = max(0, int(y))
    w = min(image_shape[1] - x, int(w))
    h = min(image_shape[0] - y, int(h))

    # Fill rectangle with 255 (foreground)
    mask[y:y+h, x:x+w] = 255

    return mask


def segmentation_to_mask(segmentation: List[List[float]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert COCO segmentation polygon to binary mask.

    Args:
        segmentation: COCO format segmentation (list of polygons, each is [x1,y1,x2,y2,...])
        image_shape: (height, width) of the image

    Returns:
        Binary mask as numpy array (height, width)
    """
    mask = np.zeros(image_shape, dtype=np.uint8)

    for polygon in segmentation:
        # Reshape flat list [x1,y1,x2,y2,...] to [(x1,y1), (x2,y2), ...]
        poly_array = np.array(polygon).reshape(-1, 2).astype(np.int32)
        # Fill polygon with 255
        cv2.fillPoly(mask, [poly_array], 255)

    return mask


def process_split(
    dataset_dir: str,
    split: str,
    output_dir: str,
    dataset_name: str,
    prompt: str = "segment taping area",
    use_segmentation: bool = True
) -> None:
    """
    Process a single split (train/val/test) of the dataset.

    Args:
        dataset_dir: Path to dataset root
        split: Split name ('train', 'valid', or 'test')
        output_dir: Path to save processed data
        dataset_name: Name of dataset (used for output folder)
        prompt: Text prompt for this dataset
        use_segmentation: If True, use segmentation polygons; if False, use bbox
    """
    dataset_path = Path(dataset_dir)
    split_dir = dataset_path / split
    annotations_path = split_dir / "_annotations.coco.json"
    output_path = Path(output_dir)

    # Check if split exists
    if not split_dir.exists():
        print(f"⚠ Warning: Split '{split}' does not exist at {split_dir}")
        return

    # Load annotations
    print(f"\nProcessing {split} split...")
    print("Loading COCO annotations...")
    coco_data = load_coco_annotations(str(annotations_path))

    # Create image_id to filename mapping
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # Create image_id to annotations mapping
    id_to_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in id_to_annotations:
            id_to_annotations[image_id] = []
        id_to_annotations[image_id].append(ann)

    # Process all images
    dataset = []

    print(f"Processing {len(id_to_filename)} images...")
    print(f"Using {'segmentation polygons' if use_segmentation else 'bounding boxes'} for masks")

    for image_id, filename in id_to_filename.items():
        image_path = split_dir / filename

        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        # Get annotations for this image
        annotations = id_to_annotations.get(image_id, [])

        # Create combined mask from all annotations
        mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

        for ann in annotations:
            if use_segmentation and ann.get('segmentation'):
                ann_mask = segmentation_to_mask(ann['segmentation'], mask.shape)
            else:
                ann_mask = bbox_to_mask(ann['bbox'], mask.shape)

            mask = np.maximum(mask, ann_mask)  # Combine masks (union)

        # Export PNG mask: single-channel (L), same spatial size as source, values {0, 255}
        mask_dir = output_path / dataset_name / split / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_stem = Path(filename).stem
        prompt_slug = prompt.replace(" ", "_")
        mask_png_path = mask_dir / f"{mask_stem}__{prompt_slug}.png"
        Image.fromarray(mask, mode='L').save(mask_png_path)

        # Store tuple: (image, mask, prompt, metadata)
        dataset.append({
            'image': img_array,
            'mask': mask,
            'prompt': prompt,
            'image_id': image_id,
            'filename': filename,
            'mask_png': str(mask_png_path),
            'num_annotations': len(annotations)
        })

    print(f"✓ Processed {len(dataset)} samples")
    print(f"✓ PNG masks saved to: {output_path / dataset_name / split / 'masks'}")

    # Save dataset as pickle
    output_pickle = output_path / dataset_name / split / f"{dataset_name}_{split}.pkl"
    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_pickle}...")
    with open(output_pickle, 'wb') as f:
        pickle.dump(dataset, f)
    print("✓ Saved dataset")

    # Print statistics
    print("\n" + "="*50)
    print(f"{split.upper()} SPLIT STATISTICS")
    print("="*50)
    print(f"Total samples: {len(dataset)}")
    print(f"Image shape: {dataset[0]['image'].shape}")
    print(f"Mask shape: {dataset[0]['mask'].shape}")
    print(f"Prompt: '{prompt}'")

    # Count masks with content
    non_empty_masks = sum(1 for sample in dataset if sample['mask'].sum() > 0)
    print(f"Samples with annotations: {non_empty_masks}/{len(dataset)}")
    print("="*50)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess COCO datasets to pickle + PNG masks")
    parser.add_argument("dataset", choices=["cracks", "drywall"], help="Dataset to process")
    parser.add_argument("--viz", action="store_true", help="Visualize samples after preprocessing")
    parser.add_argument("--viz-n", type=int, default=10, help="Number of samples to visualize (default: 10)")
    args = parser.parse_args()

    if args.dataset == "cracks":
        DATASET_DIR = "datasets/cracks.v1i.coco"
        OUTPUT_DIR  = "processed_data"
        DATASET_NAME = "cracks"
        PROMPT = "segment crack"
        USE_SEGMENTATION = True
        splits = ['train', 'valid']

        print("="*50)
        print("CRACKS DATASET PREPROCESSING")
        print("="*50)
    else:
        DATASET_DIR = "datasets/Drywall-Join-Detect.v2i.coco"
        OUTPUT_DIR  = "processed_data"
        DATASET_NAME = "drywall"
        PROMPT = "segment taping area"
        USE_SEGMENTATION = False
        splits = ['train', 'valid']

        print("="*50)
        print("DRYWALL DATASET PREPROCESSING")
        print("="*50)

    processed = {}
    for split in splits:
        process_split(
            dataset_dir=DATASET_DIR,
            split=split,
            output_dir=OUTPUT_DIR,
            dataset_name=DATASET_NAME,
            prompt=PROMPT,
            use_segmentation=USE_SEGMENTATION
        )
        if args.viz:
            pickle_path = Path(OUTPUT_DIR) / DATASET_NAME / split / f"{DATASET_NAME}_{split}.pkl"
            import pickle as _pkl
            with open(pickle_path, 'rb') as f:
                processed[split] = _pkl.load(f)

    print("\n" + "="*50)
    print("✓ All splits processed successfully!")
    print("="*50)

    if args.viz:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from visualization.visualize_dataset import visualize_samples as _viz
        VIZ_BASE = Path(OUTPUT_DIR) / DATASET_NAME
        for split, dataset in processed.items():
            print(f"\nVisualizing {split} split...")
            _viz(
                samples=dataset,
                output_dir=str(VIZ_BASE / split / "viz"),
                num_samples=args.viz_n,
                save_individual=True,
            )

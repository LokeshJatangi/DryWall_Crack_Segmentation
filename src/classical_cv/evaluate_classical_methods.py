"""
Classical CV Methods Evaluation for Drywall/Crack Segmentation
==============================================================

Evaluates classical CV techniques in THREE roles:
1. PRE-PROCESSING: Image enhancement before feeding to DL model
   - Flat-field correction, CLAHE, FFT high-pass
2. EXTRA INPUT CHANNELS: Edge/feature maps as additional model inputs
   - Sobel, Laplacian, Frangi, Gabor multi-orientation
3. POST-PROCESSING: Refine DL model predictions
   - Morphological ops, skeletonization
4. STANDALONE BASELINES: How well do classical methods do alone?
   - Otsu, Canny+Morphology, Adaptive threshold

Metrics: IoU, Dice, Precision, Recall, F1
Outputs: reports/classical_cv_evaluation/ with per-method results + visualizations

Usage:
    uv run python src/classical_cv/evaluate_classical_methods.py [--dataset cracks|drywall|both] [--num-samples 50]
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import skeletonize, disk, opening, closing


# ============================================================
# Metrics
# ============================================================

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute segmentation metrics between binary prediction and ground truth."""
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (pred_sum + gt_sum + 1e-8)
    precision = intersection / (pred_sum + 1e-8)
    recall = intersection / (gt_sum + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_pixels": int(pred_sum),
        "gt_pixels": int(gt_sum),
    }


# ============================================================
# Pre-processing methods (image enhancement)
# ============================================================

def flat_field_correction(image: np.ndarray) -> np.ndarray:
    """Remove illumination gradients using Gaussian background estimation."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray_f = gray.astype(np.float32)
    background = cv2.GaussianBlur(gray_f, (101, 101), 0)
    corrected = gray_f / (background + 1e-6) * 128.0
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    # Return as 3-channel for consistency
    return cv2.cvtColor(corrected, cv2.COLOR_GRAY2RGB)


def clahe_enhancement(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def fft_highpass(image: np.ndarray, cutoff: int = 30) -> np.ndarray:
    """FFT-based high-pass filter to enhance crack-like structures."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray_f = gray.astype(np.float32)

    # FFT
    f = np.fft.fft2(gray_f)
    fshift = np.fft.fftshift(f)

    # Create high-pass mask (Butterworth-style smooth rolloff)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    dist = np.sqrt(x * x + y * y)
    # Butterworth high-pass order=2
    hp_mask = 1.0 / (1.0 + (cutoff / (dist + 1e-8)) ** 4)

    filtered = fshift * hp_mask
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    img_back = np.abs(img_back)

    # Normalize to 0-255
    img_back = ((img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-8) * 255).astype(np.uint8)
    return cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB)


def flat_field_color(image: np.ndarray) -> np.ndarray:
    """Flat-field correction applied per-channel for color images."""
    result = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        ch = image[:, :, c].astype(np.float32)
        bg = cv2.GaussianBlur(ch, (101, 101), 0)
        result[:, :, c] = ch / (bg + 1e-6) * 128.0
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# Extra input channel methods (feature extraction)
# ============================================================

def sobel_edges(image: np.ndarray) -> np.ndarray:
    """Compute Sobel edge magnitude map."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    magnitude = (magnitude / (magnitude.max() + 1e-8) * 255).astype(np.uint8)
    return magnitude


def laplacian_edges(image: np.ndarray) -> np.ndarray:
    """Compute Laplacian edge map."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
    lap = np.abs(lap)
    lap = (lap / (lap.max() + 1e-8) * 255).astype(np.uint8)
    return lap


def frangi_filter(image: np.ndarray) -> np.ndarray:
    """Frangi vesselness filter — highlights tubular structures (cracks)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray_f = gray.astype(np.float64) / 255.0
    # Invert so cracks (dark lines) become bright ridges
    vessel = frangi(1.0 - gray_f, sigmas=range(1, 6), black_ridges=False)
    vessel = (vessel / (vessel.max() + 1e-8) * 255).astype(np.uint8)
    return vessel


def gabor_multi_orientation(image: np.ndarray) -> np.ndarray:
    """Gabor filters at 4 orientations, max-pooled."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    responses = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel(
            (31, 31), sigma=4.0, theta=theta, lambd=10.0, gamma=0.5
        )
        resp = cv2.filter2D(gray, cv2.CV_64F, kernel)
        responses.append(np.abs(resp))
    # Max-pool across orientations
    combined = np.max(responses, axis=0)
    combined = (combined / (combined.max() + 1e-8) * 255).astype(np.uint8)
    return combined


def multiscale_log(image: np.ndarray) -> np.ndarray:
    """Multi-scale Laplacian of Gaussian."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    responses = []
    for sigma in [1, 2, 4, 8]:
        ksize = int(6 * sigma + 1) | 1  # ensure odd
        blurred = cv2.GaussianBlur(gray.astype(np.float64), (ksize, ksize), sigma)
        log_resp = cv2.Laplacian(blurred, cv2.CV_64F)
        # Normalize by sigma^2 for scale invariance
        responses.append(np.abs(log_resp) * sigma ** 2)
    combined = np.max(responses, axis=0)
    combined = (combined / (combined.max() + 1e-8) * 255).astype(np.uint8)
    return combined


def combined_edge_channels(image: np.ndarray) -> np.ndarray:
    """Create 5-channel input: [R, G, B, Sobel, Laplacian]."""
    sob = sobel_edges(image)
    lap = laplacian_edges(image)
    # Return as separate channels stacked
    return np.dstack([image, sob, lap])  # shape: (H, W, 5)


# ============================================================
# Standalone classical baselines (produce binary masks)
# ============================================================

def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Otsu thresholding on inverted grayscale (cracks are dark)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    # Invert: cracks are dark, so invert to make them bright
    inverted = 255 - gray
    thresh = threshold_otsu(inverted)
    mask = (inverted > thresh).astype(np.uint8) * 255
    return mask


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Adaptive thresholding for local contrast structures."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    return mask


def canny_morphology(image: np.ndarray) -> np.ndarray:
    """Canny edge detection + morphological closing."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def frangi_threshold(image: np.ndarray) -> np.ndarray:
    """Frangi filter + Otsu threshold to produce binary mask."""
    vessel = frangi_filter(image)
    if vessel.max() == 0:
        return np.zeros_like(vessel)
    thresh = threshold_otsu(vessel)
    mask = (vessel > thresh).astype(np.uint8) * 255
    return mask


def hough_line_mask(image: np.ndarray) -> np.ndarray:
    """Hough transform line detection — good for drywall joints."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=50, maxLineGap=10
    )
    mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
    return mask


def gabor_threshold(image: np.ndarray) -> np.ndarray:
    """Gabor multi-orientation + Otsu threshold."""
    gabor = gabor_multi_orientation(image)
    if gabor.max() == 0:
        return np.zeros_like(gabor)
    thresh = threshold_otsu(gabor)
    mask = (gabor > thresh).astype(np.uint8) * 255
    return mask


# ============================================================
# Post-processing methods (applied on binary mask predictions)
# ============================================================

def morph_close(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Morphological closing to connect broken segments."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def morph_open(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Morphological opening to remove noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def morph_close_then_open(mask: np.ndarray) -> np.ndarray:
    """Close (connect) then open (denoise)."""
    closed = morph_close(mask, ksize=7)
    opened = morph_open(closed, ksize=3)
    return opened


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Skeletonize to 1-pixel wide crack lines, then dilate back."""
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return mask
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    # Dilate skeleton to make it visible / matchable
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(skeleton, kernel, iterations=1)
    return dilated


def remove_small_components(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Remove small connected components (noise)."""
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255
    return result


# ============================================================
# Data loading
# ============================================================

def load_dataset(dataset_name: str, split: str = "valid", max_samples: int = None):
    """Load processed pickle dataset."""
    base = Path(__file__).parent.parent.parent

    if dataset_name == "drywall":
        pkl_path = base / f"processed_data/drywall/{split}/drywall_{split}.pkl"
    elif dataset_name == "cracks":
        pkl_path = base / f"processed_data/cracks/{split}/cracks_{split}.pkl"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not pkl_path.exists():
        # Try train if valid not found
        if split == "valid":
            pkl_path = base / f"processed_data/{dataset_name}/train/{dataset_name}_train.pkl"
        if not pkl_path.exists():
            print(f"[WARN] Dataset not found: {pkl_path}")
            return []

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if max_samples:
        # Sample images that have actual annotations (non-empty masks)
        annotated = [s for s in data if s["mask"].sum() > 0]
        if len(annotated) > max_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(annotated), max_samples, replace=False)
            data = [annotated[i] for i in indices]
        else:
            data = annotated

    return data


def load_cracks_from_coco(split: str = "valid", max_samples: int = None):
    """Load cracks dataset directly from COCO annotations if pickle not available."""
    base = Path(__file__).parent.parent.parent
    dataset_dir = base / f"datasets/cracks.v1i.coco/{split}"
    ann_path = dataset_dir / "_annotations.coco.json"

    if not ann_path.exists():
        print(f"[WARN] Annotations not found: {ann_path}")
        return []

    with open(ann_path) as f:
        coco = json.load(f)

    # Build mappings
    img_map = {img["id"]: img for img in coco["images"]}
    ann_map = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann)

    # Filter to images with segmentation annotations
    valid_ids = [img_id for img_id, anns in ann_map.items()
                 if any(len(a.get("segmentation", [])) > 0 and len(a["segmentation"][0]) >= 6
                        for a in anns)]

    if max_samples and len(valid_ids) > max_samples:
        rng = np.random.RandomState(42)
        valid_ids = rng.choice(valid_ids, max_samples, replace=False).tolist()

    samples = []
    for img_id in valid_ids:
        img_info = img_map[img_id]
        img_path = dataset_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = img_info["height"], img_info["width"]

        # Create mask from segmentation polygons
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in ann_map.get(img_id, []):
            for seg in ann.get("segmentation", []):
                if len(seg) >= 6:
                    pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    pts = pts.astype(np.int32)
                    cv2.fillPoly(mask, [pts], 255)

        if mask.sum() == 0:
            continue

        # Resize to 640x640
        if image.shape[:2] != (640, 640):
            image = cv2.resize(image, (640, 640))
            mask = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)

        samples.append({
            "image": image,
            "mask": mask,
            "prompt": "segment crack",
            "image_id": img_id,
            "filename": img_info["file_name"],
        })

    return samples


# ============================================================
# Evaluation engine
# ============================================================

def evaluate_standalone_methods(samples: list, dataset_name: str, output_dir: Path):
    """Evaluate classical methods that produce binary masks directly."""
    methods = {
        "otsu_threshold": otsu_threshold,
        "adaptive_threshold": adaptive_threshold,
        "canny_morphology": canny_morphology,
        "frangi_threshold": frangi_threshold,
        "hough_lines": hough_line_mask,
        "gabor_threshold": gabor_threshold,
    }

    results = {}
    for method_name, method_fn in methods.items():
        print(f"  Evaluating standalone: {method_name}...")
        metrics_list = []
        times = []

        for sample in samples:
            image = sample["image"]
            gt = sample["mask"]

            t0 = time.time()
            pred = method_fn(image)
            elapsed = time.time() - t0
            times.append(elapsed)

            m = compute_metrics(pred, gt)
            metrics_list.append(m)

        # Aggregate
        avg_metrics = {}
        for key in metrics_list[0]:
            vals = [m[key] for m in metrics_list]
            avg_metrics[f"mean_{key}"] = float(np.mean(vals))
            avg_metrics[f"std_{key}"] = float(np.std(vals))

        avg_metrics["mean_time_ms"] = float(np.mean(times) * 1000)
        avg_metrics["num_samples"] = len(samples)
        results[method_name] = avg_metrics

    return results


def evaluate_preprocessing_quality(samples: list, dataset_name: str, output_dir: Path):
    """
    Evaluate pre-processing methods by measuring:
    1. Edge signal-to-noise ratio on crack regions vs background
    2. Contrast improvement in crack regions
    """
    methods = {
        "original": lambda x: x,
        "flat_field": flat_field_correction,
        "flat_field_color": flat_field_color,
        "clahe": clahe_enhancement,
        "fft_highpass": fft_highpass,
    }

    results = {}
    for method_name, method_fn in methods.items():
        print(f"  Evaluating preprocessing: {method_name}...")
        snr_values = []
        contrast_values = []
        times = []

        for sample in samples:
            image = sample["image"]
            gt = (sample["mask"] > 0)

            if gt.sum() == 0 or (~gt).sum() == 0:
                continue

            t0 = time.time()
            enhanced = method_fn(image)
            elapsed = time.time() - t0
            times.append(elapsed)

            # Convert to grayscale for analysis
            if enhanced.ndim == 3:
                gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            else:
                gray = enhanced

            gray_f = gray.astype(np.float64)

            # Edge magnitude
            edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            edge_mag = np.abs(edges)

            # SNR: mean edge in crack region / mean edge in background
            fg_edge = edge_mag[gt].mean()
            bg_edge = edge_mag[~gt].mean()
            snr = fg_edge / (bg_edge + 1e-8)
            snr_values.append(snr)

            # Contrast: difference in mean intensity
            fg_intensity = gray_f[gt].mean()
            bg_intensity = gray_f[~gt].mean()
            contrast = abs(fg_intensity - bg_intensity)
            contrast_values.append(contrast)

        results[method_name] = {
            "mean_edge_snr": float(np.mean(snr_values)) if snr_values else 0,
            "std_edge_snr": float(np.std(snr_values)) if snr_values else 0,
            "mean_contrast": float(np.mean(contrast_values)) if contrast_values else 0,
            "std_contrast": float(np.std(contrast_values)) if contrast_values else 0,
            "mean_time_ms": float(np.mean(times) * 1000) if times else 0,
            "num_samples": len(snr_values),
        }

    return results


def evaluate_feature_channels(samples: list, dataset_name: str, output_dir: Path):
    """
    Evaluate feature extraction methods by measuring:
    - How well the feature map discriminates crack vs background
    - Using ROC-AUC style: mean feature value in GT region vs background
    """
    methods = {
        "sobel": sobel_edges,
        "laplacian": laplacian_edges,
        "frangi": frangi_filter,
        "gabor": gabor_multi_orientation,
        "multiscale_log": multiscale_log,
    }

    results = {}
    for method_name, method_fn in methods.items():
        print(f"  Evaluating feature channel: {method_name}...")
        discrimination_scores = []
        times = []

        for sample in samples:
            image = sample["image"]
            gt = (sample["mask"] > 0)

            if gt.sum() == 0 or (~gt).sum() == 0:
                continue

            t0 = time.time()
            feature = method_fn(image)
            elapsed = time.time() - t0
            times.append(elapsed)

            feature_f = feature.astype(np.float64)

            # Discrimination: how much higher is feature in crack region?
            fg_mean = feature_f[gt].mean()
            bg_mean = feature_f[~gt].mean()
            fg_std = feature_f[gt].std() + 1e-8
            bg_std = feature_f[~gt].std() + 1e-8

            # Fisher's discriminant ratio
            fisher = (fg_mean - bg_mean) ** 2 / (fg_std ** 2 + bg_std ** 2)
            discrimination_scores.append(fisher)

        results[method_name] = {
            "mean_fisher_discriminant": float(np.mean(discrimination_scores)) if discrimination_scores else 0,
            "std_fisher_discriminant": float(np.std(discrimination_scores)) if discrimination_scores else 0,
            "mean_time_ms": float(np.mean(times) * 1000) if times else 0,
            "num_samples": len(discrimination_scores),
        }

    return results


def evaluate_postprocessing(samples: list, dataset_name: str, output_dir: Path):
    """
    Evaluate post-processing by applying to a simulated noisy prediction.
    We simulate model output by adding noise to GT masks.
    """
    methods = {
        "no_postprocess": lambda x: x,
        "morph_close_5": lambda x: morph_close(x, ksize=5),
        "morph_close_7": lambda x: morph_close(x, ksize=7),
        "morph_open_3": lambda x: morph_open(x, ksize=3),
        "close_then_open": morph_close_then_open,
        "remove_small_100": lambda x: remove_small_components(x, min_area=100),
        "remove_small_200": lambda x: remove_small_components(x, min_area=200),
        "skeleton_dilate": skeletonize_mask,
    }

    results = {}
    rng = np.random.RandomState(42)

    for method_name, method_fn in methods.items():
        print(f"  Evaluating postprocess: {method_name}...")
        metrics_list = []

        for sample in samples:
            gt = sample["mask"]
            gt_binary = (gt > 0).astype(np.uint8)

            if gt_binary.sum() == 0:
                continue

            # Simulate noisy prediction:
            # 1. Erode GT (miss some pixels) — simulates under-segmentation
            # 2. Add random noise — simulates false positives
            # 3. Break connectivity — simulates fragmented predictions
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            noisy_pred = cv2.erode(gt_binary * 255, kernel_erode, iterations=1)

            # Add salt noise (false positives)
            noise = (rng.random(gt.shape) < 0.005).astype(np.uint8) * 255
            noisy_pred = np.maximum(noisy_pred, noise)

            # Random erosion to break connectivity
            if rng.random() > 0.5:
                k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                noisy_pred = cv2.erode(noisy_pred, k, iterations=1)

            # Apply post-processing
            processed = method_fn(noisy_pred)

            m = compute_metrics(processed, gt)
            metrics_list.append(m)

        # Aggregate
        if metrics_list:
            avg_metrics = {}
            for key in metrics_list[0]:
                vals = [m[key] for m in metrics_list]
                avg_metrics[f"mean_{key}"] = float(np.mean(vals))
                avg_metrics[f"std_{key}"] = float(np.std(vals))
            avg_metrics["num_samples"] = len(metrics_list)
            results[method_name] = avg_metrics

    return results


# ============================================================
# Visualization
# ============================================================

def visualize_methods_comparison(samples: list, dataset_name: str, output_dir: Path, num_vis: int = 5):
    """Generate visual comparison of all methods on a few samples."""
    vis_dir = output_dir / "visualizations" / dataset_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    standalone_methods = {
        "Otsu": otsu_threshold,
        "Adaptive": adaptive_threshold,
        "Canny+Morph": canny_morphology,
        "Frangi": frangi_threshold,
        "Hough": hough_line_mask,
        "Gabor": gabor_threshold,
    }

    feature_methods = {
        "Sobel": sobel_edges,
        "Laplacian": laplacian_edges,
        "Frangi Map": frangi_filter,
        "Gabor Map": gabor_multi_orientation,
        "MS-LoG": multiscale_log,
    }

    preprocess_methods = {
        "Flat-Field": flat_field_correction,
        "CLAHE": clahe_enhancement,
        "FFT HP": fft_highpass,
    }

    for idx, sample in enumerate(samples[:num_vis]):
        image = sample["image"]
        gt = sample["mask"]
        filename = sample.get("filename", f"sample_{idx}")

        # --- Standalone methods visualization ---
        n_methods = len(standalone_methods)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(3 * (n_methods + 1), 6))

        # Top row: original + method outputs
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original", fontsize=9)
        axes[0, 0].axis("off")

        axes[1, 0].imshow(gt, cmap="gray")
        axes[1, 0].set_title("Ground Truth", fontsize=9)
        axes[1, 0].axis("off")

        for j, (name, fn) in enumerate(standalone_methods.items()):
            pred = fn(image)
            m = compute_metrics(pred, gt)

            axes[0, j + 1].imshow(pred, cmap="gray")
            axes[0, j + 1].set_title(f"{name}", fontsize=9)
            axes[0, j + 1].axis("off")

            # Overlay: green=pred, red=GT, yellow=overlap
            overlay = np.zeros((640, 640, 3), dtype=np.uint8)
            overlay[:, :, 1] = (pred > 0).astype(np.uint8) * 200  # green: pred
            overlay[:, :, 0] = (gt > 0).astype(np.uint8) * 200  # red: GT
            axes[1, j + 1].imshow(overlay)
            axes[1, j + 1].set_title(f"IoU={m['iou']:.3f}", fontsize=8)
            axes[1, j + 1].axis("off")

        plt.suptitle(f"Standalone Methods — {dataset_name} — {filename}", fontsize=11)
        plt.tight_layout()
        plt.savefig(vis_dir / f"standalone_{idx}.png", dpi=120, bbox_inches="tight")
        plt.close()

        # --- Feature channels visualization ---
        n_feat = len(feature_methods)
        fig, axes = plt.subplots(1, n_feat + 2, figsize=(3 * (n_feat + 2), 3))

        axes[0].imshow(image)
        axes[0].set_title("Original", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("GT Mask", fontsize=9)
        axes[1].axis("off")

        for j, (name, fn) in enumerate(feature_methods.items()):
            feat = fn(image)
            axes[j + 2].imshow(feat, cmap="hot")
            axes[j + 2].set_title(name, fontsize=9)
            axes[j + 2].axis("off")

        plt.suptitle(f"Feature Channels — {dataset_name} — {filename}", fontsize=11)
        plt.tight_layout()
        plt.savefig(vis_dir / f"features_{idx}.png", dpi=120, bbox_inches="tight")
        plt.close()

        # --- Pre-processing visualization ---
        n_pre = len(preprocess_methods)
        fig, axes = plt.subplots(1, n_pre + 1, figsize=(3 * (n_pre + 1), 3))

        axes[0].imshow(image)
        axes[0].set_title("Original", fontsize=9)
        axes[0].axis("off")

        for j, (name, fn) in enumerate(preprocess_methods.items()):
            enhanced = fn(image)
            if enhanced.ndim == 2:
                axes[j + 1].imshow(enhanced, cmap="gray")
            else:
                axes[j + 1].imshow(enhanced)
            axes[j + 1].set_title(name, fontsize=9)
            axes[j + 1].axis("off")

        plt.suptitle(f"Pre-processing — {dataset_name} — {filename}", fontsize=11)
        plt.tight_layout()
        plt.savefig(vis_dir / f"preprocess_{idx}.png", dpi=120, bbox_inches="tight")
        plt.close()

    print(f"  Visualizations saved to {vis_dir}/")


def print_results_table(title: str, results: dict):
    """Pretty-print results as a table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    if not results:
        print("  No results.")
        return

    # Determine columns based on first result
    first = list(results.values())[0]
    # Pick key metrics to display
    display_keys = []
    for k in first:
        if k.startswith("mean_") and k != "mean_time_ms" and "pred_pixels" not in k and "gt_pixels" not in k:
            display_keys.append(k)

    # Header
    header = f"{'Method':<25}"
    for k in display_keys:
        short = k.replace("mean_", "")
        header += f"  {short:>10}"
    header += f"  {'time_ms':>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for method_name, metrics in results.items():
        row = f"{method_name:<25}"
        for k in display_keys:
            val = metrics.get(k, 0)
            row += f"  {val:>10.4f}"
        row += f"  {metrics.get('mean_time_ms', 0):>10.1f}"
        print(row)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate classical CV methods")
    parser.add_argument("--dataset", choices=["cracks", "drywall", "both"], default="both")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--num-vis", type=int, default=5, help="Number of visualization samples")
    parser.add_argument("--skip-vis", action="store_true", help="Skip generating visualizations")
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent.parent / "reports" / "classical_cv_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_eval = []
    if args.dataset in ("cracks", "both"):
        datasets_to_eval.append("cracks")
    if args.dataset in ("drywall", "both"):
        datasets_to_eval.append("drywall")

    all_results = {}

    for dataset_name in datasets_to_eval:
        print(f"\n{'#' * 80}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#' * 80}")

        # Load data
        print(f"Loading {dataset_name} dataset...")
        if dataset_name == "cracks":
            # Try pickle first, then COCO
            samples = load_dataset("cracks", split="valid", max_samples=args.num_samples)
            if not samples:
                samples = load_dataset("cracks", split="train", max_samples=args.num_samples)
            if not samples:
                print("  Loading from COCO annotations...")
                samples = load_cracks_from_coco(split="valid", max_samples=args.num_samples)
            if not samples:
                samples = load_cracks_from_coco(split="train", max_samples=args.num_samples)
        else:
            samples = load_dataset("drywall", split="valid", max_samples=args.num_samples)
            if not samples:
                samples = load_dataset("drywall", split="train", max_samples=args.num_samples)

        if not samples:
            print(f"  [SKIP] No data found for {dataset_name}")
            continue

        print(f"  Loaded {len(samples)} samples")

        ds_results = {}

        # 1. Standalone baselines
        print("\n[1/4] Evaluating standalone classical baselines...")
        standalone_res = evaluate_standalone_methods(samples, dataset_name, output_dir)
        print_results_table(f"Standalone Baselines — {dataset_name}", standalone_res)
        ds_results["standalone"] = standalone_res

        # 2. Pre-processing quality
        print("\n[2/4] Evaluating pre-processing methods...")
        preprocess_res = evaluate_preprocessing_quality(samples, dataset_name, output_dir)
        print_results_table(f"Pre-processing Quality — {dataset_name}", preprocess_res)
        ds_results["preprocessing"] = preprocess_res

        # 3. Feature channels
        print("\n[3/4] Evaluating feature extraction channels...")
        feature_res = evaluate_feature_channels(samples, dataset_name, output_dir)
        print_results_table(f"Feature Channels (Fisher Discriminant) — {dataset_name}", feature_res)
        ds_results["feature_channels"] = feature_res

        # 4. Post-processing
        print("\n[4/4] Evaluating post-processing methods...")
        postprocess_res = evaluate_postprocessing(samples, dataset_name, output_dir)
        print_results_table(f"Post-processing (on simulated noisy pred) — {dataset_name}", postprocess_res)
        ds_results["postprocessing"] = postprocess_res

        # Visualizations
        if not args.skip_vis:
            print("\nGenerating visualizations...")
            visualize_methods_comparison(samples, dataset_name, output_dir, num_vis=args.num_vis)

        all_results[dataset_name] = ds_results

    # Save full results as JSON
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {results_path}")

    # Print recommendations
    print_recommendations(all_results)


def print_recommendations(all_results: dict):
    """Print actionable recommendations based on evaluation results."""
    print(f"\n{'=' * 80}")
    print("  RECOMMENDATIONS")
    print(f"{'=' * 80}")

    for dataset_name, ds_results in all_results.items():
        print(f"\n--- {dataset_name.upper()} ---")

        # Best standalone
        if "standalone" in ds_results:
            best = max(ds_results["standalone"].items(), key=lambda x: x[1].get("mean_iou", 0))
            print(f"\n  Best standalone baseline: {best[0]} (IoU={best[1]['mean_iou']:.4f})")

        # Best preprocessing
        if "preprocessing" in ds_results:
            # Skip 'original' for comparison
            preproc = {k: v for k, v in ds_results["preprocessing"].items() if k != "original"}
            if preproc:
                best_snr = max(preproc.items(), key=lambda x: x[1].get("mean_edge_snr", 0))
                best_contrast = max(preproc.items(), key=lambda x: x[1].get("mean_contrast", 0))
                orig = ds_results["preprocessing"].get("original", {})
                print(f"\n  Pre-processing (vs original SNR={orig.get('mean_edge_snr', 0):.3f}):")
                print(f"    Best edge SNR: {best_snr[0]} (SNR={best_snr[1]['mean_edge_snr']:.3f})")
                print(f"    Best contrast: {best_contrast[0]} (contrast={best_contrast[1]['mean_contrast']:.1f})")

        # Best feature channel
        if "feature_channels" in ds_results:
            best = max(ds_results["feature_channels"].items(),
                       key=lambda x: x[1].get("mean_fisher_discriminant", 0))
            print(f"\n  Best feature channel: {best[0]} (Fisher={best[1]['mean_fisher_discriminant']:.4f})")

        # Best post-processing
        if "postprocessing" in ds_results:
            postproc = {k: v for k, v in ds_results["postprocessing"].items() if k != "no_postprocess"}
            if postproc:
                base_iou = ds_results["postprocessing"].get("no_postprocess", {}).get("mean_iou", 0)
                best = max(postproc.items(), key=lambda x: x[1].get("mean_iou", 0))
                improvement = best[1]["mean_iou"] - base_iou
                print(f"\n  Post-processing (vs baseline IoU={base_iou:.4f}):")
                print(f"    Best: {best[0]} (IoU={best[1]['mean_iou']:.4f}, improvement={improvement:+.4f})")

    print(f"\n{'=' * 80}")
    print("  HOW TO USE THESE IN YOUR DL PIPELINE:")
    print(f"{'=' * 80}")
    print("""
  1. PRE-PROCESSING (apply before model input):
     - Use CLAHE or flat-field correction on training images
     - Improves model's ability to learn crack features vs illumination artifacts

  2. EXTRA INPUT CHANNELS (concatenate with RGB):
     - Add best feature channel(s) as additional input channels
     - Model input becomes [R, G, B, Feature1, Feature2] = 5 channels
     - Modify first conv layer: in_channels=3 → in_channels=5

  3. POST-PROCESSING (apply after model prediction):
     - Use best morphological operation on model output masks
     - Typically: morph_close → remove_small_components
     - This is cheap and often gives +1-3% IoU for free

  4. COMBINED APPROACH:
     - Pre-process input → Model → Post-process output
     - This is the recommended pipeline
""")


if __name__ == "__main__":
    main()

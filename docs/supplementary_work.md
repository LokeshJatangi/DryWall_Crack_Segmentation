# Supplementary Work

This document covers supporting work for the prompted segmentation project: classical CV baselines, visualization tooling, failure analysis approach, and data preprocessing pipeline.

## 1. Classical CV Evaluation

Six classical computer vision methods were evaluated on 50 samples from each dataset to establish non-learned baselines. These methods use hand-crafted features and thresholding — no training is involved.

### Methods

| Method | Approach | Key Parameters |
|--------|----------|---------------|
| Otsu threshold | Global thresholding on grayscale | Auto-threshold |
| Adaptive threshold | Local mean thresholding | Block size, offset |
| Canny + morphology | Edge detection + dilation | Low/high thresholds, kernel |
| Frangi filters | Vessel/ridge enhancement | Scale range, beta values |
| Hough lines | Line detection + mask generation | Threshold, min line length |
| Gabor filters | Texture-based filtering | Multiple orientations, frequencies |

### Results — Cracks Dataset (50 samples)

| Method | IoU | Dice | Precision | Recall | Time (ms) |
|--------|-----|------|-----------|--------|-----------|
| Gabor filters | 0.2448 | 0.3734 | 0.5496 | 0.4112 | 84.0 |
| Canny + morphology | 0.1414 | 0.2287 | 0.4663 | 0.3606 | 1.6 |
| Otsu threshold | 0.1155 | 0.1896 | 0.2780 | 0.5261 | 2.0 |
| Hough lines | 0.1092 | 0.1777 | 0.3197 | 0.4765 | 37.1 |
| Adaptive threshold | 0.1098 | 0.1870 | 0.3340 | 0.3173 | 3.1 |
| Frangi filters | 0.0748 | 0.1355 | 0.6022 | 0.1179 | 1275.6 |

### Results — Drywall Dataset (50 samples)

| Method | IoU | Dice | Precision | Recall | Time (ms) |
|--------|-----|------|-----------|--------|-----------|
| Hough lines | 0.0610 | 0.1127 | 0.3002 | 0.0872 | 7.7 |
| Otsu threshold | 0.0392 | 0.0695 | 0.0807 | 0.1349 | 1.8 |
| Adaptive threshold | 0.0359 | 0.0690 | 0.1593 | 0.0607 | 3.1 |
| Gabor filters | 0.0255 | 0.0468 | 0.1577 | 0.0317 | 84.4 |
| Canny + morphology | 0.0236 | 0.0452 | 0.2237 | 0.0316 | 1.1 |
| Frangi filters | 0.0092 | 0.0181 | 0.1742 | 0.0101 | 1273.6 |

### Deep Learning Improvement

| Dataset | Best Classical IoU | Best DL IoU (SegFormer) | Improvement |
|---------|-------------------|------------------------|-------------|
| Cracks | 0.2448 (Gabor) | 0.5950 | 2.4x |
| Drywall | 0.0610 (Hough) | 0.7888 | 12.9x |

Classical methods fail catastrophically on drywall — the taping area has no distinctive low-level texture or edge signature. Only learned features with spatial context can identify these regions.

Evaluation data: `reports/classical_cv_evaluation/evaluation_results.json`

## 2. Visualization Tooling

Seven visualization scripts provide comprehensive tools for dataset inspection, augmentation verification, and model evaluation.

| Script | Purpose | Output |
|--------|---------|--------|
| `visualize_dataset.py` | Matplotlib grid of raw pickle samples | PNG grids (dataset + split args) |
| `visualize_dataloader.py` | Post-augmentation DataLoader visualization | PNG grid of augmented batches |
| `visualize_interactive.py` | Interactive OpenCV browser for pickles | Live window (headless-safe) |
| `visualize_failures.py` | Failure analysis: inference + metrics + grids | failure_grid.png, best_grid.png, CSV, report |
| `visualize_tiny_annotations.py` | QA visualization: tiny/bleed/disconnected | PNG panels per mode |
| `verify_drywall.py` | 4-panel bbox-to-mask alignment check | PNG verification panels |
| `visualize_crack_aug.py` | CrackWidthAugmentation effect visualization | Before/after comparison grids |

All OpenCV-based viewers are **headless-safe** — they check `$DISPLAY`/`$WAYLAND_DISPLAY` before opening windows and fall back to file-only output on servers.

### Usage Examples

```bash
# Dataset grid export
uv run python src/visualization/visualize_dataset.py --dataset cracks --split train --n 20

# Post-augmentation visualization
uv run python src/visualization/visualize_dataloader.py --config src/configs/experiment.yaml --n 16

# Failure analysis (headless-safe)
uv run python src/visualization/visualize_failures.py \
    --checkpoint experiments/<run>/ckpts/best_model.pt \
    --config src/configs/experiment.yaml \
    --no-interactive --top-n 20
```

## 3. Failure Analysis Approach

The failure analysis pipeline (`visualize_failures.py`) provides systematic tools for understanding model errors.

### Per-Sample Metrics

For every validation image, four metrics are computed:
- **IoU** (Intersection over Union)
- **Dice coefficient**
- **Precision** (fraction of predicted foreground that is correct)
- **Recall** (fraction of ground truth foreground that is detected)

Results are saved as a sortable CSV (`per_sample_metrics.csv`) for offline analysis.

### Visualization

**4-panel display** per sample:
1. **Input** — original RGB image
2. **Input + Pred** — prediction overlay in green
3. **Input + GT** — ground truth overlay in red
4. **Combined** — red = GT only (missed), green = pred only (false positive), yellow = overlap (correct)

Two grid outputs:
- **Failure grid** (`failure_grid.png`) — top-N worst samples sorted by IoU, exposing systematic failure modes
- **Best grid** (`best_grid.png`) — top-N best predictions per prompt, showing model capabilities

### Key Findings

Analysis of failures across all 6 baseline models reveals:
- **No systematic model errors** — failures are not concentrated in a specific failure mode (over/under-segmentation)
- **Dataset noise dominates** — the worst predictions correspond to samples with noisy GT annotations (imprecise polygon boundaries, inconsistent labeling)
- **Thin cracks are inherently hard** — cracks with <2% mask coverage account for a disproportionate share of low-IoU predictions
- **The IoU gap between prompts (~18-20pp) is structural** — taping areas are large, well-defined regions while cracks are thin and variable

## 4. Data Preprocessing Pipeline

### Pipeline Overview

```
COCO JSON → preprocess.py → per-dataset pickles → merge_datasets.py → merged pickle
→ dataset.py (PyTorch Dataset) → Albumentations augmentation → DataLoader → Model
```

### Scripts (in `src/data/preprocess_cleaning/`)

| Script | Purpose |
|--------|---------|
| `preprocess.py` | COCO JSON → pickle with binary masks ({0,255}), supports both datasets via CLI arg |
| `merge_datasets.py` | Merges drywall + cracks pickles, exports PNG masks for verification |
| `dedup_drywall_v2.py` | Annotation-first drywall deduplication (polygon similarity) |
| `dedup_cracks.py` | Base-name + Union-Find cracks deduplication |
| `mask_quality_checks.py` | Consolidated QA: 14 checks + annotation cleaning |
| `verify_dedup_cracks.py` | 7-check post-dedup verification |

### 14-Point QA Checks

The consolidated `mask_quality_checks.py` performs:
1. Tiny annotations (<0.5% image coverage)
2. Mask bleed outside bounding box (>1% overflow)
3. Disconnected components in single annotations
4. Degenerate polygons (self-intersecting, <3 points)
5. All-zero masks (empty annotations)
6. All-white masks (full-coverage annotations)
7. Annotation count per image
8. Category ID consistency
9. Image dimension validation
10. Polygon coordinate bounds
11. Duplicate annotation detection
12. Missing image references
13. Missing annotation references
14. Format consistency (COCO JSON schema)

### Train-Only Cleaning Principle

All deduplication and cleaning operations are applied exclusively to training splits. Validation and test splits are never modified, ensuring evaluation metrics reflect real-world performance without data leakage.

## 5. Reports Folder

All generated reports are saved to `reports/`:

| File | Contents |
|------|----------|
| `failure_analysis_*.txt` | Per-experiment: per-class metrics, runtime/footprint |
| `mask_quality_report.txt` | Consolidated QA check results |
| `classical_cv_evaluation/` | Classical CV method evaluation results (JSON + visualizations) |

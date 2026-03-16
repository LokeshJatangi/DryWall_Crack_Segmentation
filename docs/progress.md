# Project Progress - Drywall Segmentation

**Last Updated:** 2026-03-16
**Session:** Evaluation Script, Reports & Documentation

---

## ✅ Milestone 1: Dataset Setup & Preprocessing (COMPLETE)

### 1.1 Project Structure Setup ✓
Current directory structure:
```
src/
├── data/
│   ├── dataset.py                        # PyTorch Dataset + DataLoader factory
│   └── preprocess_cleaning/              # All data prep & QA scripts
│       ├── preprocess.py                 # COCO → pickle + PNG mask export (masks {0,255})
│       ├── merge_datasets.py             # Merge drywall + cracks pickles
│       ├── dedup_drywall_v2.py           # Annotation-first drywall dedup ✅ EXECUTED
│       ├── dedup_cracks.py               # Base-name + Union-Find cracks dedup ✅ EXECUTED
│       ├── verify_dedup_cracks.py        # 7-check post-dedup verification ✅ EXECUTED
│       └── mask_quality_checks.py        # Consolidated QA: 14 checks + annotation cleaning
├── augmentations/
│   └── transforms.py                     # 5-tier Albumentations + CrackWidthAugmentation
├── models/
│   ├── prompt_encoder.py                 # Shared prompt embedding (nn.Embedding)
│   ├── film_wrapper.py                   # FiLM modulation + auxiliary decoder injection
│   ├── smp_models.py                     # U-Net, U-Net++, DeepLabV3+ factory (SMP)
│   └── segformer.py                      # SegFormer B2 with FiLM (HuggingFace)
├── training/
│   ├── trainer.py                        # Training loop: AMP, early stopping, TensorBoard
│   ├── losses.py                         # Dice, DiceBCE, Focal, Boundary, Combined
│   └── metrics.py                        # mIoU, Dice, Precision, Recall tracker
├── configs/
│   ├── config.py                         # ExperimentConfig dataclass + YAML loader
│   ├── experiment.yaml                   # UnetPlusPlus / resnet34 / full aug
│   ├── segformer_experiment.yaml         # SegFormer B2 config
│   ├── unet_resume_cosine.yaml           # Resume U-Net with cosine scheduler
│   ├── unet_resume_onecycle.yaml         # Resume U-Net with OneCycleLR
│   └── segformer_resume_onecycle.yaml    # Resume SegFormer with OneCycleLR
├── utils/
│   └── seed.py                           # set_seed() + worker_init_fn()
└── visualization/
    ├── viz_utils.py                      # Shared: load_pickle, overlay_rgb/bgr, save_grid
    ├── visualize_dataset.py              # Matplotlib grid exports
    ├── visualize_dataloader.py           # Post-augmentation DataLoader visualization
    ├── visualize_interactive.py          # Interactive OpenCV viewer for pickles
    ├── visualize_failures.py             # Failure analysis: inference + metrics + browser
    ├── verify_drywall.py                 # 4-panel bbox→mask verification
    └── visualize_tiny_annotations.py    # QA viz: tiny / bleed / disco modes

main.py                 # Training entry point
processed_data/
├── drywall/
│   ├── train/
│   │   ├── drywall_train.pkl        # 481 samples (post-dedup)
│   │   └── masks/                   # PNG masks: {stem}__segment_taping_area.png
│   └── valid/
│       ├── drywall_valid.pkl        # 202 samples
│       └── masks/                   # PNG masks
├── cracks/
│   ├── train/
│   │   ├── cracks_train.pkl         # 907 samples (post-dedup, post-clean)
│   │   └── masks/                   # PNG masks: {stem}__segment_crack.png
│   └── valid/
│       ├── cracks_valid.pkl         # 201 samples
│       └── masks/                   # PNG masks
└── merged/                          # Unified merged dataset (run merge_datasets.py)
    ├── train/         # merged_train.pkl
    ├── valid/         # merged_valid.pkl
    ├── test/          # merged_test.pkl (cracks only)
    └── masks_png/     # Individual PNG masks
```

### 1.2 Dependencies Installed ✓
- numpy >= 1.24.0
- pillow >= 10.0.0
- matplotlib >= 3.7.0
- opencv-python >= 4.8.0
- scikit-image >= 0.26.0
- torch >= 2.0.0
- torchvision >= 0.15.0
- segmentation-models-pytorch >= 0.3.3
- albumentations >= 1.3.0
- transformers >= 4.30.0
- pyyaml >= 6.0
- tqdm >= 4.65.0
- tensorboard >= 2.13.0
- scipy >= 1.10.0

### 1.3 Dataset Preprocessing Script ✓
**File:** `src/data/preprocess.py`

**Features:**
- Generic processing for train/valid/test splits
- Supports both bbox (drywall) and segmentation polygon (cracks) formats
- Creates combined masks (union) for multi-annotation images
- Exports PNG masks alongside pickle: single-channel L-mode, `{0,255}`, named `{stem}__{prompt_slug}.png`
- Saves as pickle files for training

> Note: QA checks (area mismatches, empty masks, resolution) are handled exclusively by `mask_quality_checks.py` — not duplicated here.

**Output Format:**
```python
{
    'image': np.ndarray (H, W, 3),   # RGB
    'mask': np.ndarray (H, W),        # {0, 255}
    'prompt': str,
    'image_id': int,
    'filename': str,
    'mask_png': str,                  # path to exported PNG mask
    'num_annotations': int
}
```

### 1.4 Interactive Visualizer ✓
**File:** `src/visualization/visualize_interactive.py` (316 lines)

**Features:**
- OpenCV-based interactive viewer
- Displays: Original | Mask (RED) | Overlay side-by-side
- Auto-scales to 50% screen size (configurable)
- Keyboard controls:
  - `d` / `→` : Next image
  - `a` / `←` : Previous image
  - `p` / `SPACE` : Play/Pause auto-advance
  - `q` / `ESC` : Quit
- Auto-play mode with configurable delay (default: 100ms)
- Shows mask coverage statistics
- Warns about missing masks
- Supports predicted masks (GREEN) for later model evaluation

### 1.5 Debug & Annotation Tools ✓
**File:** `src/data/debug_annotations.py` (146 lines)

**Features:**
- Analyzes COCO annotation files
- Finds images without annotations
- Detects orphaned annotations (no matching image)
- Detects bbox area mismatches
- Shows annotation distribution statistics (min/max/avg per image)

### 1.6 Duplicate Detection ✓
**File:** `src/data/dedup_drywall.py` (248 lines) — v1, quadrant-based

**Features:**
- Groups images by annotation count (single, multi, no annotation)
- Sub-groups single-annotation images by bbox quadrant location
- SSIM-based near-duplicate detection (threshold: 0.95)
- Bbox comparison within tolerance
- Dry-run mode (default) with optional `--remove` flag

**File:** `src/data/dedup_drywall_v2.py` (452 lines) — v2, annotation-first

**Features:**
- Annotation-first approach: bbox matching as primary signal
- Two SSIM thresholds: 0.70 (bbox-confirmed), 0.95 (visual-only)
- Handles lighting variants and data augmentation
- Generates visual comparison panels with KEEP/REMOVE badges
- Union-find grouping for connected duplicate components
- Saves individual duplicate pair visualizations with diffs

### 1.7 Quality Checks ✓
**File:** `src/data/mask_quality_checks.py` ← single consolidated QA authority

**Checks performed (19 total):**
1. File integrity: missing, extra, corrupted, non-RGB
2. Image-annotation size mismatch
3. Resolution stats + non-640×640 count
4. Empty masks (no annotation pixels)
5. Orphan annotations (no matching image)
6. Missing / unused categories
7. Annotation bleed (bbox beyond image bounds)
8. Tiny annotations (area < 100px)
9. Disconnected components (tiny islands < 100px in generated mask)
10. Incorrect label IDs
11. Degenerate polygons (< 4 vertices)
12. Multi-polygon annotations
13. Crowd annotations (iscrowd=1)
14. Bbox area vs JSON area mismatch

**Annotation cleaning flags (train split only):**
```bash
# Remove annotations with area < 100px
uv run python src/data/mask_quality_checks.py --dataset cracks --remove-tiny

# Remove annotations whose pixels fall entirely inside tiny disconnected components
uv run python src/data/mask_quality_checks.py --dataset cracks --remove-disconnected
```

### 1.8 Additional Visualization Tools ✓

**File:** `src/visualization/visualize_dataset.py` (202 lines)
- Publication-quality matplotlib visualizations
- Grid layout: Original | Binary Mask | Overlay
- Individual high-res PNG exports
- Mask coverage statistics

**File:** `src/visualization/verify_drywall.py` (190 lines)
- 4-panel verification: Original | BBoxes | Mask | Overlay
- Colored, numbered bboxes for multi-annotation images
- `--multi-only` flag to show only multi-annotation images
- `--save` flag for batch export

**File:** `src/visualization/visualize_tiny_annotations.py`
Three visualization modes for the cracks dataset:
- `tiny` — RED bboxes for annotations with area < 100px, GREEN for normal
- `bleed` — RED bboxes that exceed image boundaries; yellow dashes = image border
- `disco` — RED overlay on tiny disconnected mask islands; side-by-side overlay + pure mask view

```bash
uv run python src/visualization/visualize_tiny_annotations.py tiny
uv run python src/visualization/visualize_tiny_annotations.py bleed
uv run python src/visualization/visualize_tiny_annotations.py disco
uv run python src/visualization/visualize_tiny_annotations.py   # all three
```
Outputs to `processed_data/verification/{tiny_annotations,bleed,disconnected}/`

---

## 📊 Dataset Statistics

### DryWall-Join-Detect Dataset (Processed)

**Training Split (post-dedup, post-cleaning):**
- Total samples: 481 (339 duplicates removed from original 820)
- Annotations: 623 (479 images with annotations, 2 without)
- Tiny annotations removed: 11 (via `--remove-tiny`)
- Image shape: (640, 640, 3) | Mask shape: (640, 640) | Values: {0, 255}
- Prompt: `'segment taping area'`
- 2 corrupted files moved to `datasets/Drywall-Join-Detect.v2i.coco/corrupted/`

**Validation Split:**
- Total samples: 202 (all annotated)
- Prompt: `'segment taping area'`

**Files:**
- `processed_data/drywall/train/drywall_train.pkl` — 481 samples
- `processed_data/drywall/train/masks/` — PNG masks `{stem}__segment_taping_area.png`
- `processed_data/drywall/valid/drywall_valid.pkl` — 202 samples
- `processed_data/drywall/valid/masks/` — PNG masks

---

### Cracks Dataset (Processed) ✅

**Deduplication (train only):**
- Original: 5,164 images (907 source images × 2–17 Roboflow augmentation variants)
- After dedup: **907 canonical images** kept in `train/`
- **4,257 duplicates moved** to `train/duplicates/` (not deleted)
- Method: base-name grouping (`{source}.rf.{hash}.jpg`) → Union-Find connected components
- SSIM not used — Roboflow photometric augmentations make it unreliable (~0.23 for same-source)

**Known dataset quality issue (Roboflow source):**
- `525_jpg` base name appears in both `train` and `valid` (cross-split leakage)
- `2056_jpg` base name appears in both `train` and `test` (cross-split leakage)
- Not fixed — document in final report

**Training Split (post-dedup):**
- Total samples: **907** | Annotations: 1,502 | Images without annotations: 0
- Prompt: `'segment crack'`

**Validation Split:**
- Total samples: **201** | Annotations: 372 | Images without annotations: 0
- Prompt: `'segment crack'`

**Files:**
- `processed_data/cracks/train/cracks_train.pkl` — 907 samples
- `processed_data/cracks/train/masks/` — PNG masks `{stem}__segment_crack.png`
- `processed_data/cracks/valid/cracks_valid.pkl` — 201 samples
- `processed_data/cracks/valid/masks/` — PNG masks
- `processed_data/verification/cracks_duplicates/*.png` — 50 group visualization grids

---

## 🔧 Commands Reference

### Preprocessing (scripts in `src/data/preprocess_cleaning/`)
```bash
# Drywall (train + valid)
uv run python src/data/preprocess_cleaning/preprocess.py drywall

# Cracks (train + valid)
uv run python src/data/preprocess_cleaning/preprocess.py cracks

# Merge both datasets into unified pickle
uv run python src/data/preprocess_cleaning/merge_datasets.py
```

### Duplicate Detection
```bash
# Drywall — annotation-first (ALREADY EXECUTED)
uv run python src/data/preprocess_cleaning/dedup_drywall_v2.py --split train --remove

# Cracks — base-name + Union-Find (ALREADY EXECUTED)
uv run python src/data/preprocess_cleaning/dedup_cracks.py --split train --move

# Cracks — verify post-dedup state
uv run python src/data/preprocess_cleaning/verify_dedup_cracks.py --split train --report
```

### Quality Checks
```bash
# Full report (both datasets)
uv run python src/data/preprocess_cleaning/mask_quality_checks.py

# Single dataset
uv run python src/data/preprocess_cleaning/mask_quality_checks.py --dataset drywall
uv run python src/data/preprocess_cleaning/mask_quality_checks.py --dataset cracks

# Clean annotations (train only, writes in place)
uv run python src/data/preprocess_cleaning/mask_quality_checks.py --dataset cracks --remove-tiny
uv run python src/data/preprocess_cleaning/mask_quality_checks.py --dataset cracks --remove-disconnected
```

### Training
```bash
# Default (UnetPlusPlus / resnet34 / full augmentation)
uv run python main.py --config src/configs/experiment.yaml

# SegFormer B2
uv run python main.py --config src/configs/segformer_experiment.yaml

# Resume
uv run python main.py --config src/configs/unet_resume_cosine.yaml
```

### Failure Analysis (run as a single line on server — no line breaks)
```bash
uv run python src/visualization/visualize_failures.py --checkpoint experiments/<run>/ckpts/best_model.pt --config src/configs/experiment.yaml

# Filter to only bad failures, skip interactive
uv run python src/visualization/visualize_failures.py --checkpoint <ckpt> --config <cfg> --iou-threshold 0.5 --no-interactive
```

### Visualization
```bash
# DataLoader post-augmentation visualization
uv run python src/visualization/visualize_dataloader.py --split train --n 16 --tier full

# Dataset grid (matplotlib)
uv run python src/visualization/visualize_dataset.py drywall --split valid --n 10
uv run python src/visualization/visualize_dataset.py cracks --split train --n 20

# Interactive pickle viewer (requires display)
uv run python src/visualization/visualize_interactive.py --pickle_path processed_data/merged/valid/merged_valid.pkl

# Verify bbox→mask alignment
uv run python src/visualization/verify_drywall.py --multi-only

# QA annotation visualization
uv run python src/visualization/visualize_tiny_annotations.py tiny
uv run python src/visualization/visualize_tiny_annotations.py bleed
uv run python src/visualization/visualize_tiny_annotations.py disco
```

---

## 📝 Notes & Observations

1. **Image Size:** All images at 640×640 pixels
2. **Missing Annotations:** 2 drywall train images have no annotations — kept in dataset (zero mask)
3. **Annotation Format:** Drywall = COCO bbox only; Cracks = COCO segmentation polygons
4. **Mask Creation:** bbox → rectangular fill (drywall); polygon fill via cv2.fillPoly (cracks)
5. **QA Authority:** `mask_quality_checks.py` is the single source of truth — do not add QA logic to preprocessing or training scripts
6. **Train-only Cleaning:** `--remove-tiny` and `--remove-disconnected` only ever modify the train COCO JSON; valid/test are never touched
7. **Corrupted Files:** 2 drywall images physically moved to `corrupted/` subfolder; still referenced in COCO JSON
8. **Bleed tolerance:** Set to `5px` in `visualize_tiny_annotations.py`, `0px` in `mask_quality_checks.py` — align if needed
9. **`expected_categories`:** Both datasets declare only `{1: ...}` — category 0 does not appear in any annotation

---

## ✅ Milestone 2: Baseline Setup (COMPLETE)

### 2.1 Training Pipeline ✓
**Built complete training framework with FiLM prompt conditioning.**

**Models implemented (all with FiLM conditioning):**
| Model | Backend | File |
|-------|---------|------|
| U-Net | SMP (resnet34) | `src/models/smp_models.py` |
| U-Net++ | SMP (resnet34) | `src/models/smp_models.py` |
| DeepLabV3+ | SMP (resnet34) | `src/models/smp_models.py` |
| SegFormer B2 | HuggingFace | `src/models/segformer.py` |

**FiLM (Feature-wise Linear Modulation) Architecture:**
- `PromptEncoder`: Embeds prompt IDs (0=crack, 1=taping area) → dense vectors
- `FiLMBlock`: Per-scale modulation: γ * feature + β
- Auxiliary prompt injection at decoder bottleneck
- Applied at every encoder scale for prompt-dependent feature representations

### 2.2 Loss Functions ✓
| Loss | Description |
|------|-------------|
| DiceLoss | Standard soft Dice |
| DiceBCELoss | Combined Dice + BCE (configurable weights) |
| FocalLoss | α=0.25, γ=2.0 for class imbalance |
| BoundaryLoss | Distance-transform based boundary loss |
| CombinedLoss | Configurable mix of multiple losses |

### 2.3 Augmentation System ✓
5 cumulative tiers for ablation:
| Tier | Transforms |
|------|-----------|
| `baseline` | HorizontalFlip |
| `geometric` | + Rotate(30°), ShiftScaleRotate |
| `photometric` | + RandomBrightnessContrast, CLAHE, GaussNoise |
| `edge` | + Sharpen, MotionBlur |
| `full` | + CoarseDropout, ElasticTransform |

Plus: `CrackWidthAugmentation` (dilate/erode crack masks, kernel 2-5px)

### 2.4 Metrics Tracking ✓
- mIoU (foreground + background)
- Dice coefficient
- Precision, Recall, F1

### 2.5 Training Features ✓
- Mixed-precision training (AMP)
- Early stopping on val Dice
- Best model checkpointing
- TensorBoard logging
- Cosine annealing / ReduceLROnPlateau schedulers
- Prediction saving as `{id}__{prompt}.png` (values {0, 255})

### 2.6 Failure Analysis ✓
**File:** `src/visualization/visualize_failures.py`

**Features:**
- Loads any checkpoint (reads arch from `ckpt['config'].model`, not the YAML)
- Runs full inference on val set, computes per-sample IoU/Dice/Precision/Recall
- **4-panel visualization** (all panels overlay on input image):
  - Input | Input + Pred (green) | Input + GT (red) | Combined (R=GT, G=Pred, Y=overlap)
- **Per-class metrics** — crack vs taping area reported separately (mean ± std)
- **Runtime report** — total params, peak GPU mem, avg inference ms/image
- **Interactive browser** — sorted worst-first by IoU; `s` to save, headless-safe
- **Static failure grid** — top-N worst samples saved as `failure_grid.png`
- **Per-sample CSV** — `per_sample_metrics.csv` sortable by any metric
- **Auto-suggestions** — detects under/over-segmentation, class imbalance, tiny masks
- **Report file** — `reports/failure_analysis_{exp_name}.txt`

```bash
uv run python src/visualization/visualize_failures.py --checkpoint experiments/<run>/ckpts/best_model.pt --config src/configs/experiment.yaml
```

### 2.7 Visualize DataLoader ✓
**File:** `src/visualization/visualize_dataloader.py`

Visualizes post-augmentation batches to verify transforms and mask alignment.

```bash
uv run python src/visualization/visualize_dataloader.py --split train --n 16 --tier full
```

### Milestone 2 Status
- [x] U-Net, U-Net++, SegFormer B2 trained (experiments/ contains multiple runs)
- [x] Failure analysis tooling complete
- [x] Per-class crack/taping metrics tracked
- [x] Runtime/footprint reporting
- [x] Formal performance summary table across all models
- [x] Visual results table (best_grid.png + failure_grid.png)

---

## ✅ Milestone 2b: Extended Baselines & Evaluation (COMPLETE)

**Session: 2026-03-16 — Evaluation Script, Reports & Documentation**

### 2b.1 6 Baseline Experiments Evaluated ✓

| # | Model | Loss | Dice | mIoU |
|---|-------|------|------|------|
| 1 | U-Net | Dice+BCE | 0.7670 | 0.6520 |
| 2 | U-Net++ | Dice+BCE | 0.7705 | 0.6556 |
| 3 | SegFormer B2 | Dice+BCE | 0.7939 | 0.6812 |
| 4 | U-Net | Dice+Focal | 0.7760 | 0.6594 |
| 5 | U-Net++ | Dice+Focal | 0.7755 | 0.6563 |
| 6 | **SegFormer B2** | **Dice+Focal** | **0.8041** | **0.6921** |

### 2b.2 Standalone Evaluation Script ✓
**File:** `src/evaluation/evaluate.py`
- Reuses `run_evaluation()`, `build_class_metrics()`, `build_runtime()` from `visualize_failures.py`
- CLI: `--checkpoint`, `--config`, `--save`, `--csv`
- No matplotlib/OpenCV — metrics only

### 2b.3 Visualization Updates ✓
**File:** `src/visualization/visualize_failures.py` (modified)
- Added `save_best_grid()` — top-N best predictions per prompt
- Added `title` and `pre_sorted` params to `save_failure_grid()`
- Added `--best-n` CLI arg
- Removed `build_suggestions()` — failures are dataset noise, not systematic model errors

### 2b.4 Documentation ✓

| File | Description |
|------|-------------|
| `docs/final_report_v2.md` | New final report — 6 baselines, FiLM architecture diagram, per-prompt metrics, classical CV comparison, feature channel analysis |
| `docs/supplementary_work.md` | Classical CV evaluation, 7 visualization tools, failure analysis approach, preprocessing pipeline |
| `docs/additional_experiments.md` | 5 boundary/crack_aug experiments (training complete, evaluation pending) |
| `README.md` | Updated with current results and new documentation links |

### 2b.5 Additional Experiments (Training Complete, Evaluation Pending) ✓
5 experiments exploring boundary loss and crack width augmentation:
1. U-Net + Boundary Loss (dice_focal_boundary, γ=2.0)
2. SegFormer + Boundary Loss (dice_focal_boundary, γ=2.0)
3. U-Net + Boundary + Focal γ=3.0
4. SegFormer + Boundary + Focal γ=3.0
5. SegFormer + CrackWidthAug (k=2-7, p=0.6)

Evaluation commands in `docs/additional_experiments.md`.

---

## 📂 Dataset Status

| Dataset | Train | Valid | Test | Status |
|---------|-------|-------|------|--------|
| DryWall-Join-Detect | ✅ 481 (dedup + cleaned) | ✅ 202 | N/A | Pickle ready |
| Cracks | ✅ 907 (dedup done) | ✅ 201 | 4 images | Pickle ready |
| **Merged** | ✅ 1,388 | ✅ 403 | ✅ 4 | Ready |

### Commands Reference (updated paths)
```bash
# Preprocessing (scripts now in preprocess_cleaning/)
uv run python src/data/preprocess_cleaning/preprocess.py drywall
uv run python src/data/preprocess_cleaning/preprocess.py cracks

# Merge
uv run python src/data/preprocess_cleaning/merge_datasets.py

# Quality checks
uv run python src/data/preprocess_cleaning/mask_quality_checks.py --dataset cracks

# Train
uv run python main.py --config src/configs/experiment.yaml

# Failure analysis (run as single line on server)
uv run python src/visualization/visualize_failures.py --checkpoint experiments/<run>/ckpts/best_model.pt --config src/configs/experiment.yaml
```
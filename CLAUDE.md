# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Rules

- **Do NOT run scripts** — write code to files and let the user run them manually
- **Data cleaning on train dataset only** — never modify valid/test splits during cleaning
- **Preprocessing scripts live in `src/data/preprocess_cleaning/`** — dedup, merge, QA, preprocess scripts are all in that subfolder; `src/data/` only contains `dataset.py`
- **Overlapping scripts:** `mask_quality_checks.py` is the consolidated mask/annotation quality checker (replaces `debug_annotations.py`, `quality_checks_drywall.py`, and the QA portions of `visualize_tiny_annotations.py`)
- **Checkpoint arch:** Always read `ckpt['config'].model` from a checkpoint to reconstruct the correct architecture — never rely solely on the YAML config (which may differ from the trained model)
- **Visualization scripts are headless-safe** — all OpenCV-based viewers check `$DISPLAY`/`$WAYLAND_DISPLAY` before opening windows

## Project Overview

This is a prompted segmentation project for drywall quality assurance. The goal is to train/fine-tune text-conditioned segmentation models that can produce binary masks for:
- **"segment crack"** (Cracks Dataset)
- **"segment taping area"** (Drywall Join Dataset)

## Dataset Structure

Two COCO-format datasets are located in `datasets/`:

### Drywall-Join-Detect.v2i.coco
- Contains images of drywall joints for taping area segmentation
- Structure: `train/` and `valid/` directories with images and `_annotations.coco.json`
- **Train split (after dedup):** 481 images, 623 annotations (479 with annotations, 2 without)
- **Valid split:** 202 images, 202 with annotations
- Deduplication removed 339 duplicate images from train split (820 → 481)
- License: CC BY 4.0
- Source: https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect

### cracks.v1i.coco
- Contains images of cracks for crack segmentation
- Structure: `train/`, `valid/`, and `test/` directories with images and `_annotations.coco.json`
- **Train split (after dedup):** 907 images (4,257 Roboflow augmentation variants moved to `duplicates/`)
- **Valid split:** 201 images | **Test split:** 4 images
- License: CC BY 4.0
- Source: https://universe.roboflow.com/segtasks/cracks-3ii36-z95xm

### Merged Dataset (ready for training)
- `processed_data/merged/train/merged_train.pkl` — 1,388 samples (907 cracks + 481 drywall)
- `processed_data/merged/valid/merged_valid.pkl` — 403 samples (201 cracks + 202 drywall)
- `processed_data/merged/test/merged_test.pkl` — 4 samples (cracks only)

### Dataset Format
- Annotations: COCO JSON format with segmentation polygons
- Category IDs: 0 for "Drywall-Join" (drywall dataset), varies for cracks dataset
- All images resized to 640×640 for training
- Prompt IDs: `0` = "segment crack", `1` = "segment taping area"

## Training Pipeline Architecture

The training pipeline uses **FiLM (Feature-wise Linear Modulation)** for prompt conditioning across all models.

### Data Pipeline
```
COCO JSON → preprocess.py → per-dataset pickles → merge_datasets.py → merged pickle
→ dataset.py (PyTorch Dataset) → Albumentations augmentation → DataLoader → Model
```

### Prompt Conditioning (FiLM)
- `prompt_encoder.py`: Embeds prompt IDs (0=crack, 1=taping area) into dense vectors
- `film_wrapper.py`: FiLMBlock modulates encoder features at every scale (γ*feat + β), with auxiliary prompt injection at the decoder bottleneck
- All SMP models (U-Net, U-Net++, DeepLabV3+) are wrapped via `FiLMConditionedModel`
- SegFormer has its own FiLM integration reshaping between `(B, seq_len, C)` and `(B, C, H, W)`

### Key Files

| File | Purpose |
|------|---------|
| `src/data/preprocess_cleaning/preprocess.py` | COCO → pickle (masks {0,255}), supports both datasets via CLI arg |
| `src/data/preprocess_cleaning/merge_datasets.py` | Merges drywall + cracks pickles, exports PNG masks |
| `src/data/preprocess_cleaning/dedup_drywall_v2.py` | Annotation-first drywall deduplication |
| `src/data/preprocess_cleaning/dedup_cracks.py` | Base-name + Union-Find cracks deduplication |
| `src/data/preprocess_cleaning/mask_quality_checks.py` | Consolidated QA: 14 checks + annotation cleaning |
| `src/data/preprocess_cleaning/verify_dedup_cracks.py` | 7-check post-dedup verification |
| `src/data/dataset.py` | PyTorch Dataset + DataLoader factory |
| `src/augmentations/transforms.py` | 5-tier Albumentations pipelines + CrackWidthAugmentation |
| `src/models/smp_models.py` | SMP model factory (Unet, UnetPlusPlus, DeepLabV3Plus) |
| `src/models/segformer.py` | SegFormer B2 with FiLM conditioning |
| `src/models/film_wrapper.py` | FiLMBlock + FiLMConditionedModel wrapper |
| `src/models/prompt_encoder.py` | Shared prompt embedding (nn.Embedding) |
| `src/training/trainer.py` | Training loop with AMP, early stopping, TensorBoard |
| `src/training/losses.py` | Dice, BCE, Focal, Boundary, Combined losses |
| `src/training/metrics.py` | mIoU, Dice, Precision, Recall tracking |
| `src/configs/config.py` | ExperimentConfig dataclass + YAML loader |
| `src/configs/experiment.yaml` | Default experiment config (UnetPlusPlus/resnet34) |
| `src/configs/segformer_experiment.yaml` | SegFormer B2 config |
| `src/utils/seed.py` | `set_seed()` + `worker_init_fn()` for reproducibility |
| `src/visualization/viz_utils.py` | Shared utilities: load_pickle, overlay_rgb/bgr, save_grid |
| `src/visualization/visualize_dataset.py` | Matplotlib grid exports (CLI: dataset + split args) |
| `src/visualization/visualize_dataloader.py` | Post-augmentation DataLoader visualization |
| `src/visualization/visualize_interactive.py` | Interactive OpenCV viewer for dataset pickles |
| `src/visualization/visualize_failures.py` | **Failure analysis**: inference + per-class metrics + interactive browser |
| `src/visualization/verify_drywall.py` | 4-panel bbox→mask alignment verification |
| `src/visualization/visualize_tiny_annotations.py` | QA visualization: tiny/bleed/disconnected modes |
| `main.py` | Training entry point (`uv run python main.py --config ...`) |

### Augmentation Tiers (for ablation)
`baseline` → `geometric` → `photometric` → `edge` → `full` (cumulative)

## Experiment Structure

Runs are saved to `experiments/{exp_name}_{timestamp}/`:
```
experiments/{run_name}/
├── ckpts/
│   ├── best_model.pt          # best val Dice checkpoint
│   └── checkpoint_epoch_N.pt  # per-improvement checkpoints
├── logs/                      # TensorBoard event files
├── predictions/               # {idx}__{prompt_slug}.png (values {0,255})
└── failure_analysis/          # output of visualize_failures.py
    ├── failure_grid.png        # top-50 worst samples (4-panel overlay)
    ├── per_sample_metrics.csv  # per-image IoU/Dice/Prec/Recall
    └── saved_failures/         # interactively saved failure images
```

Checkpoint format:
```python
{
    'epoch': int,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'metrics': {'dice': float, 'miou': float, ...},
    'config': ExperimentConfig,   # ← arch info lives here
}
```

## Failure Analysis

```bash
# Full run (headless-safe, auto-skips interactive browser on server)
uv run python src/visualization/visualize_failures.py --checkpoint experiments/<run>/ckpts/best_model.pt --config src/configs/experiment.yaml

# Worst-50 failures, no interactive window
uv run python src/visualization/visualize_failures.py --checkpoint <ckpt> --config <cfg> --no-interactive --top-n 50

# Filter to only IoU < 0.5 failures
uv run python src/visualization/visualize_failures.py --checkpoint <ckpt> --config <cfg> --iou-threshold 0.5
```

Outputs:
- `experiments/<run>/failure_analysis/failure_grid.png` — top-N worst (4-panel: input | input+pred | input+GT | combined overlay)
- `experiments/<run>/failure_analysis/per_sample_metrics.csv` — sortable per-image scores
- `reports/failure_analysis_{exp_name}.txt` — per-class metrics + runtime + suggestions

## Training

```bash
# Train (default config: UnetPlusPlus / resnet34 / full augmentation)
uv run python main.py --config src/configs/experiment.yaml

# SegFormer
uv run python main.py --config src/configs/segformer_experiment.yaml

# Resume from checkpoint
uv run python main.py --config src/configs/unet_resume_cosine.yaml
```

## Project Milestones

### Milestone 1: Dataset Setup & Preprocessing ✅ COMPLETE
- Drywall dataset: 481 train (dedup + cleaned), 202 valid
- Cracks dataset: 907 train (dedup done), 201 valid
- Merged pickle ready: 1,388 train / 403 valid
- All visualization and QA tools built

### Milestone 2: Baseline Setup ✅ TRAINING DONE
- U-Net, U-Net++, SegFormer B2 trained with FiLM conditioning
- Failure analysis tooling complete (`visualize_failures.py`)
- Per-class (crack vs taping) metrics tracked separately
- Runtime / footprint / avg-inference-time reporting in place

### Milestone 3: Zero-Shot & Pretrained Model Analysis
- Run zero-shot inference on: SAM, SAM 2, FastSAM
- Compare qualitative results (visual & mIoU)
- Record inference time, GPU memory, FPS

### Milestone 4: Fine-Tuning & Model Comparison
- Fine-tune CNN: DeepLabV3, DINOv3
- Fine-tune Transformer: MaskFormer, Segmenter
- Ablations: image sizes (320/480/640), loss functions, augmentation tiers

### Milestone 5: Optimization & Segmentation Tricks
- Boundary / edge-aware loss
- CLAHE / Laplacian pre-processing
- Test-time augmentation (flips, multi-scale)
- Mixed precision profiling

### Milestone 6: Final Evaluation & Reporting
- Evaluate on held-out test set
- Generate Orig | GT | Pred visuals
- Performance summary table (accuracy vs latency vs model size)

### Milestone 7: Latency Optimization (Optional)
- Export to ONNX / TensorRT
- Quantize FP16/INT8
- Plot Accuracy vs Latency curve

## Key Evaluation Metrics

- **mIoU** (mean Intersection over Union)
- **Dice coefficient** (per-class: crack vs taping)
- **Inference time** (avg ms/image)
- **GPU memory usage** (peak MB)
- **Model size** (total parameters)

## Expected Outputs

### Prediction Masks
- Format: `{id}__{prompt_slug}.png` — values {0, 255}
- Prompts: `segment_crack` or `segment_taping_area`

### Reports (saved to `reports/`)
- `failure_analysis_{exp_name}.txt` — per-class metrics + runtime + suggestions
- `mask_quality_report.txt` — QA check results
- `per_sample_metrics.csv` — per-image scores (inside experiment folder)

### Visualizations
- 4-panel failure grid: Input | Input+Pred | Input+GT | Combined overlay
- Accuracy vs Latency vs Memory plots (Milestone 6)

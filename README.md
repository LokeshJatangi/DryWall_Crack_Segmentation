# Prompted Segmentation for Drywall QA

Text-conditioned segmentation models for drywall quality assurance using FiLM conditioning.

## Overview

A single model accepts an image + text prompt and produces a binary mask for:
- **"segment crack"** — crack detection in drywall/concrete surfaces
- **"segment taping area"** — drywall joint/taping area identification

**Best result: SegFormer B2 — mIoU 0.847, Dice 0.845** on validation (403 samples).

## Results Summary

| Model | mIoU | Dice | Precision | Recall | Params | Checkpoint |
|-------|------|------|-----------|--------|--------|------------|
| U-Net / ResNet-34 | 0.836 | 0.831 | 0.818 | 0.845 | 24.8M | 284 MB |
| U-Net++ / ResNet-34 | 0.831 | 0.826 | 0.820 | 0.831 | 26.4M | 303 MB |
| **SegFormer B2 / MiT-B2** | **0.847** | **0.845** | **0.816** | **0.876** | **27.7M** | **318 MB** |

All models use FiLM (Feature-wise Linear Modulation) for prompt conditioning. See [docs/final_report.md](docs/final_report.md) for full analysis.

## Quick Start

```bash
# Install dependencies
uv sync

# Train (default config: UnetPlusPlus / resnet34 / full augmentation)
uv run python main.py --config src/configs/experiment.yaml

# Train SegFormer B2
uv run python main.py --config src/configs/segformer_experiment.yaml

# Resume from checkpoint
uv run python main.py --config src/configs/unet_resume_cosine.yaml

# Failure analysis
uv run python src/visualization/visualize_failures.py \
  --checkpoint experiments/<run>/ckpts/best_model.pt \
  --config <config.yaml> --no-interactive

# Visualize dataset interactively
uv run python src/visualization/visualize_interactive.py processed_data/merged/valid/merged_valid.pkl
```

**Visualizer Controls:** `d` next | `a` prev | `p` play/pause | `q` quit

## Project Status

- ✅ **Milestone 1:** Dataset preprocessing, deduplication & QA (COMPLETE)
- ✅ **Milestone 2:** Baseline training — 3 model families + failure analysis (COMPLETE)
- ⏳ **Milestone 3:** Zero-shot evaluation — SAM, SAM 2, FastSAM (NEXT)

## Datasets

| Dataset | Train | Valid | Test | Prompt |
|---------|-------|-------|------|--------|
| Drywall-Join | 481 (dedup from 820) | 202 | — | "segment taping area" |
| Cracks | 907 (dedup from 5,164) | 201 | 4 | "segment crack" |
| **Merged** | **1,388** | **403** | **4** | both |

All images 640x640, masks binary {0, 255}, COCO-format annotations. License: CC BY 4.0.

## Project Structure

```
src/
├── data/
│   ├── dataset.py                     # PyTorch Dataset + DataLoader factory
│   └── preprocess_cleaning/           # Dedup, merge, QA, preprocess scripts
├── augmentations/transforms.py        # 5-tier Albumentations + CrackWidthAugmentation
├── models/
│   ├── prompt_encoder.py              # Prompt embedding (nn.Embedding)
│   ├── film_wrapper.py                # FiLM modulation + auxiliary decoder injection
│   ├── smp_models.py                  # SMP factory (U-Net, U-Net++, DeepLabV3+)
│   └── segformer.py                   # SegFormer B2 with FiLM
├── training/
│   ├── trainer.py                     # Training loop: AMP, early stopping, TensorBoard
│   ├── losses.py                      # Dice, DiceBCE, Focal, Boundary, Combined
│   └── metrics.py                     # mIoU, Dice, Precision, Recall tracker
├── configs/                           # YAML experiment configs
│   ├── experiment.yaml                # Default: UnetPlusPlus / resnet34
│   └── segformer_experiment.yaml      # SegFormer B2
├── utils/seed.py                      # set_seed() + worker_init_fn()
└── visualization/
    ├── viz_utils.py                   # Shared: load_pickle, overlay, save_grid
    ├── visualize_dataset.py           # Matplotlib grid exports
    ├── visualize_dataloader.py        # Post-augmentation visualization
    ├── visualize_interactive.py       # Interactive OpenCV viewer
    ├── visualize_failures.py          # Failure analysis: inference + metrics + browser
    ├── verify_drywall.py              # 4-panel bbox→mask verification
    └── visualize_tiny_annotations.py  # QA viz: tiny/bleed/disconnected modes
main.py                                # Training entry point
experiments/                           # Saved runs (ckpts, logs, predictions)
processed_data/                        # Preprocessed pickle files + PNG masks
reports/                               # QA reports, classical CV evaluation
docs/                                  # Final report, progress log, setup guide
```

## Documentation

| Document | Description |
|----------|-------------|
| [Final Report](docs/final_report.md) | Full methodology, results & analysis |
| [Setup Guide](docs/setup_guide.md) | Installation, preprocessing, visualization |
| [Project Plan](docs/Plan_Milestone.md) | 7-milestone roadmap |
| [Progress Log](docs/progress.md) | Session-by-session progress tracking |

## Development

### Prerequisites
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- GPU with CUDA support (training)

### Reproducibility
- Seed: 24 (set for Python, NumPy, PyTorch, CUDA, DataLoader workers)
- All configs: `src/configs/*.yaml`
- Checkpoints save full `ExperimentConfig` for architecture reconstruction

## License

Datasets: CC BY 4.0

# Prompted Segmentation for Drywall QA

Text-conditioned segmentation models for drywall quality assurance using FiLM conditioning.

## Overview

A single model accepts an image + text prompt and produces a binary mask for:
- **"segment crack"** — crack detection in drywall/concrete surfaces
- **"segment taping area"** — drywall joint/taping area identification

**Best result: SegFormer B2 (Dice+Focal) — Dice 0.8041, mIoU 0.6921** on validation (403 samples).

## Results Summary

| # | Model | Loss | Dice | mIoU | Precision | Recall | Params |
|---|-------|------|------|------|-----------|--------|--------|
| 1 | U-Net / ResNet-34 | Dice+BCE | 0.7670 | 0.6520 | 0.7872 | 0.8112 | 24.8M |
| 2 | U-Net++ / ResNet-34 | Dice+BCE | 0.7705 | 0.6556 | 0.7975 | 0.8048 | 26.4M |
| 3 | SegFormer B2 / MiT-B2 | Dice+BCE | 0.7939 | 0.6812 | 0.8087 | 0.8350 | 27.7M |
| 4 | U-Net / ResNet-34 | Dice+Focal | 0.7760 | 0.6594 | 0.7749 | 0.8352 | 24.8M |
| 5 | U-Net++ / ResNet-34 | Dice+Focal | 0.7755 | 0.6563 | 0.7815 | 0.8304 | 26.4M |
| 6 | **SegFormer B2 / MiT-B2** | **Dice+Focal** | **0.8041** | **0.6921** | **0.8090** | **0.8522** | **27.7M** |

All models use FiLM (Feature-wise Linear Modulation) for prompt conditioning. See [docs/final_report_v2.md](docs/final_report_v2.md) for full analysis.

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

# Evaluate a checkpoint (metrics only, no GUI)
uv run python src/evaluation/evaluate.py \
  --checkpoint experiments/<run>/ckpts/best_model.pt \
  --config <config.yaml> --save reports/eval.txt --csv reports/eval.csv

# Failure analysis (visual)
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
├── evaluation/
│   └── evaluate.py                   # Standalone evaluation (metrics only, no GUI)
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
| [Final Report](docs/final_report_v2.md) | Full methodology, 6 baselines, per-prompt metrics & analysis |
| [Additional Experiments](docs/additional_experiments.md) | Boundary loss, crack width augmentation experiments |
| [Supplementary Work](docs/supplementary_work.md) | Classical CV evaluation, visualization tooling, preprocessing pipeline |
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

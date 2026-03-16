# Additional Experiments — Boundary Loss & Crack Width Augmentation

This document covers 5 additional experiments exploring boundary-aware loss and crack-specific augmentation, building on the 6 baseline experiments from the final report.

## 1. Motivation

Analysis of baseline results revealed two areas for potential improvement:

**Boundary precision:** The standard Dice+Focal loss treats all pixels equally. For thin structures like cracks, boundary pixels are disproportionately important — a boundary-aware loss could improve segmentation of fine edges.

**Crack width variability:** The ~18-20pp IoU gap between crack and taping segmentation suggests that crack-specific augmentation (varying crack width during training) could help the model generalize to thin/thick crack variations.

## 2. Experiment Details

### Loss Configuration: Dice+Focal+Boundary

The combined loss adds a boundary loss component (20% weight):
- **Boundary loss:** Computes distance transform of GT boundary, penalizes predictions far from GT edges
- **Effective weights:** 40% Dice + 40% Focal + 20% Boundary

### Experiments

| # | Name | Architecture | Loss | Key Change | Epochs | Dir |
|---|------|-------------|------|------------|--------|-----|
| 1 | U-Net + Boundary | U-Net / ResNet-34 | dice_focal_boundary | +20% boundary loss, γ=2.0 | 75 | `unet_resnet34_boundary_20260316_041743` |
| 2 | SegFormer + Boundary | SegFormer / MiT-B2 | dice_focal_boundary | +20% boundary loss, γ=2.0 | 75 | `segformer_b2_boundary_20260316_041812` |
| 3 | U-Net + Boundary (γ=3.0) | U-Net / ResNet-34 | dice_focal_boundary | +boundary, focal γ=3.0 | 75 | `unet_resnet34_boundary_focal_g_3.0_20260316_042052` |
| 4 | SegFormer + Boundary (γ=3.0) | SegFormer / MiT-B2 | dice_focal_boundary | +boundary, focal γ=3.0 | 75 | `segformer_b2_boundary_focal_g_3.0_20260316_042152` |
| 5 | SegFormer + CrackAug | SegFormer / MiT-B2 | dice_focal | CrackWidthAug (k=2-7, p=0.6) | 50 | `segformer_b2_crack_aug_20260316_043049` |

**Note on configs:** Experiments 1-2 use focal γ=2.0, experiments 3-4 use focal γ=3.0. The YAML config files on disk were modified between runs, so the authoritative configuration is stored in the checkpoint (`ckpt['config']`), not the YAML.

### Hyperparameters (unchanged from baselines unless noted)

| Parameter | U-Net Experiments | SegFormer Experiments |
|-----------|------------------|----------------------|
| Learning rate | 1e-4 | 6e-5 |
| Batch size | 32 | 8 |
| Weight decay | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Early stopping | patience=10 | patience=10 |
| Augmentation | full tier | full tier |

### CrackWidthAugmentation Details (Experiment 5)

Applied as a post-augmentation step on crack masks only:
- **Kernel range:** 2-7 (elliptical structuring element)
- **Probability:** 0.6
- **Operation:** Random dilation or erosion (50/50 when not dilate-only)
- **Purpose:** Simulate varying crack widths to improve model robustness to thin/thick cracks

## 3. Results

*Pending evaluation — run the commands below to generate metrics.*

| # | Model | Loss | Dice | mIoU | Crack IoU | Taping IoU |
|---|-------|------|------|------|-----------|------------|
| 1 | U-Net + Boundary | dice_focal_boundary (γ=2.0) | TBD | TBD | TBD | TBD |
| 2 | SegFormer + Boundary | dice_focal_boundary (γ=2.0) | TBD | TBD | TBD | TBD |
| 3 | U-Net + Boundary (γ=3.0) | dice_focal_boundary (γ=3.0) | TBD | TBD | TBD | TBD |
| 4 | SegFormer + Boundary (γ=3.0) | dice_focal_boundary (γ=3.0) | TBD | TBD | TBD | TBD |
| 5 | SegFormer + CrackAug | dice_focal | TBD | TBD | TBD | TBD |

Best baseline for comparison: SegFormer B2 Dice+Focal — Dice 0.8041, mIoU 0.6921

## 4. Evaluation Commands

```bash
# Experiment 1: U-Net + Boundary (γ=2.0)
uv run python src/evaluation/evaluate.py \
    --checkpoint experiments/unet_resnet34_boundary_20260316_041743/ckpts/best_model.pt \
    --config src/configs/unet_boundary.yaml \
    --save reports/eval_unet_boundary.txt \
    --csv reports/eval_unet_boundary.csv

# Experiment 2: SegFormer + Boundary (γ=2.0)
uv run python src/evaluation/evaluate.py \
    --checkpoint experiments/segformer_b2_boundary_20260316_041812/ckpts/best_model.pt \
    --config src/configs/segformer_boundary.yaml \
    --save reports/eval_segformer_boundary.txt \
    --csv reports/eval_segformer_boundary.csv

# Experiment 3: U-Net + Boundary (γ=3.0)
uv run python src/evaluation/evaluate.py \
    --checkpoint experiments/unet_resnet34_boundary_focal_g_3.0_20260316_042052/ckpts/best_model.pt \
    --config src/configs/unet_boundary.yaml \
    --save reports/eval_unet_boundary_g3.txt \
    --csv reports/eval_unet_boundary_g3.csv

# Experiment 4: SegFormer + Boundary (γ=3.0)
uv run python src/evaluation/evaluate.py \
    --checkpoint experiments/segformer_b2_boundary_focal_g_3.0_20260316_042152/ckpts/best_model.pt \
    --config src/configs/segformer_boundary.yaml \
    --save reports/eval_segformer_boundary_g3.txt \
    --csv reports/eval_segformer_boundary_g3.csv

# Experiment 5: SegFormer + CrackAug
uv run python src/evaluation/evaluate.py \
    --checkpoint experiments/segformer_b2_crack_aug_20260316_043049/ckpts/best_model.pt \
    --config src/configs/segformer_crack_aug.yaml \
    --save reports/eval_segformer_crack_aug.txt \
    --csv reports/eval_segformer_crack_aug.csv
```

To also generate failure/best grids for any experiment:
```bash
uv run python src/visualization/visualize_failures.py \
    --checkpoint experiments/<dir>/ckpts/best_model.pt \
    --config <config.yaml> \
    --no-interactive --top-n 20 --best-n 5
```

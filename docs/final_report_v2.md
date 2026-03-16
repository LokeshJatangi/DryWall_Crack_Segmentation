# Prompted Segmentation for Drywall QA — Final Report

## 1. Problem Statement

The goal is to train text-conditioned segmentation models that take a 640x640 image and a prompt, and produce a binary mask (values {0, 255}) for the specified defect or region.

Two prompts are supported:
- **"segment crack"** (prompt ID = 0) — thin, low-contrast fractures in drywall/concrete surfaces
- **"segment taping area"** (prompt ID = 1) — larger drywall joint/tape regions

**Key assumption:** The prompt vocabulary is **closed and known a priori** — only the two prompts above are supported. Because the set of segmentation tasks is fixed, we use integer prompt IDs (0 = crack, 1 = taping area) fed into a learned embedding, rather than a full NLP/text-encoder pipeline. No tokenization or language model is involved. This simplification is valid because the prompts are not user-generated or dynamic — they are predetermined by the QA application.

## 2. Dataset

| Dataset | Train | Valid | Test | Source |
|---------|-------|-------|------|--------|
| Drywall-Join | 481 | 202 | — | Roboflow (CC BY 4.0) |
| Cracks | 907 | 201 | 4 | Roboflow (CC BY 4.0) |
| **Merged** | **1,388** | **403** | **4** | — |

**Format:** COCO JSON with segmentation polygons, converted to binary masks via preprocessing pipeline. All images resized to 640x640.

**Known issues:**
- 2 drywall train images have no annotations (included as negative samples)
- 2 potential train/valid leakage pairs identified in drywall dataset (not removed — borderline cases with different crops)

## 3. Data Cleaning

All cleaning operations were applied to **train splits only** — validation and test splits were never modified.

**Deduplication:**
- **Drywall:** Annotation-first dedup using polygon similarity → 339 duplicates removed (820 → 481)
- **Cracks:** Base-name + Union-Find grouping to identify Roboflow augmentation variants → 4,257 augmented copies moved to `duplicates/` (5,164 → 907)

**Quality Assurance:**
14-point automated QA via `mask_quality_checks.py`:
- Tiny annotations (<0.5% coverage)
- Mask bleed (>1% outside bounding box)
- Disconnected components
- Degenerate polygons
- Empty/all-white masks
- Format consistency checks

Results documented in `reports/mask_quality_report.txt`.

**Visual samples:** Example duplicate pairs identified and removed during cleaning are saved in `docs/data_cleaning_viz_samples/` for reference.

## 4. Methodology

### 4.1 FiLM Conditioning

All models use Feature-wise Linear Modulation (FiLM) to condition segmentation on the prompt. The prompt embedding is injected at every encoder scale plus an auxiliary injection at the decoder bottleneck.

```
Prompt ID ──→ PromptEncoder ──→ prompt_embed (B, 128)
               nn.Embedding(2,128)     |
                                       |──────────────────────────────────┐
                                       |                                  |
Image (B,3,640,640)                    |                                  |
    |                                  |                                  |
    v                                  |                                  |
┌─────────────────┐                    |                                  |
|  SMP Encoder     |                   |                                  |
|  (ResNet-34)     |                   |                                  |
└─────────────────┘                    |                                  |
    |                                  |                                  |
    |── features[0] (64ch) ──────────────────────────────────────┐       |
    |   [no FiLM — raw input]                                    |       |
    |                                                            |       |
    |── features[1] (64ch)  ──→ FiLMBlock_1(γ·f+β) ──────────┐  |       |
    |                                                         |  |       |
    |── features[2] (128ch) ──→ FiLMBlock_2(γ·f+β) ────────┐ |  |       |
    |                                                       | |  |       |
    |── features[3] (256ch) ──→ FiLMBlock_3(γ·f+β) ──────┐ | |  |       |
    |                                                     | | |  |       |
    └── features[4] (512ch) ──→ FiLMBlock_4(γ·f+β) ──┐   | | |  |       |
                                                      |   | | |  |       |
              Auxiliary Injection <────────────────────|───|─|─|──|───────┘
              Linear(128→512)+ReLU                    |   | | |  |
                        |                             |   | | |  |
                        └──────────→ (+) ─────────────┘   | | |  |
                                      |                   | | |  |
                              ┌───────┴───────────────────┴─┴─┴──┘
                              |  modulated features [0..4]
                              v
                     ┌─────────────────┐
                     |   SMP Decoder    |
                     | (skip connects)  |
                     └─────────────────┘
                              |
                              v
                     ┌─────────────────┐
                     | Segmentation    |
                     | Head            |
                     └─────────────────┘
                              |
                              v
                     Output (B, 1, 640, 640)
```

Each **FiLMBlock** learns a per-channel scale (γ) and shift (β) from the prompt embedding:
- `γ = Linear(128 → C)` applied as channel-wise multiplication
- `β = Linear(128 → C)` applied as channel-wise addition
- Result: `γ * feature + β` (broadcast over spatial dims)

The **auxiliary injection** adds a separate projection of the prompt embedding directly to the deepest encoder features, providing an additional conditioning signal at the bottleneck.

For **SegFormer**, FiLM blocks reshape between `(B, seq_len, C)` and `(B, C, H, W)` to match the transformer's sequence format.

### 4.2 Model Architectures

| Model | Framework | Encoder | Parameters |
|-------|-----------|---------|------------|
| U-Net | SMP | ResNet-34 (ImageNet) | 24,766,865 |
| U-Net++ | SMP | ResNet-34 (ImageNet) | 26,409,105 |
| SegFormer B2 | HuggingFace | MiT-B2 (ImageNet-1k) | 27,677,889 |

All architectures are wrapped with `FiLMConditionedModel` (or equivalent SegFormer integration) for prompt conditioning. Encoders are initialized from ImageNet pretrained weights.

### 4.3 Augmentation

All experiments use the `full` augmentation tier:

| Transform | Parameters |
|-----------|------------|
| HorizontalFlip | p=0.5 |
| VerticalFlip | p=0.5 |
| RandomRotate90 | p=0.5 |
| ShiftScaleRotate | shift=0.1, scale=0.15, rotate=30, p=0.6 |
| ElasticTransform | alpha=80, sigma=10, p=0.3 |
| GridDistortion | p=0.3 |
| RandomBrightnessContrast | brightness=0.2, contrast=0.2, p=0.5 |
| HueSaturationValue | hue=15, sat=20, val=15, p=0.3 |
| GaussNoise | var_limit=(5,25), p=0.3 |
| GaussianBlur | blur_limit=5, p=0.2 |
| CLAHE | clip_limit=3.0, p=0.3 |
| CoarseDropout | max_holes=6, max_height=40, max_width=40, p=0.3 |
| ImageNet Normalize | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |

## 5. Training Setup

| Parameter | Value |
|-----------|-------|
| Image size | 640 x 640 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 50 |
| Early stopping | patience=10 (on val Dice) |
| Seed | 24 |
| Mixed precision | AMP (fp16) |

**Learning rates:**
- U-Net / U-Net++: 1e-4
- SegFormer B2: 6e-5

**Batch sizes:**
- U-Net: 32
- U-Net++: 12
- SegFormer B2: 8

**Loss configurations:**
- **Runs 1-3 (Dice+BCE):** 0.5 * DiceLoss + 0.5 * BCEWithLogitsLoss
- **Runs 4-6 (Dice+Focal):** 0.5 * DiceLoss + 0.5 * FocalLoss (α=0.75, γ=2.0)

## 6. Results

### 6.1 Validation Metrics — All 6 Baselines

| # | Model | Loss | Dice | mIoU | Precision | Recall |
|---|-------|------|------|------|-----------|--------|
| 1 | U-Net | Dice+BCE | 0.7670 | 0.6520 | 0.7872 | 0.8112 |
| 2 | U-Net++ | Dice+BCE | 0.7705 | 0.6556 | 0.7975 | 0.8048 |
| 3 | SegFormer B2 | Dice+BCE | 0.7939 | 0.6812 | 0.8087 | 0.8350 |
| 4 | U-Net | Dice+Focal | 0.7760 | 0.6594 | 0.7749 | 0.8352 |
| 5 | U-Net++ | Dice+Focal | 0.7755 | 0.6563 | 0.7815 | 0.8304 |
| 6 | **SegFormer B2** | **Dice+Focal** | **0.8041** | **0.6921** | **0.8090** | **0.8522** |

**Observations:**
- SegFormer B2 outperforms both CNN architectures across all metrics
- Dice+Focal provides a consistent improvement over Dice+BCE for all architectures (+0.5-1.1pp Dice)
- Focal loss improves recall (+1-3pp) with minimal precision trade-off, consistent with its design for hard example mining

### 6.2 Per-Prompt Metrics

| # | Model | Loss | Crack Dice | Crack IoU | Taping Dice | Taping IoU |
|---|-------|------|------------|-----------|-------------|------------|
| 1 | U-Net | Dice+BCE | 0.6778 | 0.5356 | 0.8557 | 0.7678 |
| 2 | U-Net++ | Dice+BCE | 0.6895 | 0.5497 | 0.8511 | 0.7610 |
| 3 | SegFormer B2 | Dice+BCE | 0.7174 | 0.5775 | 0.8699 | 0.7843 |
| 4 | U-Net | Dice+Focal | 0.6988 | 0.5584 | 0.8528 | 0.7599 |
| 5 | U-Net++ | Dice+Focal | 0.7073 | 0.5672 | 0.8433 | 0.7449 |
| 6 | **SegFormer B2** | **Dice+Focal** | **0.7334** | **0.5950** | **0.8744** | **0.7888** |

The ~18-20pp IoU gap between taping and crack segmentation is consistent across all models and expected: taping areas are large, well-defined regions while cracks are thin, variable-width structures with inherently noisy ground truth annotations.

### 6.3 Model Footprint & Runtime

| Model | Parameters | Checkpoint | Inference (ms/img) | GPU Peak (MB) |
|-------|------------|------------|-------------------|---------------|
| U-Net (ResNet-34) | 24,766,865 | 284 MB | 17.84 | 8,142 |
| U-Net++ (ResNet-34) | 26,409,105 | 303 MB | 34.11 | 5,895 |
| SegFormer B2 (MiT-B2) | 27,677,889 | 318 MB | 56.77 | 6,086 |

U-Net has the fastest inference (17.84 ms) but highest GPU memory peak. SegFormer is slowest at inference but achieves the best accuracy. U-Net++ offers a middle ground.

### 6.4 Classical CV Baselines

For context, 6 classical computer vision methods were evaluated on 50 samples from each dataset:

| Method | Cracks IoU | Cracks Dice | Drywall IoU | Drywall Dice |
|--------|-----------|-------------|-------------|--------------|
| Gabor filters | 0.2448 | 0.3734 | 0.0255 | 0.0468 |
| Canny + morphology | 0.1414 | 0.2287 | 0.0236 | 0.0452 |
| Otsu threshold | 0.1155 | 0.1896 | 0.0392 | 0.0695 |
| Hough lines | 0.1092 | 0.1777 | 0.0610 | 0.1127 |
| Adaptive threshold | 0.1098 | 0.1870 | 0.0359 | 0.0690 |
| Frangi filters | 0.0748 | 0.1355 | 0.0092 | 0.0181 |

**Best classical:** Gabor filters on cracks (IoU 0.2448), Hough lines on drywall (IoU 0.0610)

**Deep learning improvement over best classical:**
- Cracks: 0.5950 / 0.2448 = **2.4x** IoU improvement
- Drywall: 0.7888 / 0.0610 = **12.9x** IoU improvement

Classical methods completely fail on drywall joint segmentation, which requires understanding of spatial context that only learned features can capture.

### 6.5 Feature Channel Analysis

We also evaluated whether classical CV preprocessing (edge/texture filters) could serve as additional input channels to boost segmentation. Five feature extractors were scored using **Fisher's linear discriminant** (higher = better foreground/background separability):

| Feature Channel | Cracks FLD | Drywall FLD | Time (ms) |
|-----------------|-----------|-------------|-----------|
| Multiscale LoG | 1.466 | 0.096 | 30.0 |
| Gabor filters | 0.967 | 0.045 | 84.5 |
| Sobel edges | 0.422 | 0.019 | 3.7 |
| Laplacian | 0.359 | 0.023 | 4.0 |
| Frangi filters | 0.138 | 0.007 | 1272.8 |

On cracks, Multiscale LoG and Gabor show moderate discriminative power — these could potentially be concatenated as extra input channels. On drywall, all features have near-zero FLD, confirming that low-level filters cannot distinguish taping areas from background. This was explored but not integrated into the final training pipeline, as the ImageNet-pretrained encoders already learn effective features.

### 6.6 Training Curves

TensorBoard screenshots are saved in `tensorboard_screenshots/` and show:
- **Train loss:** All models converge within 30-40 epochs, with SegFormer showing smoother loss curves
- **Val Dice:** Steady improvement with SegFormer reaching peak around epoch 46-48
- **Val mIoU:** Tracks Dice closely; SegFormer achieves consistent lead from epoch ~15 onward

## 7. Visual Examples

### Best Predictions

The top 5 crack and top 5 taping predictions from SegFormer B2 (Dice+Focal) are saved as a grid in:
```
experiments/segformer_b2_film_baseline_20260315_192614/failure_analysis/best_grid.png
```

4-panel format per sample: Input | Input+Pred (green) | Input+GT (red) | Combined (R=GT, G=Pred, Y=Both)

### Failure Cases

The 20 worst predictions (lowest IoU) from SegFormer B2 (Dice+Focal):
```
experiments/segformer_b2_film_baseline_20260315_192614/failure_analysis/failure_grid.png
```

Failure analysis shows that low-scoring samples are predominantly:
- **Thin cracks** with noisy ground truth annotations (imprecise polygon boundaries)
- **Low-contrast cracks** where even human annotation is ambiguous
- **Edge cases** in drywall dataset where the taping area extends to image boundaries

These are dataset-quality and inherent-difficulty issues, not systematic model errors.

### Per-Experiment Reports

Detailed per-class metrics and runtime reports for each experiment are in `reports/`:
- `reports/failure_analysis_unet_resnet34_baseline_20260315_192543.txt`
- `reports/failure_analysis_unet++_resnet34_baseline_20260315_135027.txt`
- `reports/failure_analysis_unet++_resnet34_baseline_20260315_192504.txt`
- `reports/failure_analysis_segformer_b2_film_baseline_20260315_135901.txt`
- `reports/failure_analysis_segformer_b2_film_baseline_20260315_192614.txt`

Per-sample metrics CSVs (sortable by IoU/Dice per image) are in each experiment's `failure_analysis/per_sample_metrics.csv`.

## 8. Additional Experiments (In Progress)

Five additional experiments exploring boundary loss and crack width augmentation have completed training and are pending evaluation:

1. **U-Net + Boundary Loss** — Dice+Focal+Boundary (20% boundary weight)
2. **SegFormer + Boundary Loss** — Dice+Focal+Boundary (20% boundary weight)
3. **U-Net + Boundary + Focal γ=3.0** — Higher focal gamma for harder example focus
4. **SegFormer + Boundary + Focal γ=3.0** — Higher focal gamma for harder example focus
5. **SegFormer + CrackWidthAug** — CrackWidthAugmentation (k=2-7, p=0.6)

Results will be updated once evaluation completes. See [docs/additional_experiments.md](additional_experiments.md) for details.

## 9. Supplementary Work

Detailed documentation of classical CV evaluation, visualization tooling, failure analysis methodology, and data preprocessing pipeline is available in [docs/supplementary_work.md](supplementary_work.md).

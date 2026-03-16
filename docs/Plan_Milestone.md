# 🧠 Prompted Segmentation for Drywall QA — Project Plan

## 🎯 Goal
Train (or fine-tune) a text-conditioned segmentation model so that, given an image and a prompt, it produces a binary mask for:
- **“segment crack”** (Cracks Dataset)
- **“segment taping area”** (Drywall Join Dataset)

---

## 📅 Milestone 1: Dataset Setup & Preprocessing
**Goal:** Prepare a clean, balanced dataset for prompted segmentation.

### ✅ Tasks
- [x] Download Dataset 1 (Drywall-Join-Detect) and Dataset 2 (Cracks-3ii36)
- [x] Merge datasets → unify format: `(image, mask, prompt)` — `merge_datasets.py`
- [x] Resize all images to 640×640
- [x] Split data into train / val / test (decide cross-val later)
- [x] Create dataset loader for CNN & Transformer models — `dataset.py` with Albumentations
- [x] Build visualization tool: input | GT | predicted mask
- [x] Implement duplicate detection — drywall (quadrant + annotation-first), cracks (`dedup_cracks.py`)
- [x] Implement comprehensive quality checks (19-point, `mask_quality_checks.py`)
- [x] Build bbox→mask verification visualizer
- [x] Build annotation issue visualizer — tiny / bleed / disconnected modes
- [x] Export PNG masks from preprocessing (`{0,255}`, single-channel, `{stem}__{prompt}.png`)
- [x] Add annotation cleaning to quality checker (`--remove-tiny`, `--remove-disconnected`)

---

## ⚙️ Milestone 2: Baseline Setup ✅ COMPLETE
**Goal:** Establish a working segmentation model and evaluation baseline.

### ✅ Tasks
- [x] Implement **U-Net baseline** — U-Net, U-Net++, DeepLabV3+, SegFormer B2 with FiLM conditioning
- [x] Use Dice + BCE loss initially — `losses.py` (Dice, DiceBCE, Focal, Boundary, Combined)
- [x] Train 6 baselines (3 arch × 2 loss configs); log mIoU, Dice, runtime
- [x] Save prediction masks in `{id}__{prompt}.png` format — `trainer.py:save_predictions()`
- [x] Validate predictions visually — `best_grid.png` + `failure_grid.png`
- [x] Create a summary table of metrics + runtime — `docs/final_report_v2.md`
- [x] Standalone evaluation script — `src/evaluation/evaluate.py`
- [x] Classical CV baseline comparison (6 methods, 2 datasets)
- [x] Feature channel analysis (Sobel, Gabor, Frangi, LoG, Laplacian)
- [x] 5 additional experiments (boundary loss, crack aug) — training complete

---

## 🔍 Milestone 3: Zero-Shot & Pretrained Model Analysis
**Goal:** Benchmark existing segmentation models without fine-tuning.

### ✅ Tasks
- [ ] Run zero-shot inference on:
  - [ ] SAM
  - [ ] SAM 2
  - [ ] FastSAM (CNN + Transformer)
- [ ] Compare qualitative results (visual & mIoU)
- [ ] Record inference time, GPU memory, FPS
- [ ] Identify best candidate models for fine-tuning

---

## 🧩 Milestone 4: Fine-Tuning & Model Comparison
**Goal:** Fine-tune multiple architectures for both prompts.

### ✅ Tasks
- [ ] Fine-tune CNN models: U-Net, DeepLabV3, DINOv3
- [ ] Fine-tune Transformer models: MaskFormer, Segmenter
- [ ] Fine-tune hybrid models: FastSAM or custom CNN+Transformer
- [ ] Keep same image size (640×640), vary later
- [ ] Perform ablations:
  - [ ] Vary image sizes (320, 480, 640)
  - [ ] Vary model size (small/medium/large)
  - [ ] Try different loss functions (Dice, BCE, Boundary, Focal, Lovasz)
- [ ] Collect results: accuracy vs latency vs model size

---

## 🧠 Milestone 5: Optimization & Segmentation Tricks
**Goal:** Improve segmentation quality and speed.

### ✅ Tasks
- [x] Implement **Boundary loss** or **Edge-aware loss** — `dice_focal_boundary` in `losses.py`
- [x] Add pre-processing filters (Laplacian, CLAHE) — evaluated as feature channels (Fisher discriminant)
- [ ] Apply test-time augmentations (flips, rotations)
- [x] Use mixed precision (FP16) — AMP enabled in all training runs
- [x] Profile GPU usage, memory, FLOPs — reported in `failure_analysis_*.txt`
- [ ] Create “Accuracy vs Latency vs Memory” table

---

## 📊 Milestone 6: Final Evaluation & Reporting ✅ COMPLETE
**Goal:** Summarize results and prepare final report.

### ✅ Tasks
- [ ] Evaluate best model on held-out test set (4 samples — pending)
- [x] Generate visual results — `best_grid.png` (top 5 crack + 5 taping) + `failure_grid.png`
- [x] Create performance summary table — 6 baselines in `docs/final_report_v2.md`
- [x] Document failure cases — failure analysis shows dataset noise, not systematic errors
- [x] Record runtime, model size, average inference/image — footprint table in report
- [x] Write report — `docs/final_report_v2.md` (9 sections, no appendices)
- [x] Supplementary documentation — `docs/supplementary_work.md`
- [x] Additional experiments documentation — `docs/additional_experiments.md`
- [x] Finalize report (Markdown + visuals)

---

## ⚡ Milestone 7 (Optional): Latency Optimization
**Goal:** Optimize for deployment or real-time QA applications.

### ✅ Tasks
- [ ] Export best model to ONNX / TensorRT
- [ ] Quantize (FP16 / INT8)
- [ ] Measure latency improvements (FPS, memory)
- [ ] Plot Accuracy vs Latency curve

---

## 📘 Deliverables
- Dataset split details (counts, sizes)
- Training/eval metrics (mIoU, Dice)
- Visual examples (orig | GT | pred)
- Report with tables, visuals, and runtime summary
- Accuracy vs Latency summary plot

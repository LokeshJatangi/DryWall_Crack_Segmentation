"""
Failure analysis and prediction visualization.

Loads a trained checkpoint, runs inference on the validation set, computes
per-sample IoU/Dice, and provides:
  - Interactive OpenCV browser sorted by worst scores (with save option)
  - Per-class (crack vs taping) metrics breakdown
  - Runtime / footprint / inference-time summary

Usage:
    uv run python src/visualization/visualize_failures.py \
        --checkpoint experiments/<run>/ckpts/best_model.pt \
        --config src/configs/experiment.yaml

Controls (interactive viewer):
    d / Right Arrow  : Next sample
    a / Left Arrow   : Previous sample
    s                : Save current sample to failure_saves/
    p / Space        : Play / Pause auto-advance
    q / ESC          : Quit
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch


def _has_display() -> bool:
    """Return True if a graphical display is available."""
    if sys.platform == "darwin":
        return True
    display = os.environ.get("DISPLAY", "")
    wayland = os.environ.get("WAYLAND_DISPLAY", "")
    return bool(display or wayland)
from torch.utils.data import DataLoader

from src.augmentations.transforms import get_val_transform
from src.configs.config import ExperimentConfig
from src.data.dataset import SegmentationDataset, PROMPT_TO_ID
from src.training.trainer import create_model
from src.utils.seed import set_seed, worker_init_fn
from src.visualization.viz_utils import overlay_bgr, mask_coverage


# ── ImageNet denormalization (mirrors visualize_dataloader.py) ──────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

ID_TO_PROMPT = {v: k for k, v in PROMPT_TO_ID.items()}


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) float tensor → (H,W,3) uint8 RGB array."""
    img = tensor.permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# ── Per-sample metrics ──────────────────────────────────────────────────
def compute_sample_metrics(pred_binary: np.ndarray, gt: np.ndarray) -> dict:
    """Compute IoU, Dice, Precision, Recall for a single (H,W) pair."""
    eps = 1e-7
    pred = pred_binary.astype(np.float64)
    gt = gt.astype(np.float64)
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


# ── Inference + collect results ─────────────────────────────────────────
def run_evaluation(
    config: ExperimentConfig,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[list[dict], dict]:
    """
    Run inference on val set and return per-sample results + runtime info.

    Returns:
        results: list of dicts with keys
            image_rgb, gt_mask, pred_mask, prompt, dataset, metrics
        runtime: dict with total_time, avg_per_image, model_params, device
    """
    # Load checkpoint first — use its stored config for model arch so we always
    # recreate the exact architecture that was trained (not the YAML on disk).
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", config)
    # Merge: keep data/training from the provided config, arch from checkpoint
    config.model = ckpt_config.model
    model = create_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model arch from checkpoint: {ckpt_config.model.arch} / {ckpt_config.model.encoder}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Val dataset (no shuffle)
    val_dataset = SegmentationDataset(
        pkl_path=config.data.val_pkl,
        transform=get_val_transform(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    results = []
    total_inference_time = 0.0
    n_images = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            prompt_ids = batch["prompt_id"].to(device)
            masks_gt = batch["mask"]  # (B, H, W) float {0,1}

            # Timed inference
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits = model(images, prompt_ids)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_inference_time += t1 - t0

            pred_prob = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            pred_binary = (pred_prob >= 0.5).astype(np.uint8)
            gt_np = masks_gt.numpy()

            for i in range(images.shape[0]):
                img_rgb = denormalize(batch["image"][i])
                gt_i = gt_np[i]
                pred_i = pred_binary[i]
                m = compute_sample_metrics(pred_i, gt_i)
                results.append(
                    {
                        "image_rgb": img_rgb,
                        "gt_mask": gt_i,
                        "pred_mask": pred_i,
                        "prompt": batch["prompt"][i],
                        "dataset": batch["dataset"][i],
                        "metrics": m,
                        "index": n_images,
                    }
                )
                n_images += 1

    # GPU memory
    gpu_mem_mb = 0.0
    if device.type == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    runtime = {
        "total_time_s": total_inference_time,
        "avg_per_image_ms": (total_inference_time / max(n_images, 1)) * 1000,
        "n_images": n_images,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "gpu_peak_mem_mb": gpu_mem_mb,
        "device": str(device),
        "checkpoint_epoch": ckpt.get("epoch", "?"),
        "checkpoint_dice": ckpt.get("metrics", {}).get("dice", "?"),
    }

    return results, runtime


# ── Aggregate per-class metrics ─────────────────────────────────────────
def build_class_metrics(results: list[dict]) -> tuple[dict, str]:
    """Return (summary dict, formatted text) for per-class metrics."""
    classes = {"segment crack": [], "segment taping area": []}
    for r in results:
        classes[r["prompt"]].append(r["metrics"])

    lines = ["\n" + "=" * 70, "PER-CLASS METRICS", "=" * 70]
    summary = {}
    for prompt, metrics_list in classes.items():
        if not metrics_list:
            continue
        n = len(metrics_list)
        avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        std = {k: np.std([m[k] for m in metrics_list]) for k in metrics_list[0]}
        tag = "crack" if "crack" in prompt else "taping"
        summary[tag] = {"n": n, "avg": avg, "std": std}
        lines += [
            f"\n  [{tag.upper()}] ({n} samples)",
            f"    Dice:      {avg['dice']:.4f} +/- {std['dice']:.4f}",
            f"    IoU:       {avg['iou']:.4f} +/- {std['iou']:.4f}",
            f"    Precision: {avg['precision']:.4f} +/- {std['precision']:.4f}",
            f"    Recall:    {avg['recall']:.4f} +/- {std['recall']:.4f}",
        ]

    all_dice = [r["metrics"]["dice"] for r in results]
    all_iou = [r["metrics"]["iou"] for r in results]
    lines += [
        f"\n  [OVERALL] ({len(results)} samples)",
        f"    Dice:  {np.mean(all_dice):.4f} +/- {np.std(all_dice):.4f}",
        f"    IoU:   {np.mean(all_iou):.4f} +/- {np.std(all_iou):.4f}",
        "=" * 70,
    ]
    return summary, "\n".join(lines)


# ── Runtime / footprint report ──────────────────────────────────────────
def build_runtime(runtime: dict) -> str:
    """Return formatted runtime/footprint text."""
    lines = [
        "\n" + "=" * 70,
        "RUNTIME & FOOTPRINT",
        "=" * 70,
        f"  Device:              {runtime['device']}",
        f"  Checkpoint epoch:    {runtime['checkpoint_epoch']}",
        f"  Checkpoint val dice: {runtime['checkpoint_dice']}",
        f"  Total params:        {runtime['total_params']:,}",
        f"  Trainable params:    {runtime['trainable_params']:,}",
        f"  Val images:          {runtime['n_images']}",
        f"  Total inference:     {runtime['total_time_s']:.2f}s",
        f"  Avg per image:       {runtime['avg_per_image_ms']:.2f}ms",
    ]
    if runtime["gpu_peak_mem_mb"] > 0:
        lines.append(f"  GPU peak memory:     {runtime['gpu_peak_mem_mb']:.1f} MB")
    lines.append("=" * 70)
    return "\n".join(lines)


# ── Interactive failure browser (OpenCV) ────────────────────────────────
def _overlay_mask(img_bgr: np.ndarray, mask: np.ndarray,
                   color: tuple, alpha: float = 0.45) -> np.ndarray:
    """Blend a single mask onto an image in the given BGR color."""
    out = img_bgr.copy().astype(np.float32)
    fg = mask > 0
    out[fg] = alpha * np.array(color, dtype=np.float32) + (1 - alpha) * out[fg]
    return out.clip(0, 255).astype(np.uint8)


def build_display(
    result: dict,
    rank: int,
    total: int,
    scale: float,
) -> np.ndarray:
    """Build a 4-panel display: Input | Input+Pred(green) | Input+GT(red) | Combined overlay."""
    img_rgb = result["image_rgb"]
    pred = result["pred_mask"]
    gt = result["gt_mask"]
    m = result["metrics"]
    H, W = img_rgb.shape[:2]

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Panel 2: Input + Pred overlay (green)
    pred_overlay = _overlay_mask(img_bgr, pred, color=(0, 255, 0))

    # Panel 3: Input + GT overlay (red)
    gt_overlay = _overlay_mask(img_bgr, gt, color=(0, 0, 255))

    # Panel 4: Combined — GT=red, Pred=green, overlap=yellow on input
    combined = img_bgr.copy().astype(np.float32)
    gt_bool = gt > 0
    pred_bool = pred > 0
    gt_only = gt_bool & ~pred_bool
    pred_only = pred_bool & ~gt_bool
    both = gt_bool & pred_bool
    combined[gt_only] = 0.5 * combined[gt_only] + 0.5 * np.array([0, 0, 255], dtype=np.float32)
    combined[pred_only] = 0.5 * combined[pred_only] + 0.5 * np.array([0, 255, 0], dtype=np.float32)
    combined[both] = 0.5 * combined[both] + 0.5 * np.array([0, 255, 255], dtype=np.float32)
    combined = combined.clip(0, 255).astype(np.uint8)

    panels = np.hstack([img_bgr, pred_overlay, gt_overlay, combined])

    # Info bar
    bar_h = 70
    bar = np.zeros((bar_h, panels.shape[1], 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    line1 = (
        f"[{rank+1}/{total}]  idx={result['index']}  "
        f"{result['dataset']} / {result['prompt']}"
    )
    line2 = (
        f"Dice={m['dice']:.4f}  IoU={m['iou']:.4f}  "
        f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}"
    )
    line3 = "d:Next  a:Prev  s:Save  p:Play  q:Quit"

    # Color dice red/yellow/green
    if m["dice"] < 0.3:
        dice_color = (0, 0, 255)
    elif m["dice"] < 0.6:
        dice_color = (0, 180, 255)
    else:
        dice_color = (0, 200, 0)

    cv2.putText(bar, line1, (10, 20), font, 0.55, (255, 255, 255), 1)
    cv2.putText(bar, line2, (10, 42), font, 0.55, dice_color, 1)
    cv2.putText(bar, line3, (10, 62), font, 0.45, (150, 150, 150), 1)

    # Column labels
    col_labels = ["Input", "Input+Pred (green)", "Input+GT (red)", "Combined (R=GT G=Pred Y=Both)"]
    for ci, label in enumerate(col_labels):
        x = ci * W + 5
        cv2.putText(bar, label, (x, 62), font, 0.4, (200, 200, 200), 1)

    canvas = np.vstack([bar, panels])

    if scale != 1.0:
        new_w = int(canvas.shape[1] * scale)
        new_h = int(canvas.shape[0] * scale)
        canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return canvas


def interactive_failure_browser(
    results: list[dict],
    save_dir: Path,
    scale: float = 1.0,
    iou_threshold: float = 1.0,
) -> None:
    """
    Browse samples sorted by worst IoU.

    Args:
        results: evaluation results list.
        save_dir: directory for saved failure PNGs.
        scale: display scale factor.
        iou_threshold: only show samples with IoU <= threshold.
    """
    # Filter and sort by IoU ascending (worst first)
    filtered = [r for r in results if r["metrics"]["iou"] <= iou_threshold]
    filtered.sort(key=lambda r: r["metrics"]["iou"])

    if not filtered:
        print(f"No samples with IoU <= {iou_threshold}")
        return

    print(f"\nBrowsing {len(filtered)} samples (IoU <= {iou_threshold}), worst first.")
    save_dir.mkdir(parents=True, exist_ok=True)

    win = "Failure Analysis"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    idx = 0
    playing = False

    while True:
        vis = build_display(filtered[idx], idx, len(filtered), scale)
        cv2.imshow(win, vis)
        wait = 300 if playing else 0
        key = cv2.waitKey(wait) & 0xFF

        if playing and key == 255:
            idx = (idx + 1) % len(filtered)
            continue

        if key == ord("d") or key == 83:
            idx = (idx + 1) % len(filtered)
        elif key == ord("a") or key == 81:
            idx = (idx - 1) % len(filtered)
        elif key == ord("s"):
            r = filtered[idx]
            fname = (
                f"fail_{idx:03d}_iou{r['metrics']['iou']:.3f}"
                f"_{r['dataset']}_{r['index']}.png"
            )
            cv2.imwrite(str(save_dir / fname), vis)
            print(f"  Saved: {fname}")
        elif key == ord("p") or key == 32:
            playing = not playing
            print("PLAYING" if playing else "PAUSED")
        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


# ── Static grid export (top-N worst) ───────────────────────────────────
def save_failure_grid(
    results: list[dict],
    out_path: Path,
    n: int = 20,
) -> None:
    """Save a matplotlib grid of the N worst-scoring samples."""
    import matplotlib.pyplot as plt

    sorted_r = sorted(results, key=lambda r: r["metrics"]["iou"])[:n]

    rows = len(sorted_r)
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Input + Pred (green)", "Input + GT (red)", "Combined (R=GT G=Pred Y=Both)"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11, fontweight="bold")

    def _overlay_rgb(base, mask, color, alpha=0.45):
        out = base.copy().astype(np.float32)
        fg = mask > 0
        out[fg] = alpha * np.array(color, dtype=np.float32) + (1 - alpha) * out[fg]
        return out.clip(0, 255).astype(np.uint8)

    for ri, r in enumerate(sorted_r):
        img = r["image_rgb"]
        pred = r["pred_mask"]
        gt = r["gt_mask"]
        m = r["metrics"]

        # Panel overlays
        pred_on_input = _overlay_rgb(img, pred, color=(0, 255, 0))
        gt_on_input = _overlay_rgb(img, gt, color=(255, 0, 0))

        # Combined overlay
        combined = img.copy().astype(np.float32)
        gt_bool = gt > 0
        pred_bool = pred > 0
        gt_only = gt_bool & ~pred_bool
        pred_only = pred_bool & ~gt_bool
        both = gt_bool & pred_bool
        combined[gt_only] = 0.5 * combined[gt_only] + 0.5 * np.array([255, 0, 0])
        combined[pred_only] = 0.5 * combined[pred_only] + 0.5 * np.array([0, 255, 0])
        combined[both] = 0.5 * combined[both] + 0.5 * np.array([255, 255, 0])
        combined = combined.clip(0, 255).astype(np.uint8)

        axes[ri, 0].imshow(img)
        axes[ri, 0].set_ylabel(
            f"#{ri+1} [{r['dataset']}]\n"
            f"IoU={m['iou']:.3f}\nDice={m['dice']:.3f}",
            fontsize=8, rotation=0, labelpad=80, va="center",
        )
        axes[ri, 0].axis("off")

        axes[ri, 1].imshow(pred_on_input)
        axes[ri, 1].axis("off")

        axes[ri, 2].imshow(gt_on_input)
        axes[ri, 2].axis("off")

        axes[ri, 3].imshow(combined)
        axes[ri, 3].set_title(
            f"P={m['precision']:.2f} R={m['recall']:.2f}", fontsize=8
        )
        axes[ri, 3].axis("off")

    plt.suptitle(f"Top-{n} Worst Failures (sorted by IoU)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved failure grid: {out_path}")


# ── Static grid export (top-N best per prompt) ───────────────────────────
def save_best_grid(
    results: list[dict],
    out_path: Path,
    n_per_prompt: int = 5,
) -> None:
    """Save a matplotlib grid of the N best-scoring samples per prompt."""
    import matplotlib.pyplot as plt

    # Split by prompt, sort by IoU descending (best first)
    crack_results = sorted(
        [r for r in results if "crack" in r["prompt"]],
        key=lambda r: r["metrics"]["iou"], reverse=True,
    )[:n_per_prompt]
    taping_results = sorted(
        [r for r in results if "taping" in r["prompt"]],
        key=lambda r: r["metrics"]["iou"], reverse=True,
    )[:n_per_prompt]
    selected = crack_results + taping_results

    rows = len(selected)
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Input + Pred (green)", "Input + GT (red)", "Combined (R=GT G=Pred Y=Both)"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11, fontweight="bold")

    def _overlay_rgb(base, mask, color, alpha=0.45):
        out = base.copy().astype(np.float32)
        fg = mask > 0
        out[fg] = alpha * np.array(color, dtype=np.float32) + (1 - alpha) * out[fg]
        return out.clip(0, 255).astype(np.uint8)

    for ri, r in enumerate(selected):
        img = r["image_rgb"]
        pred = r["pred_mask"]
        gt = r["gt_mask"]
        m = r["metrics"]

        pred_on_input = _overlay_rgb(img, pred, color=(0, 255, 0))
        gt_on_input = _overlay_rgb(img, gt, color=(255, 0, 0))

        combined = img.copy().astype(np.float32)
        gt_bool = gt > 0
        pred_bool = pred > 0
        gt_only = gt_bool & ~pred_bool
        pred_only = pred_bool & ~gt_bool
        both = gt_bool & pred_bool
        combined[gt_only] = 0.5 * combined[gt_only] + 0.5 * np.array([255, 0, 0])
        combined[pred_only] = 0.5 * combined[pred_only] + 0.5 * np.array([0, 255, 0])
        combined[both] = 0.5 * combined[both] + 0.5 * np.array([255, 255, 0])
        combined = combined.clip(0, 255).astype(np.uint8)

        prompt_label = "crack" if "crack" in r["prompt"] else "taping"
        axes[ri, 0].imshow(img)
        axes[ri, 0].set_ylabel(
            f"#{ri+1} [{prompt_label}]\n"
            f"IoU={m['iou']:.3f}\nDice={m['dice']:.3f}",
            fontsize=8, rotation=0, labelpad=80, va="center",
        )
        axes[ri, 0].axis("off")

        axes[ri, 1].imshow(pred_on_input)
        axes[ri, 1].axis("off")

        axes[ri, 2].imshow(gt_on_input)
        axes[ri, 2].axis("off")

        axes[ri, 3].imshow(combined)
        axes[ri, 3].set_title(
            f"P={m['precision']:.2f} R={m['recall']:.2f}", fontsize=8
        )
        axes[ri, 3].axis("off")

    plt.suptitle(
        f"Best Predictions — Top {n_per_prompt} per Prompt (sorted by IoU)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved best grid: {out_path}")


# ── Suggestions ─────────────────────────────────────────────────────────
def build_suggestions(class_summary: dict, results: list[dict]) -> str:
    """Return formatted failure-analysis suggestions text."""
    lines = ["\n" + "=" * 70, "FAILURE ANALYSIS SUGGESTIONS", "=" * 70]

    if "crack" in class_summary and "taping" in class_summary:
        crack_dice = class_summary["crack"]["avg"]["dice"]
        taping_dice = class_summary["taping"]["avg"]["dice"]
        gap = abs(crack_dice - taping_dice)
        worse = "crack" if crack_dice < taping_dice else "taping"
        if gap > 0.1:
            lines += [
                f"\n  1. CLASS IMBALANCE: {worse} is {gap:.3f} Dice behind.",
                f"     -> Try class-weighted loss or oversample {worse} in training.",
                f"     -> Add {worse}-specific augmentations (e.g., elastic for cracks).",
            ]

    low_recall = [r for r in results if r["metrics"]["recall"] < 0.3 and r["metrics"]["iou"] < 0.5]
    low_prec = [r for r in results if r["metrics"]["precision"] < 0.3 and r["metrics"]["iou"] < 0.5]
    if len(low_recall) > len(low_prec):
        lines += [
            "\n  2. UNDER-SEGMENTATION dominant: model misses foreground.",
            "     -> Increase recall: lower threshold, use Focal loss, or Boundary loss.",
            "     -> Check if thin/small objects are being missed (crack width augmentation).",
        ]
    elif len(low_prec) > len(low_recall):
        lines += [
            "\n  2. OVER-SEGMENTATION dominant: model produces false positives.",
            "     -> Try adding more negative examples or harder negatives.",
            "     -> Consider CLAHE/edge-aware preprocessing.",
        ]
    else:
        lines.append("\n  2. Mixed failure modes (both under- and over-segmentation).")

    tiny = [r for r in results if mask_coverage(r["gt_mask"]) < 2.0 and r["metrics"]["iou"] < 0.5]
    if tiny:
        lines += [
            f"\n  3. TINY MASKS: {len(tiny)} failures have <2% mask coverage.",
            "     -> Use CrackWidthAugmentation to dilate thin annotations during training.",
            "     -> Try Boundary loss or Lovasz loss for fine structures.",
        ]

    lines += [
        "\n  OTHER IDEAS:",
        "     - Ablate augmentation tiers (baseline -> geometric -> full)",
        "     - Try different backbones (resnet50, efficientnet-b3)",
        "     - Experiment with loss functions: Focal, Boundary, Lovasz",
        "     - Add test-time augmentation (horizontal flip, multi-scale)",
        "     - Visualize attention maps / grad-CAM on failure cases",
        "     - Check dataset quality: are GT annotations correct on failures?",
        "     - Train longer with lower LR (cosine annealing to 1e-6)",
        "=" * 70,
    ]
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Failure analysis & prediction visualization")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--scale", type=float, default=1.5, help="Display scale (default: 1.5)")
    parser.add_argument("--iou-threshold", type=float, default=1.0,
                        help="Only browse samples with IoU <= this (default: 1.0 = all)")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of worst failures to save in grid (default: 50)")
    parser.add_argument("--best-n", type=int, default=5,
                        help="Number of best predictions per prompt to save in grid (default: 5)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip interactive browser, just print metrics and save grid")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: next to checkpoint)")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    set_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output dir defaults to experiment's failure_analysis/ folder
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.checkpoint).parent.parent / "failure_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")
    print(f"Device:     {device}")

    # Run evaluation
    results, runtime = run_evaluation(config, args.checkpoint, device)

    # Build report text
    class_summary, metrics_text = build_class_metrics(results)
    runtime_text = build_runtime(runtime)
    suggestions_text = build_suggestions(class_summary, results)
    report = metrics_text + "\n" + runtime_text + "\n" + suggestions_text

    # Print to console
    print(report)

    # Save report to reports/
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    exp_name = Path(args.checkpoint).parents[1].name
    report_path = reports_dir / f"failure_analysis_{exp_name}.txt"
    with open(report_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Config:     {args.config}\n")
        f.write(report + "\n")
    print(f"\nSaved report: {report_path}")

    # Save failure grid
    grid_path = out_dir / "failure_grid.png"
    save_failure_grid(results, grid_path, n=args.top_n)

    # Save best predictions grid
    best_path = out_dir / "best_grid.png"
    save_best_grid(results, best_path, n_per_prompt=args.best_n)

    # Save metrics CSV
    csv_path = out_dir / "per_sample_metrics.csv"
    with open(csv_path, "w") as f:
        f.write("index,dataset,prompt,iou,dice,precision,recall,gt_coverage\n")
        for r in sorted(results, key=lambda x: x["metrics"]["iou"]):
            m = r["metrics"]
            cov = mask_coverage(r["gt_mask"])
            f.write(
                f"{r['index']},{r['dataset']},{r['prompt']},"
                f"{m['iou']:.6f},{m['dice']:.6f},"
                f"{m['precision']:.6f},{m['recall']:.6f},{cov:.2f}\n"
            )
    print(f"Saved per-sample CSV: {csv_path}")

    # Interactive browser — skip automatically on headless servers
    headless = not _has_display()
    if headless and not args.no_interactive:
        print("\n[headless] No display detected — skipping interactive browser.")
        print(f"  Outputs saved to: {out_dir}")
    elif not args.no_interactive:
        save_dir = out_dir / "saved_failures"
        interactive_failure_browser(results, save_dir, args.scale, args.iou_threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()

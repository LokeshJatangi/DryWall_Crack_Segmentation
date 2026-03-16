"""
Standalone evaluation script — computes metrics from a trained checkpoint.

Reuses run_evaluation(), build_class_metrics(), and build_runtime() from
visualize_failures.py.  No matplotlib or OpenCV imports — metrics only.

Usage:
    uv run python src/evaluation/evaluate.py \
        --checkpoint experiments/<run>/ckpts/best_model.pt \
        --config src/configs/experiment.yaml \
        [--save reports/eval_<name>.txt] \
        [--csv reports/eval_<name>.csv]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.configs.config import ExperimentConfig
from src.utils.seed import set_seed
from src.visualization.visualize_failures import (
    build_class_metrics,
    build_runtime,
    compute_sample_metrics,
    run_evaluation,
)
from src.visualization.viz_utils import mask_coverage


def format_evaluation_summary(
    results: list[dict],
    runtime: dict,
    checkpoint_path: str,
) -> str:
    """Build a clean evaluation summary string."""
    class_summary, _ = build_class_metrics(results)

    # Overall metrics
    all_dice = np.mean([r["metrics"]["dice"] for r in results])
    all_iou = np.mean([r["metrics"]["iou"] for r in results])
    all_prec = np.mean([r["metrics"]["precision"] for r in results])
    all_rec = np.mean([r["metrics"]["recall"] for r in results])

    # Determine arch info from checkpoint path
    exp_name = Path(checkpoint_path).parents[1].name
    arch = runtime.get("arch", "Unknown")
    encoder = runtime.get("encoder", "Unknown")

    sep = "=" * 50
    lines = [
        sep,
        f"EVALUATION SUMMARY — {exp_name}",
        sep,
        f"Checkpoint: {checkpoint_path}",
        f"Architecture: {arch} / {encoder}",
        f"Epoch: {runtime['checkpoint_epoch']}",
        "",
        f"OVERALL ({len(results)} samples):",
        f"  Dice:      {all_dice:.4f}    mIoU:      {all_iou:.4f}",
        f"  Precision: {all_prec:.4f}    Recall:    {all_rec:.4f}",
        "",
        "PER-PROMPT:",
    ]

    for tag in ["crack", "taping"]:
        if tag in class_summary:
            s = class_summary[tag]
            n = s["n"]
            avg = s["avg"]
            lines.append(f"  [{tag.upper()}] ({n} samples)")
            lines.append(f"    Dice: {avg['dice']:.4f}  IoU: {avg['iou']:.4f}  "
                         f"Prec: {avg['precision']:.4f}  Rec: {avg['recall']:.4f}")

    # Checkpoint file size
    ckpt_size_mb = 0
    if os.path.exists(checkpoint_path):
        ckpt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    lines += [
        "",
        "FOOTPRINT:",
        f"  Parameters: {runtime['total_params']:,}   "
        f"Inference: {runtime['avg_per_image_ms']:.2f} ms/img",
    ]
    if runtime["gpu_peak_mem_mb"] > 0:
        lines.append(f"  GPU peak:   {runtime['gpu_peak_mem_mb']:.1f} MB    "
                      f"Checkpoint: {ckpt_size_mb:.0f} MB")
    lines.append(sep)

    return "\n".join(lines)


def save_csv(results: list[dict], csv_path: Path) -> None:
    """Save per-sample metrics CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--save", type=str, default=None,
                        help="Save evaluation summary to this file")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save per-sample metrics CSV to this file")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    set_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")
    print(f"Device:     {device}")
    print()

    # Run evaluation
    results, runtime = run_evaluation(config, args.checkpoint, device)

    # Enrich runtime with arch info from checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config", config)
    runtime["arch"] = ckpt_config.model.arch
    runtime["encoder"] = ckpt_config.model.encoder

    # Build summary
    summary = format_evaluation_summary(results, runtime, args.checkpoint)
    print(summary)

    # Also print full per-class breakdown
    _, metrics_text = build_class_metrics(results)
    runtime_text = build_runtime(runtime)
    print(metrics_text)
    print(runtime_text)

    # Save report
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(summary + "\n")
        print(f"\nSaved report: {save_path}")

    # Save CSV
    if args.csv:
        save_csv(results, Path(args.csv))

    print("\nDone.")


if __name__ == "__main__":
    main()

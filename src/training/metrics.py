"""
Segmentation evaluation metrics.

Supports both micro-averaged (legacy) and macro-averaged (per-image) metrics.
Default: macro-averaged — computes IoU/Dice per image, then averages.
"""

import torch


class SegmentationMetrics:
    """Accumulates per-image segmentation metrics and computes macro averages."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        # Micro accumulators (legacy)
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        # Macro accumulators (per-image)
        self.sample_iou = []
        self.sample_dice = []
        self.sample_precision = []
        self.sample_recall = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch of predictions and targets.

        Args:
            pred: Raw logits (B, H, W) or (B, 1, H, W)
            target: Binary targets (B, H, W), values {0, 1}
        """
        pred = pred.squeeze(1) if pred.dim() == 4 else pred
        pred_binary = (torch.sigmoid(pred) >= self.threshold).float()
        target = target.float()

        # Micro: global pixel sums
        self.tp += (pred_binary * target).sum().item()
        self.fp += (pred_binary * (1 - target)).sum().item()
        self.fn += ((1 - pred_binary) * target).sum().item()
        self.tn += ((1 - pred_binary) * (1 - target)).sum().item()

        # Macro: per-image metrics
        eps = 1e-7
        tp = (pred_binary * target).sum(dim=(-2, -1))          # (B,)
        fp = (pred_binary * (1 - target)).sum(dim=(-2, -1))
        fn = ((1 - pred_binary) * target).sum(dim=(-2, -1))

        iou = (tp + eps) / (tp + fp + fn + eps)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)

        self.sample_iou.extend(iou.cpu().tolist())
        self.sample_dice.extend(dice.cpu().tolist())
        self.sample_precision.extend(precision.cpu().tolist())
        self.sample_recall.extend(recall.cpu().tolist())

    def compute(self) -> dict:
        """Compute macro-averaged metrics (primary) and micro-averaged (legacy)."""
        eps = 1e-7
        n = len(self.sample_iou) or 1

        # Macro-averaged (per-image mean) — foreground only
        miou = sum(self.sample_iou) / n
        dice = sum(self.sample_dice) / n
        precision = sum(self.sample_precision) / n
        recall = sum(self.sample_recall) / n

        # Micro-averaged (legacy, for reference)
        iou_fg_micro = self.tp / (self.tp + self.fp + self.fn + eps)
        iou_bg_micro = self.tn / (self.tn + self.fp + self.fn + eps)

        return {
            'miou': miou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'iou_fg': miou,  # now same as miou (both macro, fg-only)
            # Legacy micro keys for backwards compat
            'iou_fg_micro': iou_fg_micro,
            'iou_bg_micro': iou_bg_micro,
            'miou_micro': (iou_fg_micro + iou_bg_micro) / 2,
        }

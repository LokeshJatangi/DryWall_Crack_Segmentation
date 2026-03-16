"""
Loss functions for binary segmentation.

Provides Dice, BCE, Focal, Boundary losses and a configurable combined loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation (macro-averaged: per-image, then mean)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        # Per-image Dice: sum over spatial dims only, keep batch dim
        intersection = (pred * target).sum(dim=(-2, -1))
        pred_sum = pred.sum(dim=(-2, -1))
        target_sum = target.sum(dim=(-2, -1))

        dice_per_image = (2.0 * intersection + self.smooth) / (
            pred_sum + target_sum + self.smooth
        )
        return 1 - dice_per_image.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss with configurable weights."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(pred, target)
            + self.bce_weight * self.bce(pred, target)
        )


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance (macro-averaged: per-image, then mean)."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        # Per-image mean, then batch mean
        per_image = (focal_weight * bce).mean(dim=(-2, -1))
        return per_image.mean()


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal loss. Better than DiceBCE for thin structures."""

    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5,
                 focal_alpha: float = 0.75, focal_gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss(smooth)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(pred, target)
            + self.focal_weight * self.focal(pred, target)
        )


class BoundaryLoss(nn.Module):
    """
    Distance-transform based boundary loss.

    Computes distance transform of GT boundary and weights predictions
    by proximity to boundaries. Encourages accurate boundary delineation.
    """

    def __init__(self):
        super().__init__()

    def _compute_distance_map(self, target: torch.Tensor) -> torch.Tensor:
        """Compute signed distance transform for a batch of masks."""
        target_np = target.detach().cpu().numpy()
        dist_maps = []

        for mask in target_np:
            # Compute distance transform for foreground and background
            if mask.sum() == 0:
                dist_maps.append(np.zeros_like(mask))
                continue
            if mask.sum() == mask.size:
                dist_maps.append(np.zeros_like(mask))
                continue

            pos_dist = ndimage.distance_transform_edt(mask)
            neg_dist = ndimage.distance_transform_edt(1 - mask)
            # Signed distance: negative inside, positive outside
            signed_dist = neg_dist - pos_dist
            # Normalize to [-1, 1]
            max_val = max(abs(signed_dist.min()), abs(signed_dist.max()), 1.0)
            signed_dist = signed_dist / max_val
            dist_maps.append(signed_dist)

        return torch.tensor(np.stack(dist_maps), dtype=target.dtype, device=target.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        dist_map = self._compute_distance_map(target)
        return (pred_prob * dist_map).mean()


class CombinedLoss(nn.Module):
    """Configurable combination of multiple losses."""

    def __init__(self, losses: list[tuple[nn.Module, float]]):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in losses])
        self.weights = [w for _, w in losses]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = sum(w * loss(pred, target) for loss, w in zip(self.losses, self.weights))
        return total


def get_loss(config) -> nn.Module:
    """Factory function to create loss from config."""
    loss_type = config.loss.type

    if loss_type == "dice":
        return DiceLoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "dice_bce":
        return DiceBCELoss(
            dice_weight=config.loss.dice_weight,
            bce_weight=config.loss.bce_weight,
        )
    elif loss_type == "focal":
        return FocalLoss(alpha=config.loss.focal_alpha, gamma=config.loss.focal_gamma)
    elif loss_type == "dice_focal":
        return DiceFocalLoss(
            dice_weight=config.loss.dice_weight,
            focal_weight=config.loss.bce_weight,  # reuse bce_weight field for focal weight
            focal_alpha=config.loss.focal_alpha,
            focal_gamma=config.loss.focal_gamma,
        )
    elif loss_type == "boundary":
        return BoundaryLoss()
    elif loss_type == "dice_bce_boundary":
        return CombinedLoss([
            (DiceBCELoss(config.loss.dice_weight, config.loss.bce_weight), 0.8),
            (BoundaryLoss(), 0.2),
        ])
    elif loss_type == "dice_focal_boundary":
        return CombinedLoss([
            (DiceFocalLoss(
                dice_weight=config.loss.dice_weight,
                focal_weight=config.loss.bce_weight,
                focal_alpha=config.loss.focal_alpha,
                focal_gamma=config.loss.focal_gamma,
            ), 0.8),
            (BoundaryLoss(), 0.2),
        ])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

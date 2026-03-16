"""
Albumentations-based augmentation pipelines for segmentation training.

Provides tiered augmentation configs for ablation studies,
plus a crack-specific mask width augmentation.
"""

import random

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CrackWidthAugmentation:
    """
    Randomly dilate or erode crack masks to simulate varying crack widths.

    Applied as a post-processing step on the mask only (not inside A.Compose).
    Only applied to crack dataset samples during training.
    """

    def __init__(self, kernel_range: tuple = (2, 5), p: float = 0.3,
                 dilate_only: bool = False):
        self.kernel_range = kernel_range
        self.p = p
        self.dilate_only = dilate_only

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return mask

        k = random.randint(*self.kernel_range)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        if self.dilate_only or random.random() > 0.5:
            return cv2.dilate(mask, kernel, iterations=1)
        else:
            return cv2.erode(mask, kernel, iterations=1)


def _normalize_and_tensor():
    """Common tail: normalize + convert to tensor."""
    return [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]


def get_train_transform(tier: str = "full") -> A.Compose:
    """
    Get training augmentation pipeline by tier.

    Tiers (cumulative):
        baseline:     HorizontalFlip
        geometric:    + Rotate, ShiftScaleRotate
        photometric:  + RandomBrightnessContrast, CLAHE, GaussNoise
        edge:         + Sharpen, MotionBlur
        full:         + CoarseDropout, ElasticTransform, aggressive Rotate
    """
    transforms = []

    # Baseline
    transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(A.VerticalFlip(p=0.25))

    if tier in ("geometric", "photometric", "edge", "full"):
        transforms.extend([
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT, p=0.5
            ),
        ])

    if tier in ("photometric", "edge", "full"):
        transforms.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.GaussNoise(p=0.2),
            # A.GaussianBlur(p=0.2),
            A.HueSaturationValue(p=0.3),
            # A.ColorJitter(
            #     brightness=0.2,
            #     contrast=0.2,
            #     saturation=0.1
            #     )
        ])

    if tier in ("edge", "full"):
        transforms.extend([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            # A.MotionBlur(blur_limit=5, p=0.2),
        ])

    # if tier == "full":
    #     transforms.extend([
    #         # A.CoarseDropout(
    #         #     max_holes=8, max_height=32, max_width=32,
    #         #     min_holes=1, min_height=8, min_width=8,
    #         #     fill_value=0, mask_fill_value=0, p=0.3
    #         # ),  # Skipped: mask_fill_value=0 corrupts thin crack annotations
    #         # A.ElasticTransform(
    #         #     alpha=80, sigma=80 * 0.05,
    #         #     border_mode=cv2.BORDER_CONSTANT, p=0.2
    #         # ),  # Skipped: sigma=4.0 too aggressive for 640px images
    #     ])

    transforms.extend(_normalize_and_tensor())
    return A.Compose(transforms)


def get_val_transform() -> A.Compose:
    """Validation/test transform: normalize + tensor only."""
    return A.Compose(_normalize_and_tensor())


def get_crack_augmentation(train: bool = True) -> CrackWidthAugmentation | None:
    """Get crack-specific mask augmentation (train only)."""
    if train:
        return CrackWidthAugmentation(kernel_range=(2, 5), p=0.5)
    return None

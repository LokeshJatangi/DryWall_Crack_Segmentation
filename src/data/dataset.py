"""
PyTorch Dataset and DataLoader for merged segmentation data.

Loads merged pickle, applies augmentations, encodes prompts as integer IDs.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.augmentations.transforms import (
    CrackWidthAugmentation,
    get_train_transform,
    get_val_transform,
)
from src.configs.config import ExperimentConfig
from src.utils.seed import worker_init_fn


# Prompt text → integer ID mapping
PROMPT_TO_ID = {
    "segment crack": 0,
    "segment taping area": 1,
}


class SegmentationDataset(Dataset):
    """
    PyTorch dataset for prompted binary segmentation.

    Each sample returns:
        image:     (3, H, W) float32 normalized tensor
        mask:      (H, W) float32 tensor, values {0, 1}
        prompt_id: int (0=crack, 1=taping area)
        prompt:    str
        dataset:   str ('cracks' or 'drywall')
    """

    def __init__(
        self,
        pkl_path: str,
        transform=None,
        crack_augmentation=None,
    ):
        with open(pkl_path, 'rb') as f:
            self.samples = pickle.load(f)
        self.transform = transform
        self.crack_augmentation = crack_augmentation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = sample['image']  # (H, W, 3) uint8
        mask = sample['mask']    # (H, W) uint8, values {0, 255}
        prompt = sample['prompt']
        dataset = sample['dataset']

        # Apply crack width augmentation on mask (before albumentations)
        if self.crack_augmentation and dataset == 'cracks':
            mask = self.crack_augmentation(mask)

        # Apply albumentations (image + mask stay aligned)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']  # (3, H, W) float32 tensor
            mask = transformed['mask']    # (H, W) tensor

        # Convert mask: {0, 255} → {0, 1} float32
        mask = (mask > 0).float()

        prompt_id = PROMPT_TO_ID[prompt]

        return {
            'image': image,
            'mask': mask,
            'prompt_id': torch.tensor(prompt_id, dtype=torch.long),
            'prompt': prompt,
            'dataset': dataset,
        }


def create_dataloaders(config) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from config.

    Args:
        config: ExperimentConfig or path to a YAML config file.

    Returns:
        (train_loader, val_loader)
    """
    if isinstance(config, str):
        config = ExperimentConfig.from_yaml(config)

    train_transform = get_train_transform(tier=config.data.augmentation_tier)
    val_transform = get_val_transform()

    crack_aug = None
    if config.data.crack_aug_enabled:
        crack_aug = CrackWidthAugmentation(
            kernel_range=(config.data.crack_aug_kernel_min, config.data.crack_aug_kernel_max),
            p=config.data.crack_aug_prob,
            dilate_only=config.data.crack_aug_dilate_only,
        )

    train_dataset = SegmentationDataset(
        pkl_path=config.data.train_pkl,
        transform=train_transform,
        crack_augmentation=crack_aug,
    )

    val_dataset = SegmentationDataset(
        pkl_path=config.data.val_pkl,
        transform=val_transform,
        crack_augmentation=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader

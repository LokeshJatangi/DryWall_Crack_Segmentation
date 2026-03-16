"""
Experiment configuration using dataclasses + YAML loading.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    train_pkl: str = "processed_data/merged/train/merged_train.pkl"
    val_pkl: str = "processed_data/merged/valid/merged_valid.pkl"
    batch_size: int = 8
    num_workers: int = 4
    augmentation_tier: str = "full"
    crack_aug_enabled: bool = False
    crack_aug_kernel_min: int = 2
    crack_aug_kernel_max: int = 5
    crack_aug_prob: float = 0.3
    crack_aug_dilate_only: bool = False


@dataclass
class ModelConfig:
    arch: str = "Unet"  # Unet | UnetPlusPlus | DeepLabV3Plus | SegFormer
    encoder: str = "resnet34"  # For SMP models
    num_prompts: int = 2
    embed_dim: int = 128


@dataclass
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    early_stopping_patience: int = 10
    seed: int = 24
    resume_from: str = ""  # path to checkpoint .pt file to resume from


@dataclass
class LossConfig:
    type: str = "dice_bce"
    dice_weight: float = 0.5
    bce_weight: float = 0.5
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0


@dataclass
class OutputConfig:
    exp_name: str = "experiment"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        config = cls()
        if 'data' in raw:
            config.data = DataConfig(**raw['data'])
        if 'model' in raw:
            config.model = ModelConfig(**raw['model'])
        if 'training' in raw:
            config.training = TrainingConfig(**raw['training'])
        if 'loss' in raw:
            config.loss = LossConfig(**raw['loss'])
        if 'output' in raw:
            config.output = OutputConfig(**raw['output'])
        return config

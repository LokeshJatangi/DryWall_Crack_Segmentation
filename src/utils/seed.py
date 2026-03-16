"""
Global seed utility for reproducible experiments.

Call set_seed(seed) once at the start of training to seed:
  - Python random
  - NumPy
  - PyTorch CPU
  - PyTorch CUDA (single and multi-GPU)
  - cuDNN determinism flags

Use worker_init_fn() as the DataLoader worker_init_fn to ensure
each worker gets a unique but deterministic seed.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Seed all RNGs for reproducible training.

    Sets deterministic mode for cuDNN. Note: deterministic mode may
    reduce throughput slightly; disable if speed is critical and
    exact reproducibility is not required.

    Args:
        seed: Integer seed value. Default: 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   # multi-GPU

    # Deterministic cuDNN ops — trades speed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker initializer for deterministic multi-process loading.

    Each worker receives a unique seed derived from the base PyTorch seed,
    preventing all workers from producing identical random augmentations.

    Usage:
        DataLoader(..., worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)

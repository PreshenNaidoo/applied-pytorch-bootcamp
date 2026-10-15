"""Helpers for basic reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Set common random seeds.

    If `deterministic=True`, PyTorch will try to be more repeatable.
    This can make some runs slower.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

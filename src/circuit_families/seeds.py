"""Deterministic seed handling for Python, NumPy, and PyTorch."""

from __future__ import annotations

import random

import numpy as np
import torch

MAX_SEED = 2**32 - 1


def validate_seed(seed: int) -> int:
    """Validate a seed usable by Python, NumPy, and PyTorch."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    if not 0 <= seed <= MAX_SEED:
        raise ValueError(
            f"seed must be between 0 and {MAX_SEED} inclusive."
        )

    return seed


def numpy_generator(seed: int) -> np.random.Generator:
    """Return a NumPy Generator backed explicitly by PCG64."""

    return np.random.Generator(np.random.PCG64(validate_seed(seed)))


def seed_everything(
    seed: int,
    *,
    deterministic_torch: bool = True,
) -> int:
    """Seed Python, NumPy, and PyTorch deterministically."""

    seed = validate_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return seed

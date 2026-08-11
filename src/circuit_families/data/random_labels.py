"""Deterministic random-label control construction."""

from __future__ import annotations

import numpy as np

from circuit_families.seeds import numpy_generator


def generate_random_labels(
    true_labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute the complete true-label vector with NumPy PCG64.

    Returns:
        A tuple containing:
        1. the permutation applied to the true-label vector;
        2. the resulting random-label vector.
    """

    if not isinstance(true_labels, np.ndarray):
        raise TypeError("true_labels must be a NumPy array.")

    if true_labels.ndim != 1:
        raise ValueError("true_labels must be one-dimensional.")

    if true_labels.size == 0:
        raise ValueError("true_labels must not be empty.")

    generator = numpy_generator(seed)
    permutation = generator.permutation(true_labels.size).astype(np.int64)
    random_labels = true_labels[permutation].copy()

    return permutation, random_labels

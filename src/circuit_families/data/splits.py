"""Deterministic train, test, and nested control splits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from circuit_families.seeds import numpy_generator


def _validate_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} must be non-negative.")

    return value


def generate_permutation(
    total_examples: int,
    seed: int,
) -> np.ndarray:
    """Generate a deterministic permutation using NumPy PCG64."""

    total_examples = _validate_count(total_examples, "total_examples")

    if total_examples == 0:
        raise ValueError("total_examples must be greater than zero.")

    generator = numpy_generator(seed)
    return generator.permutation(total_examples).astype(np.int64)


def primary_split(
    permutation: np.ndarray,
    train_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a frozen permutation into primary train and test indices."""

    if not isinstance(permutation, np.ndarray):
        raise TypeError("permutation must be a NumPy array.")

    if permutation.ndim != 1:
        raise ValueError("permutation must be one-dimensional.")

    train_count = _validate_count(train_count, "train_count")

    if train_count > permutation.size:
        raise ValueError(
            "train_count cannot exceed the number of examples."
        )

    expected = np.arange(permutation.size, dtype=np.int64)
    if not np.array_equal(np.sort(permutation), expected):
        raise ValueError(
            "permutation must contain every index exactly once."
        )

    train_indices = permutation[:train_count].copy()
    test_indices = permutation[train_count:].copy()

    return train_indices, test_indices


def control_prefixes(
    permutation: np.ndarray,
    fractions: Sequence[float],
) -> dict[float, np.ndarray]:
    """Create nested control sets as prefixes of one frozen permutation."""

    if not isinstance(permutation, np.ndarray):
        raise TypeError("permutation must be a NumPy array.")

    if permutation.ndim != 1:
        raise ValueError("permutation must be one-dimensional.")

    if not fractions:
        raise ValueError("fractions must contain at least one value.")

    previous_fraction = 0.0
    prefixes: dict[float, np.ndarray] = {}

    for fraction in fractions:
        if isinstance(fraction, bool) or not isinstance(
            fraction,
            (int, float),
        ):
            raise TypeError("Each control fraction must be numeric.")

        fraction = float(fraction)

        if not 0.0 < fraction < 1.0:
            raise ValueError(
                "Each control fraction must be strictly between 0 and 1."
            )

        if fraction <= previous_fraction:
            raise ValueError(
                "Control fractions must be strictly increasing."
            )

        count = int(np.floor(permutation.size * fraction))
        prefixes[fraction] = permutation[:count].copy()
        previous_fraction = fraction

    return prefixes


def generate_splits(
    *,
    total_examples: int,
    split_seed: int,
    primary_train_count: int,
    control_fractions: Sequence[float],
) -> dict[str, np.ndarray]:
    """Generate the frozen permutation and all required split arrays."""

    permutation = generate_permutation(total_examples, split_seed)
    train_indices, test_indices = primary_split(
        permutation,
        primary_train_count,
    )
    controls = control_prefixes(permutation, control_fractions)

    arrays: dict[str, np.ndarray] = {
        "split_permutation": permutation,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }

    for fraction, indices in controls.items():
        percentage = round(fraction * 100)
        arrays[f"control_train_indices_{percentage:02d}pct"] = indices

    return arrays

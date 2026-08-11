"""Deterministic construction of the modular-addition dataset."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return value


def generate_ordered_pairs(modulus: int) -> np.ndarray:
    """Generate every ordered operand pair in lexicographic order."""

    modulus = _validate_positive_integer(modulus, "modulus")

    operands = np.arange(modulus, dtype=np.int16)
    first_operands = np.repeat(operands, modulus)
    second_operands = np.tile(operands, modulus)

    return np.column_stack((first_operands, second_operands))


def generate_inputs(
    pairs: np.ndarray,
    equals_token_id: int,
) -> np.ndarray:
    """Convert operand pairs to input sequences of the form [a, b, =]."""

    if not isinstance(pairs, np.ndarray):
        raise TypeError("pairs must be a NumPy array.")

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n_examples, 2).")

    equals_token_id = _validate_positive_integer(
        equals_token_id,
        "equals_token_id",
    )

    equals_column = np.full(
        (pairs.shape[0], 1),
        equals_token_id,
        dtype=np.int16,
    )

    return np.concatenate(
        (pairs.astype(np.int16, copy=False), equals_column),
        axis=1,
    )


def generate_true_labels(
    pairs: np.ndarray,
    modulus: int,
) -> np.ndarray:
    """Generate labels equal to (a + b) modulo the task modulus."""

    if not isinstance(pairs, np.ndarray):
        raise TypeError("pairs must be a NumPy array.")

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n_examples, 2).")

    modulus = _validate_positive_integer(modulus, "modulus")

    labels = (pairs[:, 0].astype(np.int64) + pairs[:, 1]) % modulus
    return labels.astype(np.int16)


def generate_modular_addition_dataset(
    *,
    modulus: int,
    equals_token_id: int,
) -> dict[str, np.ndarray]:
    """Generate ordered pairs, token inputs, and true labels."""

    pairs = generate_ordered_pairs(modulus)
    inputs = generate_inputs(pairs, equals_token_id)
    true_labels = generate_true_labels(pairs, modulus)

    return {
        "pairs": pairs,
        "inputs": inputs,
        "true_labels": true_labels,
    }


def hash_named_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash named arrays using names, dtypes, shapes, and C-order bytes."""

    digest = hashlib.sha256()

    for name in sorted(arrays):
        array = arrays[name]

        if not isinstance(array, np.ndarray):
            raise TypeError(f"arrays[{name!r}] must be a NumPy array.")

        contiguous = np.ascontiguousarray(array)

        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            ",".join(str(dimension) for dimension in contiguous.shape).encode(
                "ascii"
            )
        )
        digest.update(b"\0")
        digest.update(contiguous.tobytes(order="C"))
        digest.update(b"\0")

    return digest.hexdigest()

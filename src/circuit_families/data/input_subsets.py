"""Frozen operand-range subsets for later fidelity and transfer analysis."""

from __future__ import annotations

import numpy as np

SUBSET_NAMES = ("Q1", "Q2", "Q3", "Q4")

_LOWER_OPERAND_MAX = 56
_UPPER_OPERAND_MIN = 57
_MAX_OPERAND = 112


def generate_input_subsets(
    pairs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return indices for the frozen Q1-Q4 operand-range partition.

    The returned arrays contain row indices into ``pairs``:

    - Q1: first operand 0-56, second operand 0-56;
    - Q2: first operand 0-56, second operand 57-112;
    - Q3: first operand 57-112, second operand 0-56;
    - Q4: first operand 57-112, second operand 57-112.
    """

    if not isinstance(pairs, np.ndarray):
        raise TypeError("pairs must be a NumPy array.")

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n_examples, 2).")

    if not np.issubdtype(pairs.dtype, np.integer):
        raise TypeError("pairs must contain integer operands.")

    if np.any(pairs < 0) or np.any(pairs > _MAX_OPERAND):
        raise ValueError("pair operands must be between 0 and 112.")

    first_operands = pairs[:, 0]
    second_operands = pairs[:, 1]

    masks = {
        "Q1": (
            (first_operands <= _LOWER_OPERAND_MAX)
            & (second_operands <= _LOWER_OPERAND_MAX)
        ),
        "Q2": (
            (first_operands <= _LOWER_OPERAND_MAX)
            & (second_operands >= _UPPER_OPERAND_MIN)
        ),
        "Q3": (
            (first_operands >= _UPPER_OPERAND_MIN)
            & (second_operands <= _LOWER_OPERAND_MAX)
        ),
        "Q4": (
            (first_operands >= _UPPER_OPERAND_MIN)
            & (second_operands >= _UPPER_OPERAND_MIN)
        ),
    }

    return {
        name: np.flatnonzero(masks[name]).astype(np.int64)
        for name in SUBSET_NAMES
    }

"""Tests for deterministic Python, NumPy, and PyTorch seed handling."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from circuit_families.seeds import (
    MAX_SEED,
    numpy_generator,
    seed_everything,
    validate_seed,
)


def test_seed_everything_reproduces_random_sequences() -> None:
    seed_everything(7)

    first_python = random.random()
    first_numpy = np.random.random()
    first_torch = torch.rand(4)

    seed_everything(7)

    second_python = random.random()
    second_numpy = np.random.random()
    second_torch = torch.rand(4)

    assert first_python == second_python
    assert first_numpy == second_numpy
    assert torch.equal(first_torch, second_torch)


def test_numpy_generator_uses_reproducible_pcg64_sequence() -> None:
    first = numpy_generator(0).permutation(100)
    second = numpy_generator(0).permutation(100)

    assert np.array_equal(first, second)


def test_changing_numpy_seed_changes_sequence() -> None:
    first = numpy_generator(0).permutation(100)
    second = numpy_generator(1).permutation(100)

    assert not np.array_equal(first, second)


@pytest.mark.parametrize("invalid_seed", [True, 1.5, "1", None])
def test_non_integer_seeds_fail_clearly(invalid_seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        validate_seed(invalid_seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_seed", [-1, MAX_SEED + 1])
def test_out_of_range_seeds_fail_clearly(invalid_seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be between"):
        validate_seed(invalid_seed)

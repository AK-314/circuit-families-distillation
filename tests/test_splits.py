"""Tests for deterministic primary and nested control splits."""

from __future__ import annotations

import numpy as np

from circuit_families.data.modular_addition import hash_named_arrays
from circuit_families.data.splits import (
    control_prefixes,
    generate_permutation,
    generate_splits,
    primary_split,
)


def test_primary_split_has_frozen_counts() -> None:
    permutation = generate_permutation(12_769, seed=0)
    train_indices, test_indices = primary_split(
        permutation,
        train_count=3_830,
    )

    assert train_indices.shape == (3_830,)
    assert test_indices.shape == (8_939,)


def test_train_and_test_are_disjoint_and_exhaustive() -> None:
    permutation = generate_permutation(12_769, seed=0)
    train_indices, test_indices = primary_split(
        permutation,
        train_count=3_830,
    )

    assert np.intersect1d(train_indices, test_indices).size == 0

    combined = np.concatenate((train_indices, test_indices))
    expected = np.arange(12_769, dtype=np.int64)

    assert np.array_equal(np.sort(combined), expected)


def test_control_sets_are_prefixes_of_one_frozen_permutation() -> None:
    permutation = generate_permutation(12_769, seed=0)
    fractions = [0.05, 0.10, 0.15, 0.20, 0.25]
    controls = control_prefixes(permutation, fractions)

    expected_counts = {
        0.05: 638,
        0.10: 1_276,
        0.15: 1_915,
        0.20: 2_553,
        0.25: 3_192,
    }

    previous = np.array([], dtype=np.int64)

    for fraction in fractions:
        indices = controls[fraction]

        assert indices.shape == (expected_counts[fraction],)
        assert np.array_equal(indices, permutation[: expected_counts[fraction]])
        assert np.array_equal(indices[: previous.size], previous)

        previous = indices


def test_same_seed_reproduces_permutation_and_split_hash() -> None:
    first = generate_splits(
        total_examples=12_769,
        split_seed=0,
        primary_train_count=3_830,
        control_fractions=[0.05, 0.10, 0.15, 0.20, 0.25],
    )
    second = generate_splits(
        total_examples=12_769,
        split_seed=0,
        primary_train_count=3_830,
        control_fractions=[0.05, 0.10, 0.15, 0.20, 0.25],
    )

    for name in first:
        assert np.array_equal(first[name], second[name])

    assert hash_named_arrays(first) == hash_named_arrays(second)


def test_changing_split_seed_changes_permutation() -> None:
    first = generate_permutation(12_769, seed=0)
    second = generate_permutation(12_769, seed=1)

    assert not np.array_equal(first, second)

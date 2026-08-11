"""Tests for the frozen Q1-Q4 operand-range partition."""

from __future__ import annotations

import numpy as np

from circuit_families.data.input_subsets import (
    SUBSET_NAMES,
    generate_input_subsets,
)
from circuit_families.data.modular_addition import generate_ordered_pairs


def _pair_index(first_operand: int, second_operand: int) -> int:
    """Return the row index for a pair in the modulus-113 lexicographic array."""

    return first_operand * 113 + second_operand


def test_input_subsets_have_frozen_names_counts_and_dtype() -> None:
    pairs = generate_ordered_pairs(113)
    subsets = generate_input_subsets(pairs)

    expected_counts = {
        "Q1": 3_249,
        "Q2": 3_192,
        "Q3": 3_192,
        "Q4": 3_136,
    }

    assert tuple(subsets) == SUBSET_NAMES

    for name, expected_count in expected_counts.items():
        assert subsets[name].shape == (expected_count,)
        assert subsets[name].dtype == np.int64


def test_input_subset_boundaries_match_frozen_partition() -> None:
    pairs = generate_ordered_pairs(113)
    subsets = generate_input_subsets(pairs)
    memberships = {
        name: set(indices.tolist())
        for name, indices in subsets.items()
    }

    assert _pair_index(0, 0) in memberships["Q1"]
    assert _pair_index(56, 56) in memberships["Q1"]

    assert _pair_index(0, 57) in memberships["Q2"]
    assert _pair_index(56, 112) in memberships["Q2"]

    assert _pair_index(57, 0) in memberships["Q3"]
    assert _pair_index(112, 56) in memberships["Q3"]

    assert _pair_index(57, 57) in memberships["Q4"]
    assert _pair_index(112, 112) in memberships["Q4"]

    assert _pair_index(56, 57) not in memberships["Q1"]
    assert _pair_index(57, 56) not in memberships["Q4"]


def test_input_subsets_are_disjoint_and_exhaustive() -> None:
    pairs = generate_ordered_pairs(113)
    subsets = generate_input_subsets(pairs)

    combined_indices = np.concatenate(
        [subsets[name] for name in SUBSET_NAMES]
    )
    expected_indices = np.arange(pairs.shape[0], dtype=np.int64)

    assert combined_indices.size == 12_769
    assert np.unique(combined_indices).size == 12_769
    assert np.array_equal(np.sort(combined_indices), expected_indices)


def test_each_subset_contains_only_its_operand_ranges() -> None:
    pairs = generate_ordered_pairs(113)
    subsets = generate_input_subsets(pairs)

    q1_pairs = pairs[subsets["Q1"]]
    q2_pairs = pairs[subsets["Q2"]]
    q3_pairs = pairs[subsets["Q3"]]
    q4_pairs = pairs[subsets["Q4"]]

    assert np.all(q1_pairs[:, 0] <= 56)
    assert np.all(q1_pairs[:, 1] <= 56)

    assert np.all(q2_pairs[:, 0] <= 56)
    assert np.all(q2_pairs[:, 1] >= 57)

    assert np.all(q3_pairs[:, 0] >= 57)
    assert np.all(q3_pairs[:, 1] <= 56)

    assert np.all(q4_pairs[:, 0] >= 57)
    assert np.all(q4_pairs[:, 1] >= 57)


def test_input_subset_generation_is_deterministic() -> None:
    pairs = generate_ordered_pairs(113)

    first = generate_input_subsets(pairs)
    second = generate_input_subsets(pairs.copy())

    for name in SUBSET_NAMES:
        assert np.array_equal(first[name], second[name])

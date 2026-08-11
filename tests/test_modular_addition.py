"""Tests for deterministic modular-addition dataset construction."""

from __future__ import annotations

import numpy as np

from circuit_families.data.modular_addition import (
    generate_modular_addition_dataset,
    generate_ordered_pairs,
    generate_true_labels,
    hash_named_arrays,
)


def test_generates_all_unique_ordered_pairs() -> None:
    pairs = generate_ordered_pairs(113)

    assert pairs.shape == (12_769, 2)
    assert np.array_equal(pairs[0], np.array([0, 0]))
    assert np.array_equal(pairs[112], np.array([0, 112]))
    assert np.array_equal(pairs[113], np.array([1, 0]))
    assert np.array_equal(pairs[-1], np.array([112, 112]))
    assert np.unique(pairs, axis=0).shape[0] == 12_769


def test_every_true_label_matches_modular_addition_rule() -> None:
    pairs = generate_ordered_pairs(113)
    labels = generate_true_labels(pairs, 113)

    expected = (pairs[:, 0] + pairs[:, 1]) % 113

    assert labels.shape == (12_769,)
    assert np.array_equal(labels, expected)


def test_input_sequences_have_equals_token_in_final_position() -> None:
    dataset = generate_modular_addition_dataset(
        modulus=113,
        equals_token_id=113,
    )

    assert dataset["inputs"].shape == (12_769, 3)
    assert np.array_equal(dataset["inputs"][:, :2], dataset["pairs"])
    assert np.all(dataset["inputs"][:, 2] == 113)


def test_dataset_regeneration_produces_identical_arrays_and_hashes() -> None:
    first = generate_modular_addition_dataset(
        modulus=113,
        equals_token_id=113,
    )
    second = generate_modular_addition_dataset(
        modulus=113,
        equals_token_id=113,
    )

    for name in first:
        assert np.array_equal(first[name], second[name])

    assert hash_named_arrays(first) == hash_named_arrays(second)

"""Tests for deterministic random-label construction."""

from __future__ import annotations

import numpy as np

from circuit_families.data.modular_addition import (
    generate_modular_addition_dataset,
)
from circuit_families.data.random_labels import generate_random_labels


def test_random_labels_preserve_exact_global_class_counts() -> None:
    dataset = generate_modular_addition_dataset(
        modulus=113,
        equals_token_id=113,
    )
    true_labels = dataset["true_labels"]

    permutation, random_labels = generate_random_labels(
        true_labels,
        seed=1,
    )

    true_counts = np.bincount(true_labels, minlength=113)
    random_counts = np.bincount(random_labels, minlength=113)

    assert permutation.shape == (12_769,)
    assert random_labels.shape == (12_769,)
    assert np.array_equal(true_counts, random_counts)
    assert np.all(random_counts == 113)


def test_random_label_permutation_is_exhaustive() -> None:
    dataset = generate_modular_addition_dataset(
        modulus=113,
        equals_token_id=113,
    )

    permutation, _ = generate_random_labels(
        dataset["true_labels"],
        seed=1,
    )

    expected = np.arange(12_769, dtype=np.int64)

    assert np.array_equal(np.sort(permutation), expected)


def test_same_seed_reproduces_random_labels() -> None:
    dataset = generate_modular_addition_dataset(
        modulus=113,
        equals_token_id=113,
    )
    true_labels = dataset["true_labels"]

    first_permutation, first_labels = generate_random_labels(
        true_labels,
        seed=1,
    )
    second_permutation, second_labels = generate_random_labels(
        true_labels,
        seed=1,
    )

    assert np.array_equal(first_permutation, second_permutation)
    assert np.array_equal(first_labels, second_labels)


def test_changing_random_label_seed_changes_permutation() -> None:
    dataset = generate_modular_addition_dataset(
        modulus=113,
        equals_token_id=113,
    )
    true_labels = dataset["true_labels"]

    first_permutation, first_labels = generate_random_labels(
        true_labels,
        seed=1,
    )
    second_permutation, second_labels = generate_random_labels(
        true_labels,
        seed=2,
    )

    assert not np.array_equal(first_permutation, second_permutation)
    assert not np.array_equal(first_labels, second_labels)

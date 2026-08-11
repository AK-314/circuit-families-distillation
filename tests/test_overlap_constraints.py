"""Tests for exact Stage 12 structural-overlap rules."""

from fractions import Fraction

import pytest

from circuit_families.interpretability.masks import ComponentMask
from circuit_families.interpretability.overlap_constraints import (
    is_structurally_distinct,
    jaccard_counts,
    jaccard_fraction,
    jaccard_similarity,
    maximum_pairwise_jaccard,
    pairwise_jaccard_overlaps,
    passes_jaccard_cutoff,
    validate_jaccard_cutoff,
)


def mask(*identifiers: str) -> ComponentMask:
    return ComponentMask.from_retained_identifiers(identifiers)


def test_jaccard_identity_and_symmetry() -> None:
    left = mask("H0", "N0", "N1")
    right = mask("H0", "N1", "N2")

    assert jaccard_fraction(left, left) == Fraction(1, 1)
    assert jaccard_fraction(left, right) == jaccard_fraction(
        right,
        left,
    )


def test_jaccard_counts_and_value() -> None:
    left = mask("H0", "H1", "N0")
    right = mask("H1", "N0", "N1")

    assert jaccard_counts(left, right) == (2, 4)
    assert jaccard_fraction(left, right) == Fraction(1, 2)
    assert jaccard_similarity(left, right) == 0.5


def test_disjoint_masks_have_zero_overlap() -> None:
    left = mask("H0")
    right = mask("N0")

    assert jaccard_fraction(left, right) == Fraction(0, 1)


def test_two_empty_masks_have_unit_jaccard() -> None:
    empty = ComponentMask.all_ablated()

    assert jaccard_counts(empty, empty) == (0, 0)
    assert jaccard_fraction(empty, empty) == 1
    assert jaccard_similarity(empty, empty) == 1.0


def test_exact_cutoff_boundary_passes() -> None:
    left = mask("H0", "H1", "N0")
    right = mask("H0", "H1", "N1")

    assert jaccard_fraction(left, right) == Fraction(1, 2)
    assert passes_jaccard_cutoff(
        left,
        right,
        cutoff=Fraction(1, 2),
    )


def test_overlap_above_cutoff_fails() -> None:
    left = mask("H0", "H1", "N0")
    right = mask("H0", "H1", "N0", "N1")

    assert jaccard_fraction(left, right) == Fraction(3, 4)
    assert not passes_jaccard_cutoff(
        left,
        right,
        cutoff=0.5,
    )


def test_maximum_pairwise_overlap_not_mean_controls_rule() -> None:
    proposed = mask("H0", "H1", "N0")
    high_overlap = mask("H0", "H1", "N0", "N1")
    disjoint = mask("N2")

    overlaps = pairwise_jaccard_overlaps(
        proposed,
        (high_overlap, disjoint),
    )

    assert overlaps == (
        Fraction(3, 4),
        Fraction(0, 1),
    )
    assert maximum_pairwise_jaccard(
        proposed,
        (high_overlap, disjoint),
    ) == Fraction(3, 4)

    assert not is_structurally_distinct(
        proposed,
        (high_overlap, disjoint),
        cutoff=0.5,
    )


def test_first_circuit_has_zero_prior_overlap() -> None:
    proposed = mask("H0", "N0")

    assert maximum_pairwise_jaccard(
        proposed,
        (),
    ) == Fraction(0, 1)

    assert is_structurally_distinct(
        proposed,
        (),
        cutoff=0.25,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.25, Fraction(1, 4)),
        ("0.50", Fraction(1, 2)),
        (Fraction(3, 4), Fraction(3, 4)),
    ],
)
def test_cutoff_conversion_is_exact(
    raw: Fraction | float | str,
    expected: Fraction,
) -> None:
    assert validate_jaccard_cutoff(raw) == expected


@pytest.mark.parametrize("invalid", [-0.01, 1.01, float("inf")])
def test_invalid_cutoffs_fail(invalid: float) -> None:
    with pytest.raises(ValueError):
        validate_jaccard_cutoff(invalid)

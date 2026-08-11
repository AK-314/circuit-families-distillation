"""Tests for the frozen Stage 17 structural-distinctness sweep."""

from fractions import Fraction

import pytest

from circuit_families.analysis.distinctness_sweep import (
    DISTINCTNESS_DISPLAYS,
    DISTINCTNESS_GRID,
    distinctness_display,
    passes_exact_distinctness,
    validate_distinctness_cutoff,
)


def test_distinctness_grid_is_exact_and_ordered() -> None:
    assert DISTINCTNESS_GRID == (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
    assert DISTINCTNESS_DISPLAYS == ("0.25", "0.50", "0.75")


@pytest.mark.parametrize("cutoff", DISTINCTNESS_GRID)
def test_exact_jaccard_boundary_passes(cutoff: Fraction) -> None:
    assert passes_exact_distinctness(cutoff.numerator, cutoff.denominator, cutoff)


@pytest.mark.parametrize("cutoff", DISTINCTNESS_GRID)
def test_above_jaccard_boundary_fails(cutoff: Fraction) -> None:
    union = cutoff.denominator * 4
    intersection = cutoff.numerator * 4 + 1
    assert not passes_exact_distinctness(intersection, union, cutoff)


def test_unplanned_cutoff_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unplanned"):
        validate_distinctness_cutoff("0.05")


def test_distinctness_display_uses_frozen_format() -> None:
    assert distinctness_display(Fraction(1, 2)) == "0.50"

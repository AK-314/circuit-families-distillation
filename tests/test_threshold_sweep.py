"""Tests for the frozen Stage 17 fidelity sweep."""

from fractions import Fraction

import pytest

from circuit_families.analysis.threshold_sweep import (
    FIDELITY_DISPLAYS,
    FIDELITY_GRID,
    fidelity_display,
    passes_exact_fidelity,
    validate_fidelity_threshold,
)


def test_fidelity_grid_is_exact_and_ordered() -> None:
    assert FIDELITY_GRID == (
        Fraction(4, 5),
        Fraction(17, 20),
        Fraction(9, 10),
        Fraction(19, 20),
        Fraction(39, 40),
        Fraction(99, 100),
    )
    assert FIDELITY_DISPLAYS == (
        "0.800",
        "0.850",
        "0.900",
        "0.950",
        "0.975",
        "0.990",
    )


@pytest.mark.parametrize("threshold", FIDELITY_GRID)
def test_exact_fidelity_boundary_passes(threshold: Fraction) -> None:
    total = threshold.denominator
    assert passes_exact_fidelity(threshold.numerator, total, threshold)


@pytest.mark.parametrize("threshold", FIDELITY_GRID)
def test_below_exact_fidelity_boundary_fails(threshold: Fraction) -> None:
    total = threshold.denominator * 2
    agreement = threshold.numerator * 2 - 1
    assert not passes_exact_fidelity(agreement, total, threshold)


def test_unplanned_fidelity_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unplanned"):
        validate_fidelity_threshold("0.92")


def test_fidelity_display_uses_frozen_format() -> None:
    assert fidelity_display(Fraction(39, 40)) == "0.975"

"""Frozen exact fidelity thresholds for Stage 17."""

from __future__ import annotations

from fractions import Fraction

FIDELITY_GRID = (
    Fraction(4, 5),
    Fraction(17, 20),
    Fraction(9, 10),
    Fraction(19, 20),
    Fraction(39, 40),
    Fraction(99, 100),
)

FIDELITY_DISPLAYS = (
    "0.800",
    "0.850",
    "0.900",
    "0.950",
    "0.975",
    "0.990",
)


def validate_fidelity_threshold(value: Fraction | int | float | str) -> Fraction:
    """Return one member of the frozen Stage 17 fidelity grid."""

    if isinstance(value, bool):
        raise TypeError("Fidelity threshold must not be boolean.")

    try:
        exact = value if isinstance(value, Fraction) else Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"Invalid fidelity threshold: {value!r}") from error

    if exact not in FIDELITY_GRID:
        raise ValueError(f"Unplanned Stage 17 fidelity threshold: {value!r}")

    return exact


def fidelity_display(value: Fraction | int | float | str) -> str:
    """Return the frozen three-decimal display for a threshold."""

    exact = validate_fidelity_threshold(value)
    return FIDELITY_DISPLAYS[FIDELITY_GRID.index(exact)]


def passes_exact_fidelity(
    agreement_count: int,
    evaluated_example_count: int,
    threshold: Fraction | int | float | str,
) -> bool:
    """Apply the protocol's exact integer fidelity comparison."""

    exact = validate_fidelity_threshold(threshold)
    for name, count in (
        ("agreement_count", agreement_count),
        ("evaluated_example_count", evaluated_example_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{name} must be an integer.")
        if count < 0:
            raise ValueError(f"{name} must be non-negative.")

    if agreement_count > evaluated_example_count:
        raise ValueError("agreement_count may not exceed evaluated_example_count.")

    return agreement_count * exact.denominator >= evaluated_example_count * exact.numerator

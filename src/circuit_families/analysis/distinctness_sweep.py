"""Frozen exact structural-distinctness cutoffs for Stage 17."""

from __future__ import annotations

from fractions import Fraction

DISTINCTNESS_GRID = (
    Fraction(1, 4),
    Fraction(1, 2),
    Fraction(3, 4),
)

DISTINCTNESS_DISPLAYS = (
    "0.25",
    "0.50",
    "0.75",
)


def validate_distinctness_cutoff(value: Fraction | int | float | str) -> Fraction:
    """Return one member of the frozen Stage 17 cutoff grid."""

    if isinstance(value, bool):
        raise TypeError("Distinctness cutoff must not be boolean.")

    try:
        exact = value if isinstance(value, Fraction) else Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"Invalid distinctness cutoff: {value!r}") from error

    if exact not in DISTINCTNESS_GRID:
        raise ValueError(f"Unplanned Stage 17 distinctness cutoff: {value!r}")

    return exact


def distinctness_display(value: Fraction | int | float | str) -> str:
    """Return the frozen two-decimal display for a cutoff."""

    exact = validate_distinctness_cutoff(value)
    return DISTINCTNESS_DISPLAYS[DISTINCTNESS_GRID.index(exact)]


def passes_exact_distinctness(
    intersection_count: int,
    union_count: int,
    cutoff: Fraction | int | float | str,
) -> bool:
    """Apply the protocol's exact integer Jaccard comparison."""

    exact = validate_distinctness_cutoff(cutoff)
    for name, count in (
        ("intersection_count", intersection_count),
        ("union_count", union_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{name} must be an integer.")
        if count < 0:
            raise ValueError(f"{name} must be non-negative.")

    if intersection_count > union_count:
        raise ValueError("intersection_count may not exceed union_count.")
    if union_count == 0:
        raise ValueError("union_count must be positive.")

    return intersection_count * exact.denominator <= union_count * exact.numerator

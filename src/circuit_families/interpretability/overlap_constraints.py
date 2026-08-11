"""Exact structural-overlap rules for circuit families."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from circuit_families.interpretability.masks import ComponentMask


def _validate_mask(mask: ComponentMask) -> ComponentMask:
    if not isinstance(mask, ComponentMask):
        raise TypeError("mask must be a ComponentMask.")

    return mask


def validate_jaccard_cutoff(
    cutoff: Fraction | int | float | str,
) -> Fraction:
    """Return an exact cutoff in the closed interval [0, 1]."""

    if isinstance(cutoff, bool):
        raise TypeError("cutoff must be numeric.")

    if isinstance(cutoff, Fraction):
        value = cutoff
    elif isinstance(cutoff, int):
        value = Fraction(cutoff, 1)
    elif isinstance(cutoff, float):
        if not math.isfinite(cutoff):
            raise ValueError("cutoff must be finite.")
        value = Fraction(str(cutoff))
    elif isinstance(cutoff, str):
        value = Fraction(cutoff)
    else:
        raise TypeError("cutoff must be numeric.")

    if value < 0 or value > 1:
        raise ValueError("cutoff must lie between zero and one.")

    return value


def jaccard_counts(
    left: ComponentMask,
    right: ComponentMask,
) -> tuple[int, int]:
    """Return intersection and union component counts."""

    left = _validate_mask(left)
    right = _validate_mask(right)

    left_components = set(left.retained_component_ids)
    right_components = set(right.retained_component_ids)

    return (
        len(left_components & right_components),
        len(left_components | right_components),
    )


def jaccard_fraction(
    left: ComponentMask,
    right: ComponentMask,
) -> Fraction:
    """Return exact Jaccard similarity.

    Two empty masks have Jaccard similarity one under the frozen
    Stage 12 convention.
    """

    intersection_count, union_count = jaccard_counts(left, right)

    if union_count == 0:
        return Fraction(1, 1)

    return Fraction(intersection_count, union_count)


def jaccard_similarity(
    left: ComponentMask,
    right: ComponentMask,
) -> float:
    """Return Jaccard similarity as a reporting float."""

    return float(jaccard_fraction(left, right))


def passes_jaccard_cutoff(
    left: ComponentMask,
    right: ComponentMask,
    *,
    cutoff: Fraction | int | float | str,
) -> bool:
    """Apply the cutoff using exact integer arithmetic."""

    intersection_count, union_count = jaccard_counts(left, right)

    if union_count == 0:
        raise ValueError(
            "Jaccard overlap is undefined for two empty masks."
        )

    threshold = validate_jaccard_cutoff(cutoff)

    return (
        intersection_count * threshold.denominator
        <= threshold.numerator * union_count
    )


def pairwise_jaccard_overlaps(
    proposed: ComponentMask,
    accepted_family: Sequence[ComponentMask],
) -> tuple[Fraction, ...]:
    """Return overlap with every previously accepted circuit."""

    proposed = _validate_mask(proposed)
    family = tuple(accepted_family)

    if any(not isinstance(mask, ComponentMask) for mask in family):
        raise TypeError(
            "accepted_family must contain ComponentMask values."
        )

    return tuple(
        jaccard_fraction(proposed, previous)
        for previous in family
    )


def maximum_pairwise_jaccard(
    proposed: ComponentMask,
    accepted_family: Sequence[ComponentMask],
) -> Fraction:
    """Return maximum prior overlap, or zero for the first circuit."""

    overlaps = pairwise_jaccard_overlaps(
        proposed,
        accepted_family,
    )

    if not overlaps:
        return Fraction(0, 1)

    return max(overlaps)


def is_structurally_distinct(
    proposed: ComponentMask,
    accepted_family: Sequence[ComponentMask],
    *,
    cutoff: Fraction | int | float | str,
) -> bool:
    """Require every pairwise overlap to be no greater than the cutoff."""

    threshold = validate_jaccard_cutoff(cutoff)

    return all(
        overlap <= threshold
        for overlap in pairwise_jaccard_overlaps(
            proposed,
            accepted_family,
        )
    )

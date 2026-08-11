"""Tests for deterministic Stage 14 transfer primitives."""

from __future__ import annotations

from fractions import Fraction

import pytest

from circuit_families.analysis.transfer import (
    TransferProfile,
    complete_linkage_groups,
    grouping_sensitivity,
    pairwise_transfer_distances,
    transfer_grouping,
    transfer_profile_distance,
)


def profile(
    circuit_id: str,
    value: float,
) -> TransferProfile:
    return TransferProfile(
        circuit_id=circuit_id,
        q1_fidelity=value,
        q2_fidelity=value,
        q3_fidelity=value,
        q4_fidelity=value,
    )


def test_transfer_profile_distance_is_l_infinity() -> None:
    left = TransferProfile(
        circuit_id="C1",
        q1_fidelity=0.99,
        q2_fidelity=0.90,
        q3_fidelity=0.80,
        q4_fidelity=0.70,
    )
    right = TransferProfile(
        circuit_id="C2",
        q1_fidelity=0.98,
        q2_fidelity=0.84,
        q3_fidelity=0.82,
        q4_fidelity=0.69,
    )

    assert transfer_profile_distance(left, right) == pytest.approx(
        0.06
    )


def test_pairwise_transfer_distances_are_canonical() -> None:
    distances = pairwise_transfer_distances(
        (
            profile("C3", 0.10),
            profile("C1", 0.00),
            profile("C2", 0.04),
        )
    )

    assert tuple(distances) == (
        ("C1", "C2"),
        ("C1", "C3"),
        ("C2", "C3"),
    )
    assert distances[("C1", "C2")] == pytest.approx(0.04)
    assert distances[("C1", "C3")] == pytest.approx(0.10)
    assert distances[("C2", "C3")] == pytest.approx(0.06)


def test_complete_linkage_is_deterministic_under_tie() -> None:
    profiles = (
        profile("C", 0.08),
        profile("B", 0.04),
        profile("A", 0.00),
    )

    groups = complete_linkage_groups(
        profiles,
        tolerance=Fraction(1, 20),
    )

    assert groups == (("A", "B"), ("C",))


def test_complete_linkage_requires_every_pair_within_tolerance() -> None:
    profiles = (
        profile("A", 0.00),
        profile("B", 0.04),
        profile("C", 0.08),
    )

    assert complete_linkage_groups(
        profiles,
        tolerance=Fraction(1, 25),
    ) == (("A", "B"), ("C",))

    assert complete_linkage_groups(
        profiles,
        tolerance=Fraction(2, 25),
    ) == (("A", "B", "C"),)


def test_empty_and_singleton_group_counts_follow_protocol() -> None:
    empty = transfer_grouping(
        (),
        tolerance=Fraction(1, 20),
    )
    singleton = transfer_grouping(
        (profile("C1", 0.99),),
        tolerance=Fraction(1, 20),
    )

    assert empty.groups == ()
    assert empty.group_count is None
    assert singleton.groups == (("C1",),)
    assert singleton.group_count == 1


def test_grouping_sensitivity_uses_frozen_order() -> None:
    profiles = (
        profile("C1", 0.00),
        profile("C2", 0.04),
        profile("C3", 0.09),
    )

    results = grouping_sensitivity(profiles)

    assert tuple(result.tolerance for result in results) == (
        Fraction(1, 40),
        Fraction(1, 20),
        Fraction(1, 10),
    )
    assert tuple(result.group_count for result in results) == (
        3,
        2,
        1,
    )


def test_duplicate_circuit_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        transfer_grouping(
            (
                profile("C1", 0.1),
                profile("C1", 0.2),
            )
        )

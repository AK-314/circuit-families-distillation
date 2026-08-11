"""Deterministic functional-transfer analysis primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from circuit_families.analysis.random_label_circuit_analysis import (
    PRIMARY_TRANSFER_GROUPING_TOLERANCE,
    TRANSFER_GROUPING_SENSITIVITY_GRID,
    subset_contexts,
)
from circuit_families.data.input_subsets import SUBSET_NAMES
from circuit_families.interpretability.fidelity import (
    MaskEvaluationMetrics,
    compute_full_model_reference,
    evaluate_component_mask,
)
from circuit_families.interpretability.masks import ComponentMask


@dataclass(frozen=True)
class TransferProfile:
    """Four-subset fidelity profile for one recovered circuit."""

    circuit_id: str
    q1_fidelity: float
    q2_fidelity: float
    q3_fidelity: float
    q4_fidelity: float

    def __post_init__(self) -> None:
        if not isinstance(self.circuit_id, str) or not self.circuit_id:
            raise ValueError("circuit_id must be a non-empty string.")

        for value in self.values:
            if not math.isfinite(value):
                raise ValueError(
                    "Transfer-profile fidelities must be finite."
                )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Transfer-profile fidelities must lie in [0, 1]."
                )

    @property
    def values(self) -> tuple[float, float, float, float]:
        """Return fidelities in frozen Q1-Q4 order."""

        return (
            self.q1_fidelity,
            self.q2_fidelity,
            self.q3_fidelity,
            self.q4_fidelity,
        )

    def as_mapping(self) -> dict[str, float]:
        """Return the profile keyed by frozen subset name."""

        return dict(
            zip(
                SUBSET_NAMES,
                self.values,
                strict=True,
            )
        )


@dataclass(frozen=True)
class SubsetTransferEvaluation:
    """One circuit evaluated on one frozen transfer subset."""

    circuit_id: str
    discovery_subset: str | None
    evaluation_subset: str
    metrics: MaskEvaluationMetrics


@dataclass(frozen=True)
class TransferEvaluation:
    """Complete Q1-Q4 transfer evaluation for one circuit."""

    profile: TransferProfile
    evaluations: tuple[SubsetTransferEvaluation, ...]


@dataclass(frozen=True)
class TransferGrouping:
    """Deterministic complete-linkage grouping at one tolerance."""

    tolerance: Fraction
    groups: tuple[tuple[str, ...], ...]
    group_count: int | None


def transfer_profile_distance(
    left: TransferProfile,
    right: TransferProfile,
) -> float:
    """Return the frozen maximum absolute profile difference."""

    if not isinstance(left, TransferProfile):
        raise TypeError("left must be a TransferProfile.")

    if not isinstance(right, TransferProfile):
        raise TypeError("right must be a TransferProfile.")

    return max(
        abs(left_value - right_value)
        for left_value, right_value in zip(
            left.values,
            right.values,
            strict=True,
        )
    )


def _validate_tolerance(
    tolerance: Fraction | int | float | str,
) -> Fraction:
    if isinstance(tolerance, bool):
        raise TypeError("Grouping tolerance must not be boolean.")

    try:
        exact = (
            tolerance
            if isinstance(tolerance, Fraction)
            else Fraction(str(tolerance))
        )
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"Invalid grouping tolerance: {tolerance!r}"
        ) from error

    if exact < 0:
        raise ValueError("Grouping tolerance must be non-negative.")

    return exact


def _validated_profiles(
    profiles: Sequence[TransferProfile],
) -> tuple[TransferProfile, ...]:
    values = tuple(profiles)

    if not all(
        isinstance(profile, TransferProfile)
        for profile in values
    ):
        raise TypeError(
            "profiles must contain only TransferProfile values."
        )

    ordered = tuple(
        sorted(
            values,
            key=lambda profile: profile.circuit_id,
        )
    )

    identifiers = [
        profile.circuit_id
        for profile in ordered
    ]

    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Transfer-profile circuit IDs must be unique.")

    return ordered


def pairwise_transfer_distances(
    profiles: Sequence[TransferProfile],
) -> dict[tuple[str, str], float]:
    """Return every unordered circuit-pair transfer distance."""

    ordered = _validated_profiles(profiles)
    distances: dict[tuple[str, str], float] = {}

    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            distances[
                (left.circuit_id, right.circuit_id)
            ] = transfer_profile_distance(left, right)

    return distances


def _complete_linkage_distance(
    left_cluster: tuple[TransferProfile, ...],
    right_cluster: tuple[TransferProfile, ...],
) -> float:
    return max(
        transfer_profile_distance(left, right)
        for left in left_cluster
        for right in right_cluster
    )


def complete_linkage_groups(
    profiles: Sequence[TransferProfile],
    *,
    tolerance: Fraction | int | float | str,
) -> tuple[tuple[str, ...], ...]:
    """Run deterministic complete-linkage agglomerative clustering."""

    exact_tolerance = _validate_tolerance(tolerance)
    threshold = float(exact_tolerance)
    ordered = _validated_profiles(profiles)

    clusters: list[tuple[TransferProfile, ...]] = [
        (profile,)
        for profile in ordered
    ]

    while len(clusters) > 1:
        candidates: list[
            tuple[
                float,
                tuple[str, ...],
                tuple[str, ...],
                int,
                int,
            ]
        ] = []

        for left_index, left_cluster in enumerate(clusters):
            for right_index in range(
                left_index + 1,
                len(clusters),
            ):
                right_cluster = clusters[right_index]

                candidates.append(
                    (
                        _complete_linkage_distance(
                            left_cluster,
                            right_cluster,
                        ),
                        tuple(
                            profile.circuit_id
                            for profile in left_cluster
                        ),
                        tuple(
                            profile.circuit_id
                            for profile in right_cluster
                        ),
                        left_index,
                        right_index,
                    )
                )

        (
            minimum_distance,
            _,
            _,
            left_index,
            right_index,
        ) = min(candidates)

        if minimum_distance > threshold:
            break

        merged = tuple(
            sorted(
                (
                    *clusters[left_index],
                    *clusters[right_index],
                ),
                key=lambda profile: profile.circuit_id,
            )
        )

        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
        clusters.sort(
            key=lambda cluster: tuple(
                profile.circuit_id
                for profile in cluster
            )
        )

    return tuple(
        tuple(
            profile.circuit_id
            for profile in cluster
        )
        for cluster in clusters
    )


def transfer_grouping(
    profiles: Sequence[TransferProfile],
    *,
    tolerance: Fraction | int | float | str = (
        PRIMARY_TRANSFER_GROUPING_TOLERANCE
    ),
) -> TransferGrouping:
    """Return complete-linkage groups and the protocol group count."""

    exact_tolerance = _validate_tolerance(tolerance)
    ordered = _validated_profiles(profiles)

    if not ordered:
        return TransferGrouping(
            tolerance=exact_tolerance,
            groups=(),
            group_count=None,
        )

    groups = complete_linkage_groups(
        ordered,
        tolerance=exact_tolerance,
    )

    return TransferGrouping(
        tolerance=exact_tolerance,
        groups=groups,
        group_count=len(groups),
    )


def grouping_sensitivity(
    profiles: Sequence[TransferProfile],
    *,
    tolerances: Sequence[
        Fraction | int | float | str
    ] = TRANSFER_GROUPING_SENSITIVITY_GRID,
) -> tuple[TransferGrouping, ...]:
    """Evaluate the frozen grouping-tolerance sensitivity grid."""

    exact_tolerances = tuple(
        _validate_tolerance(value)
        for value in tolerances
    )

    if len(exact_tolerances) != len(set(exact_tolerances)):
        raise ValueError("Grouping tolerances must be unique.")

    return tuple(
        transfer_grouping(
            profiles,
            tolerance=tolerance,
        )
        for tolerance in exact_tolerances
    )


def profile_from_metrics(
    *,
    circuit_id: str,
    metrics_by_subset: Mapping[
        str,
        MaskEvaluationMetrics,
    ],
) -> TransferProfile:
    """Construct one frozen-order transfer profile from metrics."""

    if set(metrics_by_subset) != set(SUBSET_NAMES):
        raise ValueError(
            "metrics_by_subset must contain exactly "
            "Q1, Q2, Q3 and Q4."
        )

    return TransferProfile(
        circuit_id=circuit_id,
        q1_fidelity=(
            metrics_by_subset["Q1"].primary_fidelity
        ),
        q2_fidelity=(
            metrics_by_subset["Q2"].primary_fidelity
        ),
        q3_fidelity=(
            metrics_by_subset["Q3"].primary_fidelity
        ),
        q4_fidelity=(
            metrics_by_subset["Q4"].primary_fidelity
        ),
    )


def evaluate_transfer_profile(
    *,
    context: Any,
    mask: ComponentMask,
    circuit_id: str,
    batch_size: int,
    discovery_subset: str | None = None,
) -> TransferEvaluation:
    """Evaluate one fixed circuit unchanged on Q1-Q4."""

    if not isinstance(mask, ComponentMask):
        raise TypeError("mask must be a ComponentMask.")

    if (
        discovery_subset is not None
        and discovery_subset not in SUBSET_NAMES
    ):
        raise ValueError(
            f"Unknown discovery subset: {discovery_subset!r}"
        )

    contexts = subset_contexts(context)
    metrics_by_subset: dict[
        str,
        MaskEvaluationMetrics,
    ] = {}
    evaluations = []

    for subset_name in SUBSET_NAMES:
        subset = contexts[subset_name]
        reference = compute_full_model_reference(
            subset.model,
            subset.inputs,
            subset.targets,
            batch_size=batch_size,
        )
        metrics = evaluate_component_mask(
            subset.model,
            subset.inputs,
            subset.targets,
            mask,
            batch_size=batch_size,
            full_model_reference=reference,
        )
        metrics_by_subset[subset_name] = metrics
        evaluations.append(
            SubsetTransferEvaluation(
                circuit_id=circuit_id,
                discovery_subset=discovery_subset,
                evaluation_subset=subset_name,
                metrics=metrics,
            )
        )

    return TransferEvaluation(
        profile=profile_from_metrics(
            circuit_id=circuit_id,
            metrics_by_subset=metrics_by_subset,
        ),
        evaluations=tuple(evaluations),
    )

"""Stage 12 fidelity-sparsity-overlap-effort frontier."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from circuit_families.analysis.fidelity_calibration import (
    file_sha256,
    write_csv_records,
)
from circuit_families.analysis.stage12_reporting import (
    Stage12ReportCell,
)
from circuit_families.interpretability.diversity_forced_search import (
    FamilyRestartOutcome,
)

FRONTIER_COLUMNS = (
    "stage12_run_id",
    "cell_id",
    "checkpoint_step",
    "distinctness_cutoff",
    "requested_member_index",
    "member_label",
    "accepted_circuit",
    "representative_restart_index",
    "primary_fidelity",
    "retained_component_count",
    "retained_component_proportion",
    "maximum_prior_overlap_numerator",
    "maximum_prior_overlap_denominator",
    "maximum_prior_overlap",
    "mean_prior_overlap",
    "most_overlapping_prior_member_index",
    "most_overlapping_prior_member_label",
    "cumulative_exact_evaluations",
    "member_exact_evaluations",
    "member_ranking_passes",
    "runtime_seconds",
    "restart_count",
    "status",
    "failure_reason",
    "family_right_censored",
    "scientific_family_result",
    "runtime_included_in_deterministic_scientific_hashes",
    "runtime_comparison_policy",
    "frontier_scalar_score",
)


@dataclass(frozen=True)
class Stage12FrontierRuntime:
    """Runtime for one requested family member."""

    cell_id: str
    requested_member_index: int
    runtime_seconds: float


@dataclass(frozen=True)
class Stage12FrontierArtifacts:
    """Path and physical hash for the frontier table."""

    table_path: Path
    table_sha256: str
    row_count: int
    runtime_bearing: bool = True


def _validate_runtime(
    record: Stage12FrontierRuntime,
) -> None:
    if not isinstance(
        record,
        Stage12FrontierRuntime,
    ):
        raise TypeError(
            "runtime records must be "
            "Stage12FrontierRuntime values."
        )

    if not record.cell_id:
        raise ValueError(
            "Runtime cell_id must not be empty."
        )

    if (
        isinstance(record.requested_member_index, bool)
        or not isinstance(
            record.requested_member_index,
            int,
        )
        or record.requested_member_index <= 0
    ):
        raise ValueError(
            "requested_member_index must be a "
            "positive integer."
        )

    if (
        isinstance(record.runtime_seconds, bool)
        or not isinstance(
            record.runtime_seconds,
            (int, float),
        )
        or not math.isfinite(
            float(record.runtime_seconds)
        )
        or float(record.runtime_seconds) < 0.0
    ):
        raise ValueError(
            "runtime_seconds must be a finite "
            "non-negative number."
        )


def _runtime_lookup(
    records: Sequence[Stage12FrontierRuntime],
) -> dict[tuple[str, int], float]:
    lookup: dict[tuple[str, int], float] = {}

    for record in records:
        _validate_runtime(record)
        key = (
            record.cell_id,
            record.requested_member_index,
        )

        if key in lookup:
            raise ValueError(
                "Duplicate frontier runtime record for "
                f"{key!r}."
            )

        lookup[key] = float(
            record.runtime_seconds
        )

    return lookup


def _ordered_cells(
    cells: Sequence[Stage12ReportCell],
    execution_order: Sequence[Fraction],
) -> tuple[Stage12ReportCell, ...]:
    if not cells:
        raise ValueError(
            "cells must not be empty."
        )

    order = {
        cutoff: index
        for index, cutoff in enumerate(
            execution_order
        )
    }

    if len(order) != len(execution_order):
        raise ValueError(
            "execution_order contains duplicates."
        )

    identifiers: set[str] = set()

    for cell in cells:
        if not isinstance(cell, Stage12ReportCell):
            raise TypeError(
                "cells must contain "
                "Stage12ReportCell values."
            )

        if cell.cell_id in identifiers:
            raise ValueError(
                f"Duplicate cell_id: {cell.cell_id}"
            )

        identifiers.add(cell.cell_id)

        if cell.distinctness_cutoff not in order:
            raise ValueError(
                "Every cell cutoff must appear in "
                "execution_order."
            )

        if (
            cell.execution.result.distinctness_cutoff
            != cell.distinctness_cutoff
        ):
            raise ValueError(
                "Cell cutoff does not match its "
                "family-search result."
            )

    return tuple(
        sorted(
            cells,
            key=lambda cell: (
                order[cell.distinctness_cutoff],
                cell.cell_id,
            ),
        )
    )


def _group_restart_outcomes(
    outcomes: Sequence[FamilyRestartOutcome],
) -> dict[int, tuple[FamilyRestartOutcome, ...]]:
    grouped: dict[
        int,
        list[FamilyRestartOutcome],
    ] = {}

    for outcome in outcomes:
        grouped.setdefault(
            outcome.requested_member_index,
            [],
        ).append(outcome)

    return {
        member_index: tuple(
            sorted(
                values,
                key=lambda value: (
                    value.restart_index
                ),
            )
        )
        for member_index, values in grouped.items()
    }


def _mean_overlap(
    overlaps: Sequence[Fraction],
) -> float | str:
    if not overlaps:
        return ""

    return float(
        sum(overlaps, Fraction(0, 1))
        / len(overlaps)
    )


def _most_overlapping_prior_member(
    overlaps: Sequence[Fraction],
) -> tuple[int | str, str]:
    if not overlaps:
        return "", ""

    index = max(
        range(1, len(overlaps) + 1),
        key=lambda prior_index: (
            overlaps[prior_index - 1],
            -prior_index,
        ),
    )

    return index, f"C{index}"


def _overlap_fields(
    overlaps: Sequence[Fraction],
    maximum: Fraction,
) -> dict[str, Any]:
    prior_index, prior_label = (
        _most_overlapping_prior_member(
            overlaps
        )
    )

    return {
        "maximum_prior_overlap_numerator": (
            maximum.numerator
        ),
        "maximum_prior_overlap_denominator": (
            maximum.denominator
        ),
        "maximum_prior_overlap": float(maximum),
        "mean_prior_overlap": _mean_overlap(
            overlaps
        ),
        "most_overlapping_prior_member_index": (
            prior_index
        ),
        "most_overlapping_prior_member_label": (
            prior_label
        ),
    }


def _last_attempted_restart(
    outcomes: Sequence[FamilyRestartOutcome],
) -> FamilyRestartOutcome:
    if not outcomes:
        raise ValueError(
            "A failed requested alternative must "
            "contain at least one restart outcome."
        )

    return max(
        outcomes,
        key=lambda outcome: outcome.restart_index,
    )


def frontier_rows(
    *,
    stage12_run_id: str,
    cells: Sequence[Stage12ReportCell],
    execution_order: Sequence[Fraction],
    runtimes: Sequence[Stage12FrontierRuntime],
) -> list[dict[str, Any]]:
    """Build the descriptive Stage 12 frontier.

    Accepted circuits use the selected restart and final accepted
    metrics. A failed requested alternative uses the terminal metrics
    from its final attempted restart. Every restart remains recorded
    separately in the Stage 12 restart table and raw artifacts.

    Runtime is included because the protocol requires effort in this
    frontier. The table is therefore runtime-bearing and is compared
    semantically rather than byte-for-byte across independent runs.
    No scalar frontier score is constructed.
    """

    if not stage12_run_id:
        raise ValueError(
            "stage12_run_id must not be empty."
        )

    ordered = _ordered_cells(
        cells,
        execution_order,
    )
    runtime_lookup = _runtime_lookup(runtimes)
    used_runtime_keys: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []

    for cell in ordered:
        family = cell.execution.result
        outcomes_by_member = (
            _group_restart_outcomes(
                family.restart_outcomes
            )
        )
        accepted_by_member = {
            member.member_index: member
            for member in family.members
        }

        unknown_accepted = set(
            accepted_by_member
        ).difference(outcomes_by_member)

        if unknown_accepted:
            raise ValueError(
                "Accepted members have no corresponding "
                "restart outcomes: "
                f"{sorted(unknown_accepted)}"
            )

        cumulative_exact_evaluations = 0

        for requested_member_index in sorted(
            outcomes_by_member
        ):
            outcomes = outcomes_by_member[
                requested_member_index
            ]
            member_exact_evaluations = sum(
                outcome.execution.result
                .exact_evaluations_used
                for outcome in outcomes
            )
            member_ranking_passes = sum(
                outcome.execution.result
                .ranking_passes_used
                for outcome in outcomes
            )
            cumulative_exact_evaluations += (
                member_exact_evaluations
            )

            runtime_key = (
                cell.cell_id,
                requested_member_index,
            )

            if runtime_key not in runtime_lookup:
                raise ValueError(
                    "Missing frontier runtime record for "
                    f"{runtime_key!r}."
                )

            used_runtime_keys.add(runtime_key)
            runtime_seconds = runtime_lookup[
                runtime_key
            ]

            common = {
                "stage12_run_id": stage12_run_id,
                "cell_id": cell.cell_id,
                "checkpoint_step": (
                    cell.checkpoint_step
                ),
                "distinctness_cutoff": float(
                    cell.distinctness_cutoff
                ),
                "requested_member_index": (
                    requested_member_index
                ),
                "member_label": (
                    f"C{requested_member_index}"
                ),
                "cumulative_exact_evaluations": (
                    cumulative_exact_evaluations
                ),
                "member_exact_evaluations": (
                    member_exact_evaluations
                ),
                "member_ranking_passes": (
                    member_ranking_passes
                ),
                "runtime_seconds": runtime_seconds,
                "restart_count": len(outcomes),
                "family_right_censored": (
                    family.right_censored
                ),
                "runtime_included_in_"
                "deterministic_scientific_hashes": (
                    False
                ),
                "runtime_comparison_policy": (
                    "semantic"
                ),
                "frontier_scalar_score": "",
            }

            member = accepted_by_member.get(
                requested_member_index
            )

            if member is not None:
                selected = [
                    outcome
                    for outcome in outcomes
                    if (
                        outcome.restart_index
                        == member.selected_restart_index
                    )
                ]

                if len(selected) != 1:
                    raise ValueError(
                        "Accepted member does not have "
                        "exactly one selected restart."
                    )

                selected_outcome = selected[0]

                if not (
                    selected_outcome.accepted_candidate
                ):
                    raise ValueError(
                        "Selected restart is not marked "
                        "as accepted."
                    )

                rows.append(
                    {
                        **common,
                        "accepted_circuit": True,
                        "representative_restart_index": (
                            member.selected_restart_index
                        ),
                        "primary_fidelity": (
                            member.metrics
                            .primary_fidelity
                        ),
                        "retained_component_count": (
                            member.mask
                            .retained_component_count
                        ),
                        "retained_component_proportion": (
                            member.mask
                            .retained_component_proportion
                        ),
                        **_overlap_fields(
                            member.pairwise_overlaps,
                            member
                            .maximum_pairwise_overlap,
                        ),
                        "status": (
                            selected_outcome
                            .outcome_status
                        ),
                        "failure_reason": "",
                        "scientific_family_result": (
                            True
                        ),
                    }
                )
                continue

            representative = (
                _last_attempted_restart(outcomes)
            )
            search = representative.execution.result

            rows.append(
                {
                    **common,
                    "accepted_circuit": False,
                    "representative_restart_index": (
                        representative.restart_index
                    ),
                    "primary_fidelity": (
                        search.final_metrics
                        .primary_fidelity
                    ),
                    "retained_component_count": (
                        search.final_mask
                        .retained_component_count
                    ),
                    "retained_component_proportion": (
                        search.final_mask
                        .retained_component_proportion
                    ),
                    **_overlap_fields(
                        representative.pairwise_overlaps,
                        representative
                        .maximum_pairwise_overlap,
                    ),
                    "status": (
                        representative.outcome_status
                    ),
                    "failure_reason": (
                        search.failure_detail
                        if search.failure_detail
                        is not None
                        else search.stopping_reason
                    ),
                    "scientific_family_result": False,
                }
            )

    unused_runtime_keys = set(
        runtime_lookup
    ).difference(used_runtime_keys)

    if unused_runtime_keys:
        raise ValueError(
            "Frontier runtime records do not correspond "
            "to emitted rows: "
            f"{sorted(unused_runtime_keys)}"
        )

    return rows


def write_frontier_table(
    path: str | Path,
    *,
    stage12_run_id: str,
    cells: Sequence[Stage12ReportCell],
    execution_order: Sequence[Fraction],
    runtimes: Sequence[Stage12FrontierRuntime],
) -> Stage12FrontierArtifacts:
    """Write the runtime-bearing descriptive frontier table."""

    rows = frontier_rows(
        stage12_run_id=stage12_run_id,
        cells=cells,
        execution_order=execution_order,
        runtimes=runtimes,
    )
    output = write_csv_records(
        path,
        fieldnames=FRONTIER_COLUMNS,
        rows=rows,
    )

    return Stage12FrontierArtifacts(
        table_path=output,
        table_sha256=file_sha256(output),
        row_count=len(rows),
    )

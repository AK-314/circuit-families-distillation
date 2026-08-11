"""Deterministic run-level tables for Stage 12."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from circuit_families.analysis.fidelity_calibration import (
    file_sha256,
    write_csv_records,
)
from circuit_families.interpretability.diversity_forced_search import (
    CheckpointFamilySearchExecution,
    FamilyRestartOutcome,
)
from circuit_families.interpretability.overlap_constraints import (
    jaccard_counts,
    jaccard_fraction,
)

FAMILY_SUMMARY_COLUMNS = (
    "stage12_run_id",
    "cell_id",
    "checkpoint_step",
    "distinctness_cutoff",
    "status",
    "stopping_reason",
    "family_size",
    "family_target",
    "right_censored",
    "exact_evaluations_used",
    "per_cell_budget",
    "budget_remaining",
    "restart_outcome_count",
    "pseudo_target_sha256",
    "full_model_reference_sha256",
    "raw_cell_directory",
)

CIRCUIT_COLUMNS = (
    "stage12_run_id",
    "cell_id",
    "checkpoint_step",
    "distinctness_cutoff",
    "member_index",
    "member_label",
    "selected_restart_index",
    "mask_id",
    "retained_attention_head_count",
    "retained_mlp_neuron_count",
    "retained_component_count",
    "retained_component_proportion",
    "primary_fidelity",
    "prediction_agreement_count",
    "evaluated_example_count",
    "maximum_pairwise_overlap_numerator",
    "maximum_pairwise_overlap_denominator",
    "maximum_pairwise_overlap",
    "exact_evaluations_used",
    "ranking_passes_used",
    "accepted_removal_count",
    "rejected_candidate_count",
    "candidate_batches_tested",
    "locally_single_deletion_minimal",
    "meaningfully_sparse",
)

PAIRWISE_OVERLAP_COLUMNS = (
    "stage12_run_id",
    "cell_id",
    "checkpoint_step",
    "distinctness_cutoff",
    "left_member_index",
    "left_member_label",
    "right_member_index",
    "right_member_label",
    "intersection_count",
    "union_count",
    "jaccard_numerator",
    "jaccard_denominator",
    "jaccard",
    "passes_active_cutoff",
)

RESTART_COLUMNS = (
    "stage12_run_id",
    "cell_id",
    "checkpoint_step",
    "distinctness_cutoff",
    "requested_member_index",
    "requested_member_label",
    "restart_index",
    "restart_used",
    "unused_reason",
    "seed_model_seed",
    "seed_checkpoint_index",
    "seed_family_member_index",
    "seed_restart_index",
    "seed_canonical_material",
    "seed_sha256_digest",
    "seed_integer",
    "seed_bit_generator",
    "outcome_status",
    "accepted_candidate",
    "search_status",
    "stopping_reason",
    "failure_detail",
    "retained_component_count",
    "primary_fidelity",
    "maximum_pairwise_overlap_numerator",
    "maximum_pairwise_overlap_denominator",
    "maximum_pairwise_overlap",
    "exact_evaluation_budget",
    "exact_evaluations_used",
    "ranking_passes_used",
    "candidate_batches_tested",
    "rejected_candidate_count",
    "budget_remaining",
    "budget_exhausted",
    "locally_single_deletion_minimal",
    "meaningfully_sparse",
)


@dataclass(frozen=True)
class Stage12ReportCell:
    """One completed Stage 12 cell for run-level reporting."""

    cell_id: str
    checkpoint_step: int
    distinctness_cutoff: Fraction
    execution: CheckpointFamilySearchExecution
    raw_cell_directory: str


@dataclass(frozen=True)
class Stage12ReportArtifacts:
    """Paths and hashes for deterministic Stage 12 reports."""

    output_directory: Path
    family_summary_path: Path
    family_summary_sha256: str
    circuit_path: Path
    circuit_sha256: str
    pairwise_overlap_path: Path
    pairwise_overlap_sha256: str
    restart_path: Path
    restart_sha256: str
    validation_note_path: Path
    validation_note_sha256: str


def _prepare_empty_output_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    existing = tuple(path.iterdir())

    if existing:
        names = ", ".join(
            sorted(item.name for item in existing)
        )
        raise FileExistsError(
            "Stage 12 report output directory must be empty. "
            f"Existing entries: {names}"
        )

    return path


def _validate_cell(
    cell: Stage12ReportCell,
) -> None:
    if not isinstance(cell, Stage12ReportCell):
        raise TypeError(
            "cells must contain Stage12ReportCell values."
        )

    if not cell.cell_id:
        raise ValueError("cell_id must not be empty.")

    if (
        isinstance(cell.checkpoint_step, bool)
        or not isinstance(cell.checkpoint_step, int)
        or cell.checkpoint_step < 0
    ):
        raise ValueError(
            "checkpoint_step must be a non-negative integer."
        )

    if not isinstance(
        cell.execution,
        CheckpointFamilySearchExecution,
    ):
        raise TypeError(
            "cell.execution must be a "
            "CheckpointFamilySearchExecution."
        )

    if (
        cell.execution.result.distinctness_cutoff
        != cell.distinctness_cutoff
    ):
        raise ValueError(
            "Cell cutoff does not match execution result."
        )

    if not cell.raw_cell_directory:
        raise ValueError(
            "raw_cell_directory must not be empty."
        )


def _ordered_cells(
    cells: Sequence[Stage12ReportCell],
    execution_order: Sequence[Fraction],
) -> tuple[Stage12ReportCell, ...]:
    if not cells:
        raise ValueError("cells must not be empty.")

    order = {
        cutoff: index
        for index, cutoff in enumerate(
            execution_order
        )
    }

    if len(order) != len(execution_order):
        raise ValueError(
            "execution_order must not contain duplicates."
        )

    identifiers: set[str] = set()

    for cell in cells:
        _validate_cell(cell)

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

    return tuple(
        sorted(
            cells,
            key=lambda cell: (
                order[cell.distinctness_cutoff],
                cell.cell_id,
            ),
        )
    )


def _fraction_fields(
    prefix: str,
    value: Fraction,
) -> dict[str, Any]:
    return {
        f"{prefix}_numerator": value.numerator,
        f"{prefix}_denominator": value.denominator,
        prefix: float(value),
    }


def family_summary_rows(
    *,
    stage12_run_id: str,
    cells: Sequence[Stage12ReportCell],
    execution_order: Sequence[Fraction],
) -> list[dict[str, Any]]:
    ordered = _ordered_cells(
        cells,
        execution_order,
    )
    rows: list[dict[str, Any]] = []

    for cell in ordered:
        execution = cell.execution
        family = execution.result

        rows.append(
            {
                "stage12_run_id": stage12_run_id,
                "cell_id": cell.cell_id,
                "checkpoint_step": (
                    cell.checkpoint_step
                ),
                "distinctness_cutoff": float(
                    cell.distinctness_cutoff
                ),
                "status": family.status,
                "stopping_reason": (
                    family.stopping_reason
                ),
                "family_size": family.family_size,
                "family_target": family.family_target,
                "right_censored": (
                    family.right_censored
                ),
                "exact_evaluations_used": (
                    family.exact_evaluations_used
                ),
                "per_cell_budget": (
                    family.per_cell_budget
                ),
                "budget_remaining": (
                    family.budget_remaining
                ),
                "restart_outcome_count": len(
                    family.restart_outcomes
                ),
                "pseudo_target_sha256": (
                    execution.pseudo_target_sha256
                ),
                "full_model_reference_sha256": (
                    execution
                    .full_model_reference_sha256
                ),
                "raw_cell_directory": (
                    cell.raw_cell_directory
                ),
            }
        )

    return rows


def circuit_rows(
    *,
    stage12_run_id: str,
    cells: Sequence[Stage12ReportCell],
    execution_order: Sequence[Fraction],
) -> list[dict[str, Any]]:
    ordered = _ordered_cells(
        cells,
        execution_order,
    )
    rows: list[dict[str, Any]] = []

    for cell in ordered:
        for member in cell.execution.result.members:
            mask = member.mask
            metrics = member.metrics

            rows.append(
                {
                    "stage12_run_id": stage12_run_id,
                    "cell_id": cell.cell_id,
                    "checkpoint_step": (
                        cell.checkpoint_step
                    ),
                    "distinctness_cutoff": float(
                        cell.distinctness_cutoff
                    ),
                    "member_index": (
                        member.member_index
                    ),
                    "member_label": (
                        f"C{member.member_index}"
                    ),
                    "selected_restart_index": (
                        member.selected_restart_index
                    ),
                    "mask_id": mask.mask_id,
                    "retained_attention_head_count": (
                        mask.retained_attention_head_count
                    ),
                    "retained_mlp_neuron_count": (
                        mask.retained_mlp_neuron_count
                    ),
                    "retained_component_count": (
                        mask.retained_component_count
                    ),
                    "retained_component_proportion": (
                        mask.retained_component_proportion
                    ),
                    "primary_fidelity": (
                        metrics.primary_fidelity
                    ),
                    "prediction_agreement_count": (
                        metrics.prediction_agreement_count
                    ),
                    "evaluated_example_count": (
                        metrics.evaluated_example_count
                    ),
                    **_fraction_fields(
                        "maximum_pairwise_overlap",
                        member.maximum_pairwise_overlap,
                    ),
                    "exact_evaluations_used": (
                        member.search_result
                        .exact_evaluations_used
                    ),
                    "ranking_passes_used": (
                        member.search_result
                        .ranking_passes_used
                    ),
                    "accepted_removal_count": len(
                        member.search_result
                        .accepted_removals
                    ),
                    "rejected_candidate_count": (
                        member.search_result
                        .rejected_candidate_count
                    ),
                    "candidate_batches_tested": (
                        member.search_result
                        .candidate_batches_tested
                    ),
                    "locally_single_deletion_minimal": (
                        member.search_result
                        .locally_single_deletion_minimal
                    ),
                    "meaningfully_sparse": (
                        member.search_result
                        .meaningfully_sparse
                    ),
                }
            )

    return rows


def pairwise_overlap_rows(
    *,
    stage12_run_id: str,
    cells: Sequence[Stage12ReportCell],
    execution_order: Sequence[Fraction],
) -> list[dict[str, Any]]:
    ordered = _ordered_cells(
        cells,
        execution_order,
    )
    rows: list[dict[str, Any]] = []

    for cell in ordered:
        members = cell.execution.result.members

        for left_position, left in enumerate(members):
            for right in members[left_position + 1 :]:
                intersection, union = jaccard_counts(
                    left.mask,
                    right.mask,
                )
                overlap = jaccard_fraction(
                    left.mask,
                    right.mask,
                )

                rows.append(
                    {
                        "stage12_run_id": (
                            stage12_run_id
                        ),
                        "cell_id": cell.cell_id,
                        "checkpoint_step": (
                            cell.checkpoint_step
                        ),
                        "distinctness_cutoff": float(
                            cell.distinctness_cutoff
                        ),
                        "left_member_index": (
                            left.member_index
                        ),
                        "left_member_label": (
                            f"C{left.member_index}"
                        ),
                        "right_member_index": (
                            right.member_index
                        ),
                        "right_member_label": (
                            f"C{right.member_index}"
                        ),
                        "intersection_count": (
                            intersection
                        ),
                        "union_count": union,
                        **_fraction_fields(
                            "jaccard",
                            overlap,
                        ),
                        "passes_active_cutoff": (
                            overlap
                            <= cell.distinctness_cutoff
                        ),
                    }
                )

    return rows


def _used_restart_row(
    *,
    stage12_run_id: str,
    cell: Stage12ReportCell,
    outcome: FamilyRestartOutcome,
) -> dict[str, Any]:
    result = outcome.execution.result
    seed = outcome.seed_record

    seed_values = {
        "seed_model_seed": "",
        "seed_checkpoint_index": "",
        "seed_family_member_index": "",
        "seed_restart_index": "",
        "seed_canonical_material": "",
        "seed_sha256_digest": "",
        "seed_integer": "",
        "seed_bit_generator": "",
    }

    if seed is not None:
        seed_values = {
            "seed_model_seed": seed.model_seed,
            "seed_checkpoint_index": (
                seed.checkpoint_index
            ),
            "seed_family_member_index": (
                seed.family_member_index
            ),
            "seed_restart_index": (
                seed.restart_index
            ),
            "seed_canonical_material": (
                seed.canonical_material
            ),
            "seed_sha256_digest": (
                seed.sha256_digest
            ),
            "seed_integer": seed.integer_seed,
            "seed_bit_generator": (
                seed.bit_generator
            ),
        }

    return {
        "stage12_run_id": stage12_run_id,
        "cell_id": cell.cell_id,
        "checkpoint_step": cell.checkpoint_step,
        "distinctness_cutoff": float(
            cell.distinctness_cutoff
        ),
        "requested_member_index": (
            outcome.requested_member_index
        ),
        "requested_member_label": (
            f"C{outcome.requested_member_index}"
        ),
        "restart_index": outcome.restart_index,
        "restart_used": True,
        "unused_reason": "",
        **seed_values,
        "outcome_status": outcome.outcome_status,
        "accepted_candidate": (
            outcome.accepted_candidate
        ),
        "search_status": result.status,
        "stopping_reason": result.stopping_reason,
        "failure_detail": (
            ""
            if result.failure_detail is None
            else result.failure_detail
        ),
        "retained_component_count": (
            result.final_mask.retained_component_count
        ),
        "primary_fidelity": (
            result.final_metrics.primary_fidelity
        ),
        **_fraction_fields(
            "maximum_pairwise_overlap",
            outcome.maximum_pairwise_overlap,
        ),
        "exact_evaluation_budget": (
            result.exact_evaluation_budget
        ),
        "exact_evaluations_used": (
            result.exact_evaluations_used
        ),
        "ranking_passes_used": (
            result.ranking_passes_used
        ),
        "candidate_batches_tested": (
            result.candidate_batches_tested
        ),
        "rejected_candidate_count": (
            result.rejected_candidate_count
        ),
        "budget_remaining": result.budget_remaining,
        "budget_exhausted": (
            result.budget_exhausted
        ),
        "locally_single_deletion_minimal": (
            result.locally_single_deletion_minimal
        ),
        "meaningfully_sparse": (
            result.meaningfully_sparse
        ),
    }


def _unused_restart_row(
    *,
    stage12_run_id: str,
    cell: Stage12ReportCell,
    requested_member_index: int,
    restart_index: int,
    unused_reason: str,
) -> dict[str, Any]:
    return {
        "stage12_run_id": stage12_run_id,
        "cell_id": cell.cell_id,
        "checkpoint_step": cell.checkpoint_step,
        "distinctness_cutoff": float(
            cell.distinctness_cutoff
        ),
        "requested_member_index": (
            requested_member_index
        ),
        "requested_member_label": (
            f"C{requested_member_index}"
        ),
        "restart_index": restart_index,
        "restart_used": False,
        "unused_reason": unused_reason,
        "seed_model_seed": "",
        "seed_checkpoint_index": "",
        "seed_family_member_index": "",
        "seed_restart_index": "",
        "seed_canonical_material": "",
        "seed_sha256_digest": "",
        "seed_integer": "",
        "seed_bit_generator": "",
        "outcome_status": "",
        "accepted_candidate": False,
        "search_status": "",
        "stopping_reason": "",
        "failure_detail": "",
        "retained_component_count": "",
        "primary_fidelity": "",
        "maximum_pairwise_overlap_numerator": "",
        "maximum_pairwise_overlap_denominator": "",
        "maximum_pairwise_overlap": "",
        "exact_evaluation_budget": "",
        "exact_evaluations_used": "",
        "ranking_passes_used": "",
        "candidate_batches_tested": "",
        "rejected_candidate_count": "",
        "budget_remaining": "",
        "budget_exhausted": "",
        "locally_single_deletion_minimal": "",
        "meaningfully_sparse": "",
    }


def restart_rows(
    *,
    stage12_run_id: str,
    cells: Sequence[Stage12ReportCell],
    execution_order: Sequence[Fraction],
) -> list[dict[str, Any]]:
    ordered = _ordered_cells(
        cells,
        execution_order,
    )
    rows: list[dict[str, Any]] = []

    for cell in ordered:
        family = cell.execution.result
        grouped: dict[
            int,
            list[FamilyRestartOutcome],
        ] = {}

        for outcome in family.restart_outcomes:
            grouped.setdefault(
                outcome.requested_member_index,
                [],
            ).append(outcome)

        for member_index in sorted(grouped):
            outcomes = sorted(
                grouped[member_index],
                key=lambda value: value.restart_index,
            )

            for outcome in outcomes:
                rows.append(
                    _used_restart_row(
                        stage12_run_id=stage12_run_id,
                        cell=cell,
                        outcome=outcome,
                    )
                )

            if member_index == 1:
                continue

            used_indices = {
                outcome.restart_index
                for outcome in outcomes
            }
            accepted = any(
                outcome.accepted_candidate
                for outcome in outcomes
            )

            for restart_index in range(
                family.max_restarts_per_alternative
            ):
                if restart_index in used_indices:
                    continue

                if accepted:
                    reason = (
                        "earlier_valid_distinct_candidate"
                    )
                elif family.budget_remaining == 0:
                    reason = "no_cell_budget_remaining"
                else:
                    reason = "search_stopped_before_restart"

                rows.append(
                    _unused_restart_row(
                        stage12_run_id=stage12_run_id,
                        cell=cell,
                        requested_member_index=(
                            member_index
                        ),
                        restart_index=restart_index,
                        unused_reason=reason,
                    )
                )

    return rows


def _validation_note_text(
    *,
    stage12_run_id: str,
    cells: Sequence[Stage12ReportCell],
    execution_order: Sequence[Fraction],
) -> str:
    ordered = _ordered_cells(
        cells,
        execution_order,
    )

    lines = [
        "# Stage 12 diversity-search validation",
        "",
        f"- Stage 12 run ID: `{stage12_run_id}`",
        "- Scientific runtime telemetry is excluded from "
        "deterministic hashes.",
        "- Exact distinctness uses maximum pairwise Jaccard "
        "over retained components.",
        "- Unused restart slots are recorded explicitly.",
        "- No pre-grokking or transition family search was run.",
        "- Stage 13 has not begun.",
        "",
        "## Completed cells",
        "",
    ]

    for cell in ordered:
        family = cell.execution.result
        lines.append(
            "- "
            f"`{cell.cell_id}`: cutoff "
            f"`{float(cell.distinctness_cutoff):.2f}`, "
            f"status `{family.status}`, "
            f"family size `{family.family_size}`, "
            f"exact evaluations "
            f"`{family.exact_evaluations_used}`."
        )

    lines.append("")
    return "\n".join(lines)


def write_stage12_report_artifacts(
    output_directory: str | Path,
    *,
    stage12_run_id: str,
    seed: int,
    cells: Sequence[Stage12ReportCell],
    execution_order: Sequence[Fraction],
) -> Stage12ReportArtifacts:
    """Write deterministic run-level Stage 12 reports."""

    if not stage12_run_id:
        raise ValueError(
            "stage12_run_id must not be empty."
        )

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError(
            "seed must be a non-negative integer."
        )

    ordered = _ordered_cells(
        cells,
        execution_order,
    )
    output = _prepare_empty_output_directory(
        Path(output_directory)
    )

    try:
        prefix = f"seed_{seed}_stage12"

        family_summary_path = write_csv_records(
            output / f"{prefix}_family_summary.csv",
            fieldnames=FAMILY_SUMMARY_COLUMNS,
            rows=family_summary_rows(
                stage12_run_id=stage12_run_id,
                cells=ordered,
                execution_order=execution_order,
            ),
        )
        circuit_path = write_csv_records(
            output / f"{prefix}_circuits.csv",
            fieldnames=CIRCUIT_COLUMNS,
            rows=circuit_rows(
                stage12_run_id=stage12_run_id,
                cells=ordered,
                execution_order=execution_order,
            ),
        )
        pairwise_overlap_path = write_csv_records(
            output / f"{prefix}_pairwise_overlap.csv",
            fieldnames=PAIRWISE_OVERLAP_COLUMNS,
            rows=pairwise_overlap_rows(
                stage12_run_id=stage12_run_id,
                cells=ordered,
                execution_order=execution_order,
            ),
        )
        restart_path = write_csv_records(
            output / f"{prefix}_restarts.csv",
            fieldnames=RESTART_COLUMNS,
            rows=restart_rows(
                stage12_run_id=stage12_run_id,
                cells=ordered,
                execution_order=execution_order,
            ),
        )

        validation_note_path = (
            output
            / f"seed_{seed}_stage12_validation.md"
        )
        validation_note_path.write_text(
            _validation_note_text(
                stage12_run_id=stage12_run_id,
                cells=ordered,
                execution_order=execution_order,
            ),
            encoding="utf-8",
        )

        return Stage12ReportArtifacts(
            output_directory=output,
            family_summary_path=(
                family_summary_path
            ),
            family_summary_sha256=file_sha256(
                family_summary_path
            ),
            circuit_path=circuit_path,
            circuit_sha256=file_sha256(
                circuit_path
            ),
            pairwise_overlap_path=(
                pairwise_overlap_path
            ),
            pairwise_overlap_sha256=file_sha256(
                pairwise_overlap_path
            ),
            restart_path=restart_path,
            restart_sha256=file_sha256(
                restart_path
            ),
            validation_note_path=(
                validation_note_path
            ),
            validation_note_sha256=file_sha256(
                validation_note_path
            ),
        )

    except Exception:
        shutil.rmtree(output)
        raise

"""Definitive execution and reporting for Stage 17 sensitivity analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from statistics import mean
from typing import Any

from circuit_families.analysis.fidelity_calibration import write_csv_records
from circuit_families.analysis.stage12_artifacts import (
    Stage12CellArtifacts,
    write_stage12_cell_artifacts,
)
from circuit_families.analysis.stage14_random_label_reporting import (
    write_deterministic_archive,
)
from circuit_families.analysis.stage17_sensitivity import (
    CHECKPOINT_INDEX,
    CHECKPOINT_STEP,
    FAMILY_TARGET,
    MAXIMUM_RETAINED_COMPONENTS,
    MODEL_SEED,
    PER_CELL_BUDGET,
    PER_REQUESTED_CIRCUIT_BUDGET,
    PRIMARY_CELL_KEY,
    PRIMARY_TRANSFER_TOLERANCE,
    ReferenceFamily,
    Stage17Cell,
    Stage17InputValidation,
    circuit_size_summary,
    structural_overlap_summary,
    validate_stage17_inputs,
)
from circuit_families.analysis.transfer import (
    TransferProfile,
    evaluate_transfer_profile,
    pairwise_transfer_distances,
    transfer_grouping,
)
from circuit_families.data.input_subsets import SUBSET_NAMES
from circuit_families.interpretability.diversity_forced_search import (
    CheckpointFamilySearchExecution,
    run_checkpoint_family_search,
)
from circuit_families.interpretability.fidelity import (
    CheckpointEvaluationContext,
    load_checkpoint_evaluation_context,
)
from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_HOOK_NAME,
    MLP_NEURON_HOOK_NAME,
    ComponentMask,
)
from circuit_families.interpretability.overlap_constraints import jaccard_counts
from circuit_families.manifests import package_versions
from circuit_families.training import canonical_state_hash, file_sha256


@dataclass(frozen=True)
class NormalizedCircuit:
    """One accepted circuit in a fresh or referenced Stage 17 family."""

    circuit_id: str
    member_index: int
    selected_restart_index: int
    mask: ComponentMask
    mask_sha256: str
    metrics: Mapping[str, Any]
    exact_evaluations_used: int
    ranking_passes_used: int
    accepted_removal_count: int
    rejected_candidate_count: int
    candidate_batches_tested: int
    locally_single_deletion_minimal: bool
    maximum_prior_overlap: Fraction
    prior_overlaps: tuple[Fraction, ...]


@dataclass(frozen=True)
class NormalizedOverlap:
    """One exact pairwise structural-overlap record."""

    circuit_i: str
    circuit_j: str
    member_i: int
    member_j: int
    intersection_count: int
    union_count: int
    jaccard: Fraction


@dataclass(frozen=True)
class CellSearchResult:
    """Normalized family-search result for one Stage 17 cell."""

    cell: Stage17Cell
    status: str
    stopping_reason: str
    family_size: int
    right_censored: bool
    exact_evaluations_used: int
    budget_remaining: int
    circuits: tuple[NormalizedCircuit, ...]
    overlaps: tuple[NormalizedOverlap, ...]
    restart_rows: tuple[dict[str, object], ...]
    ranking_passes: int
    restarts_attempted: int
    failed_requested_member_count: int
    terminal_requested_member_index: int | None
    invalid_output_count: int
    raw_cell_directory: Path
    search_integrity: Mapping[str, Any]


@dataclass(frozen=True)
class CellTransferResult:
    """Normalized transfer and grouping result for one nonempty cell."""

    cell: Stage17Cell
    profiles: tuple[TransferProfile, ...]
    profile_rows: tuple[dict[str, object], ...]
    evaluation_rows: tuple[dict[str, object], ...]
    distance_rows: tuple[dict[str, object], ...]
    group_rows: tuple[dict[str, object], ...]
    group_count: int | None
    status: str


@dataclass(frozen=True)
class Stage17OutputPaths:
    """All definitive Stage 17 output paths."""

    raw_root: Path
    tables: Mapping[str, Path]
    figures: Mapping[str, Path]
    note: Path
    caption: Path
    archive: Path
    manifest: Path


@dataclass(frozen=True)
class Stage17ExecutionResult:
    """Locations and headline values produced by Stage 17."""

    run_id: str
    implementation_commit: str
    manifest: Path
    archive: Path
    note: Path
    runtime_table: Path
    scientific_tables: tuple[Path, ...]
    figures: tuple[Path, ...]
    classification: str


REGISTRY_COLUMNS = (
    "stage17_run_id",
    "cell_index",
    "cell_id",
    "model_seed",
    "checkpoint_step",
    "fidelity_numerator",
    "fidelity_denominator",
    "displayed_fidelity",
    "distinctness_numerator",
    "distinctness_denominator",
    "displayed_jaccard_cutoff",
    "primary_cell",
    "search_execution_mode",
    "search_source_stage",
    "search_source_run_id",
    "search_source_manifest",
    "search_source_table",
    "transfer_execution_mode",
    "transfer_source_stage",
    "transfer_source_run_id",
    "transfer_source_manifest",
    "transfer_source_table",
    "expected_search_budget",
    "output_status",
    "transfer_grouping_status",
)

FAMILY_COLUMNS = (
    "stage17_run_id",
    "cell_index",
    "cell_id",
    "model_seed",
    "checkpoint_step",
    "fidelity_numerator",
    "fidelity_denominator",
    "displayed_fidelity",
    "distinctness_numerator",
    "distinctness_denominator",
    "displayed_jaccard_cutoff",
    "primary_cell",
    "search_execution_mode",
    "source_stage",
    "source_run_id",
    "family_size",
    "family_target",
    "right_censored",
    "right_censoring_status",
    "status",
    "stopping_reason",
    "exact_evaluations_used",
    "ranking_passes",
    "restarts_attempted",
    "accepted_circuit_count",
    "failed_requested_member_count",
    "budget_remaining",
    "transfer_group_count",
    "transfer_group_status",
    "pair_count",
    "minimum_pairwise_jaccard_overlap",
    "maximum_pairwise_jaccard_overlap",
    "mean_pairwise_overlap",
    "median_pairwise_overlap",
    "minimum_structural_distance",
    "maximum_structural_distance",
    "mean_structural_distance",
    "cutoff_compliance",
)

CIRCUIT_COLUMNS = (
    "stage17_run_id",
    "cell_index",
    "cell_id",
    "fidelity_numerator",
    "fidelity_denominator",
    "displayed_fidelity",
    "distinctness_numerator",
    "distinctness_denominator",
    "displayed_jaccard_cutoff",
    "search_execution_mode",
    "source_stage",
    "source_run_id",
    "circuit_id",
    "family_member_index",
    "retained_heads",
    "retained_neurons",
    "retained_components",
    "retained_proportion",
    "fidelity",
    "agreement_count",
    "evaluated_example_count",
    "threshold_pass",
    "ground_truth_accuracy",
    "cross_entropy",
    "kl_divergence",
    "jensen_shannon_divergence",
    "local_single_deletion_minimality",
    "maximum_prior_overlap_numerator",
    "maximum_prior_overlap_denominator",
    "maximum_prior_overlap",
    "mean_prior_overlap",
    "mask_sha256",
    "exact_evaluation_count",
    "ranking_passes",
    "selected_restart",
)

OVERLAP_COLUMNS = (
    "stage17_run_id",
    "cell_index",
    "cell_id",
    "fidelity_numerator",
    "fidelity_denominator",
    "displayed_fidelity",
    "distinctness_numerator",
    "distinctness_denominator",
    "displayed_jaccard_cutoff",
    "search_execution_mode",
    "source_stage",
    "source_run_id",
    "circuit_i",
    "circuit_j",
    "member_i",
    "member_j",
    "intersection_count",
    "union_count",
    "jaccard_numerator",
    "jaccard_denominator",
    "jaccard_overlap",
    "structural_distance",
    "passes_active_cutoff",
)

RESTART_COLUMNS = (
    "stage17_run_id",
    "cell_index",
    "cell_id",
    "fidelity_numerator",
    "fidelity_denominator",
    "distinctness_numerator",
    "distinctness_denominator",
    "search_execution_mode",
    "source_stage",
    "source_run_id",
    "requested_member_index",
    "restart_index",
    "restart_used",
    "seed_integer",
    "seed_sha256_digest",
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

FAILURE_COLUMNS = (
    "stage17_run_id",
    "record_type",
    "cell_index",
    "cell_id",
    "fidelity_numerator",
    "fidelity_denominator",
    "displayed_fidelity",
    "distinctness_numerator",
    "distinctness_denominator",
    "displayed_jaccard_cutoff",
    "family_size",
    "c1_qualified",
    "failed_requested_member_count",
    "terminal_requested_member_index",
    "stopping_reason",
    "failure_category",
    "budget_exhausted",
    "per_cell_evaluations_used",
    "unused_budget",
    "ranking_passes",
    "restart_count",
    "invalid_output_count",
    "search_execution_mode",
)

SIZE_COLUMNS = (
    "stage17_run_id",
    "cell_index",
    "cell_id",
    "fidelity_numerator",
    "fidelity_denominator",
    "displayed_fidelity",
    "distinctness_numerator",
    "distinctness_denominator",
    "displayed_jaccard_cutoff",
    "circuit_count",
    "minimum_retained_components",
    "maximum_retained_components",
    "mean_retained_components",
    "median_retained_components",
    "minimum_retained_proportion",
    "maximum_retained_proportion",
    "mean_retained_proportion",
    "median_retained_proportion",
)

TRANSFER_PROFILE_COLUMNS = (
    "stage17_run_id",
    "cell_index",
    "cell_id",
    "fidelity_numerator",
    "fidelity_denominator",
    "distinctness_numerator",
    "distinctness_denominator",
    "transfer_execution_mode",
    "source_stage",
    "source_run_id",
    "circuit_id",
    "q1_fidelity",
    "q2_fidelity",
    "q3_fidelity",
    "q4_fidelity",
    "q1_accuracy",
    "q2_accuracy",
    "q3_accuracy",
    "q4_accuracy",
    "q1_cross_entropy",
    "q2_cross_entropy",
    "q3_cross_entropy",
    "q4_cross_entropy",
    "q1_kl_divergence",
    "q2_kl_divergence",
    "q3_kl_divergence",
    "q4_kl_divergence",
    "q1_jensen_shannon_divergence",
    "q2_jensen_shannon_divergence",
    "q3_jensen_shannon_divergence",
    "q4_jensen_shannon_divergence",
)

TRANSFER_DISTANCE_COLUMNS = (
    "stage17_run_id",
    "cell_index",
    "cell_id",
    "circuit_i",
    "circuit_j",
    "q1_absolute_difference",
    "q2_absolute_difference",
    "q3_absolute_difference",
    "q4_absolute_difference",
    "maximum_absolute_difference",
    "maximum_distance_subset",
    "same_primary_transfer_group",
)

TRANSFER_GROUP_COLUMNS = (
    "stage17_run_id",
    "cell_index",
    "cell_id",
    "tolerance_numerator",
    "tolerance_denominator",
    "tolerance",
    "group_count",
    "group_id",
    "ordered_members_json",
    "within_group_maximum_distance",
    "between_group_minimum_distance",
    "complete_linkage_valid",
    "transfer_execution_mode",
    "source_stage",
    "source_run_id",
)

FRONTIER_COLUMNS = (
    "stage17_run_id",
    "cell_index",
    "cell_id",
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
    "cumulative_exact_evaluations",
    "member_exact_evaluations",
    "member_ranking_passes",
    "restart_count",
    "status",
    "failure_reason",
    "family_right_censored",
    "search_execution_mode",
)

FIGURE_SOURCE_COLUMNS = (
    "stage17_run_id",
    "cell_index",
    "cell_id",
    "fidelity_numerator",
    "fidelity_denominator",
    "displayed_fidelity",
    "distinctness_numerator",
    "distinctness_denominator",
    "displayed_jaccard_cutoff",
    "family_size",
    "right_censored",
    "reference_cell",
    "primary_cell",
    "zero_family",
)

RUNTIME_COLUMNS = (
    "stage17_run_id",
    "record_type",
    "cell_index",
    "cell_id",
    "circuit_id",
    "elapsed_seconds",
    "included_in_deterministic_scientific_hashes",
)


def _stable_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")
    return path


def deterministic_stage17_run_id(config_sha256: str, implementation_commit: str) -> str:
    """Return the output-state-independent Stage 17 run identity."""

    material = json.dumps(
        {
            "experiment": "stage17_two_dimensional_sensitivity",
            "configuration_sha256": config_sha256,
            "implementation_commit": implementation_commit,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"stage17-sensitivity-s1-{digest}"


def build_output_paths(
    validation: Stage17InputValidation,
    *,
    output_root: str | Path | None = None,
) -> Stage17OutputPaths:
    """Resolve every configured output beneath one selected root."""

    root = validation.repository if output_root is None else Path(output_root).resolve()
    outputs = validation.configuration.payload["outputs"]
    run_id = deterministic_stage17_run_id(
        validation.configuration.sha256, validation.implementation_commit
    )

    def resolve(template: str) -> Path:
        return root / template.format(stage17_run_id=run_id)

    tables = {key: resolve(value) for key, value in outputs.items() if key.endswith("_table")}
    figures = {
        key: resolve(value)
        for key, value in outputs.items()
        if key in {"heatmap_png", "heatmap_pdf", "curves_png", "curves_pdf"}
    }
    return Stage17OutputPaths(
        raw_root=resolve(outputs["raw_directory_template"]),
        tables=tables,
        figures=figures,
        note=resolve(outputs["note"]),
        caption=resolve(outputs["caption"]),
        archive=resolve(outputs["archive_template"]),
        manifest=resolve(outputs["manifest_template"]),
    )


def _all_output_paths(paths: Stage17OutputPaths) -> tuple[Path, ...]:
    return (
        paths.raw_root,
        *paths.tables.values(),
        *paths.figures.values(),
        paths.note,
        paths.caption,
        paths.archive,
        paths.manifest,
    )


def validate_absent_outputs(paths: Stage17OutputPaths) -> None:
    """Reject definitive execution when any configured output exists."""

    existing = [path for path in _all_output_paths(paths) if path.exists()]
    if existing:
        raise FileExistsError(
            "Stage 17 refuses to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )


def validate_archive_member_contract(root: Path, members: Sequence[Path]) -> None:
    """Reject duplicate, missing, or out-of-root deterministic members."""

    resolved_root = root.resolve()
    resolved_members = tuple(path.resolve() for path in members)
    if len(resolved_members) != len(set(resolved_members)):
        raise ValueError("Deterministic Stage 17 archive members must be unique.")
    for path in resolved_members:
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 17 archive member: {path}")
        try:
            relative = path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"Stage 17 archive member is outside output root: {path}") from error
        if ".." in relative.parts or relative.is_absolute():
            raise ValueError(f"Unsafe Stage 17 archive member: {relative}")


def _cleanup_outputs(paths: Stage17OutputPaths) -> None:
    if paths.raw_root.exists():
        shutil.rmtree(paths.raw_root)
    for path in _all_output_paths(paths)[1:]:
        if path.is_file():
            path.unlink()


def _fraction_record(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": float(value),
    }


def _hook_counts(context: CheckpointEvaluationContext) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, len(context.model.hook_dict[name]._forward_hooks))
        for name in (ATTENTION_HEAD_HOOK_NAME, MLP_NEURON_HOOK_NAME)
    )


def _gradients_absent(context: CheckpointEvaluationContext) -> bool:
    return all(parameter.grad is None for parameter in context.model.parameters())


def _reference_restart_rows(run_id: str, family: ReferenceFamily) -> tuple[dict[str, object], ...]:
    cell = family.stage17_cell
    rows = []
    for source in family.restart_rows:
        rows.append(
            {
                "stage17_run_id": run_id,
                "cell_index": cell.cell_index,
                "cell_id": cell.cell_id,
                "fidelity_numerator": cell.fidelity_threshold.numerator,
                "fidelity_denominator": cell.fidelity_threshold.denominator,
                "distinctness_numerator": cell.distinctness_cutoff.numerator,
                "distinctness_denominator": cell.distinctness_cutoff.denominator,
                "search_execution_mode": cell.search_execution_mode,
                "source_stage": 12,
                "source_run_id": family.summary_row["stage12_run_id"],
                **{
                    key: source.get(key, "")
                    for key in RESTART_COLUMNS
                    if key
                    not in {
                        "stage17_run_id",
                        "cell_index",
                        "cell_id",
                        "fidelity_numerator",
                        "fidelity_denominator",
                        "distinctness_numerator",
                        "distinctness_denominator",
                        "search_execution_mode",
                        "source_stage",
                        "source_run_id",
                    }
                },
            }
        )
    return tuple(rows)


def _fresh_restart_rows(
    run_id: str,
    cell: Stage17Cell,
    execution: CheckpointFamilySearchExecution,
) -> tuple[dict[str, object], ...]:
    rows = []
    for outcome in execution.result.restart_outcomes:
        result = outcome.execution.result
        seed = outcome.seed_record
        overlap = outcome.maximum_pairwise_overlap
        rows.append(
            {
                "stage17_run_id": run_id,
                "cell_index": cell.cell_index,
                "cell_id": cell.cell_id,
                "fidelity_numerator": cell.fidelity_threshold.numerator,
                "fidelity_denominator": cell.fidelity_threshold.denominator,
                "distinctness_numerator": cell.distinctness_cutoff.numerator,
                "distinctness_denominator": cell.distinctness_cutoff.denominator,
                "search_execution_mode": cell.search_execution_mode,
                "source_stage": "",
                "source_run_id": "",
                "requested_member_index": outcome.requested_member_index,
                "restart_index": outcome.restart_index,
                "restart_used": True,
                "seed_integer": "" if seed is None else seed.integer_seed,
                "seed_sha256_digest": "" if seed is None else seed.sha256_digest,
                "outcome_status": outcome.outcome_status,
                "accepted_candidate": outcome.accepted_candidate,
                "search_status": result.status,
                "stopping_reason": result.stopping_reason,
                "failure_detail": result.failure_detail or "",
                "retained_component_count": result.final_mask.retained_component_count,
                "primary_fidelity": result.final_metrics.primary_fidelity,
                "maximum_pairwise_overlap_numerator": overlap.numerator,
                "maximum_pairwise_overlap_denominator": overlap.denominator,
                "maximum_pairwise_overlap": float(overlap),
                "exact_evaluation_budget": result.exact_evaluation_budget,
                "exact_evaluations_used": result.exact_evaluations_used,
                "ranking_passes_used": result.ranking_passes_used,
                "candidate_batches_tested": result.candidate_batches_tested,
                "rejected_candidate_count": result.rejected_candidate_count,
                "budget_remaining": result.budget_remaining,
                "budget_exhausted": result.budget_exhausted,
                "locally_single_deletion_minimal": result.locally_single_deletion_minimal,
                "meaningfully_sparse": result.meaningfully_sparse,
            }
        )
    return tuple(rows)


def _normalized_overlaps(circuits: Sequence[NormalizedCircuit]) -> tuple[NormalizedOverlap, ...]:
    overlaps: list[NormalizedOverlap] = []
    for left_index, left in enumerate(circuits):
        for right in circuits[left_index + 1 :]:
            intersection, union = jaccard_counts(left.mask, right.mask)
            overlaps.append(
                NormalizedOverlap(
                    circuit_i=left.circuit_id,
                    circuit_j=right.circuit_id,
                    member_i=left.member_index,
                    member_j=right.member_index,
                    intersection_count=intersection,
                    union_count=union,
                    jaccard=Fraction(intersection, union),
                )
            )
    return tuple(overlaps)


def normalize_reference_search(
    run_id: str,
    family: ReferenceFamily,
    raw_cell_directory: Path,
    source_hashes: Mapping[str, str],
) -> CellSearchResult:
    """Normalize one exact Stage 12 family without rerunning it."""

    cell = family.stage17_cell
    circuits = tuple(
        NormalizedCircuit(
            circuit_id=circuit.circuit_id,
            member_index=circuit.member_index,
            selected_restart_index=int(circuit.circuit_row["selected_restart_index"]),
            mask=circuit.mask,
            mask_sha256=circuit.mask_sha256,
            metrics=dict(circuit.member_record["metrics"]),
            exact_evaluations_used=int(circuit.circuit_row["exact_evaluations_used"]),
            ranking_passes_used=int(circuit.circuit_row["ranking_passes_used"]),
            accepted_removal_count=int(circuit.circuit_row["accepted_removal_count"]),
            rejected_candidate_count=int(circuit.circuit_row["rejected_candidate_count"]),
            candidate_batches_tested=int(circuit.circuit_row["candidate_batches_tested"]),
            locally_single_deletion_minimal=(
                circuit.circuit_row["locally_single_deletion_minimal"] == "True"
            ),
            maximum_prior_overlap=Fraction(
                int(circuit.circuit_row["maximum_pairwise_overlap_numerator"]),
                int(circuit.circuit_row["maximum_pairwise_overlap_denominator"]),
            ),
            prior_overlaps=tuple(
                Fraction(
                    int(record["jaccard"]["numerator"]),
                    int(record["jaccard"]["denominator"]),
                )
                for record in circuit.member_record["pairwise_overlaps_with_prior_members"]
            ),
        )
        for circuit in family.circuits
    )
    overlaps = _normalized_overlaps(circuits)
    expected = {
        (
            row["left_member_label"],
            row["right_member_label"],
        ): Fraction(int(row["jaccard_numerator"]), int(row["jaccard_denominator"]))
        for row in family.pairwise_rows
    }
    if {(row.circuit_i, row.circuit_j): row.jaccard for row in overlaps} != expected:
        raise RuntimeError(f"Stage 12 reference overlap parity failed for {cell.cell_id}.")

    restart_rows = _reference_restart_rows(run_id, family)
    used = tuple(row for row in restart_rows if str(row["restart_used"]).lower() == "true")
    accepted_requested = {
        int(row["requested_member_index"])
        for row in used
        if str(row["accepted_candidate"]).lower() == "true"
    }
    requested = {int(row["requested_member_index"]) for row in used}
    failed = requested - accepted_requested
    if sum(int(row["exact_evaluations_used"]) for row in used) != int(
        family.summary_row["exact_evaluations_used"]
    ):
        raise RuntimeError(f"Stage 12 reference budget sum failed for {cell.cell_id}.")
    for requested_index in requested:
        member_total = sum(
            int(row["exact_evaluations_used"])
            for row in used
            if int(row["requested_member_index"]) == requested_index
        )
        if member_total > PER_REQUESTED_CIRCUIT_BUDGET:
            raise RuntimeError(
                f"Stage 12 reference member budget failed for {cell.cell_id}/C{requested_index}."
            )
    reference_record = {
        "schema_version": 1,
        "record_type": "reference_existing_result",
        "stage17_run_id": run_id,
        "cell": cell.to_record(),
        "source_stage": 12,
        "source_run_id": family.summary_row["stage12_run_id"],
        "source_manifest": cell.search_source_manifest,
        "source_table": cell.search_source_table,
        "source_archive_sha256": source_hashes["stage12_archive"],
        "source_paths": list(family.source_paths),
        "source_summary": family.summary_row,
        "accepted_circuits": [
            {
                "circuit_id": circuit.circuit_id,
                "member_index": circuit.member_index,
                "mask_member_name": circuit.mask_member_name,
                "mask_sha256": circuit.mask_sha256,
            }
            for circuit in family.circuits
        ],
        "scientifically_rerun": False,
    }
    _stable_json(raw_cell_directory / "search_reference.json", reference_record)
    return CellSearchResult(
        cell=cell,
        status=family.summary_row["status"],
        stopping_reason=family.summary_row["stopping_reason"],
        family_size=len(circuits),
        right_censored=family.summary_row["right_censored"] == "True",
        exact_evaluations_used=int(family.summary_row["exact_evaluations_used"]),
        budget_remaining=int(family.summary_row["budget_remaining"]),
        circuits=circuits,
        overlaps=overlaps,
        restart_rows=restart_rows,
        ranking_passes=sum(int(row["ranking_passes_used"]) for row in used),
        restarts_attempted=len(used),
        failed_requested_member_count=len(failed),
        terminal_requested_member_index=max(requested, default=None),
        invalid_output_count=sum("invalid" in str(row["outcome_status"]) for row in used),
        raw_cell_directory=raw_cell_directory,
        search_integrity={
            "source_hashes_verified": True,
            "reference_parity_verified": True,
            "scientifically_rerun": False,
        },
    )


def normalize_fresh_search(
    run_id: str,
    cell: Stage17Cell,
    execution: CheckpointFamilySearchExecution,
    artifacts: Stage12CellArtifacts,
) -> CellSearchResult:
    """Normalize and validate one freshly executed family search."""

    result = execution.result
    circuits: list[NormalizedCircuit] = []
    for member in result.members:
        relative = (
            f"restarts/C{member.member_index:02d}/restart_"
            f"{member.selected_restart_index:02d}/search/final_mask.json"
        )
        mask_path = artifacts.output_directory / relative
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        circuits.append(
            NormalizedCircuit(
                circuit_id=f"C{member.member_index}",
                member_index=member.member_index,
                selected_restart_index=member.selected_restart_index,
                mask=member.mask,
                mask_sha256=file_sha256(mask_path),
                metrics=member.metrics.to_record(),
                exact_evaluations_used=member.search_result.exact_evaluations_used,
                ranking_passes_used=member.search_result.ranking_passes_used,
                accepted_removal_count=len(member.search_result.accepted_removals),
                rejected_candidate_count=member.search_result.rejected_candidate_count,
                candidate_batches_tested=member.search_result.candidate_batches_tested,
                locally_single_deletion_minimal=(
                    member.search_result.locally_single_deletion_minimal
                ),
                maximum_prior_overlap=member.maximum_pairwise_overlap,
                prior_overlaps=member.pairwise_overlaps,
            )
        )
    normalized = tuple(circuits)
    overlaps = _normalized_overlaps(normalized)
    restart_rows = _fresh_restart_rows(run_id, cell, execution)
    requested = {int(row["requested_member_index"]) for row in restart_rows}
    accepted_requested = {
        int(row["requested_member_index"])
        for row in restart_rows
        if bool(row["accepted_candidate"])
    }
    failed = requested - accepted_requested
    if sum(int(row["exact_evaluations_used"]) for row in restart_rows) != (
        result.exact_evaluations_used
    ):
        raise RuntimeError(f"Restart/cell budget identity failed for {cell.cell_id}.")
    for requested_index in requested:
        member_total = sum(
            int(row["exact_evaluations_used"])
            for row in restart_rows
            if int(row["requested_member_index"]) == requested_index
        )
        if member_total > PER_REQUESTED_CIRCUIT_BUDGET:
            raise RuntimeError(f"Requested-member budget exceeded for {cell.cell_id}.")

    if result.family_size != len(normalized):
        raise RuntimeError(f"Family-size identity failed for {cell.cell_id}.")
    if result.exact_evaluations_used > PER_CELL_BUDGET:
        raise RuntimeError(f"Cell budget exceeded for {cell.cell_id}.")
    if result.family_size > FAMILY_TARGET:
        raise RuntimeError(f"Family target exceeded for {cell.cell_id}.")
    if result.family_size == FAMILY_TARGET and not result.right_censored:
        raise RuntimeError(f"Size-ten family is not right-censored for {cell.cell_id}.")
    for circuit in normalized:
        metrics = circuit.metrics
        if int(metrics["retained_component_count"]) > MAXIMUM_RETAINED_COMPONENTS:
            raise RuntimeError(f"Sparsity identity failed for {cell.cell_id}/{circuit.circuit_id}.")
        if (
            int(metrics["prediction_agreement_count"]) * cell.fidelity_threshold.denominator
            < int(metrics["evaluated_example_count"]) * cell.fidelity_threshold.numerator
        ):
            raise RuntimeError(f"Fidelity identity failed for {cell.cell_id}/{circuit.circuit_id}.")
        if circuit.exact_evaluations_used > PER_REQUESTED_CIRCUIT_BUDGET:
            raise RuntimeError(f"Circuit budget exceeded for {cell.cell_id}/{circuit.circuit_id}.")
    if any(row.jaccard > cell.distinctness_cutoff for row in overlaps):
        raise RuntimeError(f"Distinctness identity failed for {cell.cell_id}.")
    if execution.model_state_sha256_before != execution.model_state_sha256_after:
        raise RuntimeError(f"Model state changed during {cell.cell_id}.")
    if execution.hook_counts_before != execution.hook_counts_after:
        raise RuntimeError(f"Hooks changed during {cell.cell_id}.")

    return CellSearchResult(
        cell=cell,
        status=result.status,
        stopping_reason=result.stopping_reason,
        family_size=result.family_size,
        right_censored=result.right_censored,
        exact_evaluations_used=result.exact_evaluations_used,
        budget_remaining=result.budget_remaining,
        circuits=normalized,
        overlaps=overlaps,
        restart_rows=restart_rows,
        ranking_passes=sum(int(row["ranking_passes_used"]) for row in restart_rows),
        restarts_attempted=len(restart_rows),
        failed_requested_member_count=len(failed),
        terminal_requested_member_index=max(requested, default=None),
        invalid_output_count=sum("invalid" in str(row["outcome_status"]) for row in restart_rows),
        raw_cell_directory=artifacts.output_directory,
        search_integrity={
            "model_state_sha256_before": execution.model_state_sha256_before,
            "model_state_sha256_after": execution.model_state_sha256_after,
            "hook_counts_before": execution.hook_counts_before,
            "hook_counts_after": execution.hook_counts_after,
            "pseudo_target_sha256": execution.pseudo_target_sha256,
            "full_model_reference_sha256": execution.full_model_reference_sha256,
            "cell_summary_sha256": artifacts.cell_summary_sha256,
            "hash_inventory_sha256": artifacts.hash_inventory_sha256,
        },
    )


def _empty_transfer(cell: Stage17Cell) -> CellTransferResult:
    return CellTransferResult(
        cell=cell,
        profiles=(),
        profile_rows=(),
        evaluation_rows=(),
        distance_rows=(),
        group_rows=(),
        group_count=None,
        status="not_applicable_empty_family",
    )


def _transfer_rows_from_profiles(
    run_id: str,
    cell: Stage17Cell,
    profiles: Sequence[TransferProfile],
    metrics_by_circuit: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    source_stage: int | str,
    source_run_id: str,
) -> tuple[dict[str, object], ...]:
    rows = []
    for profile in profiles:
        metrics = metrics_by_circuit[profile.circuit_id]
        row: dict[str, object] = {
            "stage17_run_id": run_id,
            "cell_index": cell.cell_index,
            "cell_id": cell.cell_id,
            "fidelity_numerator": cell.fidelity_threshold.numerator,
            "fidelity_denominator": cell.fidelity_threshold.denominator,
            "distinctness_numerator": cell.distinctness_cutoff.numerator,
            "distinctness_denominator": cell.distinctness_cutoff.denominator,
            "transfer_execution_mode": cell.transfer_execution_mode,
            "source_stage": source_stage,
            "source_run_id": source_run_id,
            "circuit_id": profile.circuit_id,
        }
        for subset, fidelity in zip(SUBSET_NAMES, profile.values, strict=True):
            key = subset.lower()
            subset_metrics = metrics[subset]
            row[f"{key}_fidelity"] = fidelity
            row[f"{key}_accuracy"] = subset_metrics["ground_truth_accuracy"]
            row[f"{key}_cross_entropy"] = subset_metrics["cross_entropy"]
            row[f"{key}_kl_divergence"] = subset_metrics["kl_divergence"]
            row[f"{key}_jensen_shannon_divergence"] = subset_metrics["jensen_shannon_divergence"]
        rows.append(row)
    return tuple(rows)


def _distance_and_group_rows(
    run_id: str,
    cell: Stage17Cell,
    profiles: Sequence[TransferProfile],
    *,
    source_stage: int | str,
    source_run_id: str,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...], int | None]:
    grouping = transfer_grouping(profiles, tolerance=PRIMARY_TRANSFER_TOLERANCE)
    distances = pairwise_transfer_distances(profiles)
    by_id = {profile.circuit_id: profile for profile in profiles}
    group_map = {
        circuit_id: f"G{index:02d}"
        for index, group in enumerate(grouping.groups, start=1)
        for circuit_id in group
    }
    distance_rows = []
    for (left, right), maximum in distances.items():
        differences = tuple(
            abs(a - b) for a, b in zip(by_id[left].values, by_id[right].values, strict=True)
        )
        maximum_index = differences.index(maximum)
        distance_rows.append(
            {
                "stage17_run_id": run_id,
                "cell_index": cell.cell_index,
                "cell_id": cell.cell_id,
                "circuit_i": left,
                "circuit_j": right,
                "q1_absolute_difference": differences[0],
                "q2_absolute_difference": differences[1],
                "q3_absolute_difference": differences[2],
                "q4_absolute_difference": differences[3],
                "maximum_absolute_difference": maximum,
                "maximum_distance_subset": SUBSET_NAMES[maximum_index],
                "same_primary_transfer_group": group_map[left] == group_map[right],
            }
        )
    group_rows = []
    for index, group in enumerate(grouping.groups, start=1):
        within = [
            distances[tuple(sorted((left, right)))]
            for left_index, left in enumerate(group)
            for right in group[left_index + 1 :]
        ]
        outside = [
            distances[tuple(sorted((left, right)))]
            for left in group
            for right in by_id
            if right not in group
        ]
        within_maximum = max(within, default=0.0)
        group_rows.append(
            {
                "stage17_run_id": run_id,
                "cell_index": cell.cell_index,
                "cell_id": cell.cell_id,
                "tolerance_numerator": PRIMARY_TRANSFER_TOLERANCE.numerator,
                "tolerance_denominator": PRIMARY_TRANSFER_TOLERANCE.denominator,
                "tolerance": float(PRIMARY_TRANSFER_TOLERANCE),
                "group_count": grouping.group_count,
                "group_id": f"G{index:02d}",
                "ordered_members_json": json.dumps(list(group), separators=(",", ":")),
                "within_group_maximum_distance": within_maximum,
                "between_group_minimum_distance": min(outside, default=""),
                "complete_linkage_valid": within_maximum <= float(PRIMARY_TRANSFER_TOLERANCE),
                "transfer_execution_mode": cell.transfer_execution_mode,
                "source_stage": source_stage,
                "source_run_id": source_run_id,
            }
        )
    return tuple(distance_rows), tuple(group_rows), grouping.group_count


def reference_primary_transfer(
    run_id: str,
    validation: Stage17InputValidation,
    search: CellSearchResult,
) -> CellTransferResult:
    """Normalize the exact primary Stage 16 transfer reference."""

    reference = validation.primary_transfer_reference
    metrics: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    evaluations = []
    for source in reference.evaluation_rows:
        circuit_id = source["circuit_id"]
        subset = source["evaluation_subset"]
        metrics[circuit_id][subset] = {
            "ground_truth_accuracy": float(source["ground_truth_accuracy"]),
            "cross_entropy": float(source["cross_entropy"]),
            "kl_divergence": float(source["kl_divergence"]),
            "jensen_shannon_divergence": float(source["jensen_shannon_divergence"]),
        }
        evaluations.append(
            {
                "stage17_run_id": run_id,
                "cell_id": search.cell.cell_id,
                "circuit_id": circuit_id,
                "evaluation_subset": subset,
                "subset_example_count": int(source["subset_example_count"]),
                "prediction_agreement_count": int(source["prediction_agreement_count"]),
                "fidelity": float(source["fidelity"]),
                "ground_truth_accuracy": float(source["ground_truth_accuracy"]),
                "cross_entropy": float(source["cross_entropy"]),
                "kl_divergence": float(source["kl_divergence"]),
                "jensen_shannon_divergence": float(source["jensen_shannon_divergence"]),
                "transfer_execution_mode": "reference_existing_result",
                "source_stage": 16,
                "source_run_id": validation.configuration.payload["source"]["stage16_run_id"],
            }
        )
    profile_rows = _transfer_rows_from_profiles(
        run_id,
        search.cell,
        reference.profiles,
        metrics,
        source_stage=16,
        source_run_id=validation.configuration.payload["source"]["stage16_run_id"],
    )
    distance_rows, group_rows, group_count = _distance_and_group_rows(
        run_id,
        search.cell,
        reference.profiles,
        source_stage=16,
        source_run_id=validation.configuration.payload["source"]["stage16_run_id"],
    )
    if tuple(profile.values for profile in reference.profiles) != tuple(
        tuple(float(row[f"{subset.lower()}_fidelity"]) for subset in SUBSET_NAMES)
        for row in profile_rows
    ):
        raise RuntimeError("Stage 16 primary transfer-profile parity failed.")
    if group_count != reference.group_count:
        raise RuntimeError("Stage 16 primary transfer-group parity failed.")
    if (
        tuple(tuple(json.loads(row["ordered_members_json"])) for row in group_rows)
        != reference.groups
    ):
        raise RuntimeError("Stage 16 primary transfer-group membership parity failed.")
    return CellTransferResult(
        cell=search.cell,
        profiles=reference.profiles,
        profile_rows=profile_rows,
        evaluation_rows=tuple(evaluations),
        distance_rows=distance_rows,
        group_rows=group_rows,
        group_count=group_count,
        status="reference_verified",
    )


def execute_fresh_transfer(
    run_id: str,
    context: CheckpointEvaluationContext,
    search: CellSearchResult,
    *,
    evaluation_batch_size: int,
    runtime_rows: list[dict[str, object]],
) -> CellTransferResult:
    """Evaluate every accepted circuit on Q1-Q4 and group at 1/20."""

    if not search.circuits:
        return _empty_transfer(search.cell)
    profiles = []
    evaluations = []
    metrics_by_circuit: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for circuit in search.circuits:
        started = time.perf_counter()
        transfer = evaluate_transfer_profile(
            context=context,
            mask=circuit.mask,
            circuit_id=circuit.circuit_id,
            batch_size=evaluation_batch_size,
        )
        runtime_rows.append(
            {
                "stage17_run_id": run_id,
                "record_type": "transfer_circuit",
                "cell_index": search.cell.cell_index,
                "cell_id": search.cell.cell_id,
                "circuit_id": circuit.circuit_id,
                "elapsed_seconds": time.perf_counter() - started,
                "included_in_deterministic_scientific_hashes": False,
            }
        )
        profiles.append(transfer.profile)
        for evaluation in transfer.evaluations:
            metrics = evaluation.metrics
            subset = evaluation.evaluation_subset
            metrics_by_circuit[circuit.circuit_id][subset] = {
                "ground_truth_accuracy": metrics.masked_accuracy,
                "cross_entropy": metrics.masked_cross_entropy,
                "kl_divergence": metrics.mean_kl_divergence,
                "jensen_shannon_divergence": metrics.mean_jensen_shannon_divergence,
            }
            evaluations.append(
                {
                    "stage17_run_id": run_id,
                    "cell_id": search.cell.cell_id,
                    "circuit_id": circuit.circuit_id,
                    "evaluation_subset": subset,
                    "subset_example_count": metrics.evaluated_example_count,
                    "prediction_agreement_count": metrics.prediction_agreement_count,
                    "fidelity": metrics.primary_fidelity,
                    "ground_truth_accuracy": metrics.masked_accuracy,
                    "cross_entropy": metrics.masked_cross_entropy,
                    "kl_divergence": metrics.mean_kl_divergence,
                    "jensen_shannon_divergence": metrics.mean_jensen_shannon_divergence,
                    "transfer_execution_mode": "fresh_execution",
                    "source_stage": "",
                    "source_run_id": "",
                }
            )
    normalized_profiles = tuple(profiles)
    profile_rows = _transfer_rows_from_profiles(
        run_id,
        search.cell,
        normalized_profiles,
        metrics_by_circuit,
        source_stage="",
        source_run_id="",
    )
    distance_rows, group_rows, group_count = _distance_and_group_rows(
        run_id,
        search.cell,
        normalized_profiles,
        source_stage="",
        source_run_id="",
    )
    if len(evaluations) != len(search.circuits) * 4:
        raise RuntimeError(f"Transfer identity failed for {search.cell.cell_id}.")
    return CellTransferResult(
        cell=search.cell,
        profiles=normalized_profiles,
        profile_rows=profile_rows,
        evaluation_rows=tuple(evaluations),
        distance_rows=distance_rows,
        group_rows=group_rows,
        group_count=group_count,
        status="complete",
    )


def _failure_category(search: CellSearchResult) -> str:
    if search.right_censored:
        return "family_target_reached"
    text = f"{search.status} {search.stopping_reason}".lower()
    for category in (
        "fidelity_failure",
        "sparsity_failure",
        "distinctness_failure",
        "budget_exhaustion",
        "invalid_masking_output",
        "search_failure",
    ):
        if category in text:
            return category
    if "no_feasible" in text:
        return "no_feasible_candidate_discovered_within_tested_search"
    return "search_failure"


def _base_cell_fields(cell: Stage17Cell) -> dict[str, object]:
    return {
        "cell_index": cell.cell_index,
        "cell_id": cell.cell_id,
        "fidelity_numerator": cell.fidelity_threshold.numerator,
        "fidelity_denominator": cell.fidelity_threshold.denominator,
        "displayed_fidelity": cell.fidelity_display,
        "distinctness_numerator": cell.distinctness_cutoff.numerator,
        "distinctness_denominator": cell.distinctness_cutoff.denominator,
        "displayed_jaccard_cutoff": cell.distinctness_display,
    }


def _registry_rows(
    run_id: str,
    searches: Sequence[CellSearchResult],
    transfers: Mapping[str, CellTransferResult],
) -> list[dict[str, object]]:
    rows = []
    for search in searches:
        cell = search.cell
        rows.append(
            {
                "stage17_run_id": run_id,
                **_base_cell_fields(cell),
                "model_seed": cell.model_seed,
                "checkpoint_step": cell.checkpoint_step,
                "primary_cell": cell.is_primary,
                "search_execution_mode": cell.search_execution_mode,
                "search_source_stage": cell.search_source_stage,
                "search_source_run_id": cell.search_source_run_id,
                "search_source_manifest": cell.search_source_manifest,
                "search_source_table": cell.search_source_table,
                "transfer_execution_mode": cell.transfer_execution_mode,
                "transfer_source_stage": cell.transfer_source_stage,
                "transfer_source_run_id": cell.transfer_source_run_id,
                "transfer_source_manifest": cell.transfer_source_manifest,
                "transfer_source_table": cell.transfer_source_table,
                "expected_search_budget": cell.expected_search_budget,
                "output_status": search.status,
                "transfer_grouping_status": transfers[cell.cell_id].status,
            }
        )
    return rows


def _scientific_rows(
    run_id: str,
    searches: Sequence[CellSearchResult],
    transfers: Mapping[str, CellTransferResult],
) -> dict[str, list[dict[str, object]]]:
    families: list[dict[str, object]] = []
    circuits: list[dict[str, object]] = []
    overlaps: list[dict[str, object]] = []
    restarts: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    sizes: list[dict[str, object]] = []
    frontier: list[dict[str, object]] = []
    figure_source: list[dict[str, object]] = []

    for search in searches:
        cell = search.cell
        transfer = transfers[cell.cell_id]
        base = _base_cell_fields(cell)
        source_stage = cell.search_source_stage or ""
        source_run_id = cell.search_source_run_id or ""
        overlap_summary = structural_overlap_summary(
            tuple(row.jaccard for row in search.overlaps),
            family_size=search.family_size,
            cutoff=cell.distinctness_cutoff,
        )
        families.append(
            {
                "stage17_run_id": run_id,
                **base,
                "model_seed": cell.model_seed,
                "checkpoint_step": cell.checkpoint_step,
                "primary_cell": cell.is_primary,
                "search_execution_mode": cell.search_execution_mode,
                "source_stage": source_stage,
                "source_run_id": source_run_id,
                "family_size": search.family_size,
                "family_target": FAMILY_TARGET,
                "right_censored": search.right_censored,
                "right_censoring_status": (
                    "right_censored_at_family_target"
                    if search.right_censored
                    else "not_right_censored"
                ),
                "status": search.status,
                "stopping_reason": search.stopping_reason,
                "exact_evaluations_used": search.exact_evaluations_used,
                "ranking_passes": search.ranking_passes,
                "restarts_attempted": search.restarts_attempted,
                "accepted_circuit_count": len(search.circuits),
                "failed_requested_member_count": search.failed_requested_member_count,
                "budget_remaining": search.budget_remaining,
                "transfer_group_count": transfer.group_count,
                "transfer_group_status": transfer.status,
                **overlap_summary,
            }
        )
        size = circuit_size_summary(
            tuple(circuit.mask.retained_component_count for circuit in search.circuits)
        )
        sizes.append({"stage17_run_id": run_id, **base, **size})
        failures.append(
            {
                "stage17_run_id": run_id,
                "record_type": "cell",
                **base,
                "family_size": search.family_size,
                "c1_qualified": search.family_size > 0,
                "failed_requested_member_count": search.failed_requested_member_count,
                "terminal_requested_member_index": search.terminal_requested_member_index,
                "stopping_reason": search.stopping_reason,
                "failure_category": _failure_category(search),
                "budget_exhausted": _failure_category(search) == "budget_exhaustion",
                "per_cell_evaluations_used": search.exact_evaluations_used,
                "unused_budget": search.budget_remaining,
                "ranking_passes": search.ranking_passes,
                "restart_count": search.restarts_attempted,
                "invalid_output_count": search.invalid_output_count,
                "search_execution_mode": cell.search_execution_mode,
            }
        )
        figure_source.append(
            {
                "stage17_run_id": run_id,
                **base,
                "family_size": search.family_size,
                "right_censored": search.right_censored,
                "reference_cell": cell.search_execution_mode == "reference_existing_result",
                "primary_cell": cell.is_primary,
                "zero_family": search.family_size == 0,
            }
        )

        for circuit in search.circuits:
            metrics = circuit.metrics
            agreement = int(metrics["prediction_agreement_count"])
            evaluated = int(metrics["evaluated_example_count"])
            circuits.append(
                {
                    "stage17_run_id": run_id,
                    **base,
                    "search_execution_mode": cell.search_execution_mode,
                    "source_stage": source_stage,
                    "source_run_id": source_run_id,
                    "circuit_id": circuit.circuit_id,
                    "family_member_index": circuit.member_index,
                    "retained_heads": circuit.mask.retained_attention_head_count,
                    "retained_neurons": circuit.mask.retained_mlp_neuron_count,
                    "retained_components": circuit.mask.retained_component_count,
                    "retained_proportion": circuit.mask.retained_component_proportion,
                    "fidelity": float(metrics["primary_fidelity"]),
                    "agreement_count": agreement,
                    "evaluated_example_count": evaluated,
                    "threshold_pass": (
                        agreement * cell.fidelity_threshold.denominator
                        >= evaluated * cell.fidelity_threshold.numerator
                    ),
                    "ground_truth_accuracy": float(metrics["masked_accuracy"]),
                    "cross_entropy": float(metrics["masked_cross_entropy"]),
                    "kl_divergence": float(metrics["mean_kl_divergence"]),
                    "jensen_shannon_divergence": float(metrics["mean_jensen_shannon_divergence"]),
                    "local_single_deletion_minimality": (circuit.locally_single_deletion_minimal),
                    "maximum_prior_overlap_numerator": (circuit.maximum_prior_overlap.numerator),
                    "maximum_prior_overlap_denominator": (
                        circuit.maximum_prior_overlap.denominator
                    ),
                    "maximum_prior_overlap": float(circuit.maximum_prior_overlap),
                    "mean_prior_overlap": (
                        mean(float(value) for value in circuit.prior_overlaps)
                        if circuit.prior_overlaps
                        else None
                    ),
                    "mask_sha256": circuit.mask_sha256,
                    "exact_evaluation_count": circuit.exact_evaluations_used,
                    "ranking_passes": circuit.ranking_passes_used,
                    "selected_restart": circuit.selected_restart_index,
                }
            )
        for overlap in search.overlaps:
            overlaps.append(
                {
                    "stage17_run_id": run_id,
                    **base,
                    "search_execution_mode": cell.search_execution_mode,
                    "source_stage": source_stage,
                    "source_run_id": source_run_id,
                    "circuit_i": overlap.circuit_i,
                    "circuit_j": overlap.circuit_j,
                    "member_i": overlap.member_i,
                    "member_j": overlap.member_j,
                    "intersection_count": overlap.intersection_count,
                    "union_count": overlap.union_count,
                    "jaccard_numerator": overlap.jaccard.numerator,
                    "jaccard_denominator": overlap.jaccard.denominator,
                    "jaccard_overlap": float(overlap.jaccard),
                    "structural_distance": 1.0 - float(overlap.jaccard),
                    "passes_active_cutoff": overlap.jaccard <= cell.distinctness_cutoff,
                }
            )
        restarts.extend(dict(row) for row in search.restart_rows)

        by_requested: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in search.restart_rows:
            if str(row["restart_used"]).lower() == "true":
                by_requested[int(row["requested_member_index"])].append(dict(row))
        circuit_by_index = {circuit.member_index: circuit for circuit in search.circuits}
        cumulative = 0
        for requested_index in sorted(by_requested):
            member_rows = by_requested[requested_index]
            member_evaluations = sum(int(row["exact_evaluations_used"]) for row in member_rows)
            member_rankings = sum(int(row["ranking_passes_used"]) for row in member_rows)
            cumulative += member_evaluations
            circuit = circuit_by_index.get(requested_index)
            representative = next(
                (row for row in member_rows if str(row["accepted_candidate"]).lower() == "true"),
                member_rows[-1],
            )
            frontier.append(
                {
                    "stage17_run_id": run_id,
                    "cell_index": cell.cell_index,
                    "cell_id": cell.cell_id,
                    "requested_member_index": requested_index,
                    "member_label": f"C{requested_index}",
                    "accepted_circuit": circuit is not None,
                    "representative_restart_index": representative["restart_index"],
                    "primary_fidelity": representative["primary_fidelity"],
                    "retained_component_count": representative["retained_component_count"],
                    "retained_component_proportion": (
                        None if circuit is None else circuit.mask.retained_component_proportion
                    ),
                    "maximum_prior_overlap_numerator": representative[
                        "maximum_pairwise_overlap_numerator"
                    ],
                    "maximum_prior_overlap_denominator": representative[
                        "maximum_pairwise_overlap_denominator"
                    ],
                    "maximum_prior_overlap": representative["maximum_pairwise_overlap"],
                    "mean_prior_overlap": (
                        None
                        if circuit is None or not circuit.prior_overlaps
                        else mean(float(value) for value in circuit.prior_overlaps)
                    ),
                    "cumulative_exact_evaluations": cumulative,
                    "member_exact_evaluations": member_evaluations,
                    "member_ranking_passes": member_rankings,
                    "restart_count": len(member_rows),
                    "status": (
                        "accepted" if circuit is not None else representative["outcome_status"]
                    ),
                    "failure_reason": "" if circuit is not None else search.stopping_reason,
                    "family_right_censored": search.right_censored,
                    "search_execution_mode": cell.search_execution_mode,
                }
            )

    return {
        "family_summary_table": families,
        "circuits_table": circuits,
        "pairwise_overlap_table": overlaps,
        "restarts_table": restarts,
        "search_failures_table": failures,
        "circuit_size_summary_table": sizes,
        "frontier_table": frontier,
        "family_size_heatmap_source_table": figure_source,
        "family_size_curves_source_table": figure_source,
        "family_size_distinctness_source_table": figure_source,
    }


def _search_failure_summary_rows(
    run_id: str, failure_rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dimensions = {
        "fidelity_threshold": lambda row: str(row["displayed_fidelity"]),
        "distinctness_cutoff": lambda row: str(row["displayed_jaccard_cutoff"]),
        "failure_reason": lambda row: str(row["failure_category"]),
    }
    for dimension, key_function in dimensions.items():
        grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in failure_rows:
            grouped[key_function(row)].append(row)
        for value in sorted(grouped):
            group = grouped[value]
            rows.append(
                {
                    "stage17_run_id": run_id,
                    "summary_dimension": dimension,
                    "summary_value": value,
                    "cell_count": len(group),
                    "zero_family_cell_count": sum(int(row["family_size"]) == 0 for row in group),
                    "c1_qualified_cell_count": sum(bool(row["c1_qualified"]) for row in group),
                    "budget_exhausted_cell_count": sum(
                        bool(row["budget_exhausted"]) for row in group
                    ),
                    "total_recovered_circuits": sum(int(row["family_size"]) for row in group),
                }
            )
    return rows


def _matrix_rows(run_id: str, searches: Sequence[CellSearchResult]) -> list[dict[str, object]]:
    by_key = {search.cell.key: search for search in searches}
    fidelities = tuple(dict.fromkeys(search.cell.fidelity_threshold for search in searches))
    cutoffs = tuple(dict.fromkeys(search.cell.distinctness_cutoff for search in searches))
    cutoff_columns = {
        cutoff: "family_size_cutoff_"
        + by_key[(fidelities[0], cutoff)].cell.distinctness_display.replace(".", "_")
        for cutoff in cutoffs
    }
    return [
        {
            "stage17_run_id": run_id,
            "fidelity_numerator": fidelity.numerator,
            "fidelity_denominator": fidelity.denominator,
            "displayed_fidelity": by_key[(fidelity, cutoffs[0])].cell.fidelity_display,
            **{
                cutoff_columns[cutoff]: by_key[(fidelity, cutoff)].family_size for cutoff in cutoffs
            },
        }
        for fidelity in fidelities
    ]


def robustness_classification(searches: Sequence[CellSearchResult]) -> str:
    """Apply the prospectively frozen qualitative classification rule."""

    by_key = {search.cell.key: search.family_size for search in searches}
    expected_keys = {
        (fidelity, cutoff)
        for fidelity in (
            Fraction(4, 5),
            Fraction(17, 20),
            Fraction(9, 10),
            Fraction(19, 20),
            Fraction(39, 40),
            Fraction(99, 100),
        )
        for cutoff in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
    }
    if set(by_key) != expected_keys:
        return "unresolved"
    if by_key.get(PRIMARY_CELL_KEY, 0) < 2:
        return "unresolved"
    if all(value >= 2 for value in by_key.values()):
        return "robust across the frozen sensitivity grid"
    neighbourhood = {
        (fidelity, cutoff)
        for fidelity in (Fraction(39, 40), Fraction(99, 100))
        for cutoff in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
    }
    if all(by_key[key] >= 2 for key in neighbourhood):
        return "robust only within a limited neighbourhood of the primary cell"
    immediate = {
        (Fraction(39, 40), Fraction(1, 2)),
        (Fraction(99, 100), Fraction(1, 4)),
        (Fraction(99, 100), Fraction(3, 4)),
    }
    if any(by_key[key] < 2 for key in immediate):
        return "threshold-sensitive or fragile"
    return "mixed"


def _write_figures(
    paths: Stage17OutputPaths,
    searches: Sequence[CellSearchResult],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ordered = tuple(searches)
    fidelities = tuple(dict.fromkeys(search.cell.fidelity_display for search in ordered))
    cutoffs = tuple(dict.fromkeys(search.cell.distinctness_display for search in ordered))
    matrix = np.array(
        [
            [
                next(
                    search.family_size
                    for search in ordered
                    if search.cell.fidelity_display == fidelity
                    and search.cell.distinctness_display == cutoff
                )
                for cutoff in cutoffs
            ]
            for fidelity in fidelities
        ],
        dtype=float,
    )
    for path in paths.figures.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=FAMILY_TARGET, interpolation="none")
    axis.set_xticks(range(len(cutoffs)), labels=cutoffs)
    axis.set_yticks(range(len(fidelities)), labels=fidelities)
    axis.set_xlabel("Maximum pairwise Jaccard overlap")
    axis.set_ylabel("Behavioural-fidelity threshold")
    axis.set_title("Recovered structural family size $F(\\tau_f, \\tau_d)$")
    for row, fidelity in enumerate(fidelities):
        for column, cutoff in enumerate(cutoffs):
            search = next(
                item
                for item in ordered
                if item.cell.fidelity_display == fidelity
                and item.cell.distinctness_display == cutoff
            )
            markers = ""
            if search.cell.search_execution_mode == "reference_existing_result":
                markers += " R"
            if search.right_censored:
                markers += " †"
            label = f"{search.family_size}{markers}"
            axis.text(column, row, label, ha="center", va="center", color="white")
            if search.cell.is_primary:
                axis.add_patch(
                    plt.Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#ffcc33",
                        linewidth=3,
                    )
                )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Recovered family size")
    metadata = {"Creator": "circuit_families Stage 17", "CreationDate": None}
    fig.savefig(paths.figures["heatmap_png"], dpi=300, metadata={"Software": "circuit_families"})
    fig.savefig(paths.figures["heatmap_pdf"], metadata=metadata)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    x_fidelity = [float(value) for value in fidelities]
    for cutoff in cutoffs:
        values = [
            next(
                search.family_size
                for search in ordered
                if search.cell.fidelity_display == fidelity
                and search.cell.distinctness_display == cutoff
            )
            for fidelity in fidelities
        ]
        axes[0].plot(x_fidelity, values, marker="o", label=f"cutoff {cutoff}")
    axes[0].set_xlabel("Behavioural-fidelity threshold")
    axes[0].set_ylabel("Recovered family size")
    axes[0].set_xticks(x_fidelity, labels=fidelities, rotation=30)
    axes[0].set_ylim(-0.5, FAMILY_TARGET + 0.5)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    x_cutoff = [float(value) for value in cutoffs]
    for fidelity in fidelities:
        values = [
            next(
                search.family_size
                for search in ordered
                if search.cell.fidelity_display == fidelity
                and search.cell.distinctness_display == cutoff
            )
            for cutoff in cutoffs
        ]
        axes[1].plot(x_cutoff, values, marker="o", label=f"fidelity {fidelity}")
    axes[1].set_xlabel("Maximum pairwise Jaccard overlap")
    axes[1].set_ylabel("Recovered family size")
    axes[1].set_xticks(x_cutoff, labels=cutoffs)
    axes[1].set_ylim(-0.5, FAMILY_TARGET + 0.5)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.savefig(paths.figures["curves_png"], dpi=300, metadata={"Software": "circuit_families"})
    fig.savefig(paths.figures["curves_pdf"], metadata=metadata)
    plt.close(fig)


def _note_and_caption(
    classification: str,
    searches: Sequence[CellSearchResult],
) -> tuple[str, str]:
    primary = next(search for search in searches if search.cell.is_primary)
    size_matrix = "\n".join(
        "| "
        + search.cell.fidelity_display
        + " | "
        + " | ".join(
            str(item.family_size)
            for item in searches
            if item.cell.fidelity_threshold == search.cell.fidelity_threshold
        )
        + " |"
        for search in searches[::3]
    )
    note = (
        "# Stage 17 two-dimensional sensitivity analysis\n\n"
        f"Scientific robustness classification: **{classification}**.\n\n"
        "The frozen primary cell remains fidelity 0.990 and maximum Jaccard overlap 0.50. "
        f"It recovered {primary.family_size} circuits under a fixed 50,000-evaluation cell "
        "budget. Reference cells reproduce Stage 12 rather than counting as fresh searches; "
        "the primary transfer profile reproduces Stage 16.\n\n"
        "| Fidelity | cutoff 0.25 | cutoff 0.50 | cutoff 0.75 |\n"
        "|---:|---:|---:|---:|\n"
        f"{size_matrix}\n\n"
        "All values are fixed-budget recovered family sizes. A value of ten is right-censored "
        "at the family target, not a complete enumeration. Zero-family cells are scientific "
        "outcomes. Lower-fidelity expansion is weaker evidence than persistence at 0.990, and "
        "growth at cutoff 0.75 permits more structural reuse rather than proving more distinct "
        "mechanisms. Structural and transfer-distinct groups do not establish different "
        "algorithms.\n\n"
        "Stage 15 remains unavailable with null endpoints. No checkpoint-grid decision was made, "
        "and Stage 18 was not started.\n"
    )
    caption = (
        "Stage 17 frozen fidelity-by-structural-distinctness sensitivity analysis. "
        "Cell values and curves show fixed-budget recovered structural family size; R marks "
        "exact Stage 12 search references, the outlined cell is the prespecified primary "
        "0.990 x 0.50 result, and a dagger marks right-censoring at the family target of ten. "
        "No interpolation is used and zero-family cells remain visible."
    )
    return note, caption + "\n"


def _write_raw_transfer(root: Path, transfer: CellTransferResult) -> tuple[Path, ...]:
    transfer_root = root / "transfer"
    paths = (
        _write_jsonl(transfer_root / "evaluations.jsonl", transfer.evaluation_rows),
        _write_jsonl(transfer_root / "profiles.jsonl", transfer.profile_rows),
        _write_jsonl(transfer_root / "distances.jsonl", transfer.distance_rows),
        _write_jsonl(transfer_root / "groups.jsonl", transfer.group_rows),
    )
    _stable_json(
        transfer_root / "summary.json",
        {
            "cell_id": transfer.cell.cell_id,
            "status": transfer.status,
            "profile_count": len(transfer.profiles),
            "group_count": transfer.group_count,
            "tolerance": _fraction_record(PRIMARY_TRANSFER_TOLERANCE),
            "execution_mode": transfer.cell.transfer_execution_mode,
        },
    )
    return (*paths, transfer_root / "summary.json")


def _write_tables(
    paths: Stage17OutputPaths,
    run_id: str,
    searches: Sequence[CellSearchResult],
    transfers: Mapping[str, CellTransferResult],
    runtime_rows: Sequence[Mapping[str, object]],
) -> tuple[Path, ...]:
    rows = _scientific_rows(run_id, searches, transfers)
    rows["cells_table"] = _registry_rows(run_id, searches, transfers)
    rows["family_size_matrix_table"] = _matrix_rows(run_id, searches)
    rows["search_failure_summary_table"] = _search_failure_summary_rows(
        run_id, rows["search_failures_table"]
    )
    rows["transfer_profiles_table"] = [
        dict(row) for transfer in transfers.values() for row in transfer.profile_rows
    ]
    rows["transfer_distances_table"] = [
        dict(row) for transfer in transfers.values() for row in transfer.distance_rows
    ]
    rows["transfer_groups_table"] = [
        dict(row) for transfer in transfers.values() for row in transfer.group_rows
    ]
    fieldnames: dict[str, Sequence[str]] = {
        "cells_table": REGISTRY_COLUMNS,
        "family_summary_table": FAMILY_COLUMNS,
        "circuits_table": CIRCUIT_COLUMNS,
        "pairwise_overlap_table": OVERLAP_COLUMNS,
        "restarts_table": RESTART_COLUMNS,
        "search_failures_table": FAILURE_COLUMNS,
        "search_failure_summary_table": (
            "stage17_run_id",
            "summary_dimension",
            "summary_value",
            "cell_count",
            "zero_family_cell_count",
            "c1_qualified_cell_count",
            "budget_exhausted_cell_count",
            "total_recovered_circuits",
        ),
        "circuit_size_summary_table": SIZE_COLUMNS,
        "transfer_profiles_table": TRANSFER_PROFILE_COLUMNS,
        "transfer_distances_table": TRANSFER_DISTANCE_COLUMNS,
        "transfer_groups_table": TRANSFER_GROUP_COLUMNS,
        "frontier_table": FRONTIER_COLUMNS,
        "family_size_heatmap_source_table": FIGURE_SOURCE_COLUMNS,
        "family_size_curves_source_table": FIGURE_SOURCE_COLUMNS,
        "family_size_distinctness_source_table": FIGURE_SOURCE_COLUMNS,
        "family_size_matrix_table": (
            "stage17_run_id",
            "fidelity_numerator",
            "fidelity_denominator",
            "displayed_fidelity",
            "family_size_cutoff_0_25",
            "family_size_cutoff_0_50",
            "family_size_cutoff_0_75",
        ),
    }
    written = []
    for key, path in paths.tables.items():
        if key == "runtime_table":
            write_csv_records(path, fieldnames=RUNTIME_COLUMNS, rows=runtime_rows)
        else:
            write_csv_records(path, fieldnames=fieldnames[key], rows=rows[key])
        written.append(path)
    return tuple(written)


def execute_stage17(
    repository_root: str | Path,
    *,
    stage12_manifest: str | Path | None = None,
    stage16_manifest: str | Path | None = None,
    checkpoint_step: int = CHECKPOINT_STEP,
    device: str = "cpu",
    output_root: str | Path | None = None,
    expected_implementation_commit: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Stage17ExecutionResult:
    """Execute the complete definitive Stage 17 workload."""

    if device != "cpu":
        raise ValueError("Definitive Stage 17 execution requires CPU; MPS is prohibited.")
    validation = validate_stage17_inputs(
        repository_root,
        stage12_manifest=stage12_manifest,
        stage16_manifest=stage16_manifest,
        checkpoint_step=checkpoint_step,
        require_clean=True,
    )
    if (
        expected_implementation_commit is not None
        and validation.implementation_commit != expected_implementation_commit
    ):
        raise ValueError(
            "Current commit does not match the expected Stage 17 implementation commit."
        )
    run_id = deterministic_stage17_run_id(
        validation.configuration.sha256, validation.implementation_commit
    )
    paths = build_output_paths(validation, output_root=output_root)
    validate_absent_outputs(paths)
    source = validation.configuration.payload["source"]
    search_config = validation.configuration.payload["search"]
    context = load_checkpoint_evaluation_context(
        repository_root=validation.repository,
        run_id=source["training_run_id"],
        checkpoint_manifest_path=source["checkpoint_manifest"],
        checkpoint_step=CHECKPOINT_STEP,
        device_override=device,
    )
    state_before = canonical_state_hash(context.model.state_dict())
    hooks_before = _hook_counts(context)
    gradients_before = _gradients_absent(context)
    if state_before != source["model_state_sha256"] or not gradients_before:
        raise RuntimeError("Stage 17 pre-execution model integrity failed.")

    searches: list[CellSearchResult] = []
    transfers: dict[str, CellTransferResult] = {}
    runtime_rows: list[dict[str, object]] = []
    reference_by_key = {family.stage17_cell.key: family for family in validation.reference_families}
    try:
        paths.raw_root.mkdir(parents=True, exist_ok=False)
        _stable_json(
            paths.raw_root / "registry.json",
            {
                "schema_version": 1,
                "stage17_run_id": run_id,
                "cells": [cell.to_record() for cell in validation.registry],
            },
        )
        _stable_json(paths.raw_root / "source_hashes.json", validation.source_hashes)

        for cell in validation.registry:
            raw_cell = paths.raw_root / "cells" / cell.cell_id
            if progress_callback is not None:
                progress_callback(
                    f"search [{cell.cell_index:02d}/18] {cell.cell_id} "
                    f"mode={cell.search_execution_mode}"
                )
            if cell.search_execution_mode == "reference_existing_result":
                raw_cell.mkdir(parents=True, exist_ok=False)
                search = normalize_reference_search(
                    run_id, reference_by_key[cell.key], raw_cell, validation.source_hashes
                )
                elapsed = None
            else:
                started = time.perf_counter()
                execution = run_checkpoint_family_search(
                    context,
                    fidelity_threshold=float(cell.fidelity_threshold),
                    distinctness_cutoff=cell.distinctness_cutoff,
                    model_seed=MODEL_SEED,
                    checkpoint_index=CHECKPOINT_INDEX,
                    ranking_batch_size=int(search_config["ranking_batch_size"]),
                    evaluation_batch_size=int(search_config["evaluation_batch_size"]),
                    family_target=FAMILY_TARGET,
                    max_restarts_per_alternative=int(
                        search_config["maximum_restarts_per_alternative"]
                    ),
                    per_requested_circuit_budget=PER_REQUESTED_CIRCUIT_BUDGET,
                    per_cell_budget=PER_CELL_BUDGET,
                    reuse_coefficient=float(search_config["reuse_coefficient"]),
                    tie_tolerance=float(search_config["tie_tolerance"]),
                )
                elapsed = time.perf_counter() - started
                artifacts = write_stage12_cell_artifacts(
                    raw_cell / "search",
                    execution,
                    cell_metadata={
                        "stage17_run_id": run_id,
                        "cell_index": cell.cell_index,
                        "cell_id": cell.cell_id,
                        "model_seed": MODEL_SEED,
                        "checkpoint_step": CHECKPOINT_STEP,
                        "fidelity_threshold": _fraction_record(cell.fidelity_threshold),
                        "distinctness_cutoff": _fraction_record(cell.distinctness_cutoff),
                        "execution_mode": cell.search_execution_mode,
                        "implementation_commit": validation.implementation_commit,
                        "independent_cell_budget": True,
                        "initial_family_size": 0,
                    },
                )
                search = normalize_fresh_search(run_id, cell, execution, artifacts)
            searches.append(search)
            runtime_rows.append(
                {
                    "stage17_run_id": run_id,
                    "record_type": "search_cell",
                    "cell_index": cell.cell_index,
                    "cell_id": cell.cell_id,
                    "circuit_id": "",
                    "elapsed_seconds": elapsed,
                    "included_in_deterministic_scientific_hashes": False,
                }
            )

        for search in searches:
            cell = search.cell
            if progress_callback is not None:
                progress_callback(
                    f"transfer [{cell.cell_index:02d}/18] {cell.cell_id} "
                    f"circuits={search.family_size} mode={cell.transfer_execution_mode}"
                )
            if search.family_size == 0:
                transfer = _empty_transfer(cell)
            elif cell.transfer_execution_mode == "reference_existing_result":
                transfer = reference_primary_transfer(run_id, validation, search)
            else:
                transfer = execute_fresh_transfer(
                    run_id,
                    context,
                    search,
                    evaluation_batch_size=int(search_config["evaluation_batch_size"]),
                    runtime_rows=runtime_rows,
                )
            transfers[cell.cell_id] = transfer
            _write_raw_transfer(paths.raw_root / "cells" / cell.cell_id, transfer)

        if len(searches) != 18 or len(transfers) != 18:
            raise RuntimeError("Stage 17 execution did not produce all 18 cells.")
        for search in searches:
            if search.family_size != len(search.circuits):
                raise RuntimeError(f"Family-size identity failed for {search.cell.cell_id}.")
            transfer = transfers[search.cell.cell_id]
            if search.family_size == 0 and transfer.group_count is not None:
                raise RuntimeError(
                    f"Empty-family transfer identity failed for {search.cell.cell_id}."
                )
            if search.family_size > 0 and len(transfer.profiles) != search.family_size:
                raise RuntimeError(f"Transfer-profile count failed for {search.cell.cell_id}.")
            if search.family_size == 1 and transfer.group_count != 1:
                raise RuntimeError(f"Singleton group identity failed for {search.cell.cell_id}.")

        _write_tables(paths, run_id, searches, transfers, runtime_rows)
        classification = robustness_classification(searches)
        note, caption = _note_and_caption(classification, searches)
        paths.note.parent.mkdir(parents=True, exist_ok=True)
        paths.note.write_text(note, encoding="utf-8")
        paths.caption.parent.mkdir(parents=True, exist_ok=True)
        paths.caption.write_text(caption, encoding="utf-8")
        _write_figures(paths, searches)

        state_after = canonical_state_hash(context.model.state_dict())
        hooks_after = _hook_counts(context)
        gradients_after = _gradients_absent(context)
        if state_after != state_before:
            raise RuntimeError("Stage 17 changed model state.")
        if hooks_after != hooks_before:
            raise RuntimeError("Stage 17 did not restore hooks.")
        if not gradients_after:
            raise RuntimeError("Stage 17 left parameter gradients populated.")

        run_record = {
            "schema_version": 1,
            "stage17_run_id": run_id,
            "implementation_commit": validation.implementation_commit,
            "cell_count": len(searches),
            "fresh_search_cell_count": sum(
                search.cell.search_execution_mode == "fresh_execution" for search in searches
            ),
            "reference_search_cell_count": sum(
                search.cell.search_execution_mode == "reference_existing_result"
                for search in searches
            ),
            "fresh_transfer_workload_count": sum(
                search.family_size > 0 and search.cell.transfer_execution_mode == "fresh_execution"
                for search in searches
            ),
            "transfer_reference_count": 1,
            "classification": classification,
            "family_sizes": {search.cell.cell_id: search.family_size for search in searches},
            "transfer_group_counts": {
                cell_id: transfer.group_count for cell_id, transfer in transfers.items()
            },
            "model_integrity": {
                "model_state_sha256_before": state_before,
                "model_state_sha256_after": state_after,
                "model_state_unchanged": state_before == state_after,
                "parameter_gradients_absent_before": gradients_before,
                "parameter_gradients_absent_after": gradients_after,
                "hook_counts_before": hooks_before,
                "hook_counts_after": hooks_after,
                "hooks_restored_to_baseline": hooks_before == hooks_after,
            },
            "runtime_included_in_deterministic_scientific_hashes": False,
            "checkpoint_grid_decision_made": False,
            "stage18_started": False,
        }
        _stable_json(paths.raw_root / "stage17_run_record.json", run_record)

        deterministic_tables = tuple(
            path for key, path in paths.tables.items() if key != "runtime_table"
        )
        raw_files = tuple(path for path in paths.raw_root.rglob("*") if path.is_file())
        scientific_members = (
            *raw_files,
            *deterministic_tables,
            *paths.figures.values(),
            paths.note,
            paths.caption,
        )
        archive_root = paths.archive.parents[2]
        validate_archive_member_contract(archive_root, scientific_members)
        write_deterministic_archive(paths.archive, root=archive_root, members=scientific_members)

        output_hashes = {
            key: {
                "path": str(path.relative_to(paths.archive.parents[2])),
                "sha256": file_sha256(path),
            }
            for key, path in {
                **paths.tables,
                **paths.figures,
                "note": paths.note,
                "caption": paths.caption,
                "archive": paths.archive,
            }.items()
        }
        manifest = {
            "schema_version": 1,
            "experiment_stage": 17,
            "stage17_run_id": run_id,
            "creation_timestamp_utc": datetime.now(UTC).isoformat(),
            "implementation_commit": validation.implementation_commit,
            "repository_clean_at_start": validation.repository_clean,
            "model_seed": MODEL_SEED,
            "source_training_run_id": source["training_run_id"],
            "checkpoint": {
                "step": CHECKPOINT_STEP,
                "path": source["checkpoint_path"],
                "sha256": source["checkpoint_sha256"],
                "model_state_sha256": source["model_state_sha256"],
            },
            "source_hashes": validation.source_hashes,
            "fidelity_grid": [
                _fraction_record(search.cell.fidelity_threshold) for search in searches[::3]
            ],
            "distinctness_grid": [
                _fraction_record(search.cell.distinctness_cutoff) for search in searches[:3]
            ],
            "primary_cell": {
                "fidelity": _fraction_record(PRIMARY_CELL_KEY[0]),
                "distinctness": _fraction_record(PRIMARY_CELL_KEY[1]),
            },
            "search_configuration": dict(search_config),
            "sparsity_boundary": MAXIMUM_RETAINED_COMPONENTS,
            "component_universe": validation.configuration.payload["component_universe"],
            "transfer": validation.configuration.payload["transfer"],
            "registry": _registry_rows(run_id, searches, transfers),
            "counts": {
                "grid_cells": 18,
                "fresh_search_cells": 15,
                "reference_search_cells": 3,
                "transfer_reference_cells": 1,
                "fresh_transfer_workloads": sum(
                    search.family_size > 0
                    and search.cell.transfer_execution_mode == "fresh_execution"
                    for search in searches
                ),
            },
            "results": {
                "classification": classification,
                "family_sizes": {search.cell.cell_id: search.family_size for search in searches},
                "stopping_reasons": {
                    search.cell.cell_id: search.stopping_reason for search in searches
                },
                "right_censored": {
                    search.cell.cell_id: search.right_censored for search in searches
                },
                "transfer_group_counts": {
                    cell_id: transfer.group_count for cell_id, transfer in transfers.items()
                },
            },
            "integrity": run_record["model_integrity"],
            "stage15_status": "unavailable",
            "checkpoint_grid_decision_made": False,
            "stage18_started": False,
            "device": device,
            "software": {
                "python_packages": package_versions(
                    ("matplotlib", "numpy", "pandas", "PyYAML", "torch", "transformer-lens")
                )
            },
            "outputs": output_hashes,
        }
        _stable_json(paths.manifest, manifest)
        return Stage17ExecutionResult(
            run_id=run_id,
            implementation_commit=validation.implementation_commit,
            manifest=paths.manifest,
            archive=paths.archive,
            note=paths.note,
            runtime_table=paths.tables["runtime_table"],
            scientific_tables=deterministic_tables,
            figures=tuple(paths.figures.values()),
            classification=classification,
        )
    except Exception:
        _cleanup_outputs(paths)
        raise


def compare_reproduction(
    reference_root: str | Path,
    reproduction_root: str | Path,
    *,
    run_id: str,
) -> dict[str, object]:
    """Compare deterministic Stage 17 outputs byte-for-byte."""

    reference = Path(reference_root).resolve()
    reproduction = Path(reproduction_root).resolve()
    prefixes = (
        Path("results/raw") / run_id,
        Path("results/tables"),
        Path("results/notes"),
        Path("results/archives") / f"{run_id}.tar.gz",
        Path("figures"),
    )
    relative_paths: list[Path] = []
    for prefix in prefixes:
        source = reference / prefix
        if source.is_file():
            relative_paths.append(prefix)
        elif source.is_dir():
            relative_paths.extend(
                path.relative_to(reference)
                for path in source.rglob("*")
                if path.is_file() and ("stage17" in path.name or run_id in path.as_posix())
            )
    relative_paths = sorted(set(relative_paths))
    relative_paths = [
        path
        for path in relative_paths
        if path.name != "seed_1_stage17_runtime.csv"
        and not path.as_posix().startswith("manifests/")
    ]
    mismatches = []
    for relative in relative_paths:
        left, right = reference / relative, reproduction / relative
        if not right.is_file() or left.read_bytes() != right.read_bytes():
            mismatches.append(relative.as_posix())

    reference_manifest = reference / "manifests" / f"stage17_sensitivity_{run_id}.json"
    reproduction_manifest = reproduction / "manifests" / f"stage17_sensitivity_{run_id}.json"
    manifest_semantically_identical: bool | None = None
    if reference_manifest.is_file() or reproduction_manifest.is_file():
        if not reference_manifest.is_file() or not reproduction_manifest.is_file():
            manifest_semantically_identical = False
        else:

            def normalized_manifest(path: Path) -> dict[str, Any]:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload.pop("creation_timestamp_utc", None)
                payload.pop("software", None)
                outputs = payload.get("outputs", {})
                if isinstance(outputs, dict):
                    outputs.pop("runtime_table", None)
                return payload

            manifest_semantically_identical = normalized_manifest(
                reference_manifest
            ) == normalized_manifest(reproduction_manifest)

    runtime_relative = Path("results/tables/seed_1_stage17_runtime.csv")
    reference_runtime = reference / runtime_relative
    reproduction_runtime = reproduction / runtime_relative
    runtime_semantically_identical: bool | None = None
    if reference_runtime.is_file() or reproduction_runtime.is_file():
        if not reference_runtime.is_file() or not reproduction_runtime.is_file():
            runtime_semantically_identical = False
        else:

            def normalized_runtime(path: Path) -> list[dict[str, str]]:
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = [dict(row) for row in csv.DictReader(handle)]
                for row in rows:
                    row.pop("elapsed_seconds", None)
                return rows

            runtime_semantically_identical = normalized_runtime(
                reference_runtime
            ) == normalized_runtime(reproduction_runtime)

    byte_identical = not mismatches
    passed = (
        byte_identical
        and manifest_semantically_identical is not False
        and runtime_semantically_identical is not False
    )
    return {
        "run_id": run_id,
        "deterministic_file_count": len(relative_paths),
        "byte_identical": byte_identical,
        "mismatches": mismatches,
        "manifest_semantically_identical": manifest_semantically_identical,
        "runtime_semantically_identical": runtime_semantically_identical,
        "passed": passed,
    }

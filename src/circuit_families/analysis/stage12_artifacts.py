"""Deterministic raw artifacts for one Stage 12 search cell."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from circuit_families.analysis.fidelity_calibration import (
    file_sha256,
)
from circuit_families.interpretability.diversity_forced_search import (
    CheckpointFamilySearchExecution,
    DerivedSearchSeed,
    DiversityRankingEntry,
    DiversityRankingResult,
    FamilyRestartOutcome,
    RecoveredFamilyMember,
)
from circuit_families.interpretability.overlap_constraints import (
    jaccard_counts,
    jaccard_fraction,
)
from circuit_families.interpretability.sparse_search import (
    SparseSearchArtifacts,
    write_sparse_search_artifacts,
)


@dataclass(frozen=True)
class Stage12CellArtifacts:
    """Paths and hashes for one deterministic Stage 12 cell."""

    output_directory: Path
    cell_summary_path: Path
    cell_summary_sha256: str
    family_members_path: Path
    family_members_sha256: str
    pairwise_overlaps_path: Path
    pairwise_overlaps_sha256: str
    restart_summary_paths: tuple[Path, ...]
    restart_summary_sha256s: tuple[str, ...]
    ranking_log_paths: tuple[Path, ...]
    ranking_log_sha256s: tuple[str, ...]
    sparse_search_artifacts: tuple[SparseSearchArtifacts, ...]
    hash_inventory_path: Path
    hash_inventory_sha256: str


def _prepare_empty_output_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    existing = tuple(path.iterdir())

    if existing:
        names = ", ".join(
            sorted(item.name for item in existing)
        )
        raise FileExistsError(
            "Stage 12 output directory must be empty. "
            f"Existing entries: {names}"
        )

    return path


def _stable_json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def _write_stable_json(
    path: Path,
    value: Mapping[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _stable_json_text(value),
        encoding="utf-8",
    )
    return path


def _write_stable_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        for record in records
    ]

    path.write_text(
        "".join(f"{line}\n" for line in lines),
        encoding="utf-8",
    )
    return path


def _relative_path(
    root: Path,
    path: Path,
) -> str:
    return path.relative_to(root).as_posix()


def _fraction_record(value: Fraction) -> dict[str, Any]:
    if not isinstance(value, Fraction):
        raise TypeError("value must be a Fraction.")

    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": float(value),
    }


def _seed_record(
    seed: DerivedSearchSeed | None,
) -> dict[str, Any] | None:
    if seed is None:
        return None

    return {
        "model_seed": seed.model_seed,
        "checkpoint_index": seed.checkpoint_index,
        "family_member_index": (
            seed.family_member_index
        ),
        "restart_index": seed.restart_index,
        "canonical_material": seed.canonical_material,
        "sha256_digest": seed.sha256_digest,
        "integer_seed": seed.integer_seed,
        "bit_generator": seed.bit_generator,
    }


def _hook_count_records(
    values: Sequence[tuple[str, int]],
) -> list[dict[str, Any]]:
    return [
        {
            "hook_name": hook_name,
            "hook_count": hook_count,
        }
        for hook_name, hook_count in values
    ]


def _ranking_entry_record(
    entry: DiversityRankingEntry,
) -> dict[str, Any]:
    return {
        "component_identifier": (
            entry.component_identifier
        ),
        "component_index": entry.component_index,
        "component_class": entry.component_class,
        "gate_gradient": entry.gate_gradient,
        "raw_estimated_removal_damage": (
            entry.raw_estimated_removal_damage
        ),
        "damage_percentile": entry.damage_percentile,
        "reuse_rate": entry.reuse_rate,
        "removal_score": entry.removal_score,
        "candidate_ordering_score": (
            entry.candidate_ordering_score
        ),
        "ranking_position": entry.ranking_position,
    }


def _ranking_pass_record(
    result: DiversityRankingResult,
    *,
    ranking_pass_index: int,
) -> dict[str, Any]:
    ranking = result.ranking_result

    return {
        "ranking_pass_index": ranking_pass_index,
        "mean_pseudo_target_loss": (
            ranking.mean_pseudo_target_loss
        ),
        "evaluated_example_count": (
            ranking.evaluated_example_count
        ),
        "ranking_batch_size": ranking.ranking_batch_size,
        "retained_component_count": (
            ranking.retained_component_count
        ),
        "model_state_sha256_before": (
            ranking.model_state_sha256_before
        ),
        "model_state_sha256_after": (
            ranking.model_state_sha256_after
        ),
        "hook_counts_before": _hook_count_records(
            ranking.hook_counts_before
        ),
        "hook_counts_after": _hook_count_records(
            ranking.hook_counts_after
        ),
        "full_model_reference_sha256": (
            ranking.full_model_reference_sha256
        ),
        "full_model_reference_example_count": (
            ranking.full_model_reference_example_count
        ),
        "full_model_reference_batch_size": (
            ranking.full_model_reference_batch_size
        ),
        "gradient_source": ranking.gradient_source,
        "score_definition": ranking.score_definition,
        "entry_count": len(result.entries),
        "entries": [
            _ranking_entry_record(entry)
            for entry in result.entries
        ],
    }


def _restart_overlap_records(
    outcome: FamilyRestartOutcome,
) -> list[dict[str, Any]]:
    return [
        {
            "prior_member_index": prior_member_index,
            "prior_member_label": (
                f"C{prior_member_index}"
            ),
            "jaccard": _fraction_record(overlap),
        }
        for prior_member_index, overlap in enumerate(
            outcome.pairwise_overlaps,
            start=1,
        )
    ]


def _member_record(
    member: RecoveredFamilyMember,
    *,
    restart_directory: Path,
    root: Path,
) -> dict[str, Any]:
    return {
        "member_index": member.member_index,
        "member_label": f"C{member.member_index}",
        "selected_restart_index": (
            member.selected_restart_index
        ),
        "restart_directory": _relative_path(
            root,
            restart_directory,
        ),
        "mask_id": member.mask.mask_id,
        "mask": member.mask.to_record(),
        "metrics": member.metrics.to_record(),
        "retained_attention_head_count": (
            member.mask.retained_attention_head_count
        ),
        "retained_mlp_neuron_count": (
            member.mask.retained_mlp_neuron_count
        ),
        "retained_component_count": (
            member.mask.retained_component_count
        ),
        "retained_component_proportion": (
            member.mask.retained_component_proportion
        ),
        "pairwise_overlaps_with_prior_members": [
            {
                "prior_member_index": prior_member_index,
                "prior_member_label": (
                    f"C{prior_member_index}"
                ),
                "jaccard": _fraction_record(overlap),
            }
            for prior_member_index, overlap in enumerate(
                member.pairwise_overlaps,
                start=1,
            )
        ],
        "maximum_pairwise_overlap": _fraction_record(
            member.maximum_pairwise_overlap
        ),
        "search_status": member.search_result.status,
        "exact_evaluations_used": (
            member.search_result.exact_evaluations_used
        ),
        "ranking_passes_used": (
            member.search_result.ranking_passes_used
        ),
        "accepted_removal_count": len(
            member.search_result.accepted_removals
        ),
        "rejected_candidate_count": (
            member.search_result.rejected_candidate_count
        ),
        "candidate_batches_tested": (
            member.search_result.candidate_batches_tested
        ),
        "locally_single_deletion_minimal": (
            member.search_result.locally_single_deletion_minimal
        ),
        "meaningfully_sparse": (
            member.search_result.meaningfully_sparse
        ),
    }


def _selected_pairwise_records(
    members: Sequence[RecoveredFamilyMember],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

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

            records.append(
                {
                    "left_member_index": left.member_index,
                    "left_member_label": (
                        f"C{left.member_index}"
                    ),
                    "right_member_index": (
                        right.member_index
                    ),
                    "right_member_label": (
                        f"C{right.member_index}"
                    ),
                    "intersection_count": intersection,
                    "union_count": union,
                    "jaccard": _fraction_record(overlap),
                }
            )

    return records


def _hash_inventory_records(
    root: Path,
    *,
    excluded: Sequence[Path],
) -> list[dict[str, Any]]:
    excluded_resolved = {
        path.resolve()
        for path in excluded
    }

    files = sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.resolve() not in excluded_resolved
        )
    )

    return [
        {
            "path": _relative_path(root, path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]


def write_stage12_cell_artifacts(
    output_directory: str | Path,
    execution: CheckpointFamilySearchExecution,
    *,
    cell_metadata: Mapping[str, Any],
) -> Stage12CellArtifacts:
    """Write deterministic raw artifacts for one Stage 12 cell.

    Wall-clock runtime is deliberately excluded and must be written by
    the execution runner to a separate non-scientific runtime table.
    """

    if not isinstance(
        execution,
        CheckpointFamilySearchExecution,
    ):
        raise TypeError(
            "execution must be a "
            "CheckpointFamilySearchExecution."
        )

    if not isinstance(cell_metadata, Mapping):
        raise TypeError(
            "cell_metadata must be a mapping."
        )

    output = _prepare_empty_output_directory(
        Path(output_directory)
    )

    try:
        family = execution.result
        restart_summary_paths: list[Path] = []
        restart_summary_sha256s: list[str] = []
        ranking_log_paths: list[Path] = []
        ranking_log_sha256s: list[str] = []
        sparse_artifacts: list[
            SparseSearchArtifacts
        ] = []
        restart_directories: dict[
            tuple[int, int],
            Path,
        ] = {}

        for outcome in family.restart_outcomes:
            restart_directory = (
                output
                / "restarts"
                / f"C{outcome.requested_member_index:02d}"
                / f"restart_{outcome.restart_index:02d}"
            )
            restart_directories[
                (
                    outcome.requested_member_index,
                    outcome.restart_index,
                )
            ] = restart_directory

            search_artifacts = (
                write_sparse_search_artifacts(
                    restart_directory / "search",
                    outcome.execution.result,
                    cell_metadata={
                        **dict(cell_metadata),
                        "requested_member_index": (
                            outcome.requested_member_index
                        ),
                        "requested_member_label": (
                            f"C{outcome.requested_member_index}"
                        ),
                        "restart_index": (
                            outcome.restart_index
                        ),
                        "stage12_outcome_status": (
                            outcome.outcome_status
                        ),
                        "accepted_candidate": (
                            outcome.accepted_candidate
                        ),
                    },
                )
            )
            sparse_artifacts.append(search_artifacts)

            ranking_records = [
                _ranking_pass_record(
                    ranking_result,
                    ranking_pass_index=pass_index,
                )
                for pass_index, ranking_result in enumerate(
                    outcome.execution.ranking_results,
                    start=1,
                )
            ]
            ranking_log_path = _write_stable_jsonl(
                restart_directory
                / "diversity_rankings.jsonl",
                ranking_records,
            )
            ranking_log_sha256 = file_sha256(
                ranking_log_path
            )
            ranking_log_paths.append(ranking_log_path)
            ranking_log_sha256s.append(
                ranking_log_sha256
            )

            restart_summary = {
                "schema_version": 1,
                "cell_metadata": dict(cell_metadata),
                "requested_member_index": (
                    outcome.requested_member_index
                ),
                "requested_member_label": (
                    f"C{outcome.requested_member_index}"
                ),
                "restart_index": outcome.restart_index,
                "seed": _seed_record(
                    outcome.seed_record
                ),
                "outcome_status": (
                    outcome.outcome_status
                ),
                "accepted_candidate": (
                    outcome.accepted_candidate
                ),
                "pairwise_overlaps_with_prior_members": (
                    _restart_overlap_records(outcome)
                ),
                "maximum_pairwise_overlap": (
                    _fraction_record(
                        outcome.maximum_pairwise_overlap
                    )
                ),
                "search": {
                    "status": (
                        outcome.execution.result.status
                    ),
                    "stopping_reason": (
                        outcome.execution.result.stopping_reason
                    ),
                    "failure_detail": (
                        outcome.execution.result.failure_detail
                    ),
                    "fidelity_threshold": (
                        outcome.execution.result.fidelity_threshold
                    ),
                    "final_mask_id": (
                        outcome.execution.result.final_mask.mask_id
                    ),
                    "retained_component_count": (
                        outcome.execution.result.final_mask
                        .retained_component_count
                    ),
                    "final_metrics": (
                        outcome.execution.result.final_metrics
                        .to_record()
                    ),
                    "exact_evaluation_budget": (
                        outcome.execution.result
                        .exact_evaluation_budget
                    ),
                    "exact_evaluations_used": (
                        outcome.execution.result
                        .exact_evaluations_used
                    ),
                    "ranking_passes_used": (
                        outcome.execution.result
                        .ranking_passes_used
                    ),
                    "candidate_batches_tested": (
                        outcome.execution.result
                        .candidate_batches_tested
                    ),
                    "rejected_candidate_count": (
                        outcome.execution.result
                        .rejected_candidate_count
                    ),
                    "budget_remaining": (
                        outcome.execution.result
                        .budget_remaining
                    ),
                    "budget_exhausted": (
                        outcome.execution.result
                        .budget_exhausted
                    ),
                    "locally_single_deletion_minimal": (
                        outcome.execution.result
                        .locally_single_deletion_minimal
                    ),
                    "meaningfully_sparse": (
                        outcome.execution.result
                        .meaningfully_sparse
                    ),
                },
                "outputs": {
                    "stage9_search_directory": (
                        _relative_path(
                            output,
                            search_artifacts.output_directory,
                        )
                    ),
                    "stage9_cell_summary": {
                        "path": _relative_path(
                            output,
                            search_artifacts.cell_summary_path,
                        ),
                        "sha256": (
                            search_artifacts
                            .cell_summary_sha256
                        ),
                    },
                    "stage9_hashes": {
                        "path": _relative_path(
                            output,
                            search_artifacts.hashes_path,
                        ),
                        "sha256": (
                            search_artifacts.hashes_sha256
                        ),
                    },
                    "stage9_final_mask": {
                        "path": _relative_path(
                            output,
                            search_artifacts.final_mask_path,
                        ),
                        "sha256": (
                            search_artifacts.final_mask_sha256
                        ),
                    },
                    "diversity_rankings": {
                        "path": _relative_path(
                            output,
                            ranking_log_path,
                        ),
                        "sha256": ranking_log_sha256,
                        "record_count": len(
                            ranking_records
                        ),
                    },
                },
            }

            restart_summary_path = _write_stable_json(
                restart_directory
                / "restart_summary.json",
                restart_summary,
            )
            restart_summary_paths.append(
                restart_summary_path
            )
            restart_summary_sha256s.append(
                file_sha256(restart_summary_path)
            )

        member_records: list[dict[str, Any]] = []

        for member in family.members:
            key = (
                member.member_index,
                member.selected_restart_index,
            )

            if key not in restart_directories:
                raise RuntimeError(
                    "Selected family member has no matching "
                    "restart artifact directory."
                )

            member_records.append(
                _member_record(
                    member,
                    restart_directory=(
                        restart_directories[key]
                    ),
                    root=output,
                )
            )

        family_members_path = _write_stable_jsonl(
            output / "family_members.jsonl",
            member_records,
        )
        family_members_sha256 = file_sha256(
            family_members_path
        )

        pairwise_records = _selected_pairwise_records(
            family.members
        )
        pairwise_overlaps_path = _write_stable_jsonl(
            output / "pairwise_overlaps.jsonl",
            pairwise_records,
        )
        pairwise_overlaps_sha256 = file_sha256(
            pairwise_overlaps_path
        )

        cell_summary = {
            "schema_version": 1,
            "experiment_type": (
                "stage12_diversity_forced_search_cell"
            ),
            "cell_metadata": dict(cell_metadata),
            "family_search": {
                "status": family.status,
                "stopping_reason": (
                    family.stopping_reason
                ),
                "fidelity_threshold": (
                    family.fidelity_threshold
                ),
                "distinctness_cutoff": (
                    _fraction_record(
                        family.distinctness_cutoff
                    )
                ),
                "family_target": family.family_target,
                "family_size": family.family_size,
                "right_censored": family.right_censored,
                "max_restarts_per_alternative": (
                    family.max_restarts_per_alternative
                ),
                "per_requested_circuit_budget": (
                    family.per_requested_circuit_budget
                ),
                "per_cell_budget": (
                    family.per_cell_budget
                ),
                "exact_evaluations_used": (
                    family.exact_evaluations_used
                ),
                "budget_remaining": (
                    family.budget_remaining
                ),
                "restart_outcome_count": len(
                    family.restart_outcomes
                ),
            },
            "checkpoint_integrity": {
                "pseudo_target_sha256": (
                    execution.pseudo_target_sha256
                ),
                "pseudo_target_count": (
                    execution.pseudo_target_count
                ),
                "ranking_batch_size": (
                    execution.ranking_batch_size
                ),
                "evaluation_batch_size": (
                    execution.evaluation_batch_size
                ),
                "full_model_reference_sha256": (
                    execution.full_model_reference_sha256
                ),
                "full_model_reference_example_count": (
                    execution
                    .full_model_reference_example_count
                ),
                "full_model_reference_batch_size": (
                    execution
                    .full_model_reference_batch_size
                ),
                "model_state_sha256_before": (
                    execution.model_state_sha256_before
                ),
                "model_state_sha256_after": (
                    execution.model_state_sha256_after
                ),
                "model_state_unchanged": (
                    execution.model_state_sha256_before
                    == execution.model_state_sha256_after
                ),
                "hook_counts_before": (
                    _hook_count_records(
                        execution.hook_counts_before
                    )
                ),
                "hook_counts_after": (
                    _hook_count_records(
                        execution.hook_counts_after
                    )
                ),
                "hook_counts_unchanged": (
                    execution.hook_counts_before
                    == execution.hook_counts_after
                ),
            },
            "outputs": {
                "family_members": {
                    "path": _relative_path(
                        output,
                        family_members_path,
                    ),
                    "sha256": family_members_sha256,
                    "record_count": len(
                        member_records
                    ),
                },
                "pairwise_overlaps": {
                    "path": _relative_path(
                        output,
                        pairwise_overlaps_path,
                    ),
                    "sha256": (
                        pairwise_overlaps_sha256
                    ),
                    "record_count": len(
                        pairwise_records
                    ),
                },
                "restart_summaries": [
                    {
                        "path": _relative_path(
                            output,
                            path,
                        ),
                        "sha256": digest,
                    }
                    for path, digest in zip(
                        restart_summary_paths,
                        restart_summary_sha256s,
                        strict=True,
                    )
                ],
                "ranking_logs": [
                    {
                        "path": _relative_path(
                            output,
                            path,
                        ),
                        "sha256": digest,
                    }
                    for path, digest in zip(
                        ranking_log_paths,
                        ranking_log_sha256s,
                        strict=True,
                    )
                ],
            },
            "runtime_telemetry": {
                "included_in_deterministic_artifacts": (
                    False
                ),
                "reason": (
                    "Wall-clock runtime varies across reruns "
                    "and is stored separately."
                ),
            },
        }

        cell_summary_path = _write_stable_json(
            output / "cell_summary.json",
            cell_summary,
        )
        cell_summary_sha256 = file_sha256(
            cell_summary_path
        )

        hash_inventory_path = (
            output / "hash_inventory.json"
        )
        inventory_records = _hash_inventory_records(
            output,
            excluded=(hash_inventory_path,),
        )
        hash_inventory = {
            "schema_version": 1,
            "file_count": len(inventory_records),
            "files": inventory_records,
        }
        _write_stable_json(
            hash_inventory_path,
            hash_inventory,
        )
        hash_inventory_sha256 = file_sha256(
            hash_inventory_path
        )

        return Stage12CellArtifacts(
            output_directory=output,
            cell_summary_path=cell_summary_path,
            cell_summary_sha256=(
                cell_summary_sha256
            ),
            family_members_path=family_members_path,
            family_members_sha256=(
                family_members_sha256
            ),
            pairwise_overlaps_path=(
                pairwise_overlaps_path
            ),
            pairwise_overlaps_sha256=(
                pairwise_overlaps_sha256
            ),
            restart_summary_paths=tuple(
                restart_summary_paths
            ),
            restart_summary_sha256s=tuple(
                restart_summary_sha256s
            ),
            ranking_log_paths=tuple(
                ranking_log_paths
            ),
            ranking_log_sha256s=tuple(
                ranking_log_sha256s
            ),
            sparse_search_artifacts=tuple(
                sparse_artifacts
            ),
            hash_inventory_path=(
                hash_inventory_path
            ),
            hash_inventory_sha256=(
                hash_inventory_sha256
            ),
        )

    except Exception:
        shutil.rmtree(output)
        raise

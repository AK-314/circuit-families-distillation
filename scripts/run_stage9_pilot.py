"""Run the frozen 18-cell Stage 9 pilot search grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_sparse_search import (
    CALIBRATION_ELIGIBLE,
    CALIBRATION_EXCLUDED,
    METHOD_DEVELOPMENT_LABEL,
    default_cell_directory,
    repository_provenance,
)

from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_HOOK_NAME,
    MLP_NEURON_HOOK_NAME,
    SEARCHABLE_COMPONENT_COUNT,
)
from circuit_families.interpretability.sparse_search import (
    CANDIDATE_BATCH_SIZE,
    DEFAULT_EXACT_EVALUATION_BUDGET,
    MEANINGFULLY_SPARSE_MAX_COMPONENTS,
    RANKING_SCORE_DEFINITION,
)
from circuit_families.manifests import package_versions, utc_timestamp
from circuit_families.training import file_sha256

CHECKPOINT_EXECUTION_ORDER = (
    (9050, "stable post-grokking"),
    (200, "pre-grokking"),
    (8150, "50%"),
)

FIDELITY_THRESHOLD_EXECUTION_ORDER = (
    0.99,
    0.975,
    0.95,
    0.90,
    0.85,
    0.80,
)

SOFTWARE_PACKAGES = (
    "numpy",
    "pandas",
    "PyYAML",
    "torch",
    "transformer-lens",
)

RESULT_TABLE_FIELDS = (
    "stage9_run_id",
    "source_training_run_id",
    "phase",
    "checkpoint_step",
    "checkpoint_sha256",
    "fidelity_threshold",
    "search_status",
    "retained_heads",
    "retained_neurons",
    "total_retained_components",
    "retained_proportion",
    "final_exact_fidelity",
    "masked_ground_truth_accuracy",
    "masked_cross_entropy",
    "mean_kl_divergence",
    "mean_jensen_shannon_divergence",
    "accepted_removals",
    "exact_evaluations_used",
    "ranking_passes",
    "budget_exhausted",
    "locally_single_deletion_minimal",
    "meaningfully_sparse",
    "threshold_calibration_eligibility",
    "final_mask_path",
    "final_mask_sha256",
    "raw_trajectory_path",
    "raw_trajectory_sha256",
    "candidate_log_path",
    "candidate_log_sha256",
    "cell_summary_path",
    "cell_summary_sha256",
)

RUNTIME_TABLE_FIELDS = (
    "stage9_run_id",
    "phase",
    "checkpoint_step",
    "fidelity_threshold",
    "elapsed_runtime_seconds",
    "runtime_telemetry_path",
    "runtime_telemetry_sha256",
)


@dataclass(frozen=True)
class PilotCell:
    """One frozen checkpoint/threshold search cell."""

    execution_index: int
    checkpoint_step: int
    expected_phase: str
    fidelity_threshold: float


@dataclass(frozen=True)
class PilotPaths:
    """Deterministic Stage 9 pilot output paths."""

    stage9_run_id: str
    raw_directory: Path
    result_table: Path
    runtime_table: Path
    manifest: Path


def parse_args() -> argparse.Namespace:
    """Parse complete-pilot arguments."""

    parser = argparse.ArgumentParser(
        description="Run the frozen Stage 9 pilot sparse-search grid."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage8-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ranking-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--exact-evaluation-budget",
        type=int,
        default=DEFAULT_EXACT_EVALUATION_BUDGET,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--method-development",
        action="store_true",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        help=(
            "Method-development only: execute the first N frozen cells."
        ),
    )
    return parser.parse_args()


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _relative(repository: Path, path: Path) -> str:
    resolved = path.resolve()

    try:
        return str(resolved.relative_to(repository.resolve()))
    except ValueError:
        return str(resolved)


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def frozen_execution_plan() -> tuple[PilotCell, ...]:
    """Return the exact 18-cell Stage 9 execution order."""

    cells: list[PilotCell] = []
    execution_index = 0

    for checkpoint_step, phase in CHECKPOINT_EXECUTION_ORDER:
        for threshold in FIDELITY_THRESHOLD_EXECUTION_ORDER:
            execution_index += 1
            cells.append(
                PilotCell(
                    execution_index=execution_index,
                    checkpoint_step=checkpoint_step,
                    expected_phase=phase,
                    fidelity_threshold=threshold,
                )
            )

    return tuple(cells)


def stage9_configuration_record(
    *,
    source_training_run_id: str,
    checkpoint_manifest_sha256: str,
    stage8_manifest_sha256: str,
    implementation_git_commit: str,
    ranking_batch_size: int,
    evaluation_batch_size: int,
    exact_evaluation_budget: int,
    device: str,
    method_development: bool,
) -> dict[str, Any]:
    """Return the run-ID-defining Stage 9 configuration."""

    return {
        "source_training_run_id": source_training_run_id,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "stage8_manifest_sha256": stage8_manifest_sha256,
        "implementation_git_commit": implementation_git_commit,
        "algorithm": "deterministic_greedy_single_deletion",
        "ranking_score_definition": RANKING_SCORE_DEFINITION,
        "candidate_batch_size": CANDIDATE_BATCH_SIZE,
        "component_count": SEARCHABLE_COMPONENT_COUNT,
        "meaningfully_sparse_max_components": (
            MEANINGFULLY_SPARSE_MAX_COMPONENTS
        ),
        "checkpoint_execution_order": [
            {
                "checkpoint_step": step,
                "phase": phase,
            }
            for step, phase in CHECKPOINT_EXECUTION_ORDER
        ],
        "fidelity_threshold_execution_order": list(
            FIDELITY_THRESHOLD_EXECUTION_ORDER
        ),
        "ranking_batch_size": ranking_batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "exact_evaluation_budget_per_cell": (
            exact_evaluation_budget
        ),
        "device": device,
        "method_development": method_development,
    }


def deterministic_stage9_run_id(
    configuration: dict[str, Any],
) -> str:
    """Return a deterministic Stage 9 run ID."""

    digest = hashlib.sha256(
        _canonical_json(configuration).encode("utf-8")
    ).hexdigest()

    source_run = configuration["source_training_run_id"]
    marker = "-s"
    seed_fragment = "unknown"

    if marker in source_run:
        suffix = source_run.split(marker, maxsplit=1)[1]
        seed_fragment = suffix.split("-", maxsplit=1)[0]

    prefix = (
        "stage9-method-development"
        if configuration["method_development"]
        else "stage9-sparse"
    )

    return f"{prefix}-s{seed_fragment}-{digest[:12]}"


def pilot_output_paths(
    repository: Path,
    *,
    stage9_run_id: str,
    source_training_run_id: str,
    method_development: bool,
) -> PilotPaths:
    """Return deterministic pilot-level output paths."""

    seed_fragment = source_training_run_id.split("-s", maxsplit=1)[1]
    seed = seed_fragment.split("-", maxsplit=1)[0]

    table_prefix = (
        f"method_development_seed_{seed}_stage9_sparse_search"
        if method_development
        else f"seed_{seed}_stage9_sparse_search"
    )
    manifest_prefix = (
        "method_development_stage9_sparse"
        if method_development
        else "stage9_sparse"
    )

    return PilotPaths(
        stage9_run_id=stage9_run_id,
        raw_directory=(
            repository / "results" / "raw" / stage9_run_id
        ),
        result_table=(
            repository / "results" / "tables" / f"{table_prefix}.csv"
        ),
        runtime_table=(
            repository
            / "results"
            / "tables"
            / f"{table_prefix}_runtime.csv"
        ),
        manifest=(
            repository
            / "manifests"
            / f"{manifest_prefix}_{stage9_run_id}.json"
        ),
    )


def build_single_cell_command(
    *,
    repository: Path,
    run_id: str,
    checkpoint_manifest: Path,
    stage8_manifest: Path,
    cell: PilotCell,
    output_directory: Path,
    ranking_batch_size: int,
    evaluation_batch_size: int,
    exact_evaluation_budget: int,
    device: str,
    implementation_git_commit: str,
    method_development: bool,
) -> list[str]:
    """Build the exact uv-based command for one independent cell."""

    command = [
        "uv",
        "run",
        "python",
        "scripts/run_sparse_search.py",
        "--run-id",
        run_id,
        "--checkpoint-manifest",
        _relative(repository, checkpoint_manifest),
        "--stage8-manifest",
        _relative(repository, stage8_manifest),
        "--checkpoint-step",
        str(cell.checkpoint_step),
        "--fidelity-threshold",
        str(cell.fidelity_threshold),
        "--ranking-batch-size",
        str(ranking_batch_size),
        "--evaluation-batch-size",
        str(evaluation_batch_size),
        "--exact-evaluation-budget",
        str(exact_evaluation_budget),
        "--device",
        device,
        "--repository-root",
        ".",
        "--output-directory",
        _relative(repository, output_directory),
    ]

    if method_development:
        command.append("--method-development")
    else:
        command.extend(
            [
                "--expected-implementation-commit",
                implementation_git_commit,
            ]
        )

    return command


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")

    return value


def build_result_row(
    *,
    repository: Path,
    stage9_run_id: str,
    cell: PilotCell,
    cell_directory: Path,
) -> dict[str, Any]:
    """Build one deterministic compact scientific result row."""

    summary_path = cell_directory / "cell_summary.json"
    summary = _load_json_object(summary_path)
    metadata = summary["cell_metadata"]
    search = summary["search"]
    final_mask = summary["final_mask"]
    metrics = summary["final_metrics"]
    outputs = summary["outputs"]

    if metadata["checkpoint_step"] != cell.checkpoint_step:
        raise ValueError("Cell summary checkpoint step mismatch.")

    if metadata["checkpoint_phase"] != cell.expected_phase:
        raise ValueError("Cell summary checkpoint phase mismatch.")

    if metadata["fidelity_threshold"] != cell.fidelity_threshold:
        raise ValueError("Cell summary threshold mismatch.")

    trajectory_relative = outputs[
        "accepted_removal_trajectory"
    ]["path"]
    candidate_relative = outputs["candidate_evaluation_log"]["path"]

    trajectory_path = cell_directory / trajectory_relative
    candidate_path = cell_directory / candidate_relative
    final_mask_path = cell_directory / final_mask["path"]

    return {
        "stage9_run_id": stage9_run_id,
        "source_training_run_id": (
            metadata["source_training_run_id"]
        ),
        "phase": metadata["checkpoint_phase"],
        "checkpoint_step": metadata["checkpoint_step"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "fidelity_threshold": metadata["fidelity_threshold"],
        "search_status": search["status"],
        "retained_heads": final_mask[
            "retained_attention_head_count"
        ],
        "retained_neurons": final_mask[
            "retained_mlp_neuron_count"
        ],
        "total_retained_components": final_mask[
            "retained_component_count"
        ],
        "retained_proportion": final_mask[
            "retained_component_proportion"
        ],
        "final_exact_fidelity": metrics["primary_fidelity"],
        "masked_ground_truth_accuracy": metrics["masked_accuracy"],
        "masked_cross_entropy": metrics["masked_cross_entropy"],
        "mean_kl_divergence": metrics["mean_kl_divergence"],
        "mean_jensen_shannon_divergence": metrics[
            "mean_jensen_shannon_divergence"
        ],
        "accepted_removals": search["accepted_removal_count"],
        "exact_evaluations_used": search["exact_evaluations_used"],
        "ranking_passes": search["ranking_passes_used"],
        "budget_exhausted": search["budget_exhausted"],
        "locally_single_deletion_minimal": search[
            "locally_single_deletion_minimal"
        ],
        "meaningfully_sparse": search["meaningfully_sparse"],
        "threshold_calibration_eligibility": metadata[
            "threshold_calibration_eligibility"
        ],
        "final_mask_path": _relative(repository, final_mask_path),
        "final_mask_sha256": final_mask["sha256"],
        "raw_trajectory_path": _relative(
            repository,
            trajectory_path,
        ),
        "raw_trajectory_sha256": outputs[
            "accepted_removal_trajectory"
        ]["sha256"],
        "candidate_log_path": _relative(
            repository,
            candidate_path,
        ),
        "candidate_log_sha256": outputs[
            "candidate_evaluation_log"
        ]["sha256"],
        "cell_summary_path": _relative(repository, summary_path),
        "cell_summary_sha256": file_sha256(summary_path),
    }


def build_runtime_row(
    *,
    repository: Path,
    stage9_run_id: str,
    cell: PilotCell,
    cell_directory: Path,
) -> dict[str, Any]:
    """Build one nondeterministic runtime-telemetry row."""

    path = cell_directory / "runtime_telemetry.json"
    record = _load_json_object(path)

    return {
        "stage9_run_id": stage9_run_id,
        "phase": cell.expected_phase,
        "checkpoint_step": cell.checkpoint_step,
        "fidelity_threshold": cell.fidelity_threshold,
        "elapsed_runtime_seconds": record[
            "elapsed_runtime_seconds"
        ],
        "runtime_telemetry_path": _relative(repository, path),
        "runtime_telemetry_sha256": file_sha256(path),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
) -> Path:
    """Write stable CSV bytes in supplied row order."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    return path


def _cell_manifest_record(
    *,
    repository: Path,
    cell: PilotCell,
    cell_directory: Path,
    result_row: dict[str, Any],
    runtime_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "execution_index": cell.execution_index,
        "checkpoint_step": cell.checkpoint_step,
        "phase": cell.expected_phase,
        "fidelity_threshold": cell.fidelity_threshold,
        "status": result_row["search_status"],
        "output_directory": _relative(repository, cell_directory),
        "final_mask": {
            "path": result_row["final_mask_path"],
            "sha256": result_row["final_mask_sha256"],
        },
        "accepted_removal_trajectory": {
            "path": result_row["raw_trajectory_path"],
            "sha256": result_row["raw_trajectory_sha256"],
        },
        "candidate_evaluation_log": {
            "path": result_row["candidate_log_path"],
            "sha256": result_row["candidate_log_sha256"],
        },
        "cell_summary": {
            "path": result_row["cell_summary_path"],
            "sha256": result_row["cell_summary_sha256"],
        },
        "runtime_telemetry": {
            "path": runtime_row["runtime_telemetry_path"],
            "sha256": runtime_row["runtime_telemetry_sha256"],
            "included_in_deterministic_scientific_hashes": False,
        },
    }


def write_manifest(
    *,
    path: Path,
    repository: Path,
    stage9_run_id: str,
    implementation_git_commit: str,
    source_training_run_id: str,
    checkpoint_manifest: Path,
    stage8_manifest: Path,
    configuration: dict[str, Any],
    result_table: Path,
    runtime_table: Path,
    cells: list[dict[str, Any]],
    command: list[str],
    method_development: bool,
) -> Path:
    """Write the complete Stage 9 pilot manifest."""

    manifest = {
        "schema_version": 1,
        "experiment_type": "stage9_sparse_search_pilot",
        "stage9_run_id": stage9_run_id,
        "source_training_run_id": source_training_run_id,
        "stage9_implementation_git_commit": (
            implementation_git_commit
        ),
        "creation_timestamp_utc": utc_timestamp(),
        "method_development": method_development,
        "development_label": (
            METHOD_DEVELOPMENT_LABEL
            if method_development
            else None
        ),
        "primary_fidelity_threshold_selected": False,
        "source_manifests": {
            "checkpoint_selection": {
                "path": _relative(repository, checkpoint_manifest),
                "sha256": file_sha256(checkpoint_manifest),
            },
            "stage8_masking": {
                "path": _relative(repository, stage8_manifest),
                "sha256": file_sha256(stage8_manifest),
            },
        },
        "component_definition": {
            "searchable_component_count": (
                SEARCHABLE_COMPONENT_COUNT
            ),
            "attention_head_hook": ATTENTION_HEAD_HOOK_NAME,
            "mlp_neuron_hook": MLP_NEURON_HOOK_NAME,
            "mask_semantics": {
                "retained": 1,
                "zero_ablated": 0,
            },
        },
        "search_configuration": configuration,
        "candidate_tie_breaking": {
            "ranking_score_tie": (
                "lower stable component index"
            ),
            "within_first_valid_batch": (
                "highest exact fidelity, then lower stable "
                "component index"
            ),
        },
        "threshold_calibration": {
            "stable_post": CALIBRATION_ELIGIBLE,
            "pre_and_transition": CALIBRATION_EXCLUDED,
            "selection_performed_in_stage9": False,
        },
        "execution_command": command,
        "outputs": {
            "deterministic_result_table": {
                "path": _relative(repository, result_table),
                "sha256": file_sha256(result_table),
            },
            "nondeterministic_runtime_table": {
                "path": _relative(repository, runtime_table),
                "sha256": file_sha256(runtime_table),
                "excluded_from_deterministic_rerun_hash_check": True,
            },
            "cells": cells,
        },
        "runtime_artifact_resolution": {
            "deterministic_scientific_table_contains_runtime": False,
            "reason": (
                "Measured wall-clock time is nondeterministic. It is "
                "recorded in a separate runtime table so mask, trajectory, "
                "candidate-log, summary, and compact scientific-table "
                "hashes can reproduce exactly."
            ),
        },
        "software": {
            "python": platform.python_version(),
            "packages": package_versions(SOFTWARE_PACKAGES),
        },
        "git_cleanliness": {
            "clean_implementation_commit_verified_before_first_cell": (
                not method_development
            ),
            "later_dirty_paths_are_generated_stage9_artifacts": (
                not method_development
            ),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    """Execute the requested Stage 9 pilot plan."""

    args = parse_args()
    repository = args.repository_root.resolve()

    if args.max_cells is not None and not args.method_development:
        raise ValueError(
            "--max-cells is permitted only with --method-development."
        )

    checkpoint_manifest = _resolve(
        repository,
        args.checkpoint_manifest,
    )
    stage8_manifest = _resolve(
        repository,
        args.stage8_manifest,
    )

    implementation_commit, _ = repository_provenance(
        repository,
        require_clean=not args.method_development,
    )

    configuration = stage9_configuration_record(
        source_training_run_id=args.run_id,
        checkpoint_manifest_sha256=file_sha256(
            checkpoint_manifest
        ),
        stage8_manifest_sha256=file_sha256(stage8_manifest),
        implementation_git_commit=implementation_commit,
        ranking_batch_size=args.ranking_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        exact_evaluation_budget=args.exact_evaluation_budget,
        device=args.device,
        method_development=args.method_development,
    )
    stage9_run_id = deterministic_stage9_run_id(configuration)
    paths = pilot_output_paths(
        repository,
        stage9_run_id=stage9_run_id,
        source_training_run_id=args.run_id,
        method_development=args.method_development,
    )

    plan = frozen_execution_plan()

    if args.max_cells is not None:
        if args.max_cells <= 0:
            raise ValueError("--max-cells must be positive.")
        plan = plan[: args.max_cells]

    result_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    cell_manifest_records: list[dict[str, Any]] = []

    complete_command = [
        "uv",
        "run",
        "python",
        "scripts/run_stage9_pilot.py",
        "--run-id",
        args.run_id,
        "--checkpoint-manifest",
        _relative(repository, checkpoint_manifest),
        "--stage8-manifest",
        _relative(repository, stage8_manifest),
        "--ranking-batch-size",
        str(args.ranking_batch_size),
        "--evaluation-batch-size",
        str(args.evaluation_batch_size),
        "--exact-evaluation-budget",
        str(args.exact_evaluation_budget),
        "--device",
        args.device,
    ]

    for cell in plan:
        cell_directory = default_cell_directory(
            repository,
            stage9_run_id=stage9_run_id,
            checkpoint_step=cell.checkpoint_step,
            fidelity_threshold=cell.fidelity_threshold,
        )

        print(
            f"[{cell.execution_index:02d}/{len(plan):02d}] "
            f"checkpoint={cell.checkpoint_step} "
            f"phase={cell.expected_phase} "
            f"threshold={cell.fidelity_threshold}"
        )

        command = build_single_cell_command(
            repository=repository,
            run_id=args.run_id,
            checkpoint_manifest=checkpoint_manifest,
            stage8_manifest=stage8_manifest,
            cell=cell,
            output_directory=cell_directory,
            ranking_batch_size=args.ranking_batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            exact_evaluation_budget=args.exact_evaluation_budget,
            device=args.device,
            implementation_git_commit=implementation_commit,
            method_development=args.method_development,
        )

        subprocess.run(
            command,
            cwd=repository,
            check=True,
        )

        result_row = build_result_row(
            repository=repository,
            stage9_run_id=stage9_run_id,
            cell=cell,
            cell_directory=cell_directory,
        )
        runtime_row = build_runtime_row(
            repository=repository,
            stage9_run_id=stage9_run_id,
            cell=cell,
            cell_directory=cell_directory,
        )

        result_rows.append(result_row)
        runtime_rows.append(runtime_row)
        cell_manifest_records.append(
            _cell_manifest_record(
                repository=repository,
                cell=cell,
                cell_directory=cell_directory,
                result_row=result_row,
                runtime_row=runtime_row,
            )
        )

        print(
            "completed: "
            f"status={result_row['search_status']} "
            f"retained={result_row['total_retained_components']} "
            f"fidelity={result_row['final_exact_fidelity']} "
            f"exact_evaluations="
            f"{result_row['exact_evaluations_used']}"
        )

    write_csv(
        paths.result_table,
        result_rows,
        fields=RESULT_TABLE_FIELDS,
    )
    write_csv(
        paths.runtime_table,
        runtime_rows,
        fields=RUNTIME_TABLE_FIELDS,
    )
    write_manifest(
        path=paths.manifest,
        repository=repository,
        stage9_run_id=stage9_run_id,
        implementation_git_commit=implementation_commit,
        source_training_run_id=args.run_id,
        checkpoint_manifest=checkpoint_manifest,
        stage8_manifest=stage8_manifest,
        configuration=configuration,
        result_table=paths.result_table,
        runtime_table=paths.runtime_table,
        cells=cell_manifest_records,
        command=complete_command,
        method_development=args.method_development,
    )

    print(f"stage9_run_id: {stage9_run_id}")
    print(
        "completed_cell_count: "
        f"{len(result_rows)}"
    )
    print(
        "result_table: "
        f"{_relative(repository, paths.result_table)}"
    )
    print(
        "result_table_sha256: "
        f"{file_sha256(paths.result_table)}"
    )
    print(
        "runtime_table: "
        f"{_relative(repository, paths.runtime_table)}"
    )
    print(
        "manifest: "
        f"{_relative(repository, paths.manifest)}"
    )
    print(
        "manifest_sha256: "
        f"{file_sha256(paths.manifest)}"
    )


if __name__ == "__main__":
    main()

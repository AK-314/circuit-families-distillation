"""Run one deterministic Stage 9 sparse-search cell."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from circuit_families.interpretability.fidelity import (
    CheckpointEvaluationContext,
    load_checkpoint_evaluation_context,
)
from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_HOOK_NAME,
    MLP_NEURON_HOOK_NAME,
    SEARCHABLE_COMPONENT_COUNT,
)
from circuit_families.interpretability.sparse_search import (
    DEFAULT_EXACT_EVALUATION_BUDGET,
    CheckpointSearchExecution,
    SparseSearchArtifacts,
    run_checkpoint_sparse_search,
    write_sparse_search_artifacts,
)
from circuit_families.training import file_sha256

METHOD_DEVELOPMENT_LABEL = (
    "METHOD DEVELOPMENT — EXCLUDED FROM SCIENTIFIC COMPARISONS"
)

STABLE_POST_PHASE = "stable post-grokking"
CALIBRATION_ELIGIBLE = (
    "eligible_for_later_stage11_primary_threshold_calibration"
)
CALIBRATION_EXCLUDED = (
    "excluded_from_primary_threshold_calibration"
)


@dataclass(frozen=True)
class SingleCellOutput:
    """Complete output references for one executed search cell."""

    context: CheckpointEvaluationContext
    execution: CheckpointSearchExecution
    artifacts: SparseSearchArtifacts
    runtime_telemetry_path: Path
    runtime_telemetry_sha256: str
    elapsed_runtime_seconds: float
    implementation_git_commit: str
    git_status_at_start: str
    stage8_manifest_sha256: str


def parse_args() -> argparse.Namespace:
    """Parse one-cell Stage 9 command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run one deterministic Stage 9 sparse-search cell."
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
        "--checkpoint-step",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--fidelity-threshold",
        type=float,
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
        "--output-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-implementation-commit",
        help=(
            "For orchestrated final runs, require this exact HEAD while "
            "allowing previously generated Stage 9 artifacts."
        ),
    )
    parser.add_argument(
        "--method-development",
        action="store_true",
        help=(
            "Label the output as method development and permit a dirty "
            "working tree. Such output is excluded from scientific use."
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


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object.")

    return value


def _git_output(
    repository: Path,
    *arguments: str,
) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository_provenance(
    repository: Path,
    *,
    require_clean: bool,
    expected_commit: str | None = None,
) -> tuple[str, str]:
    """Return HEAD and status while enforcing implementation provenance."""

    status = _git_output(repository, "status", "--short")
    commit = _git_output(repository, "rev-parse", "HEAD")

    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            "Current HEAD does not match the expected Stage 9 "
            "implementation commit: "
            f"expected {expected_commit}, found {commit}."
        )

    if require_clean and status:
        raise RuntimeError(
            "Scientific Stage 9 outputs must be generated from a clean "
            "implementation commit. Current status:\n"
            + status
        )

    return commit, status


def threshold_slug(fidelity_threshold: float) -> str:
    """Return a collision-resistant four-decimal threshold path slug."""

    if (
        isinstance(fidelity_threshold, bool)
        or not isinstance(fidelity_threshold, (int, float))
    ):
        raise TypeError("fidelity threshold must be numeric.")

    value = float(fidelity_threshold)

    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError(
            "fidelity threshold must be finite, positive, and at most one."
        )

    scaled = value * 10_000
    rounded = round(scaled)

    if not math.isclose(
        scaled,
        float(rounded),
        abs_tol=1.0e-9,
        rel_tol=0.0,
    ):
        raise ValueError(
            "fidelity threshold must be exactly representable to four "
            "decimal places."
        )

    return f"threshold_{rounded:05d}"


def default_cell_directory(
    repository: Path,
    *,
    stage9_run_id: str,
    checkpoint_step: int,
    fidelity_threshold: float,
) -> Path:
    """Return the deterministic raw directory for one final cell."""

    if not isinstance(stage9_run_id, str) or not stage9_run_id:
        raise ValueError("stage9_run_id must be a non-empty string.")

    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise ValueError(
            "checkpoint_step must be a non-negative integer."
        )

    return (
        repository
        / "results"
        / "raw"
        / stage9_run_id
        / f"step_{checkpoint_step:08d}"
        / threshold_slug(fidelity_threshold)
    )


def validate_stage8_manifest(
    *,
    repository: Path,
    stage8_manifest_path: Path,
    checkpoint_manifest_path: Path,
    run_id: str,
    checkpoint_step: int,
) -> tuple[dict[str, Any], str]:
    """Validate Stage 8 provenance for one requested search checkpoint."""

    record = _load_json_object(
        stage8_manifest_path,
        "Stage 8 manifest",
    )
    manifest_sha256 = file_sha256(stage8_manifest_path)

    if record.get("validation_status") != "passed":
        raise ValueError(
            "Stage 8 manifest validation_status must be passed."
        )

    if record.get("source_training_run_id") != run_id:
        raise ValueError(
            "Stage 8 manifest source training run does not match."
        )

    component_definitions = record.get("component_definitions")

    if not isinstance(component_definitions, dict):
        raise ValueError(
            "Stage 8 manifest component_definitions must be an object."
        )

    if (
        component_definitions.get("searchable_component_count")
        != SEARCHABLE_COMPONENT_COUNT
    ):
        raise ValueError(
            "Stage 8 searchable component count does not match Stage 9."
        )

    attention = component_definitions.get("attention_heads")
    neurons = component_definitions.get("mlp_neurons")

    if not isinstance(attention, dict) or not isinstance(neurons, dict):
        raise ValueError(
            "Stage 8 component hook definitions are missing."
        )

    if attention.get("hook_name") != ATTENTION_HEAD_HOOK_NAME:
        raise ValueError(
            "Stage 8 attention-head hook does not match Stage 9."
        )

    if neurons.get("hook_name") != MLP_NEURON_HOOK_NAME:
        raise ValueError(
            "Stage 8 MLP-neuron hook does not match Stage 9."
        )

    sources = record.get("source_manifests")

    if not isinstance(sources, dict):
        raise ValueError(
            "Stage 8 source_manifests must be an object."
        )

    checkpoint_source = sources.get("checkpoint_selection")

    if not isinstance(checkpoint_source, dict):
        raise ValueError(
            "Stage 8 checkpoint-selection source is missing."
        )

    recorded_checkpoint_path = checkpoint_source.get("path")
    recorded_checkpoint_sha256 = checkpoint_source.get("sha256")

    if not isinstance(recorded_checkpoint_path, str):
        raise ValueError(
            "Stage 8 checkpoint-manifest path is invalid."
        )

    expected_checkpoint_path = _resolve(
        repository,
        recorded_checkpoint_path,
    ).resolve()

    if expected_checkpoint_path != checkpoint_manifest_path.resolve():
        raise ValueError(
            "Stage 8 checkpoint-manifest path does not match the "
            "requested checkpoint manifest."
        )

    actual_checkpoint_manifest_sha256 = file_sha256(
        checkpoint_manifest_path
    )

    if (
        recorded_checkpoint_sha256
        != actual_checkpoint_manifest_sha256
    ):
        raise ValueError(
            "Stage 8 checkpoint-manifest hash does not match."
        )

    selected = record.get("selected_checkpoints")

    if not isinstance(selected, list):
        raise ValueError(
            "Stage 8 selected_checkpoints must be a list."
        )

    matches = [
        item
        for item in selected
        if isinstance(item, dict)
        and item.get("training_step") == checkpoint_step
    ]

    if len(matches) != 1:
        raise ValueError(
            "Requested checkpoint step must appear exactly once in the "
            "Stage 8 selected checkpoint grid."
        )

    return record, manifest_sha256


def build_cell_metadata(
    *,
    repository: Path,
    context: CheckpointEvaluationContext,
    stage8_manifest_path: Path,
    stage8_manifest_sha256: str,
    implementation_git_commit: str,
    training_git_commit: str,
    git_status_at_start: str,
    execution: CheckpointSearchExecution,
    fidelity_threshold: float,
    ranking_batch_size: int,
    evaluation_batch_size: int,
    exact_evaluation_budget: int,
    method_development: bool,
) -> dict[str, Any]:
    """Build deterministic scientific metadata for one search cell."""

    calibration_eligibility = (
        CALIBRATION_ELIGIBLE
        if context.checkpoint_phase == STABLE_POST_PHASE
        else CALIBRATION_EXCLUDED
    )

    return {
        "schema_version": 1,
        "experiment_stage": 9,
        "source_training_run_id": context.run_id,
        "checkpoint_phase": context.checkpoint_phase,
        "checkpoint_step": context.checkpoint_step,
        "checkpoint_path": _relative(
            repository,
            context.checkpoint_path,
        ),
        "checkpoint_sha256": context.checkpoint_sha256,
        "model_state_sha256": context.model_state_sha256,
        "checkpoint_manifest_path": _relative(
            repository,
            context.checkpoint_manifest_path,
        ),
        "checkpoint_manifest_sha256": (
            context.checkpoint_manifest_sha256
        ),
        "stage8_manifest_path": _relative(
            repository,
            stage8_manifest_path,
        ),
        "stage8_manifest_sha256": stage8_manifest_sha256,
        "training_provenance": {
            "manifest_path": _relative(
                repository,
                context.training_manifest_path,
            ),
            "manifest_sha256": (
                context.training_manifest_sha256
            ),
            "training_git_commit": training_git_commit,
        },
        "config_provenance": {
            "task_config_sha256": context.task_config_sha256,
            "model_config_sha256": context.model_config_sha256,
            "training_config_sha256": (
                context.training_config_sha256
            ),
            "combined_config_sha256": (
                context.combined_config_sha256
            ),
        },
        "dataset_provenance": {
            "dataset_sha256": context.dataset_sha256,
            "split_sha256": context.split_sha256,
            "dataset_archive_sha256": (
                context.dataset_archive_sha256
            ),
            "dataset_metadata_sha256": (
                context.dataset_metadata_sha256
            ),
        },
        "implementation_git_commit": implementation_git_commit,
        "implementation_provenance_policy": (
            "method_development_dirty_tree_permitted"
            if method_development
            else (
                "clean_before_orchestrated_run_then_exact_head_"
                "required"
            )
        ),
        "git_status_recorded_in_deterministic_metadata": False,
        "fidelity_threshold": float(fidelity_threshold),
        "ranking_batch_size": ranking_batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "exact_evaluation_budget": exact_evaluation_budget,
        "threshold_calibration_eligibility": (
            calibration_eligibility
        ),
        "primary_fidelity_threshold_selected": False,
        "method_development": method_development,
        "scientific_output_eligible": not method_development,
        "development_label": (
            METHOD_DEVELOPMENT_LABEL
            if method_development
            else None
        ),
        "evaluated_example_count": int(context.inputs.shape[0]),
        "example_ordering": context.example_ordering,
        "device": str(context.device),
        "pseudo_targets": {
            "sha256": execution.pseudo_target_sha256,
            "count": execution.pseudo_target_count,
        },
        "full_model_reference": {
            "sha256": (
                execution.full_model_reference_sha256
            ),
            "example_count": (
                execution.full_model_reference_example_count
            ),
            "inference_batch_size": (
                execution.full_model_reference_batch_size
            ),
        },
        "search_integrity": {
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
            "hook_counts_before": dict(
                execution.hook_counts_before
            ),
            "hook_counts_after": dict(
                execution.hook_counts_after
            ),
            "hook_counts_unchanged": (
                execution.hook_counts_before
                == execution.hook_counts_after
            ),
            "parameter_gradients_cleared": True,
        },
    }


def write_runtime_telemetry(
    path: Path,
    *,
    elapsed_runtime_seconds: float,
    execution: CheckpointSearchExecution,
    artifacts: SparseSearchArtifacts,
    method_development: bool,
    implementation_git_commit: str,
    git_status_at_start: str,
) -> Path:
    """Write nondeterministic runtime telemetry outside scientific hashes."""

    if not math.isfinite(elapsed_runtime_seconds):
        raise ValueError("elapsed runtime must be finite.")

    if elapsed_runtime_seconds < 0.0:
        raise ValueError("elapsed runtime must be non-negative.")

    record = {
        "schema_version": 1,
        "nondeterministic_runtime_telemetry": True,
        "excluded_from_deterministic_scientific_hashes": True,
        "method_development": method_development,
        "implementation_git_commit": (
            implementation_git_commit
        ),
        "git_status_at_start": (
            git_status_at_start or "clean"
        ),
        "development_label": (
            METHOD_DEVELOPMENT_LABEL
            if method_development
            else None
        ),
        "elapsed_runtime_seconds": elapsed_runtime_seconds,
        "search_status": execution.result.status,
        "accepted_removals": len(
            execution.result.accepted_removals
        ),
        "exact_evaluations_used": (
            execution.result.exact_evaluations_used
        ),
        "ranking_passes_used": (
            execution.result.ranking_passes_used
        ),
        "final_retained_component_count": (
            execution.result.final_mask.retained_component_count
        ),
        "final_exact_fidelity": (
            execution.result.final_metrics.primary_fidelity
        ),
        "scientific_artifact_hashes": {
            "final_mask_sha256": artifacts.final_mask_sha256,
            "accepted_removal_trajectory_sha256": (
                artifacts.accepted_removal_trajectory_sha256
            ),
            "candidate_evaluation_log_sha256": (
                artifacts.candidate_evaluation_log_sha256
            ),
            "cell_summary_sha256": (
                artifacts.cell_summary_sha256
            ),
            "hashes_sha256": artifacts.hashes_sha256,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            record,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def execute_single_cell(
    *,
    repository_root: str | Path,
    run_id: str,
    checkpoint_manifest_path: str | Path,
    stage8_manifest_path: str | Path,
    checkpoint_step: int,
    fidelity_threshold: float,
    ranking_batch_size: int,
    evaluation_batch_size: int,
    exact_evaluation_budget: int,
    device: str,
    output_directory: str | Path,
    method_development: bool,
    expected_implementation_commit: str | None = None,
) -> SingleCellOutput:
    """Validate provenance, run one cell, and write its artifacts."""

    repository = Path(repository_root).resolve()
    checkpoint_manifest = _resolve(
        repository,
        checkpoint_manifest_path,
    )
    stage8_manifest = _resolve(
        repository,
        stage8_manifest_path,
    )
    output = _resolve(repository, output_directory)

    implementation_commit, git_status = repository_provenance(
        repository,
        require_clean=(
            not method_development
            and expected_implementation_commit is None
        ),
        expected_commit=expected_implementation_commit,
    )

    _, stage8_manifest_sha256 = validate_stage8_manifest(
        repository=repository,
        stage8_manifest_path=stage8_manifest,
        checkpoint_manifest_path=checkpoint_manifest,
        run_id=run_id,
        checkpoint_step=checkpoint_step,
    )

    context = load_checkpoint_evaluation_context(
        repository_root=repository,
        run_id=run_id,
        checkpoint_manifest_path=checkpoint_manifest,
        checkpoint_step=checkpoint_step,
        device_override=device,
    )

    stage8_record = _load_json_object(
        stage8_manifest,
        "Stage 8 manifest",
    )
    stage8_checkpoint = next(
        item
        for item in stage8_record["selected_checkpoints"]
        if item["training_step"] == checkpoint_step
    )

    if (
        stage8_checkpoint.get("checkpoint_sha256")
        != context.checkpoint_sha256
    ):
        raise ValueError(
            "Stage 8 checkpoint hash does not match the validated "
            "checkpoint context."
        )

    if (
        stage8_checkpoint.get("model_state_sha256")
        != context.model_state_sha256
    ):
        raise ValueError(
            "Stage 8 model-state hash does not match the validated "
            "checkpoint context."
        )

    started = time.perf_counter()

    execution = run_checkpoint_sparse_search(
        context,
        fidelity_threshold=fidelity_threshold,
        ranking_batch_size=ranking_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        exact_evaluation_budget=exact_evaluation_budget,
    )

    elapsed = time.perf_counter() - started

    training_manifest_record = _load_json_object(
        context.training_manifest_path,
        "Training manifest",
    )
    training_git_commit = training_manifest_record.get(
        "git_commit"
    )

    if not isinstance(training_git_commit, str):
        raise ValueError(
            "Training manifest git_commit must be a string."
        )

    metadata = build_cell_metadata(
        repository=repository,
        context=context,
        stage8_manifest_path=stage8_manifest,
        stage8_manifest_sha256=stage8_manifest_sha256,
        implementation_git_commit=implementation_commit,
        training_git_commit=training_git_commit,
        git_status_at_start=git_status,
        execution=execution,
        fidelity_threshold=fidelity_threshold,
        ranking_batch_size=ranking_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        exact_evaluation_budget=exact_evaluation_budget,
        method_development=method_development,
    )

    artifacts = write_sparse_search_artifacts(
        output,
        execution.result,
        cell_metadata=metadata,
    )

    telemetry_path = write_runtime_telemetry(
        output / "runtime_telemetry.json",
        elapsed_runtime_seconds=elapsed,
        execution=execution,
        artifacts=artifacts,
        method_development=method_development,
        implementation_git_commit=implementation_commit,
        git_status_at_start=git_status,
    )

    return SingleCellOutput(
        context=context,
        execution=execution,
        artifacts=artifacts,
        runtime_telemetry_path=telemetry_path,
        runtime_telemetry_sha256=file_sha256(
            telemetry_path
        ),
        elapsed_runtime_seconds=elapsed,
        implementation_git_commit=implementation_commit,
        git_status_at_start=git_status,
        stage8_manifest_sha256=stage8_manifest_sha256,
    )


def main() -> None:
    """Execute one requested Stage 9 cell."""

    args = parse_args()

    output = execute_single_cell(
        repository_root=args.repository_root,
        run_id=args.run_id,
        checkpoint_manifest_path=args.checkpoint_manifest,
        stage8_manifest_path=args.stage8_manifest,
        checkpoint_step=args.checkpoint_step,
        fidelity_threshold=args.fidelity_threshold,
        ranking_batch_size=args.ranking_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        exact_evaluation_budget=args.exact_evaluation_budget,
        device=args.device,
        output_directory=args.output_directory,
        method_development=args.method_development,
        expected_implementation_commit=(
            args.expected_implementation_commit
        ),
    )

    result = output.execution.result

    print(f"checkpoint_step: {output.context.checkpoint_step}")
    print(f"checkpoint_phase: {output.context.checkpoint_phase}")
    print(f"fidelity_threshold: {result.fidelity_threshold}")
    print(f"search_status: {result.status}")
    print(
        "retained_components: "
        f"{result.final_mask.retained_component_count}"
    )
    print(f"accepted_removals: {len(result.accepted_removals)}")
    print(
        "exact_evaluations_used: "
        f"{result.exact_evaluations_used}"
    )
    print(f"ranking_passes_used: {result.ranking_passes_used}")
    print(
        "final_exact_fidelity: "
        f"{result.final_metrics.primary_fidelity:.12f}"
    )
    print(
        "elapsed_runtime_seconds: "
        f"{output.elapsed_runtime_seconds:.6f}"
    )
    print(
        "cell_summary: "
        f"{_relative(Path(args.repository_root), output.artifacts.cell_summary_path)}"
    )
    print(
        "final_mask_sha256: "
        f"{output.artifacts.final_mask_sha256}"
    )
    print(
        "trajectory_sha256: "
        f"{output.artifacts.accepted_removal_trajectory_sha256}"
    )


if __name__ == "__main__":
    main()

"""Run the frozen Stage 10 Fourier pipeline diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from circuit_families.analysis.fourier_sanity_check import (
    STABLE_POST_CHECKPOINT_STEP,
    deterministic_stage10_run_id,
    load_stable_post_stage9_circuits,
    run_stage10_analysis,
    stage10_configuration_record,
    stage10_output_paths,
    write_stage10_artifacts,
)
from circuit_families.interpretability.fidelity import (
    load_checkpoint_evaluation_context,
)
from circuit_families.training import file_sha256


def parse_args() -> argparse.Namespace:
    """Parse Stage 10 execution arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the stable-post Stage 10 Fourier sanity-check pipeline."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage9-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage9-table",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage9-archive",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
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
        "--expected-implementation-commit",
    )
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
    )
    return parser.parse_args()


def git_output(
    repository: Path,
    *arguments: str,
) -> str:
    """Return stripped Git command output."""

    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def resolve_path(
    repository: Path,
    value: Path,
) -> Path:
    """Resolve a command-line path against the repository."""

    return value if value.is_absolute() else repository / value


def require_clean_repository(repository: Path) -> str:
    """Require a clean implementation commit before scientific output."""

    status = git_output(repository, "status", "--short")

    if status:
        raise RuntimeError(
            "Stage 10 scientific outputs require a clean implementation "
            "commit. Current status:\n" + status
        )

    return git_output(repository, "rev-parse", "HEAD")


def main() -> None:
    """Validate inputs or execute the complete frozen Stage 10 analysis."""

    args = parse_args()
    repository = args.repository_root.resolve()

    checkpoint_manifest = resolve_path(
        repository,
        args.checkpoint_manifest,
    )
    stage9_manifest = resolve_path(
        repository,
        args.stage9_manifest,
    )
    stage9_table = resolve_path(
        repository,
        args.stage9_table,
    )
    stage9_archive = resolve_path(
        repository,
        args.stage9_archive,
    )

    circuits = load_stable_post_stage9_circuits(
        stage9_manifest_path=stage9_manifest,
        stage9_table_path=stage9_table,
        stage9_archive_path=stage9_archive,
    )

    head = git_output(repository, "rev-parse", "HEAD")

    if (
        args.expected_implementation_commit is not None
        and head != args.expected_implementation_commit
    ):
        raise RuntimeError(
            "Current HEAD does not match the expected Stage 10 "
            "implementation commit."
        )

    checkpoint_record = json.loads(
        checkpoint_manifest.read_text(encoding="utf-8")
    )
    stable = checkpoint_record[
        "selected_stable_post_checkpoint"
    ]

    if stable["training_step"] != STABLE_POST_CHECKPOINT_STEP:
        raise ValueError(
            "Checkpoint manifest stable-post step changed."
        )

    configuration = stage10_configuration_record(
        source_training_run_id=args.run_id,
        checkpoint_sha256=stable["checkpoint_sha256"],
        stage9_manifest_sha256=file_sha256(stage9_manifest),
        stage9_table_sha256=file_sha256(stage9_table),
        stage9_archive_sha256=file_sha256(stage9_archive),
        implementation_git_commit=head,
        device=args.device,
        batch_size=args.batch_size,
    )
    stage10_run_id = deterministic_stage10_run_id(configuration)
    paths = stage10_output_paths(
        repository,
        stage10_run_id=stage10_run_id,
    )

    print("===== STAGE 10 FOURIER SANITY CHECK =====")
    print(f"stage10_run_id: {stage10_run_id}")
    print(f"source_training_run_id: {args.run_id}")
    print(f"checkpoint_step: {STABLE_POST_CHECKPOINT_STEP}")
    print(f"implementation_git_commit: {head}")
    print(f"device: {args.device}")
    print(f"batch_size: {args.batch_size}")
    print(f"stable_post_circuit_count: {len(circuits)}")
    print(
        "stable_post_thresholds: "
        + ", ".join(
            str(circuit.fidelity_threshold)
            for circuit in circuits
        )
    )
    print(f"component_table: {paths.component_table}")
    print(f"circuit_table: {paths.circuit_table}")
    print(f"removal_table: {paths.removal_table}")
    print(f"manifest: {paths.manifest}")

    if args.validate_inputs_only:
        print("validation_status: passed")
        print("scientific_outputs_generated: false")
        return

    clean_head = require_clean_repository(repository)

    if clean_head != head:
        raise RuntimeError(
            "Repository HEAD changed during Stage 10 validation."
        )

    context = load_checkpoint_evaluation_context(
        repository_root=repository,
        run_id=args.run_id,
        checkpoint_manifest_path=checkpoint_manifest,
        checkpoint_step=STABLE_POST_CHECKPOINT_STEP,
        device_override=args.device,
    )

    result = run_stage10_analysis(
        model=context.model,
        inputs=context.inputs,
        targets=context.targets,
        circuits=circuits,
        batch_size=args.batch_size,
    )

    artifacts = write_stage10_artifacts(
        repository=repository,
        stage10_run_id=stage10_run_id,
        configuration=configuration,
        context=context,
        circuits=circuits,
        result=result,
    )

    from circuit_families.analysis.fourier_sanity_check import (
        classify_fourier_diagnostic,
    )

    print(
        "full_model_fourier_classification: "
        + classify_fourier_diagnostic(
            result.full_diagnostics
        )
    )
    print(
        "component_association_count: "
        f"{len(result.component_execution.records)}"
    )
    print(
        "circuit_evaluation_count: "
        f"{len(result.circuit_evaluations)}"
    )
    print(
        "selected_removal_evaluation_count: "
        f"{len(result.removal_evaluations)}"
    )
    print(f"manifest_sha256: {artifacts.manifest_sha256}")
    print("validation_status: passed")
    print("scientific_outputs_generated: true")
    print("primary_fidelity_threshold_selected: false")
    print("stage11_started: false")


if __name__ == "__main__":
    main()

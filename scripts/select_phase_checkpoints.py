"""Select deterministic Stage 7 phase checkpoints for one training run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from circuit_families.analysis.phase_detection import (
    build_phase_manifest,
    select_phase_checkpoints,
    validate_phase_inputs,
    write_phase_manifest,
    write_phase_table,
)
from circuit_families.training.checkpoints import file_sha256
from circuit_families.training.logging import read_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Apply frozen Stage 7 phase-selection rules."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    return parser.parse_args()


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def main() -> None:
    """Load one run, validate it, and print Stage 7 selections."""

    args = parse_args()
    repository = args.repository_root.resolve()

    manifest_path = (
        repository
        / "manifests"
        / f"training_{args.run_id}.json"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Training manifest does not exist: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.get("run_id") != args.run_id:
        raise ValueError(
            "Training manifest run ID does not match --run-id."
        )

    metrics_path = _resolve(
        repository,
        manifest["output_paths"]["metrics_jsonl"],
    )
    checkpoint_directory = _resolve(
        repository,
        manifest["output_paths"]["checkpoint_directory"],
    )

    records = read_jsonl(metrics_path)

    execution = manifest["execution"]

    validate_phase_inputs(
        records,
        checkpoint_directory,
        expected_run_id=args.run_id,
        max_steps=execution["max_steps"],
        evaluation_interval=execution["evaluation_interval"],
        include_step_zero=execution["evaluate_step_zero"],
    )

    result = select_phase_checkpoints(records)

    seed = int(manifest["seed"]["value"])

    phase_table_path = (
        repository
        / "results"
        / "tables"
        / f"seed_{seed}_phase_checkpoints.csv"
    )

    write_phase_table(
        result,
        phase_table_path,
    )

    phase_manifest_path = (
        repository
        / "manifests"
        / f"checkpoints_seed_{seed}.json"
    )

    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    phase_selection_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    manifest = build_phase_manifest(
        result=result,
        run_id=args.run_id,
        training_manifest_path=str(
            manifest_path.relative_to(repository)
        ),
        metrics_path=str(
            metrics_path.relative_to(repository)
        ),
        metrics_sha256=file_sha256(metrics_path),
        phase_table_path=str(
            phase_table_path.relative_to(repository)
        ),
        phase_table_sha256=file_sha256(phase_table_path),
        training_git_commit=manifest["git_commit"],
        phase_selection_git_commit=phase_selection_commit,
        phase_selection_git_status=(
            "clean"
            if not git_status
            else "dirty"
        ),
        creation_timestamp_utc=__import__(
            "datetime"
        ).datetime.now(
            __import__("datetime").UTC
        ).isoformat(),
    )

    write_phase_manifest(
        manifest,
        phase_manifest_path,
    )

    print("===== STAGE 7 PHASE SELECTION =====")
    print(f"phase_table: {phase_table_path}")
    print(f"phase_manifest: {phase_manifest_path}")
    print(f"run_id: {args.run_id}")
    print(f"metrics_path: {metrics_path}")
    print(f"checkpoint_directory: {checkpoint_directory}")
    print(
        "has_valid_pre_checkpoint: "
        f"{str(result.has_valid_pre_checkpoint).lower()}"
    )
    print(
        "pre_checkpoint_status: "
        f"{result.pre_checkpoint_status}"
    )
    print(
        "incomplete_grid: "
        f"{str(result.incomplete_grid).lower()}"
    )

    if result.pre_checkpoint is None:
        print("pre_checkpoint_step: none")
    else:
        print(
            "pre_checkpoint_step: "
            f"{result.pre_checkpoint.training_step}"
        )
        print(
            "pre_checkpoint_test_accuracy: "
            f"{result.pre_checkpoint.test_accuracy}"
        )

    print(
        "stable_post_sequence: "
        + ",".join(
            str(step)
            for step in result.stable_post_sequence
        )
    )

    if result.stable_post_checkpoint is None:
        print("stable_post_checkpoint_step: none")
    else:
        print(
            "stable_post_checkpoint_step: "
            f"{result.stable_post_checkpoint.training_step}"
        )
        print(
            "stable_post_checkpoint_test_accuracy: "
            f"{result.stable_post_checkpoint.test_accuracy}"
        )

    print("formal_landmarks:")
    if result.formal_landmarks:
        for label, checkpoint in result.formal_landmarks.items():
            print(
                f"  {label}: "
                f"step={checkpoint.training_step}, "
                f"test_accuracy={checkpoint.test_accuracy}"
            )
    else:
        print("  none")

    print("descriptive_landmarks:")
    if result.descriptive_landmarks:
        for label, checkpoint in result.descriptive_landmarks.items():
            print(
                f"  {label}: "
                f"step={checkpoint.training_step}, "
                f"test_accuracy={checkpoint.test_accuracy}, "
                "status=descriptive_only_missing_pre"
            )
    else:
        print("  none")


if __name__ == "__main__":
    main()

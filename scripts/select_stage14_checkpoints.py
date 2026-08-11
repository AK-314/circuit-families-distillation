"""Classify Stage 14 and match its seven checkpoints by exact step."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from circuit_families.analysis.random_label_control import (
    MAIN_MODEL_REFERENCE_CHECKPOINTS,
    build_exact_checkpoint_matches,
    classify_random_label_control,
    validate_main_checkpoint_reference,
)
from circuit_families.training.checkpoints import (
    load_checkpoint_payload,
)
from circuit_families.training.logging import read_jsonl
from circuit_families.training.random_label import (
    CHECKPOINT_INTERVAL,
    FINAL_STEP,
    MODEL_SEED,
    RANDOM_LABEL_SEED,
    RANDOM_LABELS_SHA256,
    file_sha256,
)

DEFAULT_MAIN_CHECKPOINT_MANIFEST = Path("manifests/checkpoints_seed_1.json")
DEFAULT_METRICS_TABLE = Path("results/tables/seed_0_stage14_random_label_training_metrics.csv")
DEFAULT_CHECKPOINT_TABLE = Path("results/tables/seed_0_stage14_random_label_checkpoints.csv")
DEFAULT_CHECKPOINT_MANIFEST = Path("manifests/stage14_random_label_checkpoints.json")

EXPECTED_MAIN_CHECKPOINT_MANIFEST_SHA256 = (
    "a33b856e07b9223d6ee552810d8b4eb6f20ae2eb74427934933ef022ba48fd23"
)

METRICS_COLUMNS = (
    "stage14_run_id",
    "training_step",
    "train_accuracy",
    "test_accuracy",
    "train_cross_entropy",
    "test_cross_entropy",
    "gradient_norm",
)

CHECKPOINT_COLUMNS = (
    "stage14_run_id",
    "main_model_reference_phase_label",
    "phase_label_scope",
    "requested_step",
    "selected_random_label_step",
    "absolute_step_mismatch",
    "checkpoint_path",
    "checkpoint_sha256",
    "model_state_sha256",
    "optimizer_state_sha256",
    "reload_verified",
    "train_accuracy",
    "test_accuracy",
    "train_cross_entropy",
    "test_cross_entropy",
)


def parse_args() -> argparse.Namespace:
    """Parse Stage 14 checkpoint-selection arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Classify one completed Stage 14 run and match its "
            "seven checkpoints by exact training step."
        )
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--main-checkpoint-manifest",
        type=Path,
        default=DEFAULT_MAIN_CHECKPOINT_MANIFEST,
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--metrics-table",
        type=Path,
        default=DEFAULT_METRICS_TABLE,
    )
    parser.add_argument(
        "--checkpoint-table",
        type=Path,
        default=DEFAULT_CHECKPOINT_TABLE,
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=DEFAULT_CHECKPOINT_MANIFEST,
    )
    parser.add_argument(
        "--expected-implementation-commit",
    )
    return parser.parse_args()


def resolve_path(root: Path, file_path: Path) -> Path:
    """Resolve one root-relative path."""

    return (file_path if file_path.is_absolute() else root / file_path).resolve()


def display_path(file_path: Path, root: Path) -> str:
    """Return a stable root-relative path where possible."""

    try:
        return str(file_path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(file_path.resolve())


def require_clean_repository(repository: Path) -> str:
    """Require unchanged code while permitting one training manifest."""

    unstaged = subprocess.run(
        ["/usr/bin/git", "diff", "--quiet"],
        cwd=repository,
        check=False,
    ).returncode
    staged = subprocess.run(
        ["/usr/bin/git", "diff", "--cached", "--quiet"],
        cwd=repository,
        check=False,
    ).returncode

    if unstaged or staged:
        raise RuntimeError("Tracked implementation files changed after training.")

    status = subprocess.run(
        [
            "/usr/bin/git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    allowed_training_manifests = [
        line
        for line in status
        if (
            line.startswith("?? ")
            and line[3:].startswith("manifests/training_stage14-random-label-training-s0-")
            and line[3:].endswith(".json")
        )
    ]

    forbidden = [line for line in status if line not in allowed_training_manifests]

    if forbidden:
        raise RuntimeError(
            "Unexpected repository changes are present before "
            "Stage 14 checkpoint selection:\n" + "\n".join(forbidden)
        )

    if len(allowed_training_manifests) > 1:
        raise RuntimeError("More than one untracked Stage 14 training manifest exists.")

    head = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return head.stdout.strip()


def _finite_float(value: Any, description: str) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite.")

    return result


def normalise_metric_record(
    record: dict[str, Any],
    *,
    stage14_run_id: str,
) -> dict[str, Any]:
    """Flatten one generic training metrics record."""

    if "training_step" in record:
        step = int(record["training_step"])
    elif "step" in record:
        step = int(record["step"])
    else:
        raise ValueError("Metric record is missing its training step.")

    train = record.get("train")
    test = record.get("test")

    if isinstance(train, dict) and isinstance(test, dict):
        train_accuracy = train["accuracy"]
        test_accuracy = test["accuracy"]
        train_cross_entropy = train["cross_entropy"]
        test_cross_entropy = test["cross_entropy"]
    else:
        train_accuracy = record["train_accuracy"]
        test_accuracy = record["test_accuracy"]
        train_cross_entropy = record.get(
            "train_cross_entropy",
            record.get("train_loss"),
        )
        test_cross_entropy = record.get(
            "test_cross_entropy",
            record.get("test_loss"),
        )

    if train_cross_entropy is None or test_cross_entropy is None:
        raise ValueError("Metric record is missing cross-entropy values.")

    gradient_norm = record.get("gradient_norm")

    if gradient_norm is not None:
        gradient_norm = _finite_float(
            gradient_norm,
            "gradient norm",
        )

    return {
        "stage14_run_id": stage14_run_id,
        "training_step": step,
        "train_accuracy": _finite_float(
            train_accuracy,
            "training accuracy",
        ),
        "test_accuracy": _finite_float(
            test_accuracy,
            "test accuracy",
        ),
        "train_cross_entropy": _finite_float(
            train_cross_entropy,
            "training cross-entropy",
        ),
        "test_cross_entropy": _finite_float(
            test_cross_entropy,
            "test cross-entropy",
        ),
        "gradient_norm": gradient_norm,
    }


def validate_metric_trajectory(
    records: list[dict[str, Any]],
) -> None:
    """Validate the complete Stage 14 evaluation trajectory."""

    expected_steps = list(range(0, FINAL_STEP + 1, CHECKPOINT_INTERVAL))
    observed_steps = [int(record["training_step"]) for record in records]

    if observed_steps != expected_steps:
        raise ValueError("Stage 14 metric steps differ from the frozen trajectory.")

    if len(records) != 182:
        raise ValueError("Stage 14 must contain exactly 182 metric records.")


def validate_training_manifest(
    manifest: dict[str, Any],
) -> None:
    """Validate the completed random-label training manifest."""

    if manifest.get("experiment_type") != ("stage14-random-label-training"):
        raise ValueError("Unexpected Stage 14 experiment type.")

    if manifest.get("mode") != "full":
        raise ValueError("Stage 14 manifest is not a full run.")

    if manifest.get("seed") != {
        "name": "model_seed",
        "value": MODEL_SEED,
    }:
        raise ValueError("Stage 14 model seed is not 0.")

    execution = manifest.get("execution")

    if not isinstance(execution, dict):
        raise ValueError("Stage 14 execution record is missing.")

    expected_execution = {
        "max_steps": FINAL_STEP,
        "evaluation_interval": CHECKPOINT_INTERVAL,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "evaluate_step_zero": True,
        "checkpoint_step_zero": True,
        "checkpoint_final_step": True,
    }

    for name, expected in expected_execution.items():
        if execution.get(name) != expected:
            raise ValueError(f"Stage 14 execution mismatch for {name}.")

    identity = manifest.get("run_identity")

    if not isinstance(identity, dict):
        raise ValueError("Stage 14 run identity is missing.")

    expected_identity = {
        "control_type": "random_label",
        "random_labels": True,
        "true_labels": False,
        "random_label_seed": RANDOM_LABEL_SEED,
        "model_seed": MODEL_SEED,
        "final_step": FINAL_STEP,
    }

    for name, expected in expected_identity.items():
        if identity.get(name) != expected:
            raise ValueError(f"Stage 14 run identity mismatch for {name}.")

    final_metrics = manifest.get("final_metrics")

    if not isinstance(final_metrics, dict):
        raise ValueError("Stage 14 final metrics are missing.")

    if int(final_metrics.get("training_step", -1)) != FINAL_STEP:
        raise ValueError("Stage 14 final metric step is not 9050.")

    checkpoints = manifest.get("checkpoints")

    if not isinstance(checkpoints, list):
        raise ValueError("Stage 14 checkpoint records are missing.")

    expected_steps = list(range(0, FINAL_STEP + 1, CHECKPOINT_INTERVAL))
    observed_steps = [int(record["training_step"]) for record in checkpoints]

    if len(checkpoints) != 182:
        raise ValueError("Stage 14 must contain exactly 182 checkpoints.")

    if observed_steps != expected_steps:
        raise ValueError("Stage 14 checkpoint steps differ from the frozen schedule.")

    if not all(record.get("reload_verified") is True for record in checkpoints):
        raise ValueError("Every Stage 14 checkpoint must be reload-verified.")


def verify_selected_checkpoint_files(
    *,
    training_output_root: Path,
    checkpoint_records: list[dict[str, Any]],
) -> None:
    """Verify files and payload hashes for seven matched steps."""

    matched_steps = {step for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS}

    for record in checkpoint_records:
        step = int(record["training_step"])

        if step not in matched_steps:
            continue

        checkpoint_path = resolve_path(
            training_output_root,
            Path(record["path"]),
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        if file_sha256(checkpoint_path) != record["file_sha256"]:
            raise ValueError(f"Checkpoint file SHA-256 mismatch at step {step}.")

        payload = load_checkpoint_payload(
            checkpoint_path,
            map_location="cpu",
        )

        if int(payload["training_step"]) != step:
            raise ValueError(f"Checkpoint payload step mismatch at step {step}.")

        if int(payload["model_seed"]) != MODEL_SEED:
            raise ValueError(f"Checkpoint model seed mismatch at step {step}.")

        if payload["dataset_hashes"].get("random_labels_sha256") != RANDOM_LABELS_SHA256:
            raise ValueError(f"Random-label hash mismatch at step {step}.")

        if payload["model_state_sha256"] != record["model_state_sha256"]:
            raise ValueError(f"Model-state hash mismatch at step {step}.")

        if payload["optimizer_state_sha256"] != record["optimizer_state_sha256"]:
            raise ValueError(f"Optimizer-state hash mismatch at step {step}.")


def write_csv(
    file_path: Path,
    *,
    columns: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> Path:
    """Write one deterministic CSV table."""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})

    return file_path


def write_json(
    file_path: Path,
    payload: dict[str, Any],
) -> Path:
    """Write one deterministic JSON object."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return file_path


def main() -> None:
    """Build deterministic Stage 14 selection outputs."""

    args = parse_args()
    repository = args.repository_root.resolve()
    output_root = resolve_path(repository, args.output_root)

    implementation_commit = require_clean_repository(repository)

    if (
        args.expected_implementation_commit is not None
        and implementation_commit != args.expected_implementation_commit
    ):
        raise RuntimeError(
            "Implementation commit mismatch: expected "
            f"{args.expected_implementation_commit}, found "
            f"{implementation_commit}."
        )

    training_manifest_path = resolve_path(
        repository,
        args.training_manifest,
    )
    main_manifest_path = resolve_path(
        repository,
        args.main_checkpoint_manifest,
    )

    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    main_manifest = json.loads(main_manifest_path.read_text(encoding="utf-8"))

    validate_training_manifest(training_manifest)
    validate_main_checkpoint_reference(main_manifest)

    if file_sha256(main_manifest_path) != EXPECTED_MAIN_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("Seed-1 checkpoint manifest SHA-256 mismatch.")

    run_id = str(training_manifest["run_id"])
    training_output_root = training_manifest_path.parent.parent

    metrics_path = resolve_path(
        training_output_root,
        Path(training_manifest["output_paths"]["metrics_jsonl"]),
    )

    metric_rows = [
        normalise_metric_record(
            record,
            stage14_run_id=run_id,
        )
        for record in read_jsonl(metrics_path)
    ]
    validate_metric_trajectory(metric_rows)

    checkpoint_records = training_manifest["checkpoints"]
    verify_selected_checkpoint_files(
        training_output_root=training_output_root,
        checkpoint_records=checkpoint_records,
    )

    matched_rows = build_exact_checkpoint_matches(
        checkpoint_records=checkpoint_records,
        metric_records=metric_rows,
    )

    checkpoint_by_step = {int(record["training_step"]): record for record in checkpoint_records}

    for row in matched_rows:
        step = int(row["requested_step"])
        row["stage14_run_id"] = run_id
        row["reload_verified"] = checkpoint_by_step[step]["reload_verified"]

    final_row = metric_rows[-1]
    classification = classify_random_label_control(
        final_training_accuracy=final_row["train_accuracy"],
        metric_records=metric_rows,
    )

    metrics_table_path = resolve_path(
        output_root,
        args.metrics_table,
    )
    checkpoint_table_path = resolve_path(
        output_root,
        args.checkpoint_table,
    )
    checkpoint_manifest_path = resolve_path(
        output_root,
        args.checkpoint_manifest,
    )

    write_csv(
        metrics_table_path,
        columns=METRICS_COLUMNS,
        rows=metric_rows,
    )
    write_csv(
        checkpoint_table_path,
        columns=CHECKPOINT_COLUMNS,
        rows=matched_rows,
    )

    manifest_payload = {
        "schema_version": 1,
        "stage": 14,
        "stage14_run_id": run_id,
        "implementation_commit": implementation_commit,
        "source_training_manifest": {
            "path": display_path(
                training_manifest_path,
                output_root,
            ),
            "sha256": file_sha256(training_manifest_path),
        },
        "source_metrics": {
            "path": display_path(metrics_path, output_root),
            "sha256": file_sha256(metrics_path),
            "record_count": len(metric_rows),
        },
        "main_model_checkpoint_reference": {
            "path": display_path(
                main_manifest_path,
                repository,
            ),
            "sha256": file_sha256(main_manifest_path),
            "phase_labels_apply_to": "main_model_only",
        },
        "matching_rule": "exact_training_step",
        "expected_absolute_step_mismatch": 0,
        "matched_checkpoint_count": len(matched_rows),
        "matched_checkpoints": matched_rows,
        "classification": {
            "criterion": ("final_training_accuracy_at_least_0.99"),
            "threshold": classification.memorisation_threshold,
            "final_training_accuracy": (classification.final_training_accuracy),
            "first_step_reaching_99_percent": (classification.first_step_reaching_99_percent),
            "reached_99_percent_by_step_9050": (classification.reached_99_percent_by_step_9050),
            "label": classification.classification,
        },
        "outputs": {
            "training_metrics_table": {
                "path": display_path(
                    metrics_table_path,
                    output_root,
                ),
                "sha256": file_sha256(metrics_table_path),
            },
            "checkpoint_table": {
                "path": display_path(
                    checkpoint_table_path,
                    output_root,
                ),
                "sha256": file_sha256(checkpoint_table_path),
            },
        },
        "random_label_sparse_search_started": False,
        "diversity_search_started": False,
        "stage15_started": False,
    }

    write_json(
        checkpoint_manifest_path,
        manifest_payload,
    )

    print(f"stage14_run_id: {run_id}")
    print(f"implementation_commit: {implementation_commit}")
    print(f"training_metric_records: {len(metric_rows)}")
    print(f"saved_checkpoints: {len(checkpoint_records)}")
    print(f"matched_checkpoint_records: {len(matched_rows)}")
    print("matched_steps: " + ", ".join(str(row["requested_step"]) for row in matched_rows))
    print(
        "absolute_step_mismatches: "
        + ", ".join(str(row["absolute_step_mismatch"]) for row in matched_rows)
    )
    print(f"final_training_accuracy: {classification.final_training_accuracy}")
    print(f"first_step_reaching_99_percent: {classification.first_step_reaching_99_percent}")
    print(f"control_classification: {classification.classification}")
    print(f"metrics_table: {metrics_table_path}")
    print(f"checkpoint_table: {checkpoint_table_path}")
    print(f"checkpoint_manifest: {checkpoint_manifest_path}")
    print("random_label_sparse_search_started: false")
    print("diversity_search_started: false")
    print("stage15_started: false")
    print("stage14_checkpoint_selection: passed")


if __name__ == "__main__":
    main()

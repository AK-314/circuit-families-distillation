"""Validate a completed Stage 6 training run and its figure source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from circuit_families.config import (
    load_config,
    load_model_config,
    load_training_config,
)
from circuit_families.models.transformer import (
    EXPECTED_PARAMETER_COUNT,
    build_transformer,
)
from circuit_families.plotting.training_curves import (
    validate_figure_source_csv,
)
from circuit_families.training.checkpoints import (
    file_sha256,
    load_checkpoint_payload,
    reload_and_reevaluate,
)
from circuit_families.training.data import load_training_data
from circuit_families.training.logging import read_jsonl
from circuit_families.training.stage6_validation import (
    compute_grokking_diagnostics,
    expected_event_steps,
    validate_metric_records,
)
from circuit_families.training.trainer import build_optimizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Validate a completed Stage 6 training run."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--figure-source-csv",
        type=Path,
        default=Path("results/tables/seed_0_training_metrics.csv"),
    )
    return parser.parse_args()


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _require_equal(
    actual: Any,
    expected: Any,
    message: str,
) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{message}: expected {expected!r}, received {actual!r}."
        )


def main() -> None:
    """Execute structural, trajectory, and selected-reload validation."""

    args = parse_args()
    repository = args.repository_root.resolve()

    manifest_path = (
        repository
        / "manifests"
        / f"training_{args.run_id}.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _require_equal(
        manifest["run_id"],
        args.run_id,
        "Manifest run ID mismatch",
    )

    metrics_path = _resolve(
        repository,
        manifest["output_paths"]["metrics_jsonl"],
    )
    checkpoint_directory = _resolve(
        repository,
        manifest["output_paths"]["checkpoint_directory"],
    )
    recorded_manifest_path = _resolve(
        repository,
        manifest["output_paths"]["training_manifest"],
    )

    for required_path in (
        metrics_path,
        checkpoint_directory,
        recorded_manifest_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Manifest output path does not exist: {required_path}"
            )

    records = read_jsonl(metrics_path)
    execution = manifest["execution"]

    validate_metric_records(
        records,
        max_steps=execution["max_steps"],
        evaluation_interval=execution["evaluation_interval"],
        include_step_zero=execution["evaluate_step_zero"],
    )

    expected_checkpoint_steps = expected_event_steps(
        max_steps=execution["max_steps"],
        interval=execution["checkpoint_interval"],
        include_step_zero=execution["checkpoint_step_zero"],
        include_final=execution["checkpoint_final_step"],
    )

    actual_checkpoint_paths = sorted(
        checkpoint_directory.glob("step_*.pt")
    )
    actual_checkpoint_steps = [
        int(path.stem.removeprefix("step_"))
        for path in actual_checkpoint_paths
    ]

    _require_equal(
        actual_checkpoint_steps,
        expected_checkpoint_steps,
        "Checkpoint schedule mismatch",
    )

    manifest_checkpoint_steps = [
        checkpoint["training_step"]
        for checkpoint in manifest["checkpoints"]
    ]

    _require_equal(
        manifest_checkpoint_steps,
        expected_checkpoint_steps,
        "Manifest checkpoint schedule mismatch",
    )

    if not all(
        checkpoint["reload_verified"]
        for checkpoint in manifest["checkpoints"]
    ):
        raise RuntimeError(
            "Not every manifest checkpoint is marked reload-verified."
        )

    for checkpoint in manifest["checkpoints"]:
        checkpoint_path = _resolve(repository, checkpoint["path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Manifest checkpoint path does not exist: {checkpoint_path}"
            )

    metrics_sha256 = file_sha256(metrics_path)
    _require_equal(
        metrics_sha256,
        manifest["hashes"]["metrics_jsonl_sha256"],
        "Metrics hash mismatch",
    )

    figure_source_csv = _resolve(
        repository,
        args.figure_source_csv,
    )
    validate_figure_source_csv(records, figure_source_csv)
    figure_source_sha256 = file_sha256(figure_source_csv)

    diagnostics = compute_grokking_diagnostics(records)

    if not diagnostics.met_frozen_criteria:
        raise RuntimeError(
            "The run does not meet the frozen Stage 6 grokking criteria."
        )

    task_config = load_config(
        _resolve(repository, manifest["configs"]["task"]["path"])
    )
    model_config = load_model_config(
        _resolve(repository, manifest["configs"]["model"]["path"])
    )
    training_config = load_training_config(
        _resolve(repository, manifest["configs"]["training"]["path"])
    )

    device = torch.device(manifest["device"]["selected_device"])

    data = load_training_data(
        archive_path=_resolve(
            repository,
            manifest["dataset"]["archive_path"],
        ),
        metadata_path=_resolve(
            repository,
            manifest["dataset"]["metadata_path"],
        ),
        manifest_path=_resolve(
            repository,
            manifest["dataset"]["manifest_path"],
        ),
        task_config=task_config,
        device=device,
    )

    records_by_step = {
        int(record["training_step"]): record
        for record in records
    }

    if diagnostics.first_train_999_step is None:
        raise RuntimeError("Training memorisation was not reached.")

    post_memorisation_step = next(
        int(record["training_step"])
        for record in records
        if int(record["training_step"])
        > diagnostics.first_train_999_step
    )

    early_step = expected_checkpoint_steps[1]
    transition_diagnostic_step = diagnostics.first_test_50_step

    if transition_diagnostic_step is None:
        raise RuntimeError("Test accuracy never reached 50%.")

    selected_steps = sorted(
        {
            0,
            early_step,
            post_memorisation_step,
            transition_diagnostic_step,
            execution["max_steps"],
        }
    )

    validation = training_config["validation"]

    print("===== SELECTED CHECKPOINT RELOADS =====")

    for step in selected_steps:
        checkpoint_path = (
            checkpoint_directory / f"step_{step:08d}.pt"
        )
        payload = load_checkpoint_payload(
            checkpoint_path,
            map_location=device,
        )
        record = records_by_step[step]
        checkpoint_metrics = payload["metrics"]

        _require_equal(
            checkpoint_metrics["train"]["cross_entropy"],
            record["train_loss"],
            f"Checkpoint train loss mismatch at step {step}",
        )
        _require_equal(
            checkpoint_metrics["train"]["accuracy"],
            record["train_accuracy"],
            f"Checkpoint train accuracy mismatch at step {step}",
        )
        _require_equal(
            checkpoint_metrics["test"]["cross_entropy"],
            record["test_loss"],
            f"Checkpoint test loss mismatch at step {step}",
        )
        _require_equal(
            checkpoint_metrics["test"]["accuracy"],
            record["test_accuracy"],
            f"Checkpoint test accuracy mismatch at step {step}",
        )

        model = build_transformer(
            model_config,
            seed=manifest["seed"]["value"],
            device=device,
        )
        optimizer = build_optimizer(model, training_config)

        reload_and_reevaluate(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
            train_inputs=data.train_inputs,
            train_targets=data.train_targets,
            test_inputs=data.test_inputs,
            test_targets=data.test_targets,
            absolute_tolerance=validation[
                "reload_metric_absolute_tolerance"
            ],
            relative_tolerance=validation[
                "reload_metric_relative_tolerance"
            ],
        )

        print(f"step {step}: passed")

    final_record = records[-1]
    final_checkpoint_path = (
        checkpoint_directory
        / f"step_{execution['max_steps']:08d}.pt"
    )
    final_checkpoint_sha256 = file_sha256(final_checkpoint_path)

    print()
    print("===== STAGE 6 VALIDATION SUMMARY =====")
    print(f"run_id: {args.run_id}")
    print(f"git_commit: {manifest['git_commit']}")
    print(f"device: {manifest['device']['selected_device']}")
    print(f"model_parameter_count: {EXPECTED_PARAMETER_COUNT}")
    print(f"start_training_step: {records[0]['training_step']}")
    print(f"final_training_step: {final_record['training_step']}")
    print(
        "first_99.9_percent_training_accuracy_step: "
        f"{diagnostics.first_train_999_step}"
    )
    print(
        "representative_test_accuracy_after_memorisation: "
        f"{diagnostics.representative_low_test_accuracy}"
    )
    print(
        "representative_test_accuracy_step: "
        f"{diagnostics.representative_low_test_step}"
    )
    print(
        "first_10_percent_test_accuracy_step: "
        f"{diagnostics.first_test_10_step}"
    )
    print(
        "first_50_percent_test_accuracy_step: "
        f"{diagnostics.first_test_50_step}"
    )
    print(
        "first_90_percent_test_accuracy_step: "
        f"{diagnostics.first_test_90_step}"
    )
    print(
        "earliest_terminal_stable_99_percent_step: "
        f"{diagnostics.terminal_stable_99_start_step}"
    )
    print(
        "terminal_stable_99_percent_evaluations: "
        f"{diagnostics.terminal_stable_99_count}"
    )
    print(f"final_train_loss: {final_record['train_loss']}")
    print(
        f"final_train_accuracy: {final_record['train_accuracy']}"
    )
    print(f"final_test_loss: {final_record['test_loss']}")
    print(f"final_test_accuracy: {final_record['test_accuracy']}")
    print(f"checkpoint_count: {len(actual_checkpoint_paths)}")
    print("missing_checkpoint_intervals: 0")
    print("metrics_steps_ordered_and_unique: true")
    print("all_recorded_metrics_finite: true")
    print("manifest_paths_exist: true")
    print("selected_checkpoint_reloads: passed")
    print("figure_source_matches_metrics: true")
    print(
        "frozen_grokking_criteria_met: "
        f"{diagnostics.met_frozen_criteria}"
    )
    print(f"metrics_sha256: {metrics_sha256}")
    print(
        "final_checkpoint_sha256: "
        f"{final_checkpoint_sha256}"
    )
    print(
        "figure_source_csv_sha256: "
        f"{figure_source_sha256}"
    )


if __name__ == "__main__":
    main()

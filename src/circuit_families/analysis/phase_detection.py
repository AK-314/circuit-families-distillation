"""Deterministic Stage 7 checkpoint phase selection."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from circuit_families.manifests import package_versions
from circuit_families.training.checkpoints import file_sha256
from circuit_families.training.stage6_validation import (
    validate_metric_records,
)


@dataclass(frozen=True)
class PhaseCheckpoint:
    """One selected checkpoint."""

    label: str
    training_step: int
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float
    checkpoint_path: Path
    checkpoint_sha256: str
    run_id: str


@dataclass(frozen=True)
class PhaseSelectionResult:
    """Result of deterministic Stage 7 phase selection."""

    has_valid_pre_checkpoint: bool
    pre_checkpoint: PhaseCheckpoint | None
    stable_post_checkpoint: PhaseCheckpoint | None
    stable_post_sequence: tuple[int, ...]
    formal_landmarks: dict[str, PhaseCheckpoint]
    descriptive_landmarks: dict[str, PhaseCheckpoint]
    pre_checkpoint_status: str
    incomplete_grid: bool


def validate_phase_inputs(
    records: Sequence[Mapping[str, Any]],
    checkpoint_directory: Path,
    *,
    expected_run_id: str,
    max_steps: int,
    evaluation_interval: int,
    include_step_zero: bool,
) -> None:
    """Validate metrics, run identity, and checkpoint integrity."""

    validate_metric_records(
        records,
        max_steps=max_steps,
        evaluation_interval=evaluation_interval,
        include_step_zero=include_step_zero,
    )

    if int(records[-1]["training_step"]) != max_steps:
        raise ValueError(
            "Metrics final step conflicts with the training manifest."
        )

    for record in records:
        step = int(record["training_step"])

        if record["run_id"] != expected_run_id:
            raise ValueError(
                f"Inconsistent run ID at step {step}."
            )

        raw_checkpoint_path = record.get("checkpoint_path")

        if not isinstance(raw_checkpoint_path, str) or not raw_checkpoint_path:
            raise ValueError(
                "Every metrics record must contain a checkpoint_path."
            )

        checkpoint_path = Path(raw_checkpoint_path)
        expected_name = f"step_{step:08d}.pt"

        if checkpoint_path.name != expected_name:
            raise ValueError(
                "Checkpoint filename is inconsistent with training_step."
            )

        if checkpoint_path.parent.name != expected_run_id:
            raise ValueError(
                "Checkpoint path belongs to another run."
            )

        resolved_path = checkpoint_directory / expected_name

        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {resolved_path}"
            )

        recorded_sha256 = record.get("checkpoint_sha256")

        if not isinstance(recorded_sha256, str):
            raise ValueError(
                f"Missing checkpoint SHA-256 at step {step}."
            )

        actual_sha256 = file_sha256(resolved_path)

        if actual_sha256 != recorded_sha256:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch at step {step}."
            )


def find_stable_post_sequence(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """
    Return the earliest sequence of five consecutive saved checkpoints with
    test accuracy >= 99%, together with the selected fifth checkpoint.

    Raises
    ------
    ValueError
        If no qualifying five-checkpoint sequence exists.
    """

    if len(records) < 5:
        raise ValueError("At least five checkpoints are required.")

    for index in range(len(records) - 4):
        window = records[index : index + 5]

        if all(
            float(record["test_accuracy"]) >= 0.99
            for record in window
        ):
            return window, window[-1]

    raise ValueError(
        "No five-consecutive >=99% test-accuracy sequence exists."
    )


def find_pre_grokking_checkpoint(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """
    Return the frozen pre-grokking checkpoint.

    The selected checkpoint is the latest saved checkpoint satisfying:

        train_accuracy >= 99.9%
        test_accuracy <= 5%

    before the first saved checkpoint whose test accuracy reaches 10%.

    Returns None if no valid checkpoint exists.
    """

    first_ten_index = None

    for index, record in enumerate(records):
        if float(record["test_accuracy"]) >= 0.10:
            first_ten_index = index
            break

    if first_ten_index is None:
        search_records = records
    else:
        search_records = records[:first_ten_index]

    latest = None

    for record in search_records:
        if (
            float(record["train_accuracy"]) >= 0.999
            and float(record["test_accuracy"]) <= 0.05
        ):
            latest = record

    return latest


TRANSITION_TARGETS = {
    "10%": 0.10,
    "25%": 0.25,
    "50%": 0.50,
    "75%": 0.75,
    "90%": 0.90,
}


def select_transition_landmarks(
    records: Sequence[Mapping[str, Any]],
    *,
    pre_step: int,
    stable_post_step: int,
) -> dict[str, Mapping[str, Any]]:
    """
    Select nearest saved checkpoints for the frozen transition targets.

    Only checkpoints strictly after the selected pre-grokking checkpoint and
    strictly before the selected stable post-grokking checkpoint are eligible.
    Exact ties are resolved in favour of the earlier training step.
    """

    eligible = [
        record
        for record in records
        if pre_step
        < int(record["training_step"])
        < stable_post_step
    ]

    if not eligible:
        raise ValueError(
            "No checkpoints exist between pre and stable-post endpoints."
        )

    selections: dict[str, Mapping[str, Any]] = {}

    for label, target in TRANSITION_TARGETS.items():
        selections[label] = min(
            eligible,
            key=lambda record: (
                abs(float(record["test_accuracy"]) - target),
                int(record["training_step"]),
            ),
        )

    return selections


def select_phase_checkpoints(
    records: Sequence[Mapping[str, Any]],
) -> PhaseSelectionResult:
    """Apply the frozen Stage 7 phase-selection rules."""

    pre_record = find_pre_grokking_checkpoint(records)

    try:
        stable_sequence, stable_record = find_stable_post_sequence(records)
    except ValueError:
        stable_sequence = []
        stable_record = None

    if pre_record is None:
        descriptive: dict[str, PhaseCheckpoint] = {}

        if stable_record is not None:
            descriptive_records = select_transition_landmarks(
                records,
                pre_step=int(records[0]["training_step"]) - 1,
                stable_post_step=int(stable_record["training_step"]),
            )

            descriptive = {
                label: _to_phase_checkpoint(
                    label=label,
                    record=record,
                )
                for label, record in descriptive_records.items()
            }

        return PhaseSelectionResult(
            has_valid_pre_checkpoint=False,
            pre_checkpoint=None,
            stable_post_checkpoint=(
                None
                if stable_record is None
                else _to_phase_checkpoint(
                    label="stable post-grokking",
                    record=stable_record,
                )
            ),
            stable_post_sequence=tuple(
                int(record["training_step"])
                for record in stable_sequence
            ),
            formal_landmarks={},
            descriptive_landmarks=descriptive,
            pre_checkpoint_status="no_valid_checkpoint",
            incomplete_grid=True,
        )

    if stable_record is None:
        return PhaseSelectionResult(
            has_valid_pre_checkpoint=True,
            pre_checkpoint=_to_phase_checkpoint(
                label="pre-grokking",
                record=pre_record,
            ),
            stable_post_checkpoint=None,
            stable_post_sequence=(),
            formal_landmarks={},
            descriptive_landmarks={},
            pre_checkpoint_status="selected",
            incomplete_grid=True,
        )

    formal_records = select_transition_landmarks(
        records,
        pre_step=int(pre_record["training_step"]),
        stable_post_step=int(stable_record["training_step"]),
    )

    return PhaseSelectionResult(
        has_valid_pre_checkpoint=True,
        pre_checkpoint=_to_phase_checkpoint(
            label="pre-grokking",
            record=pre_record,
        ),
        stable_post_checkpoint=_to_phase_checkpoint(
            label="stable post-grokking",
            record=stable_record,
        ),
        stable_post_sequence=tuple(
            int(record["training_step"])
            for record in stable_sequence
        ),
        formal_landmarks={
            label: _to_phase_checkpoint(
                label=label,
                record=record,
            )
            for label, record in formal_records.items()
        },
        descriptive_landmarks={},
        pre_checkpoint_status="selected",
        incomplete_grid=False,
    )


def _to_phase_checkpoint(
    *,
    label: str,
    record: Mapping[str, Any],
) -> PhaseCheckpoint:
    """Convert one validated metrics record to a phase checkpoint."""

    return PhaseCheckpoint(
        label=label,
        training_step=int(record["training_step"]),
        train_accuracy=float(record["train_accuracy"]),
        test_accuracy=float(record["test_accuracy"]),
        train_loss=float(record["train_loss"]),
        test_loss=float(record["test_loss"]),
        checkpoint_path=Path(str(record["checkpoint_path"])),
        checkpoint_sha256=str(record["checkpoint_sha256"]),
        run_id=str(record["run_id"]),
    )


_PHASE_TABLE_COLUMNS = (
    "phase_label",
    "selection_status",
    "phase_status",
    "target_test_accuracy",
    "achieved_train_accuracy",
    "achieved_test_accuracy",
    "absolute_target_error",
    "train_loss",
    "test_loss",
    "training_step",
    "checkpoint_path",
    "checkpoint_sha256",
    "run_id",
)


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_phase_table(
    result: PhaseSelectionResult,
    output_path: str | Path,
) -> Path:
    """Write the deterministic Stage 7 phase table."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    def add_checkpoint(
        *,
        label: str,
        checkpoint: PhaseCheckpoint | None,
        selection_status: str,
        phase_status: str,
        target: float | None,
    ) -> None:
        if checkpoint is None:
            rows.append(
                {
                    "phase_label": label,
                    "selection_status": selection_status,
                    "phase_status": phase_status,
                    "target_test_accuracy": target,
                    "achieved_train_accuracy": None,
                    "achieved_test_accuracy": None,
                    "absolute_target_error": None,
                    "train_loss": None,
                    "test_loss": None,
                    "training_step": None,
                    "checkpoint_path": None,
                    "checkpoint_sha256": None,
                    "run_id": None,
                }
            )
            return

        rows.append(
            {
                "phase_label": label,
                "selection_status": selection_status,
                "phase_status": phase_status,
                "target_test_accuracy": target,
                "achieved_train_accuracy": checkpoint.train_accuracy,
                "achieved_test_accuracy": checkpoint.test_accuracy,
                "absolute_target_error": (
                    None
                    if target is None
                    else abs(checkpoint.test_accuracy - target)
                ),
                "train_loss": checkpoint.train_loss,
                "test_loss": checkpoint.test_loss,
                "training_step": checkpoint.training_step,
                "checkpoint_path": checkpoint.checkpoint_path,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "run_id": checkpoint.run_id,
            }
        )

    add_checkpoint(
        label="pre-grokking",
        checkpoint=result.pre_checkpoint,
        selection_status=result.pre_checkpoint_status,
        phase_status="formal",
        target=None,
    )

    for label in ("10%", "25%", "50%", "75%", "90%"):
        if label in result.formal_landmarks:
            add_checkpoint(
                label=label,
                checkpoint=result.formal_landmarks[label],
                selection_status="selected",
                phase_status="formal",
                target=TRANSITION_TARGETS[label],
            )
        elif label in result.descriptive_landmarks:
            add_checkpoint(
                label=label,
                checkpoint=result.descriptive_landmarks[label],
                selection_status="not_formally_selected",
                phase_status="descriptive_only_missing_pre",
                target=TRANSITION_TARGETS[label],
            )
        else:
            add_checkpoint(
                label=label,
                checkpoint=None,
                selection_status="not_formally_selected",
                phase_status="missing",
                target=TRANSITION_TARGETS[label],
            )

    add_checkpoint(
        label="stable post-grokking",
        checkpoint=result.stable_post_checkpoint,
        selection_status=(
            "selected"
            if result.stable_post_checkpoint is not None
            else "missing"
        ),
        phase_status="formal",
        target=None,
    )

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_PHASE_TABLE_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: _csv_value(row[key])
                    for key in _PHASE_TABLE_COLUMNS
                }
            )

    return output



def _checkpoint_manifest_record(
    checkpoint: PhaseCheckpoint,
    *,
    target_test_accuracy: float | None = None,
    selection_status: str = "selected",
) -> dict[str, object]:
    """Return one JSON-serialisable checkpoint provenance record."""

    return {
        "phase_label": checkpoint.label,
        "selection_status": selection_status,
        "target_test_accuracy": target_test_accuracy,
        "achieved_train_accuracy": checkpoint.train_accuracy,
        "achieved_test_accuracy": checkpoint.test_accuracy,
        "absolute_target_error": (
            None
            if target_test_accuracy is None
            else abs(
                checkpoint.test_accuracy
                - target_test_accuracy
            )
        ),
        "train_loss": checkpoint.train_loss,
        "test_loss": checkpoint.test_loss,
        "training_step": checkpoint.training_step,
        "checkpoint_path": str(checkpoint.checkpoint_path),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "run_id": checkpoint.run_id,
    }


def build_phase_manifest(
    *,
    result: PhaseSelectionResult,
    run_id: str,
    training_manifest_path: str,
    metrics_path: str,
    metrics_sha256: str,
    phase_table_path: str,
    phase_table_sha256: str,
    training_git_commit: str,
    phase_selection_git_commit: str,
    phase_selection_git_status: str,
    creation_timestamp_utc: str,
) -> dict[str, object]:
    """Build the complete Stage 7 checkpoint-selection manifest."""

    formal_records = {
        label: _checkpoint_manifest_record(
            checkpoint,
            target_test_accuracy=TRANSITION_TARGETS[label],
        )
        for label, checkpoint in result.formal_landmarks.items()
    }

    descriptive_records = {
        label: _checkpoint_manifest_record(
            checkpoint,
            target_test_accuracy=TRANSITION_TARGETS[label],
            selection_status="descriptive_only_missing_pre",
        )
        for label, checkpoint in result.descriptive_landmarks.items()
    }

    stable_post_record = (
        None
        if result.stable_post_checkpoint is None
        else _checkpoint_manifest_record(
            result.stable_post_checkpoint,
        )
    )

    pre_record = (
        None
        if result.pre_checkpoint is None
        else _checkpoint_manifest_record(
            result.pre_checkpoint,
        )
    )

    return {
        "schema_version": 1,
        "creation_timestamp_utc": creation_timestamp_utc,
        "run_id": run_id,
        "source_training_manifest": training_manifest_path,
        "source_metrics": {
            "path": metrics_path,
            "sha256": metrics_sha256,
        },
        "phase_table": {
            "path": phase_table_path,
            "sha256": phase_table_sha256,
        },
        "phase_selection_rules": {
            "pre_grokking": {
                "selection": "latest saved checkpoint",
                "minimum_train_accuracy": 0.999,
                "maximum_test_accuracy": 0.05,
                "must_precede_first_test_accuracy_at_or_above": 0.10,
                "missing_rule": "no substitution",
            },
            "stable_post_grokking": {
                "minimum_test_accuracy": 0.99,
                "required_consecutive_checkpoints": 5,
                "selection": (
                    "fifth checkpoint in earliest qualifying sequence"
                ),
            },
            "transition_landmarks": {
                "targets": TRANSITION_TARGETS,
                "eligible_interval": (
                    "strictly after selected pre checkpoint and strictly "
                    "before selected stable-post checkpoint"
                ),
                "selection": "nearest saved test accuracy",
                "tie_break": "earlier training step",
                "missing_pre_rule": (
                    "no formal transition landmarks; optional candidates "
                    "are descriptive_only_missing_pre"
                ),
            },
        },
        "preferred_grid": [
            "pre-grokking",
            "10%",
            "25%",
            "50%",
            "75%",
            "90%",
            "stable post-grokking",
        ],
        "prospective_reduced_grid_fallback": {
            "grid": [
                "pre-grokking",
                "50%",
                "stable post-grokking",
            ],
            "status": "pending_stage_12_compute_projection",
            "selected_for_scaled_seeds": False,
        },
        "has_valid_pre_checkpoint": (
            result.has_valid_pre_checkpoint
        ),
        "pre_checkpoint_status": result.pre_checkpoint_status,
        "pre_checkpoint_failure_reason": (
            None
            if result.has_valid_pre_checkpoint
            else (
                "No saved checkpoint before the first test-accuracy "
                "checkpoint at or above 10% simultaneously satisfied "
                "train accuracy >=99.9% and test accuracy <=5%."
            )
        ),
        "pre_checkpoint": pre_record,
        "earliest_stable_post_sequence_steps": list(
            result.stable_post_sequence
        ),
        "selected_stable_post_checkpoint": stable_post_record,
        "formal_transition_checkpoints": formal_records,
        "descriptive_transition_candidates": descriptive_records,
        "preferred_grid_status": (
            "incomplete"
            if result.incomplete_grid
            else "complete"
        ),
        "incomplete_grid": result.incomplete_grid,
        "pilot_eligibility_from_stage_7": (
            "ineligible_incomplete_grid"
            if result.incomplete_grid
            else "eligible_subject_to_frozen_grokking_criteria"
        ),
        "provenance": {
            "training_git_commit": training_git_commit,
            "phase_selection_git_commit": (
                phase_selection_git_commit
            ),
            "phase_selection_git_status": (
                phase_selection_git_status
            ),
        },
        "software": package_versions(
            (
                "numpy",
                "pandas",
                "PyYAML",
                "torch",
                "transformer-lens",
            )
        ),
    }


def write_phase_manifest(
    manifest: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Write the Stage 7 manifest."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return output

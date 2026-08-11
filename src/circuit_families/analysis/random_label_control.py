"""Mechanical Stage 14 classification and checkpoint matching."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from circuit_families.training.random_label import (
    FINAL_STEP,
    MEMORISATION_THRESHOLD,
)

MEMORISATION_CONTROL = "memorisation_control"
OPTIMISATION_ONLY_CONTROL = "optimisation_only_control"

MAIN_MODEL_REFERENCE_CHECKPOINTS = (
    ("pre-grokking", 200),
    ("10%", 3400),
    ("25%", 7450),
    ("50%", 8150),
    ("75%", 8500),
    ("90%", 8650),
    ("stable post-grokking", 9050),
)


@dataclass(frozen=True)
class RandomLabelControlClassification:
    """Mechanical classification of the completed Stage 14 run."""

    final_training_accuracy: float
    memorisation_threshold: float
    first_step_reaching_99_percent: int | None
    reached_99_percent_by_step_9050: bool
    classification: str


def _finite_accuracy(value: float) -> float:
    accuracy = float(value)

    if not math.isfinite(accuracy):
        raise ValueError("Training accuracy must be finite")

    if accuracy < 0.0 or accuracy > 1.0:
        raise ValueError("Training accuracy must lie in [0, 1]")

    return accuracy


def first_step_reaching_memorisation(
    metric_records: Sequence[Mapping[str, Any]],
) -> int | None:
    """Return the first saved step with training accuracy at least 99%."""

    qualifying_steps: list[int] = []

    for record in metric_records:
        step = int(record["training_step"])
        accuracy = _finite_accuracy(float(record["train_accuracy"]))

        if accuracy >= MEMORISATION_THRESHOLD:
            qualifying_steps.append(step)

    return min(qualifying_steps) if qualifying_steps else None


def classify_random_label_control(
    *,
    final_training_accuracy: float,
    metric_records: Sequence[Mapping[str, Any]],
    final_step: int = FINAL_STEP,
) -> RandomLabelControlClassification:
    """Apply the frozen 99% criterion without altering the trajectory."""

    if final_step != FINAL_STEP:
        raise ValueError(f"Stage 14 classification requires final step {FINAL_STEP}")

    final_accuracy = _finite_accuracy(final_training_accuracy)
    first_step = first_step_reaching_memorisation(metric_records)
    reached_by_horizon = first_step is not None and first_step <= FINAL_STEP

    classification = (
        MEMORISATION_CONTROL
        if final_accuracy >= MEMORISATION_THRESHOLD
        else OPTIMISATION_ONLY_CONTROL
    )

    return RandomLabelControlClassification(
        final_training_accuracy=final_accuracy,
        memorisation_threshold=MEMORISATION_THRESHOLD,
        first_step_reaching_99_percent=first_step,
        reached_99_percent_by_step_9050=reached_by_horizon,
        classification=classification,
    )


def _extract_main_reference_steps(
    checkpoint_manifest: Mapping[str, Any],
) -> dict[str, int]:
    formal = checkpoint_manifest["formal_transition_checkpoints"]

    return {
        "pre-grokking": int(checkpoint_manifest["pre_checkpoint"]["training_step"]),
        "10%": int(formal["10%"]["training_step"]),
        "25%": int(formal["25%"]["training_step"]),
        "50%": int(formal["50%"]["training_step"]),
        "75%": int(formal["75%"]["training_step"]),
        "90%": int(formal["90%"]["training_step"]),
        "stable post-grokking": int(
            checkpoint_manifest["selected_stable_post_checkpoint"]["training_step"]
        ),
    }


def validate_main_checkpoint_reference(
    checkpoint_manifest: Mapping[str, Any],
) -> None:
    """Verify the frozen seven-step seed-1 reference trajectory."""

    observed = _extract_main_reference_steps(checkpoint_manifest)
    expected = dict(MAIN_MODEL_REFERENCE_CHECKPOINTS)

    if observed != expected:
        raise ValueError(
            "Seed-1 checkpoint manifest differs from the frozen Stage 14 "
            f"reference: expected {expected}, got {observed}"
        )


def _record_step(record: Mapping[str, Any]) -> int:
    for key in ("training_step", "step", "checkpoint_step"):
        if key in record:
            return int(record[key])

    raise ValueError(f"Checkpoint record has no step field: {record}")


def _optional_value(
    record: Mapping[str, Any],
    *names: str,
) -> Any:
    for name in names:
        if name in record:
            return record[name]

    return None


def build_exact_checkpoint_matches(
    *,
    checkpoint_records: Sequence[Mapping[str, Any]],
    metric_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the exact seven-row training-step matching table."""

    checkpoints_by_step: dict[int, Mapping[str, Any]] = {}

    for record in checkpoint_records:
        step = _record_step(record)

        if step in checkpoints_by_step:
            raise ValueError(f"Duplicate checkpoint record for step {step}")

        checkpoints_by_step[step] = record

    metrics_by_step: dict[int, Mapping[str, Any]] = {}

    for record in metric_records:
        step = int(record["training_step"])

        if step in metrics_by_step:
            raise ValueError(f"Duplicate metric record for step {step}")

        metrics_by_step[step] = record

    rows: list[dict[str, Any]] = []

    for phase_label, requested_step in MAIN_MODEL_REFERENCE_CHECKPOINTS:
        if requested_step not in checkpoints_by_step:
            raise ValueError(f"Missing Stage 14 checkpoint at step {requested_step}")

        if requested_step not in metrics_by_step:
            raise ValueError(f"Missing Stage 14 metric record at step {requested_step}")

        checkpoint = checkpoints_by_step[requested_step]
        metrics = metrics_by_step[requested_step]

        rows.append(
            {
                "main_model_reference_phase_label": phase_label,
                "phase_label_scope": "main_model_reference_only",
                "requested_step": requested_step,
                "selected_random_label_step": requested_step,
                "absolute_step_mismatch": 0,
                "checkpoint_path": _optional_value(
                    checkpoint,
                    "checkpoint_path",
                    "path",
                ),
                "checkpoint_sha256": _optional_value(
                    checkpoint,
                    "checkpoint_sha256",
                    "file_sha256",
                    "sha256",
                ),
                "model_state_sha256": _optional_value(
                    checkpoint,
                    "model_state_sha256",
                ),
                "optimizer_state_sha256": _optional_value(
                    checkpoint,
                    "optimizer_state_sha256",
                    "optimiser_state_sha256",
                ),
                "train_accuracy": float(metrics["train_accuracy"]),
                "test_accuracy": float(metrics["test_accuracy"]),
                "train_cross_entropy": float(
                    _optional_value(
                        metrics,
                        "train_cross_entropy",
                        "train_loss",
                    )
                ),
                "test_cross_entropy": float(
                    _optional_value(
                        metrics,
                        "test_cross_entropy",
                        "test_loss",
                    )
                ),
            }
        )

    return rows

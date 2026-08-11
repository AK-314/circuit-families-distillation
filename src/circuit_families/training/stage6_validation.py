"""Stage 6 training-trajectory validation and descriptive diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GrokkingDiagnostics:
    """Descriptive evidence for the frozen Stage 6 grokking criteria."""

    first_train_999_step: int | None
    representative_low_test_accuracy: float | None
    representative_low_test_step: int | None
    below_10_after_memorisation_count: int
    below_10_after_memorisation_first_step: int | None
    below_10_after_memorisation_last_step: int | None
    first_test_10_step: int | None
    first_test_50_step: int | None
    first_test_90_step: int | None
    first_test_99_step: int | None
    terminal_stable_99_start_step: int | None
    terminal_stable_99_count: int
    met_frozen_criteria: bool


def expected_event_steps(
    *,
    max_steps: int,
    interval: int,
    include_step_zero: bool,
    include_final: bool,
) -> list[int]:
    """Return the exact expected event-step sequence."""

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative.")

    if interval <= 0:
        raise ValueError("interval must be positive.")

    steps: list[int] = []

    if include_step_zero:
        steps.append(0)

    for step in range(1, max_steps + 1):
        if step % interval == 0:
            steps.append(step)

    if include_final and max_steps not in steps:
        steps.append(max_steps)

    return steps


def validate_metric_records(
    records: Sequence[Mapping[str, Any]],
    *,
    max_steps: int,
    evaluation_interval: int,
    include_step_zero: bool,
) -> None:
    """Validate ordering, schedule, uniqueness, bounds, and finiteness."""

    if not records:
        raise ValueError("Metrics records must not be empty.")

    expected_steps = expected_event_steps(
        max_steps=max_steps,
        interval=evaluation_interval,
        include_step_zero=include_step_zero,
        include_final=True,
    )

    actual_steps: list[int] = []
    run_ids: set[str] = set()

    finite_fields = (
        "learning_rate",
        "weight_norm",
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
    )

    for record in records:
        step = record.get("training_step")

        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("Every training_step must be an integer.")

        actual_steps.append(step)

        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Every metrics record must have a run_id.")
        run_ids.add(run_id)

        for field in finite_fields:
            value = record.get(field)

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be numeric.")

            if not math.isfinite(float(value)):
                raise FloatingPointError(
                    f"{field} is non-finite at step {step}."
                )

        gradient_norm = record.get("gradient_norm")
        if gradient_norm is not None:
            if isinstance(gradient_norm, bool) or not isinstance(
                gradient_norm,
                (int, float),
            ):
                raise TypeError("gradient_norm must be numeric or null.")

            if not math.isfinite(float(gradient_norm)):
                raise FloatingPointError(
                    f"gradient_norm is non-finite at step {step}."
                )

        for accuracy_field in ("train_accuracy", "test_accuracy"):
            accuracy = float(record[accuracy_field])
            if not 0.0 <= accuracy <= 1.0:
                raise ValueError(
                    f"{accuracy_field} is outside [0, 1] at step {step}."
                )

    if len(run_ids) != 1:
        raise ValueError("Metrics records contain multiple run IDs.")

    if len(actual_steps) != len(set(actual_steps)):
        raise ValueError("Metrics records contain duplicate steps.")

    if actual_steps != sorted(actual_steps):
        raise ValueError("Metrics steps are not ordered.")

    if actual_steps != expected_steps:
        missing = sorted(set(expected_steps) - set(actual_steps))
        unexpected = sorted(set(actual_steps) - set(expected_steps))
        raise ValueError(
            "Metrics schedule does not match the expected schedule. "
            f"Missing={missing}; unexpected={unexpected}."
        )


def _first_step_at_or_above(
    records: Sequence[Mapping[str, Any]],
    field: str,
    threshold: float,
) -> int | None:
    for record in records:
        if float(record[field]) >= threshold:
            return int(record["training_step"])

    return None


def compute_grokking_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> GrokkingDiagnostics:
    """Compute descriptive Stage 6 grokking evidence.

    The terminal stable sequence is the final uninterrupted suffix with test
    accuracy at least 99%. Requiring more than one evaluation prevents the
    final checkpoint alone from being treated as evidence of stability.
    """

    if not records:
        raise ValueError("Metrics records must not be empty.")

    first_train_999_step = _first_step_at_or_above(
        records,
        "train_accuracy",
        0.999,
    )
    first_test_10_step = _first_step_at_or_above(
        records,
        "test_accuracy",
        0.10,
    )
    first_test_50_step = _first_step_at_or_above(
        records,
        "test_accuracy",
        0.50,
    )
    first_test_90_step = _first_step_at_or_above(
        records,
        "test_accuracy",
        0.90,
    )
    first_test_99_step = _first_step_at_or_above(
        records,
        "test_accuracy",
        0.99,
    )

    if first_train_999_step is None:
        post_memorisation: list[Mapping[str, Any]] = []
    else:
        post_memorisation = [
            record
            for record in records
            if int(record["training_step"]) >= first_train_999_step
        ]

    below_10 = [
        record
        for record in post_memorisation
        if float(record["test_accuracy"]) < 0.10
    ]

    representative_low = (
        min(
            post_memorisation,
            key=lambda record: float(record["test_accuracy"]),
        )
        if post_memorisation
        else None
    )

    terminal_stable_records: list[Mapping[str, Any]] = []

    for record in reversed(records):
        if float(record["test_accuracy"]) >= 0.99:
            terminal_stable_records.append(record)
        else:
            break

    terminal_stable_records.reverse()

    terminal_stable_start = (
        int(terminal_stable_records[0]["training_step"])
        if terminal_stable_records
        else None
    )

    delayed_period = len(below_10) >= 2
    later_clear_rise = (
        bool(below_10)
        and first_test_90_step is not None
        and first_test_90_step
        > int(below_10[-1]["training_step"])
    )
    stable_post_grokking = len(terminal_stable_records) >= 2

    met_frozen_criteria = (
        first_train_999_step is not None
        and delayed_period
        and later_clear_rise
        and stable_post_grokking
    )

    return GrokkingDiagnostics(
        first_train_999_step=first_train_999_step,
        representative_low_test_accuracy=(
            float(representative_low["test_accuracy"])
            if representative_low is not None
            else None
        ),
        representative_low_test_step=(
            int(representative_low["training_step"])
            if representative_low is not None
            else None
        ),
        below_10_after_memorisation_count=len(below_10),
        below_10_after_memorisation_first_step=(
            int(below_10[0]["training_step"]) if below_10 else None
        ),
        below_10_after_memorisation_last_step=(
            int(below_10[-1]["training_step"]) if below_10 else None
        ),
        first_test_10_step=first_test_10_step,
        first_test_50_step=first_test_50_step,
        first_test_90_step=first_test_90_step,
        first_test_99_step=first_test_99_step,
        terminal_stable_99_start_step=terminal_stable_start,
        terminal_stable_99_count=len(terminal_stable_records),
        met_frozen_criteria=met_frozen_criteria,
    )

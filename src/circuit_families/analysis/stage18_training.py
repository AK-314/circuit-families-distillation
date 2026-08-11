"""Frozen Stage 18 training classification and extension policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


STANDARD_HORIZON = 40_000
EXTENSION_INCREMENT = 10_000
ABSOLUTE_MAXIMUM = 80_000
METRIC_INTERVAL = 50
TRAIN_MEMORISATION_THRESHOLD = 0.999
LOW_TEST_THRESHOLD = 0.10
STABLE_TEST_THRESHOLD = 0.99
STABLE_SEQUENCE_LENGTH = 5


@dataclass(frozen=True)
class GrokkingClassification:
    status: str
    eligible: bool
    first_memorisation_step: int | None
    delayed_low_test_duration_steps: int | None
    first_ten_percent_test_step: int | None
    stable_sequence_start_step: int | None
    stable_post_step: int | None
    final_step: int


def _ordered(records: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    ordered = tuple(sorted(records, key=lambda row: int(row["training_step"])))
    steps = tuple(int(row["training_step"]) for row in ordered)
    if len(steps) != len(set(steps)):
        raise ValueError("Training metric steps must be unique.")
    if not ordered:
        raise ValueError("Training metrics must not be empty.")
    return ordered


def stable_post_sequence(
    records: Sequence[Mapping[str, Any]],
) -> tuple[int, int] | None:
    ordered = _ordered(records)
    for index in range(STABLE_SEQUENCE_LENGTH - 1, len(ordered)):
        window = ordered[index - STABLE_SEQUENCE_LENGTH + 1 : index + 1]
        steps = tuple(int(row["training_step"]) for row in window)
        if any(
            right - left != METRIC_INTERVAL for left, right in zip(steps, steps[1:], strict=False)
        ):
            continue
        if all(float(row["test_accuracy"]) >= STABLE_TEST_THRESHOLD for row in window):
            return steps[0], steps[-1]
    return None


def classify_grokking_run(
    records: Sequence[Mapping[str, Any]],
) -> GrokkingClassification:
    ordered = _ordered(records)
    final_step = int(ordered[-1]["training_step"])
    memorised = next(
        (
            int(row["training_step"])
            for row in ordered
            if float(row["train_accuracy"]) >= TRAIN_MEMORISATION_THRESHOLD
        ),
        None,
    )
    if memorised is None:
        return GrokkingClassification(
            "failed_to_memorise", False, None, None, None, None, None, final_step
        )
    later = [row for row in ordered if int(row["training_step"]) >= memorised]
    low = [row for row in later if float(row["test_accuracy"]) < LOW_TEST_THRESHOLD]
    first_ten = next(
        (
            int(row["training_step"])
            for row in later
            if float(row["test_accuracy"]) >= LOW_TEST_THRESHOLD
        ),
        None,
    )
    if not low or first_ten is None or min(int(row["training_step"]) for row in low) >= first_ten:
        return GrokkingClassification(
            "no_delayed_generalisation_interval",
            False,
            memorised,
            0,
            first_ten,
            None,
            None,
            final_step,
        )
    delayed_duration = first_ten - memorised
    stable = stable_post_sequence(ordered)
    if stable is None:
        status = (
            "failed_to_generalise_by_40000"
            if final_step == STANDARD_HORIZON
            else "failed_to_grok_by_80000"
            if final_step >= ABSOLUTE_MAXIMUM
            else "failed_to_generalise_by_40000"
        )
        return GrokkingClassification(
            status,
            False,
            memorised,
            delayed_duration,
            first_ten,
            None,
            None,
            final_step,
        )
    status = "complete_grokking_seed" if stable[1] <= STANDARD_HORIZON else "extended_and_grokked"
    return GrokkingClassification(
        status,
        True,
        memorised,
        delayed_duration,
        first_ten,
        stable[0],
        stable[1],
        final_step,
    )


def requires_extension(records: Sequence[Mapping[str, Any]]) -> bool:
    ordered = _ordered(records)
    final_step = int(ordered[-1]["training_step"])
    if final_step < STANDARD_HORIZON or final_step % EXTENSION_INCREMENT != 0:
        raise ValueError("Extension decisions are allowed only at frozen horizons.")
    return stable_post_sequence(ordered) is None and final_step < ABSOLUTE_MAXIMUM


def next_training_horizon(records: Sequence[Mapping[str, Any]]) -> int:
    ordered = _ordered(records)
    final_step = int(ordered[-1]["training_step"])
    return final_step + EXTENSION_INCREMENT if requires_extension(ordered) else final_step

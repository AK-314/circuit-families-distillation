"""Curve-only selection of the matched no-generalisation control."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

CANDIDATE_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25)
DESCENDING_CANDIDATE_FRACTIONS = tuple(
    reversed(CANDIDATE_FRACTIONS)
)

MATCHED_HORIZON = 9_050
EVALUATION_INTERVAL = 50
TRAIN_ACCURACY_THRESHOLD = Decimal("0.999")
TRAIN_ACCURACY_DEADLINE = 5_000
PERSISTENCE_THRESHOLD = Decimal("0.90")
TEST_ACCURACY_CEILING = Decimal("0.10")
TEST_ACCURACY_WINDOW_LIMIT = Decimal("0.02")
TEST_LOSS_FALL_LIMIT = Decimal("0.10")

FINAL_WINDOW_START_EXCLUSIVE = MATCHED_HORIZON - 1_000
PRECEDING_WINDOW_START_EXCLUSIVE = MATCHED_HORIZON - 2_000
PRECEDING_WINDOW_END_INCLUSIVE = MATCHED_HORIZON - 1_000
LOSS_COMPARISON_START_STEP = MATCHED_HORIZON - 5_000


@dataclass(frozen=True)
class CandidateQualification:
    """Mechanically evaluated qualification record for one candidate."""

    candidate_fraction: float
    training_example_count: int
    required_horizon: int
    selection_rank: int

    first_train_accuracy_step: int | None
    criterion1_reached_by_step_5000: bool

    persistence_numerator: int
    persistence_denominator: int
    persistence_proportion: float
    criterion2_persistence: bool

    maximum_test_accuracy: float
    maximum_test_accuracy_step: int
    criterion3_test_accuracy_ceiling: bool

    mean_test_accuracy_final_window: float
    mean_test_accuracy_preceding_window: float
    test_accuracy_window_difference: float
    criterion4_test_accuracy_plateau: bool

    test_cross_entropy_at_4050: float
    test_cross_entropy_at_9050: float
    test_cross_entropy_fractional_fall: float
    criterion5_test_loss_plateau: bool

    overall_qualification: bool

    def as_row(
        self,
        *,
        selected_fraction: float | None,
    ) -> dict[str, Any]:
        """Return the stable one-row selection-table representation."""

        return {
            "fraction": self.candidate_fraction,
            "exact_training_example_count": (
                self.training_example_count
            ),
            "required_horizon": self.required_horizon,
            "selection_rank": self.selection_rank,
            "first_train_accuracy_step": (
                self.first_train_accuracy_step
            ),
            "criterion1_reached_by_step_5000": (
                self.criterion1_reached_by_step_5000
            ),
            "persistence_numerator": self.persistence_numerator,
            "persistence_denominator": self.persistence_denominator,
            "persistence_proportion": self.persistence_proportion,
            "criterion2_persistence": self.criterion2_persistence,
            "maximum_test_accuracy": self.maximum_test_accuracy,
            "maximum_test_accuracy_step": (
                self.maximum_test_accuracy_step
            ),
            "criterion3_test_accuracy_ceiling": (
                self.criterion3_test_accuracy_ceiling
            ),
            "mean_test_accuracy_final_window": (
                self.mean_test_accuracy_final_window
            ),
            "mean_test_accuracy_preceding_window": (
                self.mean_test_accuracy_preceding_window
            ),
            "test_accuracy_window_difference": (
                self.test_accuracy_window_difference
            ),
            "criterion4_test_accuracy_plateau": (
                self.criterion4_test_accuracy_plateau
            ),
            "test_cross_entropy_at_4050": (
                self.test_cross_entropy_at_4050
            ),
            "test_cross_entropy_at_9050": (
                self.test_cross_entropy_at_9050
            ),
            "test_cross_entropy_fractional_fall": (
                self.test_cross_entropy_fractional_fall
            ),
            "criterion5_test_loss_plateau": (
                self.criterion5_test_loss_plateau
            ),
            "overall_qualification": self.overall_qualification,
            "selected_control": (
                selected_fraction == self.candidate_fraction
            ),
        }


def expected_evaluation_steps(
    required_horizon: int = MATCHED_HORIZON,
) -> tuple[int, ...]:
    """Return the complete frozen saved-evaluation grid."""

    _validate_required_horizon(required_horizon)
    return tuple(
        range(
            0,
            required_horizon + 1,
            EVALUATION_INTERVAL,
        )
    )


def final_accuracy_window_steps() -> tuple[int, ...]:
    """Return steps satisfying H - 1000 < step <= H."""

    return tuple(
        step
        for step in expected_evaluation_steps()
        if FINAL_WINDOW_START_EXCLUSIVE < step <= MATCHED_HORIZON
    )


def preceding_accuracy_window_steps() -> tuple[int, ...]:
    """Return steps satisfying H - 2000 < step <= H - 1000."""

    return tuple(
        step
        for step in expected_evaluation_steps()
        if (
            PRECEDING_WINDOW_START_EXCLUSIVE
            < step
            <= PRECEDING_WINDOW_END_INCLUSIVE
        )
    )


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _validate_required_horizon(required_horizon: int) -> None:
    if (
        isinstance(required_horizon, bool)
        or not isinstance(required_horizon, int)
    ):
        raise TypeError("required_horizon must be an integer.")

    if required_horizon != MATCHED_HORIZON:
        raise ValueError(
            "Stage 13 required_horizon must equal the frozen "
            f"matched horizon {MATCHED_HORIZON}."
        )


def _validate_candidate_fraction(candidate_fraction: float) -> None:
    if (
        isinstance(candidate_fraction, bool)
        or not isinstance(candidate_fraction, (int, float))
    ):
        raise TypeError("candidate_fraction must be numeric.")

    if float(candidate_fraction) not in CANDIDATE_FRACTIONS:
        raise ValueError(
            "candidate_fraction is outside the frozen Stage 13 grid."
        )


def _validate_example_count(training_example_count: int) -> None:
    if (
        isinstance(training_example_count, bool)
        or not isinstance(training_example_count, int)
    ):
        raise TypeError("training_example_count must be an integer.")

    if training_example_count <= 0:
        raise ValueError("training_example_count must be positive.")


def _validate_metric_value(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(f"{name} values must be numeric.")

    converted = float(value)

    if not math.isfinite(converted):
        raise ValueError(f"{name} values must be finite.")

    return converted


def _validate_accuracy(value: float, name: str) -> float:
    converted = _validate_metric_value(value, name)

    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} values must lie in [0, 1].")

    return converted


def _decimal_mean(values: Sequence[float]) -> Decimal:
    if not values:
        raise ValueError("Cannot calculate a mean from an empty window.")

    return sum((_decimal(value) for value in values), Decimal(0)) / len(
        values
    )


def evaluate_candidate(
    *,
    training_steps: Sequence[int],
    train_accuracy: Sequence[float],
    test_accuracy: Sequence[float],
    test_cross_entropy: Sequence[float],
    candidate_fraction: float,
    training_example_count: int,
    required_horizon: int,
) -> CandidateQualification:
    """Evaluate one candidate from training and test curves only."""

    _validate_candidate_fraction(candidate_fraction)
    _validate_example_count(training_example_count)
    _validate_required_horizon(required_horizon)

    lengths = {
        len(training_steps),
        len(train_accuracy),
        len(test_accuracy),
        len(test_cross_entropy),
    }
    if len(lengths) != 1:
        raise ValueError("All curve inputs must have equal lengths.")

    filtered: list[tuple[int, float, float, float]] = []

    for step, train_value, test_value, loss_value in zip(
        training_steps,
        train_accuracy,
        test_accuracy,
        test_cross_entropy,
        strict=True,
    ):
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("training_steps values must be integers.")

        if step > required_horizon:
            continue

        if step < 0:
            raise ValueError("training_steps values must be non-negative.")

        filtered.append(
            (
                step,
                _validate_accuracy(train_value, "train_accuracy"),
                _validate_accuracy(test_value, "test_accuracy"),
                _validate_metric_value(
                    loss_value,
                    "test_cross_entropy",
                ),
            )
        )

    steps = tuple(record[0] for record in filtered)
    expected_steps = expected_evaluation_steps(required_horizon)

    if steps != expected_steps:
        raise ValueError(
            "Saved evaluation checkpoints are incomplete, duplicated, "
            "unordered, or do not follow the frozen 50-step interval."
        )

    train_by_step = {
        step: value
        for step, value, _, _ in filtered
    }
    test_by_step = {
        step: value
        for step, _, value, _ in filtered
    }
    loss_by_step = {
        step: value
        for step, _, _, value in filtered
    }

    qualifying_train_steps = [
        step
        for step in expected_steps
        if (
            step <= TRAIN_ACCURACY_DEADLINE
            and _decimal(train_by_step[step])
            >= TRAIN_ACCURACY_THRESHOLD
        )
    ]

    first_train_accuracy_step = (
        qualifying_train_steps[0]
        if qualifying_train_steps
        else None
    )
    criterion1 = first_train_accuracy_step is not None

    if first_train_accuracy_step is None:
        persistence_steps: tuple[int, ...] = ()
        persistence_numerator = 0
        persistence_denominator = 0
        persistence_proportion_decimal = Decimal(0)
        criterion2 = False
    else:
        persistence_steps = tuple(
            step
            for step in expected_steps
            if step > first_train_accuracy_step
        )
        persistence_numerator = sum(
            _decimal(train_by_step[step])
            >= TRAIN_ACCURACY_THRESHOLD
            for step in persistence_steps
        )
        persistence_denominator = len(persistence_steps)

        if persistence_denominator == 0:
            persistence_proportion_decimal = Decimal(0)
            criterion2 = False
        else:
            persistence_proportion_decimal = (
                Decimal(persistence_numerator)
                / persistence_denominator
            )
            criterion2 = (
                persistence_proportion_decimal
                >= PERSISTENCE_THRESHOLD
            )

    maximum_test_accuracy_step = max(
        expected_steps,
        key=lambda step: test_by_step[step],
    )
    maximum_test_accuracy = test_by_step[
        maximum_test_accuracy_step
    ]
    criterion3 = (
        _decimal(maximum_test_accuracy)
        <= TEST_ACCURACY_CEILING
    )

    final_steps = final_accuracy_window_steps()
    preceding_steps = preceding_accuracy_window_steps()

    if (
        not final_steps
        or not preceding_steps
        or set(final_steps).intersection(preceding_steps)
        or len(final_steps) != len(preceding_steps)
    ):
        raise RuntimeError(
            "Frozen accuracy windows are invalid."
        )

    final_mean_decimal = _decimal_mean(
        [test_by_step[step] for step in final_steps]
    )
    preceding_mean_decimal = _decimal_mean(
        [test_by_step[step] for step in preceding_steps]
    )
    window_difference_decimal = (
        final_mean_decimal - preceding_mean_decimal
    )
    criterion4 = (
        window_difference_decimal
        <= TEST_ACCURACY_WINDOW_LIMIT
    )

    loss_start = loss_by_step[LOSS_COMPARISON_START_STEP]
    loss_end = loss_by_step[MATCHED_HORIZON]

    if loss_start <= 0.0:
        raise ValueError(
            "test cross-entropy at step 4050 must be positive."
        )

    loss_fall_decimal = (
        _decimal(loss_start) - _decimal(loss_end)
    ) / _decimal(loss_start)
    criterion5 = loss_fall_decimal <= TEST_LOSS_FALL_LIMIT

    criteria = (
        criterion1,
        criterion2,
        criterion3,
        criterion4,
        criterion5,
    )

    fraction = float(candidate_fraction)

    return CandidateQualification(
        candidate_fraction=fraction,
        training_example_count=training_example_count,
        required_horizon=required_horizon,
        selection_rank=(
            DESCENDING_CANDIDATE_FRACTIONS.index(fraction) + 1
        ),
        first_train_accuracy_step=first_train_accuracy_step,
        criterion1_reached_by_step_5000=criterion1,
        persistence_numerator=persistence_numerator,
        persistence_denominator=persistence_denominator,
        persistence_proportion=float(
            persistence_proportion_decimal
        ),
        criterion2_persistence=criterion2,
        maximum_test_accuracy=maximum_test_accuracy,
        maximum_test_accuracy_step=maximum_test_accuracy_step,
        criterion3_test_accuracy_ceiling=criterion3,
        mean_test_accuracy_final_window=float(
            final_mean_decimal
        ),
        mean_test_accuracy_preceding_window=float(
            preceding_mean_decimal
        ),
        test_accuracy_window_difference=float(
            window_difference_decimal
        ),
        criterion4_test_accuracy_plateau=criterion4,
        test_cross_entropy_at_4050=loss_start,
        test_cross_entropy_at_9050=loss_end,
        test_cross_entropy_fractional_fall=float(
            loss_fall_decimal
        ),
        criterion5_test_loss_plateau=criterion5,
        overall_qualification=all(criteria),
    )


def select_largest_qualifying_fraction(
    qualifications: Sequence[CandidateQualification],
) -> float | None:
    """Select the first qualifying candidate in descending order."""

    by_fraction: dict[float, CandidateQualification] = {}

    for qualification in qualifications:
        fraction = qualification.candidate_fraction

        if fraction not in CANDIDATE_FRACTIONS:
            raise ValueError(
                "Qualification contains a non-frozen fraction."
            )

        if fraction in by_fraction:
            raise ValueError(
                "Qualification fractions must be unique."
            )

        by_fraction[fraction] = qualification

    if set(by_fraction) != set(CANDIDATE_FRACTIONS):
        raise ValueError(
            "Selection requires exactly the five frozen candidates."
        )

    for fraction in DESCENDING_CANDIDATE_FRACTIONS:
        if by_fraction[fraction].overall_qualification:
            return fraction

    return None

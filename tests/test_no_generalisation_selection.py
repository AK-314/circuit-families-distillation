"""Tests for frozen Stage 13 curve-only control selection."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from circuit_families.analysis.no_generalisation_selection import (
    CANDIDATE_FRACTIONS,
    DESCENDING_CANDIDATE_FRACTIONS,
    MATCHED_HORIZON,
    evaluate_candidate,
    expected_evaluation_steps,
    final_accuracy_window_steps,
    preceding_accuracy_window_steps,
    select_largest_qualifying_fraction,
)


def _passing_curves() -> dict[str, list[int] | list[float]]:
    steps = list(expected_evaluation_steps())
    train_accuracy = [
        0.998 if step < 5_000 else 0.999
        for step in steps
    ]
    test_accuracy = [0.05 for _ in steps]
    test_cross_entropy = [5.0 for _ in steps]

    return {
        "training_steps": steps,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_cross_entropy": test_cross_entropy,
    }


def _evaluate(
    *,
    fraction: float = 0.25,
    count: int = 3_192,
    curves: dict[str, list[int] | list[float]] | None = None,
):
    selected = _passing_curves() if curves is None else curves

    return evaluate_candidate(
        training_steps=selected["training_steps"],
        train_accuracy=selected["train_accuracy"],
        test_accuracy=selected["test_accuracy"],
        test_cross_entropy=selected["test_cross_entropy"],
        candidate_fraction=fraction,
        training_example_count=count,
        required_horizon=MATCHED_HORIZON,
    )


def test_frozen_candidate_grid_is_exact() -> None:
    assert CANDIDATE_FRACTIONS == (
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
    )
    assert DESCENDING_CANDIDATE_FRACTIONS == (
        0.25,
        0.20,
        0.15,
        0.10,
        0.05,
    )


def test_matched_horizon_and_interval_are_exact() -> None:
    steps = expected_evaluation_steps()

    assert steps[0] == 0
    assert steps[-1] == 9_050
    assert all(
        right - left == 50
        for left, right in zip(steps, steps[1:], strict=False)
    )


def test_final_and_preceding_windows_are_disjoint_and_equal() -> None:
    final_steps = final_accuracy_window_steps()
    preceding_steps = preceding_accuracy_window_steps()

    assert final_steps == tuple(range(8_100, 9_051, 50))
    assert preceding_steps == tuple(range(7_100, 8_051, 50))
    assert len(final_steps) == 20
    assert len(preceding_steps) == 20
    assert set(final_steps).isdisjoint(preceding_steps)


def test_reaching_train_threshold_at_step_5000_passes() -> None:
    result = _evaluate()

    assert result.first_train_accuracy_step == 5_000
    assert result.criterion1_reached_by_step_5000


def test_first_reaching_train_threshold_at_step_5050_fails() -> None:
    curves = _passing_curves()
    curves["train_accuracy"] = [
        0.998 if step < 5_050 else 0.999
        for step in curves["training_steps"]
    ]

    result = _evaluate(curves=curves)

    assert result.first_train_accuracy_step is None
    assert not result.criterion1_reached_by_step_5000
    assert not result.criterion2_persistence


def test_persistence_exactly_090_passes() -> None:
    curves = _passing_curves()
    steps = curves["training_steps"]
    accuracies = curves["train_accuracy"]

    for index, step in enumerate(steps):
        accuracies[index] = 0.998 if step < 4_050 else 0.999

    subsequent_indices = [
        index
        for index, step in enumerate(steps)
        if step > 4_050
    ]
    assert len(subsequent_indices) == 100

    for index in subsequent_indices[-10:]:
        accuracies[index] = 0.998

    result = _evaluate(curves=curves)

    assert result.first_train_accuracy_step == 4_050
    assert result.persistence_numerator == 90
    assert result.persistence_denominator == 100
    assert result.persistence_proportion == 0.90
    assert result.criterion2_persistence


def test_persistence_below_090_fails() -> None:
    curves = _passing_curves()
    steps = curves["training_steps"]
    accuracies = curves["train_accuracy"]

    subsequent_indices = [
        index
        for index, step in enumerate(steps)
        if step > 5_000
    ]

    for index in subsequent_indices[-9:]:
        accuracies[index] = 0.998

    result = _evaluate(curves=curves)

    assert result.persistence_numerator == 72
    assert result.persistence_denominator == 81
    assert result.persistence_proportion < 0.90
    assert not result.criterion2_persistence


def test_test_accuracy_exactly_010_passes() -> None:
    curves = _passing_curves()
    curves["test_accuracy"][-1] = 0.10

    result = _evaluate(curves=curves)

    assert result.maximum_test_accuracy == 0.10
    assert result.criterion3_test_accuracy_ceiling


def test_test_accuracy_above_010_fails() -> None:
    curves = _passing_curves()
    curves["test_accuracy"][-1] = 0.1000001

    result = _evaluate(curves=curves)

    assert not result.criterion3_test_accuracy_ceiling


def test_test_accuracy_difference_exactly_002_passes() -> None:
    curves = _passing_curves()
    final_steps = set(final_accuracy_window_steps())

    curves["test_accuracy"] = [
        0.07 if step in final_steps else 0.05
        for step in curves["training_steps"]
    ]

    result = _evaluate(curves=curves)

    assert result.test_accuracy_window_difference == 0.02
    assert result.criterion4_test_accuracy_plateau


def test_test_accuracy_difference_above_002_fails() -> None:
    curves = _passing_curves()
    final_steps = set(final_accuracy_window_steps())

    curves["test_accuracy"] = [
        0.070001 if step in final_steps else 0.05
        for step in curves["training_steps"]
    ]

    result = _evaluate(curves=curves)

    assert not result.criterion4_test_accuracy_plateau


def test_decreasing_test_accuracy_passes_criterion4() -> None:
    curves = _passing_curves()
    final_steps = set(final_accuracy_window_steps())
    preceding_steps = set(preceding_accuracy_window_steps())

    curves["test_accuracy"] = [
        (
            0.03
            if step in final_steps
            else 0.05
            if step in preceding_steps
            else 0.04
        )
        for step in curves["training_steps"]
    ]

    result = _evaluate(curves=curves)

    assert result.test_accuracy_window_difference == -0.02
    assert result.criterion4_test_accuracy_plateau


def test_test_loss_fall_exactly_010_passes() -> None:
    curves = _passing_curves()
    steps = curves["training_steps"]
    losses = curves["test_cross_entropy"]

    losses[steps.index(4_050)] = 5.0
    losses[steps.index(9_050)] = 4.5

    result = _evaluate(curves=curves)

    assert result.test_cross_entropy_fractional_fall == 0.10
    assert result.criterion5_test_loss_plateau


def test_test_loss_fall_above_010_fails() -> None:
    curves = _passing_curves()
    steps = curves["training_steps"]
    losses = curves["test_cross_entropy"]

    losses[steps.index(4_050)] = 5.0
    losses[steps.index(9_050)] = 4.499

    result = _evaluate(curves=curves)

    assert not result.criterion5_test_loss_plateau


def test_increasing_test_loss_passes_criterion5() -> None:
    curves = _passing_curves()
    steps = curves["training_steps"]
    losses = curves["test_cross_entropy"]

    losses[steps.index(4_050)] = 5.0
    losses[steps.index(9_050)] = 6.0

    result = _evaluate(curves=curves)

    assert result.test_cross_entropy_fractional_fall == -0.2
    assert result.criterion5_test_loss_plateau


def test_non_positive_loss_denominator_fails() -> None:
    curves = _passing_curves()
    steps = curves["training_steps"]
    curves["test_cross_entropy"][steps.index(4_050)] = 0.0

    with pytest.raises(ValueError, match="4050 must be positive"):
        _evaluate(curves=curves)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_accuracy", math.nan),
        ("test_accuracy", math.inf),
        ("test_cross_entropy", -math.inf),
    ],
)
def test_non_finite_metrics_fail(field: str, value: float) -> None:
    curves = _passing_curves()
    curves[field][0] = value

    with pytest.raises(ValueError, match="finite"):
        _evaluate(curves=curves)


def test_missing_checkpoint_fails() -> None:
    curves = _passing_curves()

    for values in curves.values():
        values.pop(100)

    with pytest.raises(ValueError, match="checkpoints are incomplete"):
        _evaluate(curves=curves)


def test_duplicate_checkpoint_fails() -> None:
    curves = _passing_curves()
    curves["training_steps"][100] = curves["training_steps"][99]

    with pytest.raises(ValueError, match="checkpoints are incomplete"):
        _evaluate(curves=curves)


def test_metrics_beyond_horizon_are_ignored() -> None:
    curves = _passing_curves()

    curves["training_steps"].append(9_100)
    curves["train_accuracy"].append(math.nan)
    curves["test_accuracy"].append(math.nan)
    curves["test_cross_entropy"].append(math.nan)

    result = _evaluate(curves=curves)

    assert result.overall_qualification


def test_all_five_criteria_are_required() -> None:
    passing = _evaluate()
    fields = (
        "criterion1_reached_by_step_5000",
        "criterion2_persistence",
        "criterion3_test_accuracy_ceiling",
        "criterion4_test_accuracy_plateau",
        "criterion5_test_loss_plateau",
    )

    assert passing.overall_qualification
    assert all(getattr(passing, field) for field in fields)


def test_largest_qualifying_fraction_is_selected() -> None:
    qualifications = [
        _evaluate(fraction=0.05, count=638),
        _evaluate(fraction=0.10, count=1_276),
        _evaluate(fraction=0.15, count=1_915),
        _evaluate(fraction=0.20, count=2_553),
        _evaluate(fraction=0.25, count=3_192),
    ]

    qualifications[-1] = replace(
        qualifications[-1],
        overall_qualification=False,
    )

    assert select_largest_qualifying_fraction(
        qualifications
    ) == 0.20


def test_lower_candidate_cannot_replace_qualifying_larger_candidate() -> None:
    qualifications = [
        _evaluate(fraction=0.05, count=638),
        _evaluate(fraction=0.10, count=1_276),
        _evaluate(fraction=0.15, count=1_915),
        _evaluate(fraction=0.20, count=2_553),
        _evaluate(fraction=0.25, count=3_192),
    ]

    assert select_largest_qualifying_fraction(
        list(reversed(qualifications))
    ) == 0.25


def test_no_qualifying_fraction_is_explicit() -> None:
    qualifications = [
        replace(
            _evaluate(fraction=fraction, count=count),
            overall_qualification=False,
        )
        for fraction, count in (
            (0.05, 638),
            (0.10, 1_276),
            (0.15, 1_915),
            (0.20, 2_553),
            (0.25, 3_192),
        )
    ]

    assert select_largest_qualifying_fraction(
        qualifications
    ) is None


def test_selection_rejects_new_fraction() -> None:
    curves = _passing_curves()

    with pytest.raises(ValueError, match="outside the frozen"):
        _evaluate(fraction=0.30, count=3_830, curves=curves)


def test_selection_requires_exactly_five_candidates() -> None:
    qualifications = [
        _evaluate(fraction=0.05, count=638),
        _evaluate(fraction=0.10, count=1_276),
    ]

    with pytest.raises(ValueError, match="exactly the five"):
        select_largest_qualifying_fraction(qualifications)


def test_selection_inputs_reject_circuit_family_fields() -> None:
    curves = _passing_curves()

    with pytest.raises(TypeError):
        evaluate_candidate(
            training_steps=curves["training_steps"],
            train_accuracy=curves["train_accuracy"],
            test_accuracy=curves["test_accuracy"],
            test_cross_entropy=curves["test_cross_entropy"],
            candidate_fraction=0.25,
            training_example_count=3_192,
            required_horizon=MATCHED_HORIZON,
            circuit_family_size=7,
        )

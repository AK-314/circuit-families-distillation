"""Tests for Stage 6 trajectory validation and diagnostics."""

from __future__ import annotations

import pytest

from circuit_families.training.stage6_validation import (
    compute_grokking_diagnostics,
    validate_metric_records,
)


def _record(
    step: int,
    *,
    train_accuracy: float,
    test_accuracy: float,
) -> dict[str, object]:
    return {
        "run_id": "test-run",
        "mode": "full",
        "training_step": step,
        "learning_rate": 0.001,
        "weight_norm": 1.0,
        "gradient_norm": None if step == 0 else 0.1,
        "train_loss": 1.0,
        "test_loss": 1.0,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }


def _grokking_records() -> list[dict[str, object]]:
    return [
        _record(0, train_accuracy=0.01, test_accuracy=0.01),
        _record(50, train_accuracy=1.0, test_accuracy=0.05),
        _record(100, train_accuracy=1.0, test_accuracy=0.06),
        _record(150, train_accuracy=1.0, test_accuracy=0.20),
        _record(200, train_accuracy=1.0, test_accuracy=0.60),
        _record(250, train_accuracy=1.0, test_accuracy=0.995),
        _record(300, train_accuracy=1.0, test_accuracy=1.0),
    ]


def test_grokking_diagnostics_capture_delayed_generalisation() -> None:
    records = _grokking_records()

    validate_metric_records(
        records,
        max_steps=300,
        evaluation_interval=50,
        include_step_zero=True,
    )

    diagnostics = compute_grokking_diagnostics(records)

    assert diagnostics.first_train_999_step == 50
    assert diagnostics.below_10_after_memorisation_count == 2
    assert diagnostics.first_test_10_step == 150
    assert diagnostics.first_test_50_step == 200
    assert diagnostics.first_test_90_step == 250
    assert diagnostics.terminal_stable_99_start_step == 250
    assert diagnostics.terminal_stable_99_count == 2
    assert diagnostics.met_frozen_criteria


def test_metric_validation_rejects_missing_interval() -> None:
    records = _grokking_records()
    del records[3]

    with pytest.raises(
        ValueError,
        match="Metrics schedule",
    ):
        validate_metric_records(
            records,
            max_steps=300,
            evaluation_interval=50,
            include_step_zero=True,
        )


def test_final_checkpoint_alone_is_not_stable_evidence() -> None:
    records = _grokking_records()
    records[-2]["test_accuracy"] = 0.98

    diagnostics = compute_grokking_diagnostics(records)

    assert diagnostics.terminal_stable_99_count == 1
    assert not diagnostics.met_frozen_criteria

from __future__ import annotations

from circuit_families.analysis.stage18_training import (
    classify_grokking_run,
    next_training_horizon,
    requires_extension,
    stable_post_sequence,
)


def _row(step: int, train: float, test: float) -> dict[str, float | int]:
    return {"training_step": step, "train_accuracy": train, "test_accuracy": test}


def test_fifth_consecutive_checkpoint_defines_stable_post() -> None:
    records = [_row(0, 0.01, 0.01), _row(100, 1.0, 0.05), _row(200, 1.0, 0.2)]
    records.extend(_row(step, 1.0, 0.995) for step in (300, 350, 400, 450, 500))
    assert stable_post_sequence(records) == (300, 500)
    result = classify_grokking_run(records)
    assert result.status == "complete_grokking_seed"
    assert result.stable_post_step == 500


def test_extension_policy_uses_exact_frozen_horizons() -> None:
    records = [_row(0, 0.01, 0.01), _row(100, 1.0, 0.05), _row(40_000, 1.0, 0.2)]
    assert requires_extension(records) is True
    assert next_training_horizon(records) == 50_000


def test_no_extension_after_stable_post() -> None:
    records = [_row(0, 0.01, 0.01), _row(100, 1.0, 0.05), _row(200, 1.0, 0.2)]
    records.extend(_row(step, 1.0, 0.995) for step in (39_800, 39_850, 39_900, 39_950, 40_000))
    assert requires_extension(records) is False
    assert next_training_horizon(records) == 40_000


def test_failure_classifications_remain_explicit() -> None:
    never_memorises = [_row(0, 0.01, 0.01), _row(40_000, 0.8, 0.01)]
    assert classify_grokking_run(never_memorises).status == "failed_to_memorise"
    never_groks = [_row(0, 0.01, 0.01), _row(100, 1.0, 0.05), _row(80_000, 1.0, 0.2)]
    assert classify_grokking_run(never_groks).status == "failed_to_grok_by_80000"

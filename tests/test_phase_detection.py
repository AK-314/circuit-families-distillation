"""Tests for deterministic Stage 7 checkpoint phase selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from circuit_families.analysis.phase_detection import (
    PhaseCheckpoint,
    PhaseSelectionResult,
    build_phase_manifest,
    find_pre_grokking_checkpoint,
    find_stable_post_sequence,
    select_phase_checkpoints,
    select_transition_landmarks,
    validate_phase_inputs,
    write_phase_manifest,
    write_phase_table,
)


def _record(step: int, test_accuracy: float) -> dict[str, object]:
    return {
        "training_step": step,
        "test_accuracy": test_accuracy,
    }


def test_stable_post_selects_fifth_checkpoint_in_earliest_sequence() -> None:
    records = [
        _record(0, 0.01),
        _record(50, 0.99),
        _record(100, 0.995),
        _record(150, 1.0),
        _record(200, 0.999),
        _record(250, 0.99),
        _record(300, 1.0),
    ]

    sequence, selected = find_stable_post_sequence(records)

    assert [record["training_step"] for record in sequence] == [
        50,
        100,
        150,
        200,
        250,
    ]
    assert selected["training_step"] == 250


def test_isolated_99_percent_crossing_does_not_qualify() -> None:
    records = [
        _record(0, 0.01),
        _record(50, 0.99),
        _record(100, 0.98),
        _record(150, 0.995),
        _record(200, 0.997),
        _record(250, 0.999),
        _record(300, 1.0),
        _record(350, 0.98),
    ]

    try:
        find_stable_post_sequence(records)
    except ValueError as error:
        assert "No five-consecutive" in str(error)
    else:
        raise AssertionError(
            "An isolated or interrupted 99% crossing must not qualify."
        )


def _phase_record(
    step: int,
    *,
    train_accuracy: float,
    test_accuracy: float,
) -> dict[str, object]:
    return {
        "training_step": step,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }


def test_pre_grokking_selects_latest_valid_checkpoint() -> None:
    records = [
        _phase_record(
            0,
            train_accuracy=0.01,
            test_accuracy=0.01,
        ),
        _phase_record(
            50,
            train_accuracy=0.999,
            test_accuracy=0.04,
        ),
        _phase_record(
            100,
            train_accuracy=1.0,
            test_accuracy=0.05,
        ),
        _phase_record(
            150,
            train_accuracy=1.0,
            test_accuracy=0.08,
        ),
        _phase_record(
            200,
            train_accuracy=1.0,
            test_accuracy=0.10,
        ),
        _phase_record(
            250,
            train_accuracy=1.0,
            test_accuracy=0.03,
        ),
    ]

    selected = find_pre_grokking_checkpoint(records)

    assert selected is not None
    assert selected["training_step"] == 100


def test_pre_grokking_does_not_accept_accuracy_above_five_percent() -> None:
    records = [
        _phase_record(
            0,
            train_accuracy=0.01,
            test_accuracy=0.01,
        ),
        _phase_record(
            50,
            train_accuracy=0.998,
            test_accuracy=0.03,
        ),
        _phase_record(
            100,
            train_accuracy=0.999,
            test_accuracy=0.05201924,
        ),
        _phase_record(
            150,
            train_accuracy=1.0,
            test_accuracy=0.08,
        ),
        _phase_record(
            200,
            train_accuracy=1.0,
            test_accuracy=0.10,
        ),
    ]

    selected = find_pre_grokking_checkpoint(records)

    assert selected is None


def test_transition_landmarks_use_only_internal_checkpoints() -> None:
    records = [
        _phase_record(0, train_accuracy=1.0, test_accuracy=0.02),
        _phase_record(50, train_accuracy=1.0, test_accuracy=0.10),
        _phase_record(100, train_accuracy=1.0, test_accuracy=0.24),
        _phase_record(150, train_accuracy=1.0, test_accuracy=0.51),
        _phase_record(200, train_accuracy=1.0, test_accuracy=0.74),
        _phase_record(250, train_accuracy=1.0, test_accuracy=0.91),
        _phase_record(300, train_accuracy=1.0, test_accuracy=1.00),
    ]

    selected = select_transition_landmarks(
        records,
        pre_step=0,
        stable_post_step=300,
    )

    assert selected["10%"]["training_step"] == 50
    assert selected["25%"]["training_step"] == 100
    assert selected["50%"]["training_step"] == 150
    assert selected["75%"]["training_step"] == 200
    assert selected["90%"]["training_step"] == 250


def test_transition_ties_choose_earlier_checkpoint() -> None:
    records = [
        _phase_record(0, train_accuracy=1.0, test_accuracy=0.00),
        _phase_record(50, train_accuracy=1.0, test_accuracy=0.49),
        _phase_record(100, train_accuracy=1.0, test_accuracy=0.51),
        _phase_record(150, train_accuracy=1.0, test_accuracy=1.00),
    ]

    selected = select_transition_landmarks(
        records,
        pre_step=0,
        stable_post_step=150,
    )

    assert selected["50%"]["training_step"] == 50



def _complete_record(
    step: int,
    *,
    train_accuracy: float,
    test_accuracy: float,
) -> dict[str, object]:
    return {
        "training_step": step,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_loss": 1.0,
        "test_loss": 1.0,
        "checkpoint_path": f"checkpoints/test-run/step_{step:08d}.pt",
        "checkpoint_sha256": "a" * 64,
        "run_id": "test-run",
    }


def test_missing_pre_produces_no_formal_transition_landmarks() -> None:
    records = [
        _complete_record(
            0,
            train_accuracy=0.01,
            test_accuracy=0.01,
        ),
        _complete_record(
            50,
            train_accuracy=0.998,
            test_accuracy=0.03,
        ),
        _complete_record(
            100,
            train_accuracy=0.999,
            test_accuracy=0.05201924,
        ),
        _complete_record(
            150,
            train_accuracy=1.0,
            test_accuracy=0.10,
        ),
        _complete_record(
            200,
            train_accuracy=1.0,
            test_accuracy=0.25,
        ),
        _complete_record(
            250,
            train_accuracy=1.0,
            test_accuracy=0.50,
        ),
        _complete_record(
            300,
            train_accuracy=1.0,
            test_accuracy=0.75,
        ),
        _complete_record(
            350,
            train_accuracy=1.0,
            test_accuracy=0.90,
        ),
        _complete_record(
            400,
            train_accuracy=1.0,
            test_accuracy=0.99,
        ),
        _complete_record(
            450,
            train_accuracy=1.0,
            test_accuracy=0.995,
        ),
        _complete_record(
            500,
            train_accuracy=1.0,
            test_accuracy=1.0,
        ),
        _complete_record(
            550,
            train_accuracy=1.0,
            test_accuracy=1.0,
        ),
        _complete_record(
            600,
            train_accuracy=1.0,
            test_accuracy=1.0,
        ),
    ]

    result = select_phase_checkpoints(records)

    assert not result.has_valid_pre_checkpoint
    assert result.pre_checkpoint is None
    assert result.pre_checkpoint_status == "no_valid_checkpoint"
    assert result.incomplete_grid
    assert result.formal_landmarks == {}
    assert set(result.descriptive_landmarks) == {
        "10%",
        "25%",
        "50%",
        "75%",
        "90%",
    }
    assert result.stable_post_sequence == (
        400,
        450,
        500,
        550,
        600,
    )
    assert result.stable_post_checkpoint is not None
    assert result.stable_post_checkpoint.training_step == 600


def test_complete_endpoints_produce_formal_landmarks_only() -> None:
    records = [
        _complete_record(
            0,
            train_accuracy=0.01,
            test_accuracy=0.01,
        ),
        _complete_record(
            50,
            train_accuracy=0.999,
            test_accuracy=0.04,
        ),
        _complete_record(
            100,
            train_accuracy=1.0,
            test_accuracy=0.10,
        ),
        _complete_record(
            150,
            train_accuracy=1.0,
            test_accuracy=0.25,
        ),
        _complete_record(
            200,
            train_accuracy=1.0,
            test_accuracy=0.50,
        ),
        _complete_record(
            250,
            train_accuracy=1.0,
            test_accuracy=0.75,
        ),
        _complete_record(
            300,
            train_accuracy=1.0,
            test_accuracy=0.90,
        ),
        _complete_record(
            350,
            train_accuracy=1.0,
            test_accuracy=0.99,
        ),
        _complete_record(
            400,
            train_accuracy=1.0,
            test_accuracy=0.995,
        ),
        _complete_record(
            450,
            train_accuracy=1.0,
            test_accuracy=1.0,
        ),
        _complete_record(
            500,
            train_accuracy=1.0,
            test_accuracy=1.0,
        ),
        _complete_record(
            550,
            train_accuracy=1.0,
            test_accuracy=1.0,
        ),
    ]

    result = select_phase_checkpoints(records)

    assert result.has_valid_pre_checkpoint
    assert result.pre_checkpoint is not None
    assert result.pre_checkpoint.training_step == 50
    assert not result.incomplete_grid
    assert result.descriptive_landmarks == {}
    assert {
        label: checkpoint.training_step
        for label, checkpoint in result.formal_landmarks.items()
    } == {
        "10%": 100,
        "25%": 150,
        "50%": 200,
        "75%": 250,
        "90%": 300,
    }



def _validation_record(
    step: int,
    *,
    run_id: str = "test-run",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "mode": "full",
        "training_step": step,
        "learning_rate": 0.001,
        "weight_norm": 1.0,
        "gradient_norm": None if step == 0 else 0.1,
        "train_loss": 1.0,
        "test_loss": 1.0,
        "train_accuracy": 1.0,
        "test_accuracy": 0.5,
        "checkpoint_path": (
            f"checkpoints/{run_id}/step_{step:08d}.pt"
        ),
        "checkpoint_sha256": "unused",
    }


def _write_validation_checkpoints(
    directory: Path,
    records: list[dict[str, object]],
) -> None:
    directory.mkdir(parents=True)

    for record in records:
        step = int(record["training_step"])
        checkpoint = directory / f"step_{step:08d}.pt"
        checkpoint.write_bytes(f"checkpoint-{step}".encode())
        record["checkpoint_sha256"] = __import__(
            "hashlib"
        ).sha256(checkpoint.read_bytes()).hexdigest()


def test_phase_validation_rejects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    records = [
        _validation_record(0),
        _validation_record(50),
    ]
    checkpoint_directory = tmp_path / "test-run"
    _write_validation_checkpoints(
        checkpoint_directory,
        records[:1],
    )

    with pytest.raises(
        FileNotFoundError,
        match="Missing checkpoint",
    ):
        validate_phase_inputs(
            records,
            checkpoint_directory,
            expected_run_id="test-run",
            max_steps=50,
            evaluation_interval=50,
            include_step_zero=True,
        )


def test_phase_validation_rejects_inconsistent_run_id(
    tmp_path: Path,
) -> None:
    records = [
        _validation_record(0),
        _validation_record(50, run_id="other-run"),
    ]

    checkpoint_directory = tmp_path / "test-run"
    checkpoint_directory.mkdir(parents=True)
    checkpoint = checkpoint_directory / "step_00000000.pt"
    checkpoint.write_bytes(b"checkpoint-0")
    records[0]["checkpoint_sha256"] = __import__(
        "hashlib"
    ).sha256(checkpoint.read_bytes()).hexdigest()

    with pytest.raises(
        ValueError,
        match="multiple run IDs|Inconsistent run ID",
    ):
        validate_phase_inputs(
            records,
            checkpoint_directory,
            expected_run_id="test-run",
            max_steps=50,
            evaluation_interval=50,
            include_step_zero=True,
        )


def test_phase_validation_rejects_duplicate_steps(
    tmp_path: Path,
) -> None:
    records = [
        _validation_record(0),
        _validation_record(0),
    ]

    with pytest.raises(
        ValueError,
        match="duplicate steps",
    ):
        validate_phase_inputs(
            records,
            tmp_path / "test-run",
            expected_run_id="test-run",
            max_steps=0,
            evaluation_interval=50,
            include_step_zero=True,
        )


def test_phase_validation_rejects_unordered_steps(
    tmp_path: Path,
) -> None:
    records = [
        _validation_record(50),
        _validation_record(0),
    ]

    with pytest.raises(
        ValueError,
        match="not ordered|expected schedule",
    ):
        validate_phase_inputs(
            records,
            tmp_path / "test-run",
            expected_run_id="test-run",
            max_steps=50,
            evaluation_interval=50,
            include_step_zero=True,
        )


def test_phase_table_generation_is_deterministic(
    tmp_path: Path,
) -> None:
    checkpoint = PhaseCheckpoint(
        label="stable post-grokking",
        training_step=5900,
        train_accuracy=1.0,
        test_accuracy=0.995,
        train_loss=0.01,
        test_loss=0.02,
        checkpoint_path=Path(
            "checkpoints/test-run/step_00005900.pt"
        ),
        checkpoint_sha256="a" * 64,
        run_id="test-run",
    )

    result = PhaseSelectionResult(
        has_valid_pre_checkpoint=False,
        pre_checkpoint=None,
        stable_post_checkpoint=checkpoint,
        stable_post_sequence=(
            5700,
            5750,
            5800,
            5850,
            5900,
        ),
        formal_landmarks={},
        descriptive_landmarks={
            "10%": checkpoint,
            "25%": checkpoint,
            "50%": checkpoint,
            "75%": checkpoint,
            "90%": checkpoint,
        },
        pre_checkpoint_status="no_valid_checkpoint",
        incomplete_grid=True,
    )

    first = tmp_path / "phase.csv"
    second = tmp_path / "phase_second.csv"

    write_phase_table(result, first)
    write_phase_table(result, second)

    assert first.read_bytes() == second.read_bytes()

    text = first.read_text(encoding="utf-8")

    assert "pre-grokking" in text
    assert "stable post-grokking" in text
    assert "descriptive_only_missing_pre" in text
    assert "not_formally_selected" in text



def test_phase_manifest_generation_is_deterministic(
    tmp_path: Path,
) -> None:
    checkpoint = PhaseCheckpoint(
        label="stable post-grokking",
        training_step=5900,
        train_accuracy=1.0,
        test_accuracy=0.995,
        train_loss=0.01,
        test_loss=0.02,
        checkpoint_path=Path(
            "checkpoints/test-run/step_00005900.pt"
        ),
        checkpoint_sha256="a" * 64,
        run_id="test-run",
    )

    result = PhaseSelectionResult(
        has_valid_pre_checkpoint=False,
        pre_checkpoint=None,
        stable_post_checkpoint=checkpoint,
        stable_post_sequence=(
            5700,
            5750,
            5800,
            5850,
            5900,
        ),
        formal_landmarks={},
        descriptive_landmarks={
            label: checkpoint
            for label in ("10%", "25%", "50%", "75%", "90%")
        },
        pre_checkpoint_status="no_valid_checkpoint",
        incomplete_grid=True,
    )

    kwargs = {
        "result": result,
        "run_id": "test-run",
        "training_manifest_path": "manifests/training_test-run.json",
        "metrics_path": "results/raw/test-run/metrics.jsonl",
        "metrics_sha256": "b" * 64,
        "phase_table_path": "results/tables/test.csv",
        "phase_table_sha256": "c" * 64,
        "training_git_commit": "d" * 40,
        "phase_selection_git_commit": "e" * 40,
        "phase_selection_git_status": "clean",
        "creation_timestamp_utc": "2026-07-16T23:00:00+00:00",
    }

    first_manifest = build_phase_manifest(**kwargs)
    second_manifest = build_phase_manifest(**kwargs)

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_phase_manifest(first_manifest, first_path)
    write_phase_manifest(second_manifest, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_manifest["formal_transition_checkpoints"] == {}
    assert first_manifest["preferred_grid_status"] == "incomplete"
    assert (
        first_manifest["pilot_eligibility_from_stage_7"]
        == "ineligible_incomplete_grid"
    )

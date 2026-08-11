"""Tests for the Stage 14 random-label foundation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from circuit_families.analysis.random_label_control import (
    MAIN_MODEL_REFERENCE_CHECKPOINTS,
    MEMORISATION_CONTROL,
    OPTIMISATION_ONLY_CONTROL,
    build_exact_checkpoint_matches,
    classify_random_label_control,
    validate_main_checkpoint_reference,
)
from circuit_families.training.random_label import (
    CHECKPOINT_INTERVAL,
    CLASS_COUNT,
    EVALUATION_INTERVAL,
    EXAMPLES_PER_CLASS,
    FINAL_STEP,
    MODEL_SEED,
    RANDOM_LABEL_SEED,
    TEST_EXAMPLE_COUNT,
    TOTAL_EXAMPLE_COUNT,
    TRAIN_EXAMPLE_COUNT,
    validate_frozen_random_label_dataset,
    validate_stage14_training_settings,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _metric_record(
    step: int,
    train_accuracy: float,
) -> dict[str, float]:
    return {
        "training_step": step,
        "train_accuracy": train_accuracy,
        "test_accuracy": 0.01,
        "train_loss": 1.0,
        "test_loss": 5.0,
    }


def test_frozen_stage14_constants() -> None:
    assert MODEL_SEED == 0
    assert RANDOM_LABEL_SEED == 1
    assert FINAL_STEP == 9050
    assert CHECKPOINT_INTERVAL == 50
    assert EVALUATION_INTERVAL == 50
    assert TOTAL_EXAMPLE_COUNT == 12_769
    assert TRAIN_EXAMPLE_COUNT == 3_830
    assert TEST_EXAMPLE_COUNT == 8_939


def test_frozen_random_label_dataset_validates() -> None:
    result = validate_frozen_random_label_dataset(
        archive_path=(REPOSITORY / "data/generated/modular_addition_m113.npz"),
        metadata_path=(REPOSITORY / "data/generated/modular_addition_m113.metadata.json"),
        manifest_path=(
            REPOSITORY / "manifests/dataset_modular-addition-dataset-s0-7ef9c73ff18f.json"
        ),
        task_config_path=REPOSITORY / "configs/task.yaml",
    )

    assert result.random_label_seed == 1
    assert result.bit_generator == "PCG64"
    assert result.total_example_count == 12_769
    assert result.train_example_count == 3_830
    assert result.test_example_count == 8_939
    assert len(result.class_counts) == CLASS_COUNT
    assert set(result.class_counts) == {EXAMPLES_PER_CLASS}
    assert result.accidental_true_label_match_count >= 0


def test_stage14_training_settings_match_frozen_config() -> None:
    training_config = yaml.safe_load(
        (REPOSITORY / "configs/training.yaml").read_text(encoding="utf-8")
    )

    validate_stage14_training_settings(
        training_config,
        model_seed=0,
        random_label_seed=1,
        final_step=9050,
    )


@pytest.mark.parametrize(
    ("final_accuracy", "expected_classification"),
    [
        (0.99, MEMORISATION_CONTROL),
        (1.0, MEMORISATION_CONTROL),
        (0.989999999, OPTIMISATION_ONLY_CONTROL),
        (0.0, OPTIMISATION_ONLY_CONTROL),
    ],
)
def test_memorisation_classification_boundary(
    final_accuracy: float,
    expected_classification: str,
) -> None:
    records = [
        _metric_record(0, 0.01),
        _metric_record(9000, final_accuracy),
        _metric_record(9050, final_accuracy),
    ]

    result = classify_random_label_control(
        final_training_accuracy=final_accuracy,
        metric_records=records,
    )

    assert result.classification == expected_classification


def test_first_99_percent_step_is_mechanical() -> None:
    records = [
        _metric_record(0, 0.01),
        _metric_record(100, 0.98),
        _metric_record(150, 0.99),
        _metric_record(200, 0.995),
        _metric_record(9050, 0.999),
    ]

    result = classify_random_label_control(
        final_training_accuracy=0.999,
        metric_records=records,
    )

    assert result.first_step_reaching_99_percent == 150
    assert result.reached_99_percent_by_step_9050 is True
    assert records[-1]["train_accuracy"] == 0.999


def test_seed_1_checkpoint_manifest_matches_frozen_grid() -> None:
    payload = json.loads(
        (REPOSITORY / "manifests/checkpoints_seed_1.json").read_text(encoding="utf-8")
    )

    validate_main_checkpoint_reference(payload)


def test_exact_checkpoint_matching_has_seven_zero_mismatch_rows() -> None:
    checkpoint_records = []
    metric_records = []

    for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS:
        checkpoint_records.append(
            {
                "training_step": step,
                "checkpoint_path": f"checkpoints/step_{step}.pt",
                "checkpoint_sha256": f"checkpoint-{step}",
                "model_state_sha256": f"model-{step}",
                "optimizer_state_sha256": f"optimizer-{step}",
            }
        )
        metric_records.append(
            {
                "training_step": step,
                "train_accuracy": 0.5,
                "test_accuracy": 0.01,
                "train_loss": 1.0,
                "test_loss": 5.0,
            }
        )

    rows = build_exact_checkpoint_matches(
        checkpoint_records=checkpoint_records,
        metric_records=metric_records,
    )

    assert len(rows) == 7
    assert [row["requested_step"] for row in rows] == [
        step for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS
    ]
    assert all(row["absolute_step_mismatch"] == 0 for row in rows)
    assert all(row["phase_label_scope"] == "main_model_reference_only" for row in rows)
    assert rows[0]["main_model_reference_phase_label"] == "pre-grokking"
    assert rows[-1]["main_model_reference_phase_label"] == ("stable post-grokking")


def test_checkpoint_matching_rejects_missing_step() -> None:
    checkpoint_records = [
        {"training_step": step} for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS if step != 7450
    ]
    metric_records = [_metric_record(step, 0.5) for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS]

    with pytest.raises(ValueError, match="7450"):
        build_exact_checkpoint_matches(
            checkpoint_records=checkpoint_records,
            metric_records=metric_records,
        )


def test_random_label_and_model_seeds_are_distinct() -> None:
    assert MODEL_SEED != RANDOM_LABEL_SEED

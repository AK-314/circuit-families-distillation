"""Tests for smoke-run orchestration and provenance outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from circuit_families.training.data import TrainingData
from circuit_families.training.logging import read_jsonl
from circuit_families.training.run import run_training

DUMMY_HASHES = {
    "dataset_sha256": "a" * 64,
    "split_sha256": "b" * 64,
    "random_labels_sha256": "c" * 64,
    "random_label_permutation_sha256": "d" * 64,
}


def _tiny_training_data(device: str | torch.device) -> TrainingData:
    selected = torch.device(device)

    return TrainingData(
        train_inputs=torch.tensor(
            [
                [0, 0, 113],
                [1, 2, 113],
                [56, 57, 113],
                [112, 112, 113],
            ],
            dtype=torch.long,
            device=selected,
        ),
        train_targets=torch.tensor(
            [0, 3, 0, 111],
            dtype=torch.long,
            device=selected,
        ),
        test_inputs=torch.tensor(
            [
                [4, 9, 113],
                [17, 22, 113],
                [88, 103, 113],
            ],
            dtype=torch.long,
            device=selected,
        ),
        test_targets=torch.tensor(
            [13, 39, 78],
            dtype=torch.long,
            device=selected,
        ),
        dataset_hashes=DUMMY_HASHES,
        archive_path=Path("dummy/dataset.npz"),
        metadata_path=Path("dummy/metadata.json"),
        manifest_path=Path("dummy/manifest.json"),
        archive_sha256="e" * 64,
        metadata_sha256="f" * 64,
        total_count=7,
        train_count=4,
        test_count=3,
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    output_root: Path,
    *,
    overwrite: bool = False,
):
    def fake_loader(**kwargs):
        return _tiny_training_data(kwargs["device"])

    monkeypatch.setattr(
        "circuit_families.training.run.load_training_data",
        fake_loader,
    )

    return run_training(
        repository_root=".",
        task_config_path="configs/task.yaml",
        model_config_path="configs/model.yaml",
        training_config_path="configs/training.yaml",
        dataset_archive_path="unused.npz",
        dataset_metadata_path="unused.json",
        dataset_manifest_path="unused-manifest.json",
        model_seed=0,
        smoke=True,
        device_override="cpu",
        output_root=output_root,
        overwrite=overwrite,
    )


def test_smoke_run_writes_dense_metrics_and_verified_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _run(monkeypatch, tmp_path)

    assert result.mode == "smoke"
    assert result.device == "cpu"
    assert result.final_step == 5
    assert result.checkpoint_count == 6
    assert result.run_id.startswith("modular-addition-training-smoke-s0-")

    records = read_jsonl(result.metrics_path)

    assert [record["training_step"] for record in records] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    assert records[0]["gradient_norm"] is None
    assert all(record["checkpoint_path"] is not None for record in records)
    assert all(0.0 <= record["train_accuracy"] <= 1.0 for record in records)
    assert all(0.0 <= record["test_accuracy"] <= 1.0 for record in records)

    checkpoints = sorted(result.checkpoint_directory.glob("step_*.pt"))
    assert len(checkpoints) == 6
    assert checkpoints[0].name == "step_00000000.pt"
    assert checkpoints[-1].name == "step_00000005.pt"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["run_id"] == result.run_id
    assert manifest["mode"] == "smoke"
    assert manifest["execution"]["max_steps"] == 5
    assert manifest["dataset"]["train_count"] == 4
    assert manifest["dataset"]["test_count"] == 3
    assert manifest["acceptance"] == {
        "checkpoint_reload_verification": "passed",
        "verified_checkpoint_count": 6,
    }
    assert all(checkpoint["reload_verified"] for checkpoint in manifest["checkpoints"])


def test_two_cpu_smoke_runs_have_identical_canonical_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _run(monkeypatch, tmp_path / "first")
    second = _run(monkeypatch, tmp_path / "second")

    first_records = read_jsonl(first.metrics_path)
    second_records = read_jsonl(second.metrics_path)

    for first_record, second_record in zip(
        first_records,
        second_records,
        strict=True,
    ):
        for key in (
            "training_step",
            "learning_rate",
            "weight_norm",
            "gradient_norm",
            "train_loss",
            "test_loss",
            "train_accuracy",
            "test_accuracy",
            "model_state_sha256",
            "optimizer_state_sha256",
        ):
            assert first_record[key] == second_record[key]


def test_existing_run_requires_explicit_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _run(monkeypatch, tmp_path)

    with pytest.raises(
        FileExistsError,
        match="already exist",
    ):
        _run(monkeypatch, tmp_path)

    replaced = _run(
        monkeypatch,
        tmp_path,
        overwrite=True,
    )

    assert replaced.checkpoint_count == 6


def test_optional_conditional_extension_continues_same_model_and_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_loader(**kwargs):
        return _tiny_training_data(kwargs["device"])

    monkeypatch.setattr(
        "circuit_families.training.run.load_training_data",
        fake_loader,
    )

    result = run_training(
        repository_root=".",
        task_config_path="configs/task.yaml",
        model_config_path="configs/model.yaml",
        training_config_path="configs/training.yaml",
        dataset_archive_path="unused.npz",
        dataset_metadata_path="unused.json",
        dataset_manifest_path="unused-manifest.json",
        model_seed=2,
        smoke=False,
        max_steps_override=3,
        device_override="cpu",
        output_root=tmp_path,
        checkpoint_verification_steps=(),
        extension_increment=2,
        absolute_max_steps=7,
        extension_decider=lambda rows: int(rows[-1]["training_step"]) < 7,
    )

    assert result.final_step == 7
    assert [row["training_step"] for row in read_jsonl(result.metrics_path)] == [
        0,
        3,
        5,
        7,
    ]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["execution"]["max_steps"] == 3
    assert manifest["conditional_extension"] == {
        "absolute_max_steps": 7,
        "decision": "frozen_stable_post_criterion_at_horizon_boundaries",
        "effective_final_step": 7,
        "executed_extension_horizons": [5, 7],
        "extension_increment": 2,
        "maximum_reached": True,
        "standard_horizon": 3,
    }

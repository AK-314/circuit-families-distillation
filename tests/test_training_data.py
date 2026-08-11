"""Tests for validated training-data loading and JSONL logging."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from circuit_families.config import load_config
from circuit_families.training.data import load_training_data
from circuit_families.training.logging import append_jsonl, read_jsonl


def test_load_training_data_matches_frozen_archive() -> None:
    data = load_training_data(
        archive_path="data/generated/modular_addition_m113.npz",
        metadata_path=(
            "data/generated/modular_addition_m113.metadata.json"
        ),
        manifest_path=(
            "manifests/"
            "dataset_modular-addition-dataset-s0-7ef9c73ff18f.json"
        ),
        task_config=load_config("configs/task.yaml"),
        device="cpu",
    )

    assert data.total_count == 12_769
    assert data.train_count == 3_830
    assert data.test_count == 8_939

    assert data.full_inputs is not None
    assert data.full_targets is not None
    assert data.full_inputs.shape == (12_769, 3)
    assert data.full_targets.shape == (12_769,)

    assert data.train_inputs.shape == (3_830, 3)
    assert data.train_targets.shape == (3_830,)
    assert data.test_inputs.shape == (8_939, 3)
    assert data.test_targets.shape == (8_939,)

    assert data.full_inputs.dtype == torch.long
    assert data.full_targets.dtype == torch.long
    assert data.train_inputs.dtype == torch.long
    assert data.train_targets.dtype == torch.long
    assert data.test_inputs.dtype == torch.long
    assert data.test_targets.dtype == torch.long

    assert data.full_inputs.device.type == "cpu"
    assert data.full_targets.device.type == "cpu"
    assert data.train_inputs.device.type == "cpu"
    assert torch.all(data.full_inputs[:, -1] == 113)
    assert torch.all(data.train_inputs[:, -1] == 113)
    assert int(data.full_targets.min()) >= 0
    assert int(data.full_targets.max()) < 113
    assert int(data.train_targets.min()) >= 0
    assert int(data.train_targets.max()) < 113

    expected_left = torch.arange(113).repeat_interleave(113)
    expected_right = torch.arange(113).repeat(113)

    assert torch.equal(data.full_inputs[:, 0], expected_left)
    assert torch.equal(data.full_inputs[:, 1], expected_right)
    assert torch.equal(
        data.full_targets,
        (expected_left + expected_right) % 113,
    )

    assert data.dataset_hashes["dataset_sha256"] == (
        "af13d2181f5f1122bc528c6dfadbdc67"
        "b0a38ea02c10b4fd504a492aca8afafa"
    )
    assert data.dataset_hashes["split_sha256"] == (
        "c83ac398724817fae6a0d137d0f1c6d"
        "0b8786eee43efaff5c3d34de0a891b7f2"
    )


def test_loader_rejects_incorrect_physical_archive_hash(
    tmp_path: Path,
) -> None:
    archive_copy = tmp_path / "dataset.npz"
    archive_copy.write_bytes(
        Path(
            "data/generated/modular_addition_m113.npz"
        ).read_bytes()
        + b"tampered"
    )

    with pytest.raises(
        ValueError,
        match="archive physical hash",
    ):
        load_training_data(
            archive_path=archive_copy,
            metadata_path=(
                "data/generated/modular_addition_m113.metadata.json"
            ),
            manifest_path=(
                "manifests/"
                "dataset_modular-addition-dataset-s0-7ef9c73ff18f.json"
            ),
            task_config=load_config("configs/task.yaml"),
            device="cpu",
        )


def test_jsonl_round_trip_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"

    append_jsonl(path, {"step": 0, "loss": 4.7})
    append_jsonl(path, {"step": 1, "loss": 4.6})

    assert read_jsonl(path) == [
        {"loss": 4.7, "step": 0},
        {"loss": 4.6, "step": 1},
    ]

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"loss":4.7,"step":0}'
    assert lines[1] == '{"loss":4.6,"step":1}'


def test_jsonl_rejects_non_finite_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        append_jsonl(
            tmp_path / "metrics.jsonl",
            {"loss": math.nan},
        )


def test_jsonl_reader_rejects_non_object_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps([1, 2, 3]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must contain an object",
    ):
        read_jsonl(path)

"""Tests for JSON provenance manifests."""

from __future__ import annotations

import json
from pathlib import Path

from circuit_families.manifests import create_manifest, write_manifest


def test_manifest_contains_required_provenance_fields(
    tmp_path: Path,
) -> None:
    manifest = create_manifest(
        run_id="test-run-s0-abcdef123456",
        experiment_type="modular_addition_dataset",
        repository_root=".",
        config_path="configs/task.yaml",
        config_sha256="a" * 64,
        seed_name="split_seed",
        seed=0,
        output_paths={
            "dataset": "data/generated/test.npz",
            "metadata": "data/generated/test.json",
        },
        hashes={
            "dataset_sha256": "b" * 64,
            "split_sha256": "c" * 64,
        },
        details={
            "dataset_seed": None,
            "random_label_seed": 1,
        },
    )

    assert manifest["timestamp_utc"]
    assert manifest["git_commit"]
    assert manifest["config"] == {
        "path": "configs/task.yaml",
        "sha256": "a" * 64,
    }
    assert manifest["software"]["python"]
    assert manifest["software"]["packages"]
    assert manifest["device"]["selected_device"]
    assert manifest["seed"] == {
        "name": "split_seed",
        "value": 0,
    }
    assert manifest["output_paths"]["dataset"] == (
        "data/generated/test.npz"
    )
    assert manifest["hashes"]["dataset_sha256"] == "b" * 64
    assert manifest["details"]["random_label_seed"] == 1

    output_path = tmp_path / "manifest.json"
    write_manifest(output_path, manifest)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == manifest


def test_written_manifest_is_valid_json(tmp_path: Path) -> None:
    manifest = create_manifest(
        run_id="test-run",
        experiment_type="test",
        repository_root=".",
        config_path="configs/task.yaml",
        config_sha256="d" * 64,
        seed_name="dataset_seed",
        seed=3,
        output_paths={"result": "results/test.json"},
    )

    path = write_manifest(tmp_path / "nested" / "manifest.json", manifest)

    assert path.is_file()
    json.loads(path.read_text(encoding="utf-8"))


def test_manifest_output_paths_are_sorted() -> None:
    manifest = create_manifest(
        run_id="test-run",
        experiment_type="test",
        repository_root=".",
        config_path="configs/task.yaml",
        config_sha256="e" * 64,
        seed_name="seed",
        seed=0,
        output_paths={
            "z_output": "z.json",
            "a_output": "a.json",
        },
    )

    assert list(manifest["output_paths"]) == ["a_output", "z_output"]

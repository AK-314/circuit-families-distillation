"""End-to-end tests for deterministic dataset generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_dataset.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_dataset_script",
    SCRIPT_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
generate_dataset = MODULE.generate_dataset


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {
            name: archive[name].copy()
            for name in archive.files
        }


def test_generation_is_reproducible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_directory = tmp_path / "configs"
    config_directory.mkdir()

    source_config = Path("configs/task.yaml").resolve()
    config_path = config_directory / "task.yaml"
    config_path.write_text(
        source_config.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    first = generate_dataset(Path("configs/task.yaml"))
    dataset_path = tmp_path / first["dataset_archive"]
    first_arrays = _load_arrays(dataset_path)

    second = generate_dataset(Path("configs/task.yaml"))
    second_arrays = _load_arrays(dataset_path)

    assert first["run_id"] == second["run_id"]
    assert first["config_sha256"] == second["config_sha256"]
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["split_sha256"] == second["split_sha256"]
    assert (
        first["random_labels_sha256"]
        == second["random_labels_sha256"]
    )
    assert (
        first["random_label_permutation_sha256"]
        == second["random_label_permutation_sha256"]
    )

    assert first_arrays.keys() == second_arrays.keys()
    for name in first_arrays:
        assert np.array_equal(first_arrays[name], second_arrays[name])


def test_generated_manifest_contains_required_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_directory = tmp_path / "configs"
    config_directory.mkdir()

    source_config = Path("configs/task.yaml").resolve()
    config_path = config_directory / "task.yaml"
    config_path.write_text(
        source_config.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    outputs = generate_dataset(Path("configs/task.yaml"))

    manifest_path = tmp_path / outputs["dataset_manifest"]
    metadata_path = tmp_path / outputs["dataset_metadata"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert manifest["timestamp_utc"]
    assert manifest["git_commit"]
    assert manifest["config"]["path"] == "configs/task.yaml"
    assert manifest["config"]["sha256"] == outputs["config_sha256"]
    assert manifest["software"]["python"]
    assert manifest["software"]["packages"]
    assert manifest["device"]["selected_device"]
    assert manifest["seed"] == {
        "name": "split_seed",
        "value": 0,
    }

    assert manifest["details"]["dataset_seed"] is None
    assert manifest["details"]["split_seed"] == 0
    assert manifest["details"]["random_label_seed"] == 1
    assert manifest["details"]["primary_training_count"] == 3_830
    assert manifest["details"]["test_count"] == 8_939

    assert (
        manifest["hashes"]["dataset_sha256"]
        == outputs["dataset_sha256"]
    )
    assert (
        manifest["hashes"]["split_sha256"]
        == outputs["split_sha256"]
    )

    assert metadata["counts"]["total_examples"] == 12_769
    assert metadata["counts"]["primary_train"] == 3_830
    assert metadata["counts"]["primary_test"] == 8_939
    assert metadata["seeds"]["split_seed"] == 0
    assert metadata["seeds"]["random_label_seed"] == 1

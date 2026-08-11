"""Validated loading of the frozen modular-addition training data."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from circuit_families.config import config_hash, validate_task_config
from circuit_families.data.modular_addition import hash_named_arrays
from circuit_families.training.checkpoints import file_sha256


@dataclass(frozen=True)
class TrainingData:
    """Validated full, train, and test tensors with provenance."""

    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    test_inputs: torch.Tensor
    test_targets: torch.Tensor
    dataset_hashes: dict[str, str]
    archive_path: Path
    metadata_path: Path
    manifest_path: Path
    archive_sha256: str
    metadata_sha256: str
    total_count: int
    train_count: int
    test_count: int
    full_inputs: torch.Tensor | None = None
    full_targets: torch.Tensor | None = None
    training_subset: dict[str, Any] | None = None


def _load_json_mapping(path: str | Path, name: str) -> dict[str, Any]:
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {file_path}")

    value = json.loads(file_path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object.")

    return value


def _require_mapping(
    value: Any,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")

    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{name} must be a lowercase SHA-256 hex digest."
        )

    return value


def _require_array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"Dataset archive is missing array {name!r}.")

    array = archive[name]

    if array.shape != shape:
        raise ValueError(
            f"{name} has shape {array.shape}; expected {shape}."
        )

    if array.dtype != dtype:
        raise ValueError(
            f"{name} has dtype {array.dtype}; expected {dtype}."
        )

    return array


def _control_array_names(
    task_config: Mapping[str, Any],
) -> tuple[str, ...]:
    fractions = task_config["split"]["control_train_fractions"]

    return tuple(
        f"control_train_indices_{round(fraction * 100):02d}pct"
        for fraction in fractions
    )


def load_training_data(
    *,
    archive_path: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    task_config: Mapping[str, Any],
    device: str | torch.device,
) -> TrainingData:
    """Load and validate the frozen primary train/test split."""

    validate_task_config(task_config)

    archive_file = Path(archive_path)
    metadata_file = Path(metadata_path)
    manifest_file = Path(manifest_path)
    selected_device = torch.device(device)

    if not archive_file.is_file():
        raise FileNotFoundError(
            f"Dataset archive does not exist: {archive_file}"
        )

    metadata = _load_json_mapping(
        metadata_file,
        "dataset metadata",
    )
    manifest = _load_json_mapping(
        manifest_file,
        "dataset manifest",
    )

    expected_config_hash = config_hash(task_config)

    if metadata.get("config_sha256") != expected_config_hash:
        raise ValueError(
            "Dataset metadata task-config hash does not match."
        )

    manifest_config = _require_mapping(
        manifest.get("config"),
        "dataset manifest config",
    )

    if manifest_config.get("sha256") != expected_config_hash:
        raise ValueError(
            "Dataset manifest task-config hash does not match."
        )

    if metadata.get("run_id") != manifest.get("run_id"):
        raise ValueError(
            "Dataset metadata and manifest run IDs do not match."
        )

    manifest_hashes = _require_mapping(
        manifest.get("hashes"),
        "dataset manifest hashes",
    )
    metadata_hashes = _require_mapping(
        metadata.get("hashes"),
        "dataset metadata hashes",
    )

    actual_archive_sha256 = file_sha256(archive_file)
    actual_metadata_sha256 = file_sha256(metadata_file)

    expected_archive_sha256 = _require_sha256(
        manifest_hashes.get("archive_sha256"),
        "dataset manifest archive_sha256",
    )
    expected_metadata_sha256 = _require_sha256(
        manifest_hashes.get("metadata_sha256"),
        "dataset manifest metadata_sha256",
    )

    if actual_archive_sha256 != expected_archive_sha256:
        raise ValueError("Dataset archive physical hash does not match.")

    if actual_metadata_sha256 != expected_metadata_sha256:
        raise ValueError("Dataset metadata physical hash does not match.")

    task = task_config["task"]
    split = task_config["split"]

    total_count = task["expected_pair_count"]
    train_count = split["primary_train_count"]
    test_count = split["primary_test_count"]

    control_names = _control_array_names(task_config)

    with np.load(archive_file, allow_pickle=False) as archive:
        inputs = _require_array(
            archive,
            "inputs",
            shape=(total_count, 3),
            dtype=np.dtype(np.int16),
        )
        pairs = _require_array(
            archive,
            "pairs",
            shape=(total_count, 2),
            dtype=np.dtype(np.int16),
        )
        true_labels = _require_array(
            archive,
            "true_labels",
            shape=(total_count,),
            dtype=np.dtype(np.int16),
        )
        split_permutation = _require_array(
            archive,
            "split_permutation",
            shape=(total_count,),
            dtype=np.dtype(np.int64),
        )
        train_indices = _require_array(
            archive,
            "train_indices",
            shape=(train_count,),
            dtype=np.dtype(np.int64),
        )
        test_indices = _require_array(
            archive,
            "test_indices",
            shape=(test_count,),
            dtype=np.dtype(np.int64),
        )

        control_arrays = {
            name: archive[name]
            for name in control_names
            if name in archive.files
        }

        if set(control_arrays) != set(control_names):
            missing = sorted(set(control_names).difference(control_arrays))
            raise ValueError(
                "Dataset archive is missing control arrays: "
                + ", ".join(missing)
            )

        dataset_sha256 = hash_named_arrays(
            {
                "inputs": inputs,
                "pairs": pairs,
                "true_labels": true_labels,
            }
        )
        split_sha256 = hash_named_arrays(
            {
                "split_permutation": split_permutation,
                "train_indices": train_indices,
                "test_indices": test_indices,
                **control_arrays,
            }
        )

        if not np.array_equal(inputs[:, :2], pairs):
            raise ValueError(
                "Dataset input operands do not match the pair array."
            )

        if not np.all(inputs[:, 2] == task["equals_token_id"]):
            raise ValueError(
                "Dataset inputs do not use the frozen equals token."
            )

        if true_labels.min() < 0 or true_labels.max() >= task["modulus"]:
            raise ValueError(
                "Dataset labels are outside the output vocabulary."
            )

        if (
            train_indices.min() < 0
            or test_indices.min() < 0
            or train_indices.max() >= total_count
            or test_indices.max() >= total_count
        ):
            raise ValueError("Dataset split indices are out of range.")

        if np.unique(train_indices).size != train_count:
            raise ValueError("Training indices contain duplicates.")

        if np.unique(test_indices).size != test_count:
            raise ValueError("Test indices contain duplicates.")

        if np.intersect1d(train_indices, test_indices).size:
            raise ValueError("Training and test indices overlap.")

        if np.unique(
            np.concatenate((train_indices, test_indices))
        ).size != total_count:
            raise ValueError(
                "Training and test indices are not exhaustive."
            )

        full_inputs_array = inputs.copy()
        full_targets_array = true_labels.copy()
        train_inputs_array = inputs[train_indices].copy()
        train_targets_array = true_labels[train_indices].copy()
        test_inputs_array = inputs[test_indices].copy()
        test_targets_array = true_labels[test_indices].copy()

    expected_dataset_sha256 = _require_sha256(
        metadata_hashes.get("dataset_sha256"),
        "dataset metadata dataset_sha256",
    )
    expected_split_sha256 = _require_sha256(
        metadata_hashes.get("split_sha256"),
        "dataset metadata split_sha256",
    )

    if dataset_sha256 != expected_dataset_sha256:
        raise ValueError("Canonical dataset-array hash does not match.")

    if split_sha256 != expected_split_sha256:
        raise ValueError("Canonical split-array hash does not match.")

    if manifest_hashes.get("dataset_sha256") != dataset_sha256:
        raise ValueError(
            "Dataset manifest canonical dataset hash does not match."
        )

    if manifest_hashes.get("split_sha256") != split_sha256:
        raise ValueError(
            "Dataset manifest canonical split hash does not match."
        )

    validated_hashes = {
        name: _require_sha256(
            value,
            f"dataset metadata hashes.{name}",
        )
        for name, value in sorted(metadata_hashes.items())
    }

    return TrainingData(
        full_inputs=torch.as_tensor(
            full_inputs_array,
            dtype=torch.long,
            device=selected_device,
        ),
        full_targets=torch.as_tensor(
            full_targets_array,
            dtype=torch.long,
            device=selected_device,
        ),
        train_inputs=torch.as_tensor(
            train_inputs_array,
            dtype=torch.long,
            device=selected_device,
        ),
        train_targets=torch.as_tensor(
            train_targets_array,
            dtype=torch.long,
            device=selected_device,
        ),
        test_inputs=torch.as_tensor(
            test_inputs_array,
            dtype=torch.long,
            device=selected_device,
        ),
        test_targets=torch.as_tensor(
            test_targets_array,
            dtype=torch.long,
            device=selected_device,
        ),
        dataset_hashes=validated_hashes,
        archive_path=archive_file,
        metadata_path=metadata_file,
        manifest_path=manifest_file,
        archive_sha256=actual_archive_sha256,
        metadata_sha256=actual_metadata_sha256,
        total_count=total_count,
        train_count=train_count,
        test_count=test_count,
    )

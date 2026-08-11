"""Frozen Stage 14 random-label training support."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from circuit_families.data.modular_addition import hash_named_arrays
from circuit_families.training.data import TrainingData, load_training_data

RANDOM_LABEL_SEED = 1
MODEL_SEED = 0
FINAL_STEP = 9050
CHECKPOINT_INTERVAL = 50
EVALUATION_INTERVAL = 50
MEMORISATION_THRESHOLD = 0.99

TOTAL_EXAMPLE_COUNT = 12_769
TRAIN_EXAMPLE_COUNT = 3_830
TEST_EXAMPLE_COUNT = 8_939
CLASS_COUNT = 113
EXAMPLES_PER_CLASS = 113

DATASET_ARCHIVE_SHA256 = "76c499f9d90d8561d12186f8cc7080f8c5cf7460c9f471c1e7b327b0b15337c5"
DATASET_METADATA_SHA256 = "acbfe6a61b48eb2a2ad522389185085858f72c960906e13291ee85f8bc95564f"
DATASET_MANIFEST_SHA256 = "8c0d9531267181cd3bbf9d247085b6463efec04c8477a4e6c4137a37c54fd667"
CANONICAL_DATASET_SHA256 = "af13d2181f5f1122bc528c6dfadbdc67b0a38ea02c10b4fd504a492aca8afafa"
SPLIT_SHA256 = "c83ac398724817fae6a0d137d0f1c6d0b8786eee43efaff5c3d34de0a891b7f2"
RANDOM_LABELS_SHA256 = "5a4f92635efff86f168c8999b426be000cbdf5e6194e6e7b5d537243583ac5c9"
RANDOM_LABEL_PERMUTATION_SHA256 = "6133ae3b6535595aae903ba9197c6b136b2667a50553fbda3a49331765cb30e5"

CANONICAL_DATASET_ARRAY_NAMES = (
    "inputs",
    "pairs",
    "true_labels",
)


@dataclass(frozen=True)
class RandomLabelDatasetValidation:
    """Validated identity and properties of the frozen random-label data."""

    archive_path: Path
    archive_sha256: str
    metadata_path: Path
    metadata_sha256: str
    manifest_path: Path
    manifest_sha256: str
    canonical_dataset_sha256: str
    split_sha256: str
    random_labels_sha256: str
    random_label_permutation_sha256: str
    random_label_seed: int
    bit_generator: str
    total_example_count: int
    train_example_count: int
    test_example_count: int
    class_counts: tuple[int, ...]
    accidental_true_label_match_count: int


def file_sha256(file_path: str | Path) -> str:
    """Return the SHA-256 digest of one physical file."""

    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(file_path: Path) -> dict[str, Any]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {file_path}")
    return payload


def _require_equal(
    actual: object,
    expected: object,
    description: str,
) -> None:
    if actual != expected:
        raise ValueError(f"{description} mismatch: expected {expected!r}, got {actual!r}")


def _require_file_hash(
    file_path: Path,
    expected_sha256: str,
    description: str,
) -> str:
    actual_sha256 = file_sha256(file_path)
    _require_equal(actual_sha256, expected_sha256, description)
    return actual_sha256


def _validate_array_shapes(arrays: Mapping[str, np.ndarray]) -> None:
    expected_shapes = {
        "pairs": (TOTAL_EXAMPLE_COUNT, 2),
        "inputs": (TOTAL_EXAMPLE_COUNT, 3),
        "true_labels": (TOTAL_EXAMPLE_COUNT,),
        "split_permutation": (TOTAL_EXAMPLE_COUNT,),
        "train_indices": (TRAIN_EXAMPLE_COUNT,),
        "test_indices": (TEST_EXAMPLE_COUNT,),
        "random_label_permutation": (TOTAL_EXAMPLE_COUNT,),
        "random_labels": (TOTAL_EXAMPLE_COUNT,),
    }

    for name, expected_shape in expected_shapes.items():
        if name not in arrays:
            raise ValueError(f"Frozen dataset is missing array {name!r}")

        _require_equal(
            tuple(arrays[name].shape),
            expected_shape,
            f"{name} shape",
        )


def _validate_pair_ordering(arrays: Mapping[str, np.ndarray]) -> None:
    expected_pairs = np.asarray(
        [(left, right) for left in range(CLASS_COUNT) for right in range(CLASS_COUNT)],
        dtype=arrays["pairs"].dtype,
    )

    if not np.array_equal(arrays["pairs"], expected_pairs):
        raise ValueError("Ordered input pairs are not lexicographic")

    if not np.array_equal(arrays["inputs"][:, :2], arrays["pairs"]):
        raise ValueError("Tokenised inputs do not preserve pair ordering")


def _validate_split(arrays: Mapping[str, np.ndarray]) -> None:
    train_indices = arrays["train_indices"]
    test_indices = arrays["test_indices"]
    split_permutation = arrays["split_permutation"]

    if not np.array_equal(
        split_permutation[:TRAIN_EXAMPLE_COUNT],
        train_indices,
    ):
        raise ValueError("Training-index ordering differs from the frozen split")

    if not np.array_equal(
        split_permutation[TRAIN_EXAMPLE_COUNT:],
        test_indices,
    ):
        raise ValueError("Test-index ordering differs from the frozen split")

    combined = np.concatenate((train_indices, test_indices))
    expected_indices = np.arange(TOTAL_EXAMPLE_COUNT, dtype=combined.dtype)

    if not np.array_equal(np.sort(combined), expected_indices):
        raise ValueError("Train and test indices do not partition the dataset")


def _validate_random_labels(
    arrays: Mapping[str, np.ndarray],
) -> tuple[tuple[int, ...], int]:
    true_labels = arrays["true_labels"]
    permutation = arrays["random_label_permutation"]
    random_labels = arrays["random_labels"]

    expected_permutation = np.arange(
        TOTAL_EXAMPLE_COUNT,
        dtype=permutation.dtype,
    )

    if not np.array_equal(np.sort(permutation), expected_permutation):
        raise ValueError("Random-label permutation is not a full permutation")

    if not np.array_equal(random_labels, true_labels[permutation]):
        raise ValueError("Frozen random labels do not equal the recorded label permutation")

    if np.array_equal(random_labels, true_labels):
        raise ValueError("Random labels were silently replaced by true labels")

    counts = Counter(int(value) for value in random_labels.tolist())

    if set(counts) != set(range(CLASS_COUNT)):
        raise ValueError("Random-label classes are not exactly 0 through 112")

    ordered_counts = tuple(counts[index] for index in range(CLASS_COUNT))

    if any(count != EXAMPLES_PER_CLASS for count in ordered_counts):
        raise ValueError("Random-label global class balance differs from 113")

    accidental_matches = int(np.count_nonzero(random_labels == true_labels))
    return ordered_counts, accidental_matches


def _validate_recorded_split_hash(
    arrays: Mapping[str, np.ndarray],
) -> None:
    split_names = [
        name
        for name in arrays
        if name == "split_permutation"
        or name == "train_indices"
        or name == "test_indices"
        or name.startswith("control_train_indices_")
    ]

    candidate_hashes: set[str] = set()

    for count in range(1, len(split_names) + 1):
        for selected_names in combinations(sorted(split_names), count):
            candidate_hashes.add(hash_named_arrays({name: arrays[name] for name in selected_names}))

    if SPLIT_SHA256 not in candidate_hashes:
        raise ValueError("Frozen split arrays do not reproduce the split hash")


def validate_frozen_random_label_dataset(
    *,
    archive_path: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    task_config_path: str | Path,
) -> RandomLabelDatasetValidation:
    """Validate committed random-label arrays without regenerating them."""

    archive = Path(archive_path)
    metadata_file = Path(metadata_path)
    manifest_file = Path(manifest_path)
    task_config_file = Path(task_config_path)

    for file_path in (
        archive,
        metadata_file,
        manifest_file,
        task_config_file,
    ):
        if not file_path.is_file():
            raise FileNotFoundError(file_path)

    archive_sha256 = _require_file_hash(
        archive,
        DATASET_ARCHIVE_SHA256,
        "dataset archive SHA-256",
    )
    metadata_sha256 = _require_file_hash(
        metadata_file,
        DATASET_METADATA_SHA256,
        "dataset metadata SHA-256",
    )
    manifest_sha256 = _require_file_hash(
        manifest_file,
        DATASET_MANIFEST_SHA256,
        "dataset manifest SHA-256",
    )

    metadata = _load_json(metadata_file)
    manifest = _load_json(manifest_file)
    task_config = yaml.safe_load(task_config_file.read_text(encoding="utf-8"))

    random_label_config = task_config["random_labels"]

    _require_equal(
        random_label_config["generator"],
        "PCG64",
        "random-label bit generator",
    )
    _require_equal(
        int(random_label_config["seed"]),
        RANDOM_LABEL_SEED,
        "random-label seed",
    )

    metadata_hashes = metadata["hashes"]
    manifest_hashes = manifest["hashes"]

    expected_recorded_hashes = {
        "dataset_sha256": CANONICAL_DATASET_SHA256,
        "split_sha256": SPLIT_SHA256,
        "random_labels_sha256": RANDOM_LABELS_SHA256,
        "random_label_permutation_sha256": (RANDOM_LABEL_PERMUTATION_SHA256),
    }

    for name, expected_value in expected_recorded_hashes.items():
        _require_equal(
            metadata_hashes[name],
            expected_value,
            f"metadata {name}",
        )
        _require_equal(
            manifest_hashes[name],
            expected_value,
            f"manifest {name}",
        )

    _require_equal(
        manifest_hashes["archive_sha256"],
        archive_sha256,
        "manifest archive SHA-256",
    )
    _require_equal(
        manifest_hashes["metadata_sha256"],
        metadata_sha256,
        "manifest metadata SHA-256",
    )

    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}

    _validate_array_shapes(arrays)
    _validate_pair_ordering(arrays)
    _validate_split(arrays)
    class_counts, accidental_matches = _validate_random_labels(arrays)

    canonical_dataset_sha256 = hash_named_arrays(
        {name: arrays[name] for name in CANONICAL_DATASET_ARRAY_NAMES}
    )
    _require_equal(
        canonical_dataset_sha256,
        CANONICAL_DATASET_SHA256,
        "canonical dataset SHA-256",
    )

    random_labels_sha256 = hash_named_arrays({"random_labels": arrays["random_labels"]})
    _require_equal(
        random_labels_sha256,
        RANDOM_LABELS_SHA256,
        "random-label table SHA-256",
    )

    permutation_sha256 = hash_named_arrays(
        {"random_label_permutation": arrays["random_label_permutation"]}
    )
    _require_equal(
        permutation_sha256,
        RANDOM_LABEL_PERMUTATION_SHA256,
        "random-label permutation SHA-256",
    )

    _validate_recorded_split_hash(arrays)

    return RandomLabelDatasetValidation(
        archive_path=archive,
        archive_sha256=archive_sha256,
        metadata_path=metadata_file,
        metadata_sha256=metadata_sha256,
        manifest_path=manifest_file,
        manifest_sha256=manifest_sha256,
        canonical_dataset_sha256=canonical_dataset_sha256,
        split_sha256=SPLIT_SHA256,
        random_labels_sha256=random_labels_sha256,
        random_label_permutation_sha256=permutation_sha256,
        random_label_seed=RANDOM_LABEL_SEED,
        bit_generator="PCG64",
        total_example_count=TOTAL_EXAMPLE_COUNT,
        train_example_count=TRAIN_EXAMPLE_COUNT,
        test_example_count=TEST_EXAMPLE_COUNT,
        class_counts=class_counts,
        accidental_true_label_match_count=accidental_matches,
    )


def load_random_label_training_data(
    *,
    archive_path: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    task_config_path: str | Path,
    task_config: Mapping[str, Any],
    device: str | torch.device,
) -> TrainingData:
    """Load the fixed split while replacing only the target labels."""

    validation = validate_frozen_random_label_dataset(
        archive_path=archive_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        task_config_path=task_config_path,
    )

    primary = load_training_data(
        archive_path=archive_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        task_config=task_config,
        device=device,
    )

    if primary.full_inputs is None or primary.full_targets is None:
        raise ValueError("Validated primary data did not retain the complete arrays.")

    selected_device = torch.device(device)

    with np.load(archive_path, allow_pickle=False) as archive:
        train_indices = np.asarray(
            archive["train_indices"],
            dtype=np.int64,
        ).copy()
        test_indices = np.asarray(
            archive["test_indices"],
            dtype=np.int64,
        ).copy()
        random_labels = np.asarray(
            archive["random_labels"],
            dtype=np.int64,
        ).copy()
        true_labels = np.asarray(
            archive["true_labels"],
            dtype=np.int64,
        ).copy()

    if np.array_equal(random_labels, true_labels):
        raise ValueError("Random labels were replaced with true labels.")

    train_index_tensor = torch.as_tensor(
        train_indices,
        dtype=torch.long,
        device=selected_device,
    )
    test_index_tensor = torch.as_tensor(
        test_indices,
        dtype=torch.long,
        device=selected_device,
    )
    full_random_targets = torch.as_tensor(
        random_labels,
        dtype=torch.long,
        device=selected_device,
    )

    train_inputs = primary.full_inputs.index_select(
        0,
        train_index_tensor,
    )
    test_inputs = primary.full_inputs.index_select(
        0,
        test_index_tensor,
    )
    train_targets = full_random_targets.index_select(
        0,
        train_index_tensor,
    )
    test_targets = full_random_targets.index_select(
        0,
        test_index_tensor,
    )

    if not torch.equal(train_inputs, primary.train_inputs):
        raise ValueError("Random-label training inputs differ from the primary split.")

    if not torch.equal(test_inputs, primary.test_inputs):
        raise ValueError("Random-label test inputs differ from the primary split.")

    dataset_hashes = dict(primary.dataset_hashes)

    required_hashes = {
        "dataset_sha256": validation.canonical_dataset_sha256,
        "split_sha256": validation.split_sha256,
        "random_labels_sha256": validation.random_labels_sha256,
        "random_label_permutation_sha256": (validation.random_label_permutation_sha256),
    }

    for name, expected in required_hashes.items():
        if dataset_hashes.get(name) != expected:
            raise ValueError(f"Loaded dataset hash mismatch for {name}: expected {expected!r}.")

    return TrainingData(
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        dataset_hashes=dataset_hashes,
        archive_path=primary.archive_path,
        metadata_path=primary.metadata_path,
        manifest_path=primary.manifest_path,
        archive_sha256=primary.archive_sha256,
        metadata_sha256=primary.metadata_sha256,
        total_count=primary.total_count,
        train_count=primary.train_count,
        test_count=primary.test_count,
        full_inputs=primary.full_inputs,
        full_targets=full_random_targets,
    )


def validate_stage14_training_settings(
    training_config: Mapping[str, Any],
    *,
    model_seed: int,
    random_label_seed: int,
    final_step: int,
) -> None:
    """Reject deviations from the complete frozen Stage 14 policy."""

    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise TypeError("Stage 14 model seed must be an integer.")

    if isinstance(random_label_seed, bool) or not isinstance(
        random_label_seed,
        int,
    ):
        raise TypeError("Random-label seed must be an integer.")

    if isinstance(final_step, bool) or not isinstance(final_step, int):
        raise TypeError("Stage 14 final step must be an integer.")

    _require_equal(model_seed, MODEL_SEED, "model seed")
    _require_equal(
        random_label_seed,
        RANDOM_LABEL_SEED,
        "random-label seed",
    )
    _require_equal(final_step, FINAL_STEP, "final training step")

    optimizer = training_config.get("optimizer")
    schedule = training_config.get("schedule")
    training = training_config.get("training")
    device = training_config.get("device")

    if not isinstance(optimizer, Mapping):
        raise ValueError("Training optimizer configuration is missing.")

    if not isinstance(schedule, Mapping):
        raise ValueError("Training schedule configuration is missing.")

    if not isinstance(training, Mapping):
        raise ValueError("Training configuration is missing.")

    if not isinstance(device, Mapping):
        raise ValueError("Training device configuration is missing.")

    expected_optimizer = {
        "name": "AdamW",
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.98,
        "epsilon": 1.0e-8,
        "weight_decay": 1.0,
    }

    for name, expected in expected_optimizer.items():
        _require_equal(
            optimizer.get(name),
            expected,
            f"optimizer setting {name}",
        )

    expected_schedule = {
        "name": "constant",
        "warmup_steps": 0,
    }

    for name, expected in expected_schedule.items():
        _require_equal(
            schedule.get(name),
            expected,
            f"schedule setting {name}",
        )

    expected_training = {
        "batch_mode": "full_training_set",
        "precision": "float32",
        "gradient_clipping": None,
        "evaluation_interval": EVALUATION_INTERVAL,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "evaluate_step_zero": True,
        "checkpoint_step_zero": True,
        "checkpoint_final_step": True,
    }

    for name, expected in expected_training.items():
        _require_equal(
            training.get(name),
            expected,
            f"training setting {name}",
        )

    configured_horizon = training.get("max_steps")

    if (
        isinstance(configured_horizon, bool)
        or not isinstance(configured_horizon, int)
        or configured_horizon < FINAL_STEP
    ):
        raise ValueError("Primary training horizon is shorter than Stage 14.")

    _require_equal(
        device.get("priority"),
        ["cuda", "cpu"],
        "device priority",
    )

    if "mps" in device.get("priority", ()):
        raise ValueError("MPS must not be available to Stage 14.")

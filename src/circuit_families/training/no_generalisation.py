"""Validated Stage 13 training-data prefixes and provenance."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from circuit_families.analysis.no_generalisation_selection import (
    CANDIDATE_FRACTIONS,
    DESCENDING_CANDIDATE_FRACTIONS,
    MATCHED_HORIZON,
)
from circuit_families.config import (
    mapping_hash,
    stable_run_id_from_hash,
    validate_task_config,
)
from circuit_families.data.modular_addition import hash_named_arrays
from circuit_families.training.data import TrainingData, load_training_data

FROZEN_SUBSET_RECORDS = (
    (0.05, "05pct", 638),
    (0.10, "10pct", 1_276),
    (0.15, "15pct", 1_915),
    (0.20, "20pct", 2_553),
    (0.25, "25pct", 3_192),
)

STAGE13_MODEL_SEED = 0
STAGE13_MATCHED_HORIZON = MATCHED_HORIZON
STAGE13_EXECUTION_ORDER = DESCENDING_CANDIDATE_FRACTIONS
STAGE13_CHECKPOINT_VALIDATION_STEPS = (
    0,
    200,
    3_400,
    4_050,
    5_000,
    7_450,
    8_150,
    8_500,
    8_650,
    9_050,
)

_FRACTION_LABELS = {
    fraction: label
    for fraction, label, _ in FROZEN_SUBSET_RECORDS
}

_FROZEN_COUNTS = {
    fraction: count
    for fraction, _, count in FROZEN_SUBSET_RECORDS
}


@dataclass(frozen=True)
class NoGeneralisationSubset:
    """One exact frozen no-generalisation training prefix."""

    fraction: float
    array_name: str
    subset_identifier: str
    exact_example_count: int
    subset_sha256: str
    source_permutation_sha256: str
    indices: np.ndarray

    def manifest_record(self) -> dict[str, Any]:
        """Return the stable provenance representation."""

        return {
            "fraction": self.fraction,
            "array_name": self.array_name,
            "subset_identifier": self.subset_identifier,
            "exact_example_count": self.exact_example_count,
            "subset_sha256": self.subset_sha256,
            "source_permutation_sha256": (
                self.source_permutation_sha256
            ),
            "nested_prefix": True,
            "true_labels": True,
            "random_labels": False,
        }


def fraction_label(fraction: float) -> str:
    """Return the exact committed percentage label."""

    value = float(fraction)

    try:
        return _FRACTION_LABELS[value]
    except KeyError as error:
        raise ValueError(
            "Fraction is outside the frozen Stage 13 grid."
        ) from error


def frozen_example_count(fraction: float) -> int:
    """Return the exact committed example count without rounding."""

    value = float(fraction)

    try:
        return _FROZEN_COUNTS[value]
    except KeyError as error:
        raise ValueError(
            "Fraction is outside the frozen Stage 13 grid."
        ) from error


def control_array_name(fraction: float) -> str:
    """Return the exact committed dataset-array name."""

    return f"control_train_indices_{fraction_label(fraction)}"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object.")

    return value


def _require_index_array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    expected_count: int,
) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"Dataset archive is missing {name!r}.")

    array = archive[name]

    if array.shape != (expected_count,):
        raise ValueError(
            f"{name} has shape {array.shape}; "
            f"expected {(expected_count,)}."
        )

    if array.dtype != np.dtype(np.int64):
        raise ValueError(
            f"{name} has dtype {array.dtype}; expected int64."
        )

    return array.copy()


def load_and_validate_control_subsets(
    *,
    archive_path: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    task_config: Mapping[str, Any],
) -> tuple[NoGeneralisationSubset, ...]:
    """Load and validate all five exact frozen nested prefixes."""

    validate_task_config(task_config)

    archive_file = Path(archive_path)
    metadata_file = Path(metadata_path)
    manifest_file = Path(manifest_path)

    metadata = _load_json(metadata_file, "dataset metadata")
    manifest = _load_json(manifest_file, "dataset manifest")

    configured_fractions = tuple(
        task_config["split"]["control_train_fractions"]
    )

    if configured_fractions != CANDIDATE_FRACTIONS:
        raise ValueError(
            "Task configuration does not contain the exact "
            "frozen Stage 13 fraction grid."
        )

    try:
        manifest_counts = manifest[
            "details"
        ]["control_training_counts"]
        metadata_counts = metadata[
            "counts"
        ]["control_train"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Dataset records do not contain frozen control counts."
        ) from error

    expected_labels = {
        label
        for _, label, _ in FROZEN_SUBSET_RECORDS
    }

    if set(manifest_counts) != expected_labels:
        raise ValueError(
            "Dataset manifest control-count labels do not match "
            "the frozen Stage 13 grid."
        )

    if set(metadata_counts) != expected_labels:
        raise ValueError(
            "Dataset metadata control-count labels do not match "
            "the frozen Stage 13 grid."
        )

    primary_train_count = int(
        task_config["split"]["primary_train_count"]
    )
    test_count = int(
        task_config["split"]["primary_test_count"]
    )
    total_count = int(
        task_config["task"]["expected_pair_count"]
    )

    subsets: list[NoGeneralisationSubset] = []

    with np.load(archive_file, allow_pickle=False) as archive:
        split_permutation = _require_index_array(
            archive,
            "split_permutation",
            total_count,
        )
        primary_train = _require_index_array(
            archive,
            "train_indices",
            primary_train_count,
        )
        test_indices = _require_index_array(
            archive,
            "test_indices",
            test_count,
        )

        if "true_labels" not in archive.files:
            raise ValueError(
                "Dataset archive is missing 'true_labels'."
            )

        true_labels = archive["true_labels"]

        if true_labels.shape != (total_count,):
            raise ValueError(
                "True-label array has the wrong shape."
            )

        if true_labels.dtype != np.dtype(np.int16):
            raise ValueError(
                "True-label array must use the frozen int16 dtype."
            )

        if not np.array_equal(
            primary_train,
            split_permutation[:primary_train_count],
        ):
            raise ValueError(
                "Primary training indices are not the frozen "
                "permutation prefix."
            )

        if not np.array_equal(
            test_indices,
            split_permutation[primary_train_count:],
        ):
            raise ValueError(
                "Test indices do not match the frozen permutation tail."
            )

        if np.intersect1d(
            primary_train,
            test_indices,
        ).size:
            raise ValueError(
                "Primary training and test partitions overlap."
            )

        source_permutation_sha256 = hash_named_arrays(
            {"split_permutation": split_permutation}
        )

        previous_indices: np.ndarray | None = None

        for fraction, label, committed_count in FROZEN_SUBSET_RECORDS:
            manifest_count = manifest_counts[label]
            metadata_count = metadata_counts[label]

            if (
                isinstance(manifest_count, bool)
                or not isinstance(manifest_count, int)
                or isinstance(metadata_count, bool)
                or not isinstance(metadata_count, int)
            ):
                raise ValueError(
                    f"Frozen count for {label} must be an integer."
                )

            if (
                manifest_count != committed_count
                or metadata_count != committed_count
            ):
                raise ValueError(
                    f"Frozen example count mismatch for {label}."
                )

            name = control_array_name(fraction)
            indices = _require_index_array(
                archive,
                name,
                committed_count,
            )

            if not np.array_equal(
                indices,
                split_permutation[:committed_count],
            ):
                raise ValueError(
                    f"{label} is not the exact frozen "
                    "permutation prefix."
                )

            if not np.array_equal(
                indices,
                primary_train[:committed_count],
            ):
                raise ValueError(
                    f"{label} is not a prefix of the primary "
                    "training partition."
                )

            if np.unique(indices).size != committed_count:
                raise ValueError(
                    f"{label} contains duplicate indices."
                )

            if (
                indices.min() < 0
                or indices.max() >= total_count
            ):
                raise ValueError(
                    f"{label} contains out-of-range indices."
                )

            if np.intersect1d(
                indices,
                test_indices,
            ).size:
                raise ValueError(
                    f"{label} overlaps the test partition."
                )

            if (
                previous_indices is not None
                and not np.array_equal(
                    indices[: previous_indices.size],
                    previous_indices,
                )
            ):
                raise ValueError(
                    "Frozen control subsets are not nested."
                )

            subset_sha256 = hash_named_arrays(
                {name: indices}
            )

            subsets.append(
                NoGeneralisationSubset(
                    fraction=fraction,
                    array_name=name,
                    subset_identifier=(
                        f"frozen_{label}_training_prefix"
                    ),
                    exact_example_count=committed_count,
                    subset_sha256=subset_sha256,
                    source_permutation_sha256=(
                        source_permutation_sha256
                    ),
                    indices=indices,
                )
            )

            previous_indices = indices

    return tuple(subsets)


def subset_by_fraction(
    subsets: Sequence[NoGeneralisationSubset],
    fraction: float,
) -> NoGeneralisationSubset:
    """Return one exact validated candidate subset."""

    requested = float(fraction)

    if requested not in CANDIDATE_FRACTIONS:
        raise ValueError(
            "Fraction is outside the frozen Stage 13 grid."
        )

    matches = [
        subset
        for subset in subsets
        if subset.fraction == requested
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one validated subset for the "
            "candidate fraction."
        )

    return matches[0]


def load_no_generalisation_training_data(
    *,
    archive_path: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    task_config: Mapping[str, Any],
    device: str | torch.device,
    subset: NoGeneralisationSubset,
) -> TrainingData:
    """Load the unchanged test set and exact smaller train prefix."""

    canonical_subsets = load_and_validate_control_subsets(
        archive_path=archive_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        task_config=task_config,
    )
    canonical_subset = subset_by_fraction(
        canonical_subsets,
        subset.fraction,
    )

    if (
        subset.array_name != canonical_subset.array_name
        or subset.subset_identifier
        != canonical_subset.subset_identifier
        or subset.exact_example_count
        != canonical_subset.exact_example_count
        or subset.subset_sha256
        != canonical_subset.subset_sha256
        or subset.source_permutation_sha256
        != canonical_subset.source_permutation_sha256
        or not np.array_equal(
            subset.indices,
            canonical_subset.indices,
        )
    ):
        raise ValueError(
            "Requested control subset does not match the "
            "validated frozen prefix."
        )

    primary = load_training_data(
        archive_path=archive_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        task_config=task_config,
        device=device,
    )

    if (
        primary.full_inputs is None
        or primary.full_targets is None
    ):
        raise ValueError(
            "Validated primary data did not retain full arrays."
        )

    selected_device = torch.device(device)
    index_tensor = torch.as_tensor(
        canonical_subset.indices,
        dtype=torch.long,
        device=selected_device,
    )

    train_inputs = primary.full_inputs.index_select(
        0,
        index_tensor,
    )
    train_targets = primary.full_targets.index_select(
        0,
        index_tensor,
    )

    dataset_hashes = dict(primary.dataset_hashes)
    dataset_hashes.update(
        {
            "control_subset_sha256": (
                canonical_subset.subset_sha256
            ),
            "source_permutation_sha256": (
                canonical_subset.source_permutation_sha256
            ),
        }
    )

    return TrainingData(
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=primary.test_inputs,
        test_targets=primary.test_targets,
        dataset_hashes=dataset_hashes,
        archive_path=primary.archive_path,
        metadata_path=primary.metadata_path,
        manifest_path=primary.manifest_path,
        archive_sha256=primary.archive_sha256,
        metadata_sha256=primary.metadata_sha256,
        total_count=primary.total_count,
        train_count=canonical_subset.exact_example_count,
        test_count=primary.test_count,
        full_inputs=primary.full_inputs,
        full_targets=primary.full_targets,
        training_subset=canonical_subset.manifest_record(),
    )

def validate_requested_fractions(
    fractions: Sequence[float],
) -> tuple[float, ...]:
    """Validate the complete grid and return execution order."""

    converted: list[float] = []

    for fraction in fractions:
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
        ):
            raise TypeError(
                "Stage 13 candidate fractions must be numeric."
            )

        converted.append(float(fraction))

    if len(converted) != len(CANDIDATE_FRACTIONS):
        raise ValueError(
            "Stage 13 requires exactly five candidate fractions."
        )

    if len(set(converted)) != len(converted):
        raise ValueError(
            "Stage 13 candidate fractions must be unique."
        )

    if set(converted) != set(CANDIDATE_FRACTIONS):
        raise ValueError(
            "Stage 13 fractions must equal the frozen "
            "0.05, 0.10, 0.15, 0.20 and 0.25 grid."
        )

    return STAGE13_EXECUTION_ORDER


def validate_stage13_training_settings(
    training_config: Mapping[str, Any],
    *,
    model_seed: int,
    final_step: int,
) -> None:
    """Validate the frozen settings used by every Stage 13 pilot."""

    if (
        isinstance(model_seed, bool)
        or not isinstance(model_seed, int)
    ):
        raise TypeError("Stage 13 model seed must be an integer.")

    if model_seed != STAGE13_MODEL_SEED:
        raise ValueError("Stage 13 model seed must equal 0.")

    if isinstance(final_step, bool) or not isinstance(final_step, int):
        raise TypeError("Stage 13 final step must be an integer.")

    if final_step != STAGE13_MATCHED_HORIZON:
        raise ValueError(
            "Stage 13 final step must equal the frozen "
            f"matched horizon {STAGE13_MATCHED_HORIZON}."
        )

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
        if optimizer.get(name) != expected:
            raise ValueError(
                "Stage 13 optimizer setting mismatch for "
                f"{name}: expected {expected!r}."
            )

    expected_schedule = {
        "name": "constant",
        "warmup_steps": 0,
    }

    for name, expected in expected_schedule.items():
        if schedule.get(name) != expected:
            raise ValueError(
                "Stage 13 schedule setting mismatch for "
                f"{name}: expected {expected!r}."
            )

    expected_training = {
        "batch_mode": "full_training_set",
        "precision": "float32",
        "gradient_clipping": None,
        "evaluation_interval": 50,
        "checkpoint_interval": 50,
        "evaluate_step_zero": True,
        "checkpoint_step_zero": True,
        "checkpoint_final_step": True,
    }

    for name, expected in expected_training.items():
        if training.get(name) != expected:
            raise ValueError(
                "Stage 13 training setting mismatch for "
                f"{name}: expected {expected!r}."
            )

    if training.get("max_steps") < STAGE13_MATCHED_HORIZON:
        raise ValueError(
            "Primary training horizon is shorter than Stage 13."
        )

    if device.get("priority") != ["cuda", "cpu"]:
        raise ValueError(
            "Stage 13 device priority must remain CUDA then CPU."
        )

    if "mps" in device.get("priority", ()):
        raise ValueError("MPS must not be available to Stage 13.")


def deterministic_stage13_run_id(
    configuration: Mapping[str, Any],
) -> str:
    """Derive the Stage 13 pilot-grid ID from frozen provenance."""

    return stable_run_id_from_hash(
        "stage13_no_generalisation",
        STAGE13_MODEL_SEED,
        mapping_hash(dict(configuration)),
    )

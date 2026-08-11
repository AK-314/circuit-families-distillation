"""Generate the frozen modular-addition dataset and provenance records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from circuit_families.config import config_hash, load_config, stable_run_id
from circuit_families.data.modular_addition import (
    generate_modular_addition_dataset,
    hash_named_arrays,
)
from circuit_families.data.random_labels import generate_random_labels
from circuit_families.data.splits import generate_splits
from circuit_families.manifests import create_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate the frozen modular-addition dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the task YAML configuration.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def array_metadata(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return JSON-safe shape and dtype metadata for saved arrays."""

    return {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        for name, array in sorted(arrays.items())
    }


def generate_dataset(config_path: Path) -> dict[str, str]:
    """Generate arrays, metadata, and the dataset manifest."""

    repository_root = Path.cwd()
    config = load_config(config_path)

    task = config["task"]
    split_config = config["split"]
    random_label_config = config["random_labels"]
    outputs = config["outputs"]

    dataset_arrays = generate_modular_addition_dataset(
        modulus=task["modulus"],
        equals_token_id=task["equals_token_id"],
    )

    split_arrays = generate_splits(
        total_examples=task["expected_pair_count"],
        split_seed=split_config["seed"],
        primary_train_count=split_config["primary_train_count"],
        control_fractions=split_config["control_train_fractions"],
    )

    random_label_permutation, random_labels = generate_random_labels(
        dataset_arrays["true_labels"],
        seed=random_label_config["seed"],
    )
    random_label_arrays = {
        "random_label_permutation": random_label_permutation,
        "random_labels": random_labels,
    }

    all_arrays = {
        **dataset_arrays,
        **split_arrays,
        **random_label_arrays,
    }

    dataset_sha256 = hash_named_arrays(dataset_arrays)
    split_sha256 = hash_named_arrays(split_arrays)
    random_labels_sha256 = hash_named_arrays(
        {"random_labels": random_labels}
    )
    random_label_permutation_sha256 = hash_named_arrays(
        {"random_label_permutation": random_label_permutation}
    )
    config_sha256 = config_hash(config)

    run_id = stable_run_id(
        config["experiment_type"],
        split_config["seed"],
        config,
    )

    data_directory = repository_root / outputs["data_directory"]
    manifest_directory = repository_root / outputs["manifest_directory"]
    data_directory.mkdir(parents=True, exist_ok=True)
    manifest_directory.mkdir(parents=True, exist_ok=True)

    dataset_path = data_directory / outputs["dataset_filename"]
    metadata_path = data_directory / outputs["metadata_filename"]
    manifest_path = manifest_directory / f"dataset_{run_id}.json"

    np.savez_compressed(dataset_path, **all_arrays)

    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_type": config["experiment_type"],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "task": {
            "name": task["name"],
            "modulus": task["modulus"],
            "pair_order": task["pair_order"],
            "equals_token_id": task["equals_token_id"],
            "target_rule": "(a + b) % 113",
        },
        "counts": {
            "total_examples": task["expected_pair_count"],
            "primary_train": split_config["primary_train_count"],
            "primary_test": split_config["primary_test_count"],
            "control_train": {
                f"{round(fraction * 100):02d}pct": int(
                    np.floor(task["expected_pair_count"] * fraction)
                )
                for fraction in split_config["control_train_fractions"]
            },
        },
        "seeds": {
            "dataset_seed": None,
            "split_seed": split_config["seed"],
            "random_label_seed": random_label_config["seed"],
        },
        "hashes": {
            "dataset_sha256": dataset_sha256,
            "split_sha256": split_sha256,
            "random_labels_sha256": random_labels_sha256,
            "random_label_permutation_sha256": (
                random_label_permutation_sha256
            ),
        },
        "arrays": array_metadata(all_arrays),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    archive_sha256 = file_sha256(dataset_path)
    metadata_sha256 = file_sha256(metadata_path)

    manifest = create_manifest(
        run_id=run_id,
        experiment_type=config["experiment_type"],
        repository_root=repository_root,
        config_path=config_path,
        config_sha256=config_sha256,
        seed_name="split_seed",
        seed=split_config["seed"],
        output_paths={
            "dataset_archive": dataset_path.relative_to(repository_root),
            "dataset_metadata": metadata_path.relative_to(repository_root),
            "dataset_manifest": manifest_path.relative_to(repository_root),
        },
        hashes={
            "archive_sha256": archive_sha256,
            "config_sha256": config_sha256,
            "dataset_sha256": dataset_sha256,
            "metadata_sha256": metadata_sha256,
            "random_label_permutation_sha256": (
                random_label_permutation_sha256
            ),
            "random_labels_sha256": random_labels_sha256,
            "split_sha256": split_sha256,
        },
        details={
            "dataset_seed": None,
            "split_seed": split_config["seed"],
            "random_label_seed": random_label_config["seed"],
            "total_example_count": task["expected_pair_count"],
            "primary_training_count": split_config[
                "primary_train_count"
            ],
            "test_count": split_config["primary_test_count"],
            "control_training_counts": metadata["counts"][
                "control_train"
            ],
            "numpy_generator": split_config["generator"],
            "random_label_method": random_label_config["method"],
        },
    )
    write_manifest(manifest_path, manifest)

    return {
        "run_id": run_id,
        "dataset_archive": str(dataset_path.relative_to(repository_root)),
        "dataset_metadata": str(metadata_path.relative_to(repository_root)),
        "dataset_manifest": str(manifest_path.relative_to(repository_root)),
        "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "random_labels_sha256": random_labels_sha256,
        "random_label_permutation_sha256": (
            random_label_permutation_sha256
        ),
        "archive_sha256": archive_sha256,
    }


def main() -> None:
    """Generate the dataset and print its reproducibility identifiers."""

    args = parse_args()
    results = generate_dataset(args.config)

    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

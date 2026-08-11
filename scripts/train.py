"""Run frozen modular-addition Transformer training."""

from __future__ import annotations

import argparse
from pathlib import Path

from circuit_families.training.run import run_training


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train the frozen modular-addition Transformer."
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path("configs/task.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model.yaml"),
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training.yaml"),
    )
    parser.add_argument(
        "--dataset-archive",
        type=Path,
        default=Path(
            "data/generated/modular_addition_m113.npz"
        ),
    )
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=Path(
            "data/generated/"
            "modular_addition_m113.metadata.json"
        ),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path(
            "manifests/"
            "dataset_modular-addition-dataset-s0-"
            "7ef9c73ff18f.json"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Model initialisation seed.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Optional explicit device override.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the frozen five-step smoke schedule.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Root under which configured output directories are created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing run with the same stable run ID.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute training and print its reproducibility identifiers."""

    args = parse_args()

    result = run_training(
        repository_root=Path.cwd(),
        task_config_path=args.task_config,
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        dataset_archive_path=args.dataset_archive,
        dataset_metadata_path=args.dataset_metadata,
        dataset_manifest_path=args.dataset_manifest,
        model_seed=args.seed,
        smoke=args.smoke,
        device_override=args.device,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )

    print(f"run_id: {result.run_id}")
    print(f"mode: {result.mode}")
    print(f"device: {result.device}")
    print(f"final_step: {result.final_step}")
    print(f"checkpoint_count: {result.checkpoint_count}")
    print(
        "combined_config_sha256: "
        f"{result.combined_config_sha256}"
    )
    print(f"metrics_path: {result.metrics_path}")
    print(f"manifest_path: {result.manifest_path}")
    print(
        "checkpoint_directory: "
        f"{result.checkpoint_directory}"
    )


if __name__ == "__main__":
    main()

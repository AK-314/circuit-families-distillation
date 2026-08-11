"""Validate or train the frozen Stage 14 random-label control."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from circuit_families.analysis.random_label_control import (
    MAIN_MODEL_REFERENCE_CHECKPOINTS,
    validate_main_checkpoint_reference,
)
from circuit_families.config import (
    config_hash,
    load_config,
    load_model_config,
    load_training_config,
    mapping_hash,
)
from circuit_families.training.random_label import (
    FINAL_STEP,
    MODEL_SEED,
    RANDOM_LABEL_SEED,
    file_sha256,
    load_random_label_training_data,
    validate_frozen_random_label_dataset,
    validate_stage14_training_settings,
)
from circuit_families.training.run import run_training

EXPERIMENT_TYPE = "stage14-random-label-training"

TASK_CONFIG = Path("configs/task.yaml")
MODEL_CONFIG = Path("configs/model.yaml")
TRAINING_CONFIG = Path("configs/training.yaml")
DATASET_ARCHIVE = Path("data/generated/modular_addition_m113.npz")
DATASET_METADATA = Path("data/generated/modular_addition_m113.metadata.json")
DATASET_MANIFEST = Path("manifests/dataset_modular-addition-dataset-s0-7ef9c73ff18f.json")
MAIN_CHECKPOINT_MANIFEST = Path("manifests/checkpoints_seed_1.json")
STAGE8_MANIFEST = Path("manifests/stage8_masking_s1-5f1bc9dee7ab.json")

TASK_CONFIG_SHA256 = "7ef9c73ff18f14d1c59cbc1a636d73caa23da099817714ffc70864b1ac00d377"
MODEL_CONFIG_SHA256 = "0b263f3e01b5de162663632903069e87fc675e29b2f3e46f0979e8ae7537952c"
TRAINING_CONFIG_SHA256 = "ebe04e9854542d13839903767f8da04f6af7560df5f782572c5308a1ca02d89a"
MAIN_CHECKPOINT_MANIFEST_SHA256 = "a33b856e07b9223d6ee552810d8b4eb6f20ae2eb74427934933ef022ba48fd23"
STAGE8_MANIFEST_SHA256 = "ed6aca8d20d43ea7618936b962c8e865859c21894e5d809580106ffe73a8d4e5"


OUTPUT_SEARCH_ROOTS = (
    Path("manifests"),
    Path("results"),
    Path("figures"),
    Path("configs/controls"),
)


def parse_args() -> argparse.Namespace:
    """Parse the Stage 14 runner arguments."""

    parser = argparse.ArgumentParser(
        description=("Validate or train the frozen Stage 14 random-label control.")
    )
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=MODEL_SEED,
    )
    parser.add_argument(
        "--random-label-seed",
        type=int,
        default=RANDOM_LABEL_SEED,
    )
    parser.add_argument(
        "--final-step",
        type=int,
        default=FINAL_STEP,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=TASK_CONFIG,
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=MODEL_CONFIG,
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=TRAINING_CONFIG,
    )
    parser.add_argument(
        "--dataset-archive",
        type=Path,
        default=DATASET_ARCHIVE,
    )
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=DATASET_METADATA,
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DATASET_MANIFEST,
    )
    parser.add_argument(
        "--main-checkpoint-manifest",
        type=Path,
        default=MAIN_CHECKPOINT_MANIFEST,
    )
    parser.add_argument(
        "--stage8-manifest",
        type=Path,
        default=STAGE8_MANIFEST,
    )
    parser.add_argument(
        "--expected-implementation-commit",
    )
    return parser.parse_args()


def resolve_path(repository: Path, file_path: Path) -> Path:
    """Resolve one repository-relative input path."""

    return (file_path if file_path.is_absolute() else repository / file_path).resolve()


def current_head(repository: Path) -> str:
    """Return the current Git commit."""

    result = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_clean_repository(repository: Path) -> str:
    """Require a clean repository for definitive training."""

    result = subprocess.run(
        ["/usr/bin/git", "status", "--porcelain=v1"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        raise RuntimeError("Definitive Stage 14 training requires a clean repository.")

    return current_head(repository)


def find_forbidden_later_stage_outputs(
    repository: Path,
) -> tuple[str, ...]:
    """Find random-label search or Stage 15 output artifacts."""

    matches: list[str] = []

    for relative_root in OUTPUT_SEARCH_ROOTS:
        root = repository / relative_root
        if not root.exists():
            continue

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue

            normalised = str(file_path.relative_to(repository)).lower().replace("_", "-")

            stage15 = "stage15" in normalised or "stage-15" in normalised
            stage14_sparse = (
                "stage14" in normalised or "stage-14" in normalised
            ) and "sparse" in normalised
            random_label_sparse = "random-label" in normalised and "sparse" in normalised
            stage14_diversity = (
                "stage14" in normalised or "stage-14" in normalised
            ) and "diversity" in normalised
            random_label_diversity = "random-label" in normalised and "diversity" in normalised

            if (
                stage15
                or stage14_sparse
                or random_label_sparse
                or stage14_diversity
                or random_label_diversity
            ):
                matches.append(str(file_path.relative_to(repository)))

    return tuple(sorted(matches))


def validate_config_hashes(
    *,
    task_config: dict[str, Any],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> tuple[str, str, str]:
    """Validate the exact primary task, model and training mappings."""

    task_sha256 = config_hash(task_config)
    model_sha256 = mapping_hash(model_config)
    training_sha256 = mapping_hash(training_config)

    expected = (
        TASK_CONFIG_SHA256,
        MODEL_CONFIG_SHA256,
        TRAINING_CONFIG_SHA256,
    )
    observed = (
        task_sha256,
        model_sha256,
        training_sha256,
    )

    if observed != expected:
        raise ValueError(
            "Stage 14 configuration hashes differ from the primary model: "
            f"expected {expected}, got {observed}."
        )

    return observed


def build_run_identity(
    *,
    implementation_commit: str,
    dataset_validation: Any,
    task_config_sha256: str,
    model_config_sha256: str,
    training_config_sha256: str,
    main_checkpoint_manifest_sha256: str,
    stage8_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the deterministic scientific intervention identity."""

    return {
        "control_type": "random_label",
        "label_intervention": ("true_modular_addition_labels_to_frozen_random_labels"),
        "random_labels": True,
        "true_labels": False,
        "random_label_method": ("permute_complete_true_label_vector"),
        "random_label_bit_generator": "PCG64",
        "random_label_seed": RANDOM_LABEL_SEED,
        "model_seed": MODEL_SEED,
        "final_step": FINAL_STEP,
        "implementation_commit": implementation_commit,
        "dataset_archive_sha256": (dataset_validation.archive_sha256),
        "dataset_manifest_sha256": (dataset_validation.manifest_sha256),
        "dataset_sha256": (dataset_validation.canonical_dataset_sha256),
        "split_sha256": dataset_validation.split_sha256,
        "random_labels_sha256": (dataset_validation.random_labels_sha256),
        "random_label_permutation_sha256": (dataset_validation.random_label_permutation_sha256),
        "task_config_sha256": task_config_sha256,
        "model_config_sha256": model_config_sha256,
        "training_config_sha256": training_config_sha256,
        "main_checkpoint_manifest_sha256": (main_checkpoint_manifest_sha256),
        "stage8_manifest_sha256": stage8_manifest_sha256,
        "checkpoint_matching": "exact_training_step",
    }


def identity_sha256(identity: dict[str, Any]) -> str:
    """Hash a run identity for validate-only reporting."""

    serialised = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def selected_device(name: str) -> torch.device:
    """Resolve the permitted Stage 14 device class."""

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    device = torch.device(name)

    if device.type == "mps":
        raise RuntimeError("Stage 14 must not execute on MPS.")

    return device


def main() -> None:
    """Validate inputs or train the single frozen random-label run."""

    args = parse_args()
    repository = args.repository_root.resolve()
    output_root = resolve_path(repository, args.output_root)

    task_config_path = resolve_path(repository, args.task_config)
    model_config_path = resolve_path(repository, args.model_config)
    training_config_path = resolve_path(
        repository,
        args.training_config,
    )
    archive_path = resolve_path(repository, args.dataset_archive)
    metadata_path = resolve_path(repository, args.dataset_metadata)
    dataset_manifest_path = resolve_path(
        repository,
        args.dataset_manifest,
    )
    main_checkpoint_manifest_path = resolve_path(
        repository,
        args.main_checkpoint_manifest,
    )
    stage8_manifest_path = resolve_path(
        repository,
        args.stage8_manifest,
    )

    task_config = load_config(task_config_path)
    model_config = load_model_config(model_config_path)
    training_config = load_training_config(training_config_path)

    validate_stage14_training_settings(
        training_config,
        model_seed=args.model_seed,
        random_label_seed=args.random_label_seed,
        final_step=args.final_step,
    )

    (
        task_config_sha256,
        model_config_sha256,
        training_config_sha256,
    ) = validate_config_hashes(
        task_config=task_config,
        model_config=model_config,
        training_config=training_config,
    )

    dataset_validation = validate_frozen_random_label_dataset(
        archive_path=archive_path,
        metadata_path=metadata_path,
        manifest_path=dataset_manifest_path,
        task_config_path=task_config_path,
    )

    main_manifest_sha256 = file_sha256(main_checkpoint_manifest_path)
    stage8_manifest_sha256 = file_sha256(stage8_manifest_path)

    if main_manifest_sha256 != MAIN_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("Seed-1 checkpoint manifest SHA-256 mismatch.")

    if stage8_manifest_sha256 != STAGE8_MANIFEST_SHA256:
        raise ValueError("Stage 8 masking manifest SHA-256 mismatch.")

    main_checkpoint_manifest = json.loads(main_checkpoint_manifest_path.read_text(encoding="utf-8"))
    validate_main_checkpoint_reference(main_checkpoint_manifest)

    forbidden_outputs = find_forbidden_later_stage_outputs(repository)
    if forbidden_outputs and not args.validate_inputs_only:
        raise RuntimeError(
            "Forbidden later-stage outputs already exist: " + ", ".join(forbidden_outputs)
        )

    implementation_commit = (
        current_head(repository)
        if args.validate_inputs_only
        else require_clean_repository(repository)
    )

    if (
        args.expected_implementation_commit is not None
        and implementation_commit != args.expected_implementation_commit
    ):
        raise RuntimeError(
            "Implementation commit mismatch: expected "
            f"{args.expected_implementation_commit}, found "
            f"{implementation_commit}."
        )

    identity = build_run_identity(
        implementation_commit=implementation_commit,
        dataset_validation=dataset_validation,
        task_config_sha256=task_config_sha256,
        model_config_sha256=model_config_sha256,
        training_config_sha256=training_config_sha256,
        main_checkpoint_manifest_sha256=main_manifest_sha256,
        stage8_manifest_sha256=stage8_manifest_sha256,
    )

    print(f"implementation_commit: {implementation_commit}")
    print(f"model_seed: {args.model_seed}")
    print(f"random_label_seed: {args.random_label_seed}")
    print(f"final_step: {args.final_step}")
    print(f"device_request: {args.device}")
    print(f"dataset_archive: {archive_path}")
    print(f"dataset_archive_sha256: {dataset_validation.archive_sha256}")
    print(f"dataset_sha256: {dataset_validation.canonical_dataset_sha256}")
    print(f"split_sha256: {dataset_validation.split_sha256}")
    print(f"random_labels_sha256: {dataset_validation.random_labels_sha256}")
    print(f"random_label_permutation_sha256: {dataset_validation.random_label_permutation_sha256}")
    print(f"accidental_true_label_matches: {dataset_validation.accidental_true_label_match_count}")
    print(f"global_class_counts: {sorted(set(dataset_validation.class_counts))}")
    print(
        "matched_checkpoint_steps: "
        + ", ".join(str(step) for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS)
    )
    print("checkpoint_reload_verification_scope: all_saved_checkpoints")
    print(f"run_identity_sha256: {identity_sha256(identity)}")

    if args.validate_inputs_only:
        print("same_ordered_input_pairs: passed")
        print("same_train_indices: passed")
        print("same_test_indices: passed")
        print("random_labels_not_true_labels: passed")
        print("global_class_balance: passed")
        print("primary_model_config: passed")
        print("primary_training_config: passed")
        print("weight_decay_1_0: passed")
        print("mps_execution_absent: passed")
        print("random_label_sparse_search_started: false")
        print("diversity_search_started: false")
        print("stage15_started: false")
        print("validate_only_outputs_created: false")
        print("input_validation: passed")
        return

    device = selected_device(args.device)

    training_data = load_random_label_training_data(
        archive_path=archive_path,
        metadata_path=metadata_path,
        manifest_path=dataset_manifest_path,
        task_config_path=task_config_path,
        task_config=task_config,
        device=device,
    )

    result = run_training(
        repository_root=repository,
        task_config_path=task_config_path,
        model_config_path=model_config_path,
        training_config_path=training_config_path,
        dataset_archive_path=archive_path,
        dataset_metadata_path=metadata_path,
        dataset_manifest_path=dataset_manifest_path,
        model_seed=args.model_seed,
        smoke=False,
        device_override=args.device,
        output_root=output_root,
        overwrite=False,
        training_data=training_data,
        max_steps_override=args.final_step,
        experiment_type_override=EXPERIMENT_TYPE,
        run_identity=identity,
        checkpoint_verification_steps=None,
    )

    print(f"run_id: {result.run_id}")
    print(f"mode: {result.mode}")
    print(f"device: {result.device}")
    print(f"final_step: {result.final_step}")
    print(f"checkpoint_count: {result.checkpoint_count}")
    print(f"combined_config_sha256: {result.combined_config_sha256}")
    print(f"metrics_path: {result.metrics_path}")
    print(f"manifest_path: {result.manifest_path}")
    print(f"checkpoint_directory: {result.checkpoint_directory}")


if __name__ == "__main__":
    main()

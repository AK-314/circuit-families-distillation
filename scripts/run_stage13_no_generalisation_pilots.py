"""Train the frozen Stage 13 no-generalisation pilot grid."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from circuit_families.analysis.no_generalisation_selection import (
    CANDIDATE_FRACTIONS,
)
from circuit_families.config import (
    combined_config_hash,
    config_hash,
    load_config,
    load_model_config,
    load_training_config,
    mapping_hash,
    stable_run_id_from_hash,
)
from circuit_families.training.checkpoints import file_sha256
from circuit_families.training.device import resolve_device
from circuit_families.training.no_generalisation import (
    STAGE13_CHECKPOINT_VALIDATION_STEPS,
    STAGE13_EXECUTION_ORDER,
    STAGE13_MATCHED_HORIZON,
    STAGE13_MODEL_SEED,
    NoGeneralisationSubset,
    deterministic_stage13_run_id,
    load_and_validate_control_subsets,
    load_no_generalisation_training_data,
    subset_by_fraction,
    validate_requested_fractions,
    validate_stage13_training_settings,
)
from circuit_families.training.run import (
    build_execution_plan,
    run_training,
)

EXPERIMENT_TYPE = "stage13_no_generalisation_training"


def parse_args() -> argparse.Namespace:
    """Parse Stage 13 pilot-grid arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate or train the frozen Stage 13 "
            "no-generalisation pilot grid."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
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
        "--fractions",
        type=float,
        nargs="+",
        default=list(CANDIDATE_FRACTIONS),
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=STAGE13_MODEL_SEED,
    )
    parser.add_argument(
        "--final-step",
        type=int,
        default=STAGE13_MATCHED_HORIZON,
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--expected-implementation-commit",
        default=None,
    )
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
    )
    return parser.parse_args()


def resolve_path(repository: Path, value: Path) -> Path:
    """Resolve a path relative to the repository."""

    return value if value.is_absolute() else repository / value


def display_path(path: Path, root: Path) -> str:
    """Render a repository-relative path when possible."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def git_output(repository: Path, *arguments: str) -> str:
    """Run Git and return stripped standard output."""

    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_clean_repository(repository: Path) -> str:
    """Require a clean implementation commit for scientific outputs."""

    status = git_output(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )

    if status:
        raise RuntimeError(
            "Stage 13 scientific outputs require a clean "
            "implementation commit. Current status:\n"
            + status
        )

    return git_output(repository, "rev-parse", "HEAD")


def current_head(repository: Path) -> str:
    """Return HEAD without requiring a clean worktree."""

    return git_output(repository, "rev-parse", "HEAD")


def build_stage13_configuration(
    *,
    repository: Path,
    task_config_path: Path,
    model_config_path: Path,
    training_config_path: Path,
    dataset_manifest_path: Path,
    subsets: tuple[NoGeneralisationSubset, ...],
    implementation_commit: str,
) -> dict[str, Any]:
    """Build deterministic Stage 13 grid provenance."""

    task_config = load_config(task_config_path)
    model_config = load_model_config(model_config_path)
    training_config = load_training_config(training_config_path)
    plan = build_execution_plan(
        training_config,
        smoke=False,
        max_steps_override=STAGE13_MATCHED_HORIZON,
    )

    return {
        "source_dataset_manifest": {
            "path": display_path(
                dataset_manifest_path,
                repository,
            ),
            "sha256": file_sha256(dataset_manifest_path),
        },
        "task_config": {
            "path": display_path(
                task_config_path,
                repository,
            ),
            "sha256": config_hash(task_config),
        },
        "model_config": {
            "path": display_path(
                model_config_path,
                repository,
            ),
            "sha256": mapping_hash(model_config),
        },
        "training_config": {
            "path": display_path(
                training_config_path,
                repository,
            ),
            "sha256": mapping_hash(training_config),
        },
        "model_seed": STAGE13_MODEL_SEED,
        "matched_horizon": STAGE13_MATCHED_HORIZON,
        "execution": asdict(plan),
        "execution_order": list(STAGE13_EXECUTION_ORDER),
        "checkpoint_validation_steps": list(
            STAGE13_CHECKPOINT_VALIDATION_STEPS
        ),
        "candidate_subsets": [
            subset.manifest_record()
            for subset in subsets
        ],
        "implementation_commit": implementation_commit,
        "curve_only_selection": True,
        "circuit_family_metrics_permitted": False,
        "stage14_started": False,
        "stage15_started": False,
    }


def candidate_run_identity(
    *,
    stage13_run_id: str,
    subset: NoGeneralisationSubset,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Build one candidate's deterministic training identity."""

    return {
        "stage": 13,
        "stage13_run_id": stage13_run_id,
        "source_dataset_manifest_path": (
            configuration["source_dataset_manifest"]["path"]
        ),
        "source_dataset_manifest_sha256": (
            configuration["source_dataset_manifest"]["sha256"]
        ),
        "candidate_fraction": subset.fraction,
        "exact_training_example_count": (
            subset.exact_example_count
        ),
        "subset_identifier": subset.subset_identifier,
        "subset_sha256": subset.subset_sha256,
        "source_permutation_sha256": (
            subset.source_permutation_sha256
        ),
        "model_config_sha256": (
            configuration["model_config"]["sha256"]
        ),
        "training_config_sha256": (
            configuration["training_config"]["sha256"]
        ),
        "model_seed": configuration["model_seed"],
        "matched_horizon": configuration["matched_horizon"],
        "implementation_commit": (
            configuration["implementation_commit"]
        ),
        "true_labels": True,
        "random_labels": False,
        "test_set_unchanged": True,
    }


def candidate_training_run_id(
    *,
    task_config: dict[str, Any],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    run_identity: dict[str, Any],
) -> tuple[str, str]:
    """Return the exact candidate run ID and combined hash."""

    plan = build_execution_plan(
        training_config,
        smoke=False,
        max_steps_override=STAGE13_MATCHED_HORIZON,
    )
    combined_sha256 = combined_config_hash(
        {
            "task": task_config,
            "model": model_config,
            "training": training_config,
            "execution": asdict(plan),
            "run_identity": run_identity,
        }
    )
    run_id = stable_run_id_from_hash(
        EXPERIMENT_TYPE,
        STAGE13_MODEL_SEED,
        combined_sha256,
    )
    return run_id, combined_sha256


def expected_output_paths(
    *,
    output_root: Path,
    training_config: dict[str, Any],
    run_id: str,
) -> tuple[Path, Path, Path]:
    """Return the three existing-run collision paths."""

    outputs = training_config["outputs"]

    return (
        output_root
        / outputs["checkpoint_directory"]
        / run_id,
        output_root
        / outputs["results_directory"]
        / run_id,
        output_root
        / outputs["manifest_directory"]
        / f"training_{run_id}.json",
    )


def validate_absent_candidate_outputs(
    *,
    output_root: Path,
    training_config: dict[str, Any],
    run_ids: tuple[str, ...],
) -> None:
    """Refuse partial or colliding definitive candidate outputs."""

    existing: list[Path] = []

    for run_id in run_ids:
        existing.extend(
            path
            for path in expected_output_paths(
                output_root=output_root,
                training_config=training_config,
                run_id=run_id,
            )
            if path.exists()
        )

    if existing:
        raise FileExistsError(
            "Stage 13 refuses to overwrite existing candidate "
            "outputs:\n"
            + "\n".join(str(path) for path in existing)
        )



def stable_json_write(
    path: Path,
    value: dict[str, Any],
) -> Path:
    """Write stable JSON without runtime-dependent formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def candidate_registry_path(
    *,
    output_root: Path,
    stage13_run_id: str,
) -> Path:
    """Return the deterministic complete-grid registry path."""

    return (
        output_root
        / "results"
        / "raw"
        / stage13_run_id
        / "candidate_runs.json"
    )

def main() -> None:
    """Validate inputs or train all five frozen candidates."""

    args = parse_args()
    repository = args.repository_root.resolve()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else repository / args.output_root
    ).resolve()

    task_config_path = resolve_path(
        repository,
        args.task_config,
    )
    model_config_path = resolve_path(
        repository,
        args.model_config,
    )
    training_config_path = resolve_path(
        repository,
        args.training_config,
    )
    archive_path = resolve_path(
        repository,
        args.dataset_archive,
    )
    metadata_path = resolve_path(
        repository,
        args.dataset_metadata,
    )
    dataset_manifest_path = resolve_path(
        repository,
        args.dataset_manifest,
    )

    execution_order = validate_requested_fractions(
        args.fractions
    )
    task_config = load_config(task_config_path)
    model_config = load_model_config(model_config_path)
    training_config = load_training_config(
        training_config_path
    )

    validate_stage13_training_settings(
        training_config,
        model_seed=args.model_seed,
        final_step=args.final_step,
    )

    subsets = load_and_validate_control_subsets(
        archive_path=archive_path,
        metadata_path=metadata_path,
        manifest_path=dataset_manifest_path,
        task_config=task_config,
    )

    implementation_commit = (
        current_head(repository)
        if args.validate_inputs_only
        else require_clean_repository(repository)
    )

    if (
        args.expected_implementation_commit is not None
        and implementation_commit
        != args.expected_implementation_commit
    ):
        raise RuntimeError(
            "Implementation commit mismatch: expected "
            f"{args.expected_implementation_commit}, found "
            f"{implementation_commit}."
        )

    configuration = build_stage13_configuration(
        repository=repository,
        task_config_path=task_config_path,
        model_config_path=model_config_path,
        training_config_path=training_config_path,
        dataset_manifest_path=dataset_manifest_path,
        subsets=subsets,
        implementation_commit=implementation_commit,
    )
    stage13_run_id = deterministic_stage13_run_id(
        configuration
    )

    candidate_records: list[
        tuple[
            NoGeneralisationSubset,
            dict[str, Any],
            str,
            str,
        ]
    ] = []

    for fraction in execution_order:
        subset = subset_by_fraction(subsets, fraction)
        identity = candidate_run_identity(
            stage13_run_id=stage13_run_id,
            subset=subset,
            configuration=configuration,
        )
        run_id, combined_sha256 = candidate_training_run_id(
            task_config=task_config,
            model_config=model_config,
            training_config=training_config,
            run_identity=identity,
        )
        candidate_records.append(
            (
                subset,
                identity,
                run_id,
                combined_sha256,
            )
        )

    print(f"stage13_run_id: {stage13_run_id}")
    print(
        "implementation_commit: "
        f"{implementation_commit}"
    )
    print(f"model_seed: {args.model_seed}")
    print(f"matched_horizon: {args.final_step}")
    print(
        "execution_order: "
        + ", ".join(
            f"{fraction:.2f}"
            for fraction in execution_order
        )
    )
    print(
        "checkpoint_validation_steps: "
        + ", ".join(
            str(step)
            for step in STAGE13_CHECKPOINT_VALIDATION_STEPS
        )
    )

    for rank, (
        subset,
        _,
        run_id,
        combined_sha256,
    ) in enumerate(candidate_records, start=1):
        print(
            "candidate: "
            f"rank={rank} "
            f"fraction={subset.fraction:.2f} "
            f"count={subset.exact_example_count} "
            f"subset_identifier={subset.subset_identifier} "
            f"subset_sha256={subset.subset_sha256} "
            f"run_id={run_id} "
            f"combined_config_sha256={combined_sha256}"
        )

    if args.validate_inputs_only:
        print("true_labels: passed")
        print("test_set_unchanged: passed")
        print("nested_prefixes: passed")
        print("weight_decay_1_0: passed")
        print("random_label_data_absent: passed")
        print("mps_execution_absent: passed")
        print("stage14_started: false")
        print("stage15_started: false")
        print("input_validation: passed")
        return

    selected_device = resolve_device(args.device)

    if selected_device.type == "mps":
        raise RuntimeError("Stage 13 must not execute on MPS.")

    validate_absent_candidate_outputs(
        output_root=output_root,
        training_config=training_config,
        run_ids=tuple(
            record[2]
            for record in candidate_records
        ),
    )

    registry_path = candidate_registry_path(
        output_root=output_root,
        stage13_run_id=stage13_run_id,
    )

    if registry_path.exists():
        raise FileExistsError(
            "Stage 13 candidate registry already exists: "
            f"{registry_path}"
        )

    print(f"device: {selected_device}")

    completed_candidates: list[dict[str, Any]] = []

    for subset, identity, expected_run_id, _ in candidate_records:
        data = load_no_generalisation_training_data(
            archive_path=archive_path,
            metadata_path=metadata_path,
            manifest_path=dataset_manifest_path,
            task_config=task_config,
            device=selected_device,
            subset=subset,
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
            device_override=selected_device.type,
            output_root=output_root,
            overwrite=False,
            training_data=data,
            max_steps_override=args.final_step,
            experiment_type_override=EXPERIMENT_TYPE,
            run_identity=identity,
            checkpoint_verification_steps=(
                STAGE13_CHECKPOINT_VALIDATION_STEPS
            ),
        )

        if result.run_id != expected_run_id:
            raise RuntimeError(
                "Candidate training run ID differs from its "
                "validated preflight identity."
            )

        completed_candidates.append(
            {
                "fraction": subset.fraction,
                "exact_training_example_count": (
                    subset.exact_example_count
                ),
                "subset_identifier": (
                    subset.subset_identifier
                ),
                "subset_sha256": subset.subset_sha256,
                "source_permutation_sha256": (
                    subset.source_permutation_sha256
                ),
                "candidate_run_id": result.run_id,
                "combined_config_sha256": (
                    result.combined_config_sha256
                ),
                "training_manifest_path": display_path(
                    result.manifest_path,
                    repository,
                ),
                "training_manifest_sha256": file_sha256(
                    result.manifest_path
                ),
                "metrics_path": display_path(
                    result.metrics_path,
                    repository,
                ),
                "metrics_sha256": file_sha256(
                    result.metrics_path
                ),
                "checkpoint_directory": display_path(
                    result.checkpoint_directory,
                    repository,
                ),
                "checkpoint_count": result.checkpoint_count,
                "verified_checkpoint_steps": list(
                    STAGE13_CHECKPOINT_VALIDATION_STEPS
                ),
            }
        )

        print(
            "completed_candidate: "
            f"fraction={subset.fraction:.2f} "
            f"run_id={result.run_id} "
            f"device={result.device} "
            f"final_step={result.final_step} "
            f"checkpoint_count={result.checkpoint_count} "
            f"metrics_path={result.metrics_path} "
            f"manifest_path={result.manifest_path}"
        )

    registry = {
        "schema_version": 1,
        "stage13_run_id": stage13_run_id,
        "implementation_commit": implementation_commit,
        "configuration": configuration,
        "device": selected_device.type,
        "model_seed": args.model_seed,
        "matched_horizon": args.final_step,
        "candidate_execution_order": list(
            execution_order
        ),
        "candidates": completed_candidates,
        "curve_only_selection_pending": True,
        "no_control_circuit_metric_inspected": True,
        "stage14_started": False,
        "stage15_started": False,
    }
    stable_json_write(registry_path, registry)

    print(
        "candidate_registry: "
        f"{display_path(registry_path, repository)}"
    )
    print(
        "candidate_registry_sha256: "
        f"{file_sha256(registry_path)}"
    )
    print("complete_pilot_grid: passed")


if __name__ == "__main__":
    main()

"""Validate Stage 8 component masking and generate provenance artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

from circuit_families.interpretability.fidelity import (
    DEFAULT_LOGIT_ABSOLUTE_TOLERANCE,
    DEFAULT_LOGIT_RELATIVE_TOLERANCE,
    MaskEvaluationMetrics,
    evaluate_component_mask,
    load_checkpoint_evaluation_context,
)
from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_COUNT,
    ATTENTION_HEAD_HOOK_NAME,
    ATTENTION_HEAD_IDS,
    MLP_NEURON_COUNT,
    MLP_NEURON_HOOK_NAME,
    MLP_NEURON_IDS,
    SEARCHABLE_COMPONENT_COUNT,
    ComponentMask,
    load_component_mask,
    save_component_mask,
)
from circuit_families.manifests import (
    package_versions,
    utc_timestamp,
)
from circuit_families.training import (
    canonical_state_hash,
    device_record,
    file_sha256,
)

PILOT_CHECKPOINT_STEPS = (
    200,
    3400,
    7450,
    8150,
    8500,
    8650,
    9050,
)
STABLE_POST_CHECKPOINT_STEP = 9050

ARBITRARY_ABLATED_IDENTIFIERS = (
    "H1",
    "H3",
    "N0",
    "N17",
    "N255",
    "N511",
)

VALIDATION_TABLE_FIELDS = (
    "run_id",
    "checkpoint_phase",
    "checkpoint_step",
    "checkpoint_sha256",
    "model_state_sha256",
    "mask_id",
    "mask_type",
    "mask_path",
    "retained_attention_heads",
    "retained_mlp_neurons",
    "retained_components",
    "retained_proportion",
    "prediction_agreement_count",
    "primary_fidelity",
    "full_accuracy",
    "masked_accuracy",
    "accuracy_change",
    "full_cross_entropy",
    "masked_cross_entropy",
    "cross_entropy_change",
    "mean_kl_divergence",
    "mean_jensen_shannon_divergence",
    "maximum_absolute_logit_difference",
    "evaluated_examples",
    "evaluation_batch_size",
    "validation_status",
)

SOFTWARE_PACKAGES = (
    "numpy",
    "pandas",
    "PyYAML",
    "torch",
    "transformer-lens",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate Stage 8 masking at the frozen seed-1 checkpoints."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    return parser.parse_args()


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _relative(repository: Path, path: Path) -> str:
    resolved = path.resolve()

    try:
        return str(resolved.relative_to(repository))
    except ValueError:
        return str(resolved)


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object.")

    return value


def _git_output(
    repository: Path,
    *arguments: str,
) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    status = _git_output(repository, "status", "--short")

    if status:
        raise RuntimeError(
            "Stage 8 artifacts must be generated from a clean "
            "implementation commit. Current status:\n"
            + status
        )

    return _git_output(repository, "rev-parse", "HEAD")


def stage8_output_paths(
    repository: Path,
    *,
    seed: int,
    combined_config_sha256: str,
) -> dict[str, Path | str]:
    """Return deterministic Stage 8 run and artifact paths."""

    suffix = f"s{seed}-{combined_config_sha256[:12]}"
    stage8_run_id = f"stage8-masking-{suffix}"

    return {
        "stage8_run_id": stage8_run_id,
        "raw_directory": (
            repository / "results" / "raw" / stage8_run_id
        ),
        "mask_directory": (
            repository
            / "results"
            / "raw"
            / stage8_run_id
            / "masks"
        ),
        "result_table": (
            repository
            / "results"
            / "tables"
            / f"seed_{seed}_stage8_mask_validation.csv"
        ),
        "manifest": (
            repository
            / "manifests"
            / f"stage8_masking_{suffix}.json"
        ),
    }


def build_validation_case_keys() -> tuple[tuple[int, str], ...]:
    """Return the exact ordered 11-row Stage 8 validation plan."""

    identity_cases = tuple(
        (step, "all_retained")
        for step in PILOT_CHECKPOINT_STEPS
    )
    stable_post_cases = (
        (STABLE_POST_CHECKPOINT_STEP, "all_ablated"),
        (STABLE_POST_CHECKPOINT_STEP, "head_H0_ablated"),
        (STABLE_POST_CHECKPOINT_STEP, "neuron_N0_ablated"),
        (
            STABLE_POST_CHECKPOINT_STEP,
            "saved_arbitrary_reloaded",
        ),
    )
    return identity_cases + stable_post_cases


def representative_masks() -> dict[str, ComponentMask]:
    """Return the five deterministic representative masks."""

    return {
        "all_retained": ComponentMask.all_retained(),
        "all_ablated": ComponentMask.all_ablated(),
        "head_H0_ablated": (
            ComponentMask.one_head_ablated("H0")
        ),
        "neuron_N0_ablated": (
            ComponentMask.one_neuron_ablated("N0")
        ),
        "saved_arbitrary_reloaded": (
            ComponentMask.from_ablated_identifiers(
                ARBITRARY_ABLATED_IDENTIFIERS
            )
        ),
    }


def representative_mask_filenames() -> dict[str, str]:
    """Return deterministic filenames for representative masks."""

    return {
        "all_retained": "all_retained.json",
        "all_ablated": "all_ablated.json",
        "head_H0_ablated": "head_H0_ablated.json",
        "neuron_N0_ablated": "neuron_N0_ablated.json",
        "saved_arbitrary_reloaded": (
            "saved_arbitrary_reloaded.json"
        ),
    }


def _hook_counts(model: Any) -> dict[str, int]:
    return {
        name: len(model.hook_dict[name]._forward_hooks)
        for name in (
            ATTENTION_HEAD_HOOK_NAME,
            MLP_NEURON_HOOK_NAME,
        )
    }


def _evaluate_without_mutation(
    context: Any,
    mask: ComponentMask,
    *,
    batch_size: int,
) -> MaskEvaluationMetrics:
    model_hash_before = canonical_state_hash(
        context.model.state_dict()
    )
    hook_counts_before = _hook_counts(context.model)

    metrics = evaluate_component_mask(
        context.model,
        context.inputs,
        context.targets,
        mask,
        batch_size=batch_size,
    )

    model_hash_after = canonical_state_hash(
        context.model.state_dict()
    )
    hook_counts_after = _hook_counts(context.model)

    if model_hash_after != model_hash_before:
        raise RuntimeError(
            "Masked evaluation changed the model-state hash."
        )

    if hook_counts_after != hook_counts_before:
        raise RuntimeError(
            "Masked evaluation leaked TransformerLens hooks."
        )

    return metrics


def _close(
    actual: float,
    expected: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    return math.isclose(
        actual,
        expected,
        abs_tol=absolute_tolerance,
        rel_tol=relative_tolerance,
    )


def _validate_all_retained(
    metrics: MaskEvaluationMetrics,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if metrics.primary_fidelity != 1.0:
        raise RuntimeError(
            "All-retained primary fidelity is not exactly 1.0."
        )

    if (
        metrics.prediction_agreement_count
        != metrics.evaluated_example_count
    ):
        raise RuntimeError(
            "All-retained top-one predictions do not all agree."
        )

    if not _close(
        metrics.masked_accuracy,
        metrics.full_accuracy,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    ):
        raise RuntimeError(
            "All-retained ground-truth accuracy mismatch."
        )

    if not _close(
        metrics.masked_cross_entropy,
        metrics.full_cross_entropy,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    ):
        raise RuntimeError(
            "All-retained cross-entropy mismatch."
        )

    if (
        metrics.maximum_absolute_logit_difference
        > absolute_tolerance
    ):
        raise RuntimeError(
            "All-retained logits exceed the absolute tolerance."
        )

    if abs(metrics.mean_kl_divergence) > absolute_tolerance:
        raise RuntimeError(
            "All-retained KL divergence exceeds tolerance."
        )

    if (
        abs(metrics.mean_jensen_shannon_divergence)
        > absolute_tolerance
    ):
        raise RuntimeError(
            "All-retained Jensen-Shannon divergence "
            "exceeds tolerance."
        )


def _validation_row(
    *,
    repository: Path,
    context: Any,
    mask: ComponentMask,
    mask_type: str,
    mask_path: Path,
    metrics: MaskEvaluationMetrics,
) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "checkpoint_phase": context.checkpoint_phase,
        "checkpoint_step": context.checkpoint_step,
        "checkpoint_sha256": context.checkpoint_sha256,
        "model_state_sha256": context.model_state_sha256,
        "mask_id": mask.mask_id,
        "mask_type": mask_type,
        "mask_path": _relative(repository, mask_path),
        "retained_attention_heads": (
            metrics.retained_attention_head_count
        ),
        "retained_mlp_neurons": (
            metrics.retained_mlp_neuron_count
        ),
        "retained_components": (
            metrics.retained_component_count
        ),
        "retained_proportion": (
            metrics.retained_component_proportion
        ),
        "prediction_agreement_count": (
            metrics.prediction_agreement_count
        ),
        "primary_fidelity": metrics.primary_fidelity,
        "full_accuracy": metrics.full_accuracy,
        "masked_accuracy": metrics.masked_accuracy,
        "accuracy_change": metrics.accuracy_change,
        "full_cross_entropy": metrics.full_cross_entropy,
        "masked_cross_entropy": metrics.masked_cross_entropy,
        "cross_entropy_change": (
            metrics.cross_entropy_change
        ),
        "mean_kl_divergence": metrics.mean_kl_divergence,
        "mean_jensen_shannon_divergence": (
            metrics.mean_jensen_shannon_divergence
        ),
        "maximum_absolute_logit_difference": (
            metrics.maximum_absolute_logit_difference
        ),
        "evaluated_examples": metrics.evaluated_example_count,
        "evaluation_batch_size": metrics.evaluation_batch_size,
        "validation_status": "passed",
    }


def write_validation_table(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write the compact deterministic Stage 8 validation table."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=VALIDATION_TABLE_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def write_stage8_manifest(
    path: str | Path,
    manifest: dict[str, Any],
) -> Path:
    """Write a stable JSON Stage 8 manifest."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    output_path.write_text(serialised + "\n", encoding="utf-8")
    return output_path


def build_stage8_manifest(
    *,
    repository: Path,
    stage8_run_id: str,
    source_run_id: str,
    masking_git_commit: str,
    checkpoint_manifest_path: Path,
    training_manifest_path: Path,
    contexts: dict[int, Any],
    result_table_path: Path,
    mask_paths: dict[str, Path],
    masks: dict[str, ComponentMask],
    batch_size: int,
) -> dict[str, Any]:
    """Build the complete provenance-bearing Stage 8 manifest."""

    first = contexts[PILOT_CHECKPOINT_STEPS[0]]

    selected_checkpoints = [
        {
            "phase": contexts[step].checkpoint_phase,
            "training_step": step,
            "path": _relative(
                repository,
                contexts[step].checkpoint_path,
            ),
            "checkpoint_sha256": (
                contexts[step].checkpoint_sha256
            ),
            "model_state_sha256": (
                contexts[step].model_state_sha256
            ),
        }
        for step in PILOT_CHECKPOINT_STEPS
    ]

    saved_masks = [
        {
            "mask_type": mask_type,
            "mask_id": masks[mask_type].mask_id,
            "path": _relative(
                repository,
                mask_paths[mask_type],
            ),
            "sha256": file_sha256(mask_paths[mask_type]),
            "retained_component_count": (
                masks[mask_type].retained_component_count
            ),
        }
        for mask_type in representative_mask_filenames()
    ]

    return {
        "schema_version": 1,
        "stage8_run_id": stage8_run_id,
        "experiment_type": "component_masking_validation",
        "source_training_run_id": source_run_id,
        "masking_code_git_commit": masking_git_commit,
        "creation_timestamp_utc": utc_timestamp(),
        "git_cleanliness": {
            "working_tree_clean_at_start": True,
            "status_at_start": "clean",
        },
        "source_manifests": {
            "training": {
                "path": _relative(
                    repository,
                    training_manifest_path,
                ),
                "sha256": first.training_manifest_sha256,
            },
            "checkpoint_selection": {
                "path": _relative(
                    repository,
                    checkpoint_manifest_path,
                ),
                "sha256": first.checkpoint_manifest_sha256,
            },
        },
        "selected_checkpoints": selected_checkpoints,
        "dataset": {
            "dataset_sha256": first.dataset_sha256,
            "split_sha256": first.split_sha256,
            "archive_sha256": first.dataset_archive_sha256,
            "metadata_sha256": (
                first.dataset_metadata_sha256
            ),
            "example_count": 12_769,
            "example_ordering": "lexicographic",
            "includes_training_and_test_examples": True,
        },
        "configuration_hashes": {
            "task_sha256": first.task_config_sha256,
            "model_sha256": first.model_config_sha256,
            "training_sha256": (
                first.training_config_sha256
            ),
            "combined_sha256": first.combined_config_sha256,
        },
        "component_definitions": {
            "attention_heads": {
                "count": ATTENTION_HEAD_COUNT,
                "identifiers": list(ATTENTION_HEAD_IDS),
                "hook_name": ATTENTION_HEAD_HOOK_NAME,
                "activation_axis": 2,
                "masked_value": (
                    "complete per-head output vector"
                ),
            },
            "mlp_neurons": {
                "count": MLP_NEURON_COUNT,
                "first_identifier": MLP_NEURON_IDS[0],
                "last_identifier": MLP_NEURON_IDS[-1],
                "hook_name": MLP_NEURON_HOOK_NAME,
                "activation_axis": 2,
                "masked_value": (
                    "individual post-ReLU activation"
                ),
            },
            "searchable_component_count": (
                SEARCHABLE_COMPONENT_COUNT
            ),
            "fixed_non_searchable_components": [
                "token_embeddings",
                "positional_embeddings",
                "residual_connections",
                "attention_matrices_within_retained_heads",
                "mlp_weights_within_retained_neurons",
                "unembedding",
                "biases",
            ],
        },
        "mask_convention": {
            "retained": 1,
            "ablated": 0,
            "ablation_baseline": "zero",
        },
        "evaluation": {
            "primary_fidelity": (
                "matching masked and full top-one predictions "
                "divided by evaluated examples"
            ),
            "evaluated_sequence_position": -1,
            "valid_output_classes": {
                "minimum": 0,
                "maximum": 112,
                "count": 113,
                "equals_token_eligible": False,
            },
            "full_model_reference": {
                "method": "computed_live",
                "checkpoint_specific": True,
                "cached": False,
            },
            "batch_size": batch_size,
            "batch_ordering": "fixed_lexicographic_no_shuffle",
            "absolute_tolerance": (
                DEFAULT_LOGIT_ABSOLUTE_TOLERANCE
            ),
            "relative_tolerance": (
                DEFAULT_LOGIT_RELATIVE_TOLERANCE
            ),
            "validation_case_count": 11,
        },
        "outputs": {
            "validation_table": {
                "path": _relative(
                    repository,
                    result_table_path,
                ),
                "sha256": file_sha256(result_table_path),
            },
            "saved_masks": saved_masks,
        },
        "software": {
            "python": platform.python_version(),
            "packages": package_versions(SOFTWARE_PACKAGES),
        },
        "device": device_record(first.device),
        "validation_status": "passed",
        "scientific_interpretation": (
            "machinery validation only; no mask is classified "
            "as a sparse circuit"
        ),
    }


def main() -> None:
    """Run all 11 frozen Stage 8 validation cases."""

    args = parse_args()
    repository = args.repository_root.resolve()
    masking_git_commit = _require_clean_repository(repository)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    checkpoint_manifest_path = _resolve(
        repository,
        args.checkpoint_manifest,
    )
    checkpoint_manifest = _load_json_object(
        checkpoint_manifest_path,
        "checkpoint manifest",
    )

    if checkpoint_manifest.get("run_id") != args.run_id:
        raise ValueError(
            "Checkpoint manifest run ID does not match --run-id."
        )

    if checkpoint_manifest.get("preferred_grid_status") != "complete":
        raise ValueError(
            "Stage 7 preferred checkpoint grid is not complete."
        )

    training_manifest_path = (
        repository
        / "manifests"
        / f"training_{args.run_id}.json"
    )
    training_manifest = _load_json_object(
        training_manifest_path,
        "training manifest",
    )

    seed = int(training_manifest["seed"]["value"])
    combined_hash = str(
        training_manifest["configs"]["combined_sha256"]
    )

    outputs = stage8_output_paths(
        repository,
        seed=seed,
        combined_config_sha256=combined_hash,
    )
    mask_directory = Path(outputs["mask_directory"])
    mask_directory.mkdir(parents=True, exist_ok=True)

    masks = representative_masks()
    filenames = representative_mask_filenames()
    mask_paths: dict[str, Path] = {}

    for mask_type, filename in filenames.items():
        mask_paths[mask_type] = save_component_mask(
            mask_directory / filename,
            masks[mask_type],
        )

    reloaded_arbitrary = load_component_mask(
        mask_paths["saved_arbitrary_reloaded"]
    )

    if reloaded_arbitrary != masks["saved_arbitrary_reloaded"]:
        raise RuntimeError(
            "Saved arbitrary mask did not reload exactly."
        )

    contexts = {
        step: load_checkpoint_evaluation_context(
            repository_root=repository,
            run_id=args.run_id,
            checkpoint_manifest_path=checkpoint_manifest_path,
            checkpoint_step=step,
            device_override=args.device,
        )
        for step in PILOT_CHECKPOINT_STEPS
    }

    rows: list[dict[str, Any]] = []

    for step in PILOT_CHECKPOINT_STEPS:
        context = contexts[step]
        mask = masks["all_retained"]
        metrics = _evaluate_without_mutation(
            context,
            mask,
            batch_size=args.batch_size,
        )
        _validate_all_retained(
            metrics,
            absolute_tolerance=(
                DEFAULT_LOGIT_ABSOLUTE_TOLERANCE
            ),
            relative_tolerance=(
                DEFAULT_LOGIT_RELATIVE_TOLERANCE
            ),
        )
        rows.append(
            _validation_row(
                repository=repository,
                context=context,
                mask=mask,
                mask_type="all_retained",
                mask_path=mask_paths["all_retained"],
                metrics=metrics,
            )
        )

    stable_context = contexts[STABLE_POST_CHECKPOINT_STEP]

    all_ablated_first = _evaluate_without_mutation(
        stable_context,
        masks["all_ablated"],
        batch_size=args.batch_size,
    )
    all_ablated_second = _evaluate_without_mutation(
        stable_context,
        masks["all_ablated"],
        batch_size=args.batch_size,
    )

    if all_ablated_first != all_ablated_second:
        raise RuntimeError(
            "All-ablated evaluation was not deterministic."
        )

    if (
        all_ablated_first.maximum_absolute_logit_difference
        <= 0.0
    ):
        raise RuntimeError(
            "All-ablated outputs did not differ from the full model."
        )

    rows.append(
        _validation_row(
            repository=repository,
            context=stable_context,
            mask=masks["all_ablated"],
            mask_type="all_ablated",
            mask_path=mask_paths["all_ablated"],
            metrics=all_ablated_first,
        )
    )

    for mask_type in (
        "head_H0_ablated",
        "neuron_N0_ablated",
    ):
        metrics = _evaluate_without_mutation(
            stable_context,
            masks[mask_type],
            batch_size=args.batch_size,
        )
        rows.append(
            _validation_row(
                repository=repository,
                context=stable_context,
                mask=masks[mask_type],
                mask_type=mask_type,
                mask_path=mask_paths[mask_type],
                metrics=metrics,
            )
        )

    arbitrary_original_metrics = _evaluate_without_mutation(
        stable_context,
        masks["saved_arbitrary_reloaded"],
        batch_size=args.batch_size,
    )
    arbitrary_reloaded_metrics = _evaluate_without_mutation(
        stable_context,
        reloaded_arbitrary,
        batch_size=args.batch_size,
    )

    if arbitrary_original_metrics != arbitrary_reloaded_metrics:
        raise RuntimeError(
            "Saved and reloaded arbitrary masks produced "
            "different metrics."
        )

    rows.append(
        _validation_row(
            repository=repository,
            context=stable_context,
            mask=reloaded_arbitrary,
            mask_type="saved_arbitrary_reloaded",
            mask_path=mask_paths[
                "saved_arbitrary_reloaded"
            ],
            metrics=arbitrary_reloaded_metrics,
        )
    )

    actual_case_keys = tuple(
        (
            int(row["checkpoint_step"]),
            str(row["mask_type"]),
        )
        for row in rows
    )

    if actual_case_keys != build_validation_case_keys():
        raise RuntimeError(
            "Generated Stage 8 validation cases do not match "
            "the frozen 11-case plan."
        )

    result_table_path = write_validation_table(
        Path(outputs["result_table"]),
        rows,
    )

    manifest = build_stage8_manifest(
        repository=repository,
        stage8_run_id=str(outputs["stage8_run_id"]),
        source_run_id=args.run_id,
        masking_git_commit=masking_git_commit,
        checkpoint_manifest_path=checkpoint_manifest_path,
        training_manifest_path=training_manifest_path,
        contexts=contexts,
        result_table_path=result_table_path,
        mask_paths=mask_paths,
        masks=masks,
        batch_size=args.batch_size,
    )
    manifest_path = write_stage8_manifest(
        Path(outputs["manifest"]),
        manifest,
    )

    print("===== STAGE 8 COMPONENT MASKING VALIDATION =====")
    print(f"stage8_run_id: {outputs['stage8_run_id']}")
    print(f"source_training_run_id: {args.run_id}")
    print(f"masking_git_commit: {masking_git_commit}")
    print(f"device: {args.device}")
    print(f"evaluation_batch_size: {args.batch_size}")
    print("evaluated_examples_per_case: 12769")
    print(f"validation_case_count: {len(rows)}")
    print("all_retained_checkpoint_steps:")
    for step in PILOT_CHECKPOINT_STEPS:
        print(f"  {step}: passed")
    print("stable_post_interventions:")
    print("  all_ablated: passed")
    print("  H0_ablated: passed")
    print("  N0_ablated: passed")
    print("  saved_arbitrary_reloaded: passed")
    print(
        "result_table: "
        f"{_relative(repository, result_table_path)}"
    )
    print(
        "result_table_sha256: "
        f"{file_sha256(result_table_path)}"
    )
    print(
        "stage8_manifest: "
        f"{_relative(repository, manifest_path)}"
    )
    print(
        "stage8_manifest_sha256: "
        f"{file_sha256(manifest_path)}"
    )
    print("validation_status: passed")


if __name__ == "__main__":
    main()

"""Validate Stage 8 masking machinery on Stage 14 checkpoints."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from circuit_families.analysis.random_label_control import (
    MAIN_MODEL_REFERENCE_CHECKPOINTS,
)
from circuit_families.config import (
    combined_config_hash,
    config_hash,
    load_config,
    load_model_config,
    load_training_config,
    mapping_hash,
)
from circuit_families.interpretability.fidelity import (
    DEFAULT_LOGIT_ABSOLUTE_TOLERANCE,
    DEFAULT_LOGIT_RELATIVE_TOLERANCE,
    MaskEvaluationMetrics,
)
from circuit_families.interpretability.masks import (
    ComponentMask,
    load_component_mask,
    save_component_mask,
)
from circuit_families.models import build_transformer
from circuit_families.training import (
    canonical_state_hash,
    file_sha256,
    load_checkpoint_payload,
    resolve_device,
)
from circuit_families.training.random_label import (
    FINAL_STEP,
    MODEL_SEED,
    RANDOM_LABEL_PERMUTATION_SHA256,
    RANDOM_LABELS_SHA256,
    load_random_label_training_data,
)

DEFAULT_CHECKPOINT_MANIFEST = Path("manifests/stage14_random_label_checkpoints.json")
DEFAULT_RESULT_TABLE = Path("results/tables/seed_0_stage14_random_label_masking_validation.csv")
SOURCE_STAGE8_MANIFEST = Path("manifests/stage8_masking_s1-5f1bc9dee7ab.json")
SOURCE_STAGE8_MANIFEST_SHA256 = "ed6aca8d20d43ea7618936b962c8e865859c21894e5d809580106ffe73a8d4e5"

EVALUATED_EXAMPLE_COUNT = 12_769
OUTPUT_CLASS_COUNT = 113
EVALUATED_TOKEN_POSITION = -1

TABLE_COLUMNS = (
    "stage14_run_id",
    "main_model_reference_phase_label",
    "phase_label_scope",
    "checkpoint_step",
    "checkpoint_path",
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
    "output_class_minimum",
    "output_class_maximum",
    "output_class_count",
    "equals_token_output_excluded",
    "evaluated_token_position",
    "model_state_sha256_before",
    "model_state_sha256_after",
    "model_state_unchanged",
    "parameter_gradient_count_before",
    "parameter_gradient_count_after",
    "parameter_gradients_absent",
    "hook_counts_before",
    "hook_counts_after",
    "hooks_removed",
    "saved_reloaded_mask_match",
    "validation_status",
)


@dataclass(frozen=True)
class Stage14MaskingContext:
    """Validated model and random-label data for one checkpoint."""

    run_id: str
    checkpoint_phase: str
    checkpoint_step: int
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_manifest_path: Path
    checkpoint_manifest_sha256: str
    training_manifest_path: Path
    training_manifest_sha256: str
    model_state_sha256: str
    task_config_sha256: str
    model_config_sha256: str
    training_config_sha256: str
    combined_config_sha256: str
    dataset_sha256: str
    split_sha256: str
    dataset_archive_sha256: str
    dataset_metadata_sha256: str
    example_ordering: str
    model: Any
    inputs: torch.Tensor
    targets: torch.Tensor
    device: torch.device


def parse_args() -> argparse.Namespace:
    """Parse Stage 14 masking-validation arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the existing Stage 8 masking machinery on "
            "the seven Stage 14 random-label checkpoints."
        )
    )
    parser.add_argument(
        "--stage14-checkpoint-manifest",
        type=Path,
        default=DEFAULT_CHECKPOINT_MANIFEST,
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--result-table",
        type=Path,
        default=DEFAULT_RESULT_TABLE,
    )
    parser.add_argument(
        "--expected-implementation-commit",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    """Resolve one root-relative path."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def resolve_recorded_path(
    *,
    repository: Path,
    output_root: Path,
    value: str | Path,
) -> Path:
    """Resolve a path recorded relative to either relevant root."""

    path = Path(value)

    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved

    candidates = (
        (output_root / path).resolve(),
        (repository / path).resolve(),
    )
    existing = []

    for candidate in candidates:
        if candidate.exists() and candidate not in existing:
            existing.append(candidate)

    if len(existing) != 1:
        raise ValueError(
            f"Recorded path {value!r} resolved to {existing}; expected exactly one existing path."
        )

    return existing[0]


def relative_or_absolute(root: Path, file_path: Path) -> str:
    """Return a stable path representation."""

    try:
        return str(file_path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(file_path.resolve())


def require_unchanged_implementation(
    repository: Path,
    *,
    expected_commit: str | None,
) -> str:
    """Allow only the expected untracked Stage 14 output set."""

    unstaged = subprocess.run(
        ["/usr/bin/git", "diff", "--quiet"],
        cwd=repository,
        check=False,
    ).returncode
    staged = subprocess.run(
        ["/usr/bin/git", "diff", "--cached", "--quiet"],
        cwd=repository,
        check=False,
    ).returncode

    if unstaged or staged:
        raise RuntimeError("Tracked implementation files changed after Stage 14 training.")

    head = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    if expected_commit is not None and head != expected_commit:
        raise RuntimeError(
            f"Implementation commit mismatch: expected {expected_commit}, found {head}."
        )

    status = subprocess.run(
        [
            "/usr/bin/git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    allowed_exact = {
        "manifests/stage14_random_label_checkpoints.json",
        ("results/tables/seed_0_stage14_random_label_training_metrics.csv"),
        ("results/tables/seed_0_stage14_random_label_checkpoints.csv"),
    }

    forbidden = []

    for line in status:
        if not line.startswith("?? "):
            forbidden.append(line)
            continue

        path = line[3:]

        training_manifest = path.startswith(
            "manifests/training_stage14-random-label-training-s0-"
        ) and path.endswith(".json")

        if not training_manifest and path not in allowed_exact:
            forbidden.append(line)

    if forbidden:
        raise RuntimeError(
            "Unexpected repository changes are present before "
            "Stage 14 masking validation:\n" + "\n".join(forbidden)
        )

    return head


def load_stage8_validator(repository: Path) -> Any:
    """Load the existing Stage 8 validator for helper reuse."""

    file_path = repository / "scripts/validate_component_masking.py"
    spec = importlib.util.spec_from_file_location(
        "stage14_reused_stage8_validator",
        file_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Stage 8 validator.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


def validation_case_keys() -> tuple[tuple[int, str], ...]:
    """Return the frozen seven identity and four stable cases."""

    all_retained = tuple((step, "all_retained") for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS)
    stable = (
        (FINAL_STEP, "all_ablated"),
        (FINAL_STEP, "head_H0_ablated"),
        (FINAL_STEP, "neuron_N0_ablated"),
        (FINAL_STEP, "saved_arbitrary_reloaded"),
    )
    return all_retained + stable


def parameter_gradient_count(model: Any) -> int:
    """Count parameters retaining gradient tensors."""

    return sum(parameter.grad is not None for parameter in model.parameters())


def hook_counts_are_zero(counts: Any) -> bool:
    """Return whether a Stage 8 hook-count record is all zero."""

    if isinstance(counts, dict):
        return all(hook_counts_are_zero(value) for value in counts.values())

    if isinstance(counts, (list, tuple)):
        return all(hook_counts_are_zero(value) for value in counts)

    return int(counts) == 0


def validate_checkpoint_manifest(
    manifest: dict[str, Any],
) -> None:
    """Validate the Stage 14 exact-step selection record."""

    if manifest.get("stage") != 14:
        raise ValueError("Checkpoint manifest stage is not 14.")

    if manifest.get("matching_rule") != "exact_training_step":
        raise ValueError("Checkpoint matching rule is not exact-step.")

    if manifest.get("matched_checkpoint_count") != 7:
        raise ValueError("Checkpoint manifest does not contain seven rows.")

    if manifest.get("expected_absolute_step_mismatch") != 0:
        raise ValueError("Expected checkpoint mismatch is not zero.")

    if manifest.get("random_label_sparse_search_started") is not False:
        raise ValueError("Random-label sparse search has begun.")

    if manifest.get("diversity_search_started") is not False:
        raise ValueError("Diversity search has begun.")

    if manifest.get("stage15_started") is not False:
        raise ValueError("Stage 15 has begun.")

    rows = manifest.get("matched_checkpoints")

    if not isinstance(rows, list):
        raise ValueError("Matched checkpoint rows are missing.")

    expected_steps = [step for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS]
    observed_steps = [int(row["requested_step"]) for row in rows]

    if observed_steps != expected_steps:
        raise ValueError("Matched checkpoint step grid differs.")

    if any(int(row["absolute_step_mismatch"]) != 0 for row in rows):
        raise ValueError("A matched checkpoint has nonzero mismatch.")

    if any(row.get("phase_label_scope") != "main_model_reference_only" for row in rows):
        raise ValueError("Checkpoint phase labels are not scoped to the main model.")


def load_context(
    *,
    repository: Path,
    output_root: Path,
    checkpoint_manifest_path: Path,
    checkpoint_manifest: dict[str, Any],
    training_manifest_path: Path,
    training_manifest: dict[str, Any],
    matched_record: dict[str, Any],
    device: torch.device,
) -> Stage14MaskingContext:
    """Load one checkpoint through existing repository APIs."""

    run_id = str(checkpoint_manifest["stage14_run_id"])
    step = int(matched_record["requested_step"])

    if step != int(matched_record["selected_random_label_step"]):
        raise ValueError("Random-label checkpoint is not an exact match.")

    training_output_root = training_manifest_path.parent.parent
    checkpoint_path = resolve_path(
        training_output_root,
        matched_record["checkpoint_path"],
    )

    if file_sha256(checkpoint_path) != matched_record["checkpoint_sha256"]:
        raise ValueError(f"Physical checkpoint hash mismatch at step {step}.")

    configs = training_manifest["configs"]

    task_config_path = resolve_recorded_path(
        repository=repository,
        output_root=output_root,
        value=configs["task"]["path"],
    )
    model_config_path = resolve_recorded_path(
        repository=repository,
        output_root=output_root,
        value=configs["model"]["path"],
    )
    training_config_path = resolve_recorded_path(
        repository=repository,
        output_root=output_root,
        value=configs["training"]["path"],
    )

    task_config = load_config(task_config_path)
    model_config = load_model_config(model_config_path)
    training_config = load_training_config(training_config_path)

    task_sha256 = config_hash(task_config)
    model_sha256 = mapping_hash(model_config)
    training_sha256 = mapping_hash(training_config)

    if task_sha256 != configs["task"]["sha256"]:
        raise ValueError("Task configuration hash mismatch.")

    if model_sha256 != configs["model"]["sha256"]:
        raise ValueError("Model configuration hash mismatch.")

    if training_sha256 != configs["training"]["sha256"]:
        raise ValueError("Training configuration hash mismatch.")

    combined_sha256 = combined_config_hash(
        {
            "task": task_config,
            "model": model_config,
            "training": training_config,
            "execution": training_manifest["execution"],
            "run_identity": training_manifest["run_identity"],
        }
    )

    if combined_sha256 != configs["combined_sha256"]:
        raise ValueError("Combined configuration hash mismatch.")

    dataset = training_manifest["dataset"]

    archive_path = resolve_recorded_path(
        repository=repository,
        output_root=output_root,
        value=dataset["archive_path"],
    )
    metadata_path = resolve_recorded_path(
        repository=repository,
        output_root=output_root,
        value=dataset["metadata_path"],
    )
    dataset_manifest_path = resolve_recorded_path(
        repository=repository,
        output_root=output_root,
        value=dataset["manifest_path"],
    )

    data = load_random_label_training_data(
        archive_path=archive_path,
        metadata_path=metadata_path,
        manifest_path=dataset_manifest_path,
        task_config_path=task_config_path,
        task_config=task_config,
        device=device,
    )

    if data.full_inputs is None or data.full_targets is None:
        raise RuntimeError("Complete random-label evaluation tensors are unavailable.")

    if data.total_count != EVALUATED_EXAMPLE_COUNT:
        raise ValueError("Complete evaluation count differs from 12,769.")

    canonical_hashes = dataset["canonical_hashes"]

    for name in (
        "dataset_sha256",
        "split_sha256",
        "random_labels_sha256",
        "random_label_permutation_sha256",
    ):
        if data.dataset_hashes.get(name) != canonical_hashes.get(name):
            raise ValueError(f"Dataset hash mismatch for {name}.")

    if data.dataset_hashes["random_labels_sha256"] != RANDOM_LABELS_SHA256:
        raise ValueError("Frozen random-label hash mismatch.")

    if data.dataset_hashes["random_label_permutation_sha256"] != RANDOM_LABEL_PERMUTATION_SHA256:
        raise ValueError("Frozen random-label permutation hash mismatch.")

    payload = load_checkpoint_payload(
        checkpoint_path,
        map_location=device,
    )

    if int(payload["training_step"]) != step:
        raise ValueError("Checkpoint payload step mismatch.")

    if int(payload["model_seed"]) != MODEL_SEED:
        raise ValueError("Checkpoint model seed mismatch.")

    if payload["model_config"] != model_config:
        raise ValueError("Checkpoint model configuration mismatch.")

    if payload["training_config"] != training_config:
        raise ValueError("Checkpoint training configuration mismatch.")

    for name, expected in data.dataset_hashes.items():
        if payload["dataset_hashes"].get(name) != expected:
            raise ValueError(f"Checkpoint dataset hash mismatch for {name}.")

    if payload["model_state_sha256"] != matched_record["model_state_sha256"]:
        raise ValueError("Checkpoint model-state hash mismatch.")

    if payload["optimizer_state_sha256"] != matched_record["optimizer_state_sha256"]:
        raise ValueError("Checkpoint optimiser-state hash mismatch.")

    model = build_transformer(
        model_config,
        seed=MODEL_SEED,
        device=device,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    if canonical_state_hash(model.state_dict()) != payload["model_state_sha256"]:
        raise ValueError("Restored model-state hash mismatch.")

    if int(model.cfg.d_vocab_out) != OUTPUT_CLASS_COUNT:
        raise ValueError("Model output vocabulary is not classes 0–112.")

    return Stage14MaskingContext(
        run_id=run_id,
        checkpoint_phase=str(matched_record["main_model_reference_phase_label"]),
        checkpoint_step=step,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=str(matched_record["checkpoint_sha256"]),
        checkpoint_manifest_path=checkpoint_manifest_path,
        checkpoint_manifest_sha256=file_sha256(checkpoint_manifest_path),
        training_manifest_path=training_manifest_path,
        training_manifest_sha256=file_sha256(training_manifest_path),
        model_state_sha256=str(matched_record["model_state_sha256"]),
        task_config_sha256=task_sha256,
        model_config_sha256=model_sha256,
        training_config_sha256=training_sha256,
        combined_config_sha256=combined_sha256,
        dataset_sha256=str(data.dataset_hashes["dataset_sha256"]),
        split_sha256=str(data.dataset_hashes["split_sha256"]),
        dataset_archive_sha256=data.archive_sha256,
        dataset_metadata_sha256=data.metadata_sha256,
        example_ordering="lexicographic",
        model=model,
        inputs=data.full_inputs,
        targets=data.full_targets,
        device=device,
    )


def evaluate_case(
    *,
    stage8: Any,
    artifact_root: Path,
    context: Stage14MaskingContext,
    mask: ComponentMask,
    mask_type: str,
    mask_path: Path,
    batch_size: int,
    saved_reloaded_mask_match: bool | None,
) -> tuple[dict[str, Any], MaskEvaluationMetrics]:
    """Evaluate one mask with explicit integrity evidence."""

    state_before = canonical_state_hash(context.model.state_dict())
    gradients_before = parameter_gradient_count(context.model)
    hooks_before = stage8._hook_counts(context.model)

    if state_before != context.model_state_sha256:
        raise RuntimeError("Model state changed before mask evaluation.")

    if gradients_before != 0:
        raise RuntimeError("Parameter gradients were present before mask evaluation.")

    if not hook_counts_are_zero(hooks_before):
        raise RuntimeError("Hooks were present before mask evaluation.")

    metrics = stage8._evaluate_without_mutation(
        context,
        mask,
        batch_size=batch_size,
    )

    state_after = canonical_state_hash(context.model.state_dict())
    gradients_after = parameter_gradient_count(context.model)
    hooks_after = stage8._hook_counts(context.model)

    state_unchanged = state_before == state_after == context.model_state_sha256
    gradients_absent = gradients_before == 0 and gradients_after == 0
    hooks_removed = hook_counts_are_zero(hooks_before) and hook_counts_are_zero(hooks_after)

    if not state_unchanged:
        raise RuntimeError("Mask evaluation changed the model state.")

    if not gradients_absent:
        raise RuntimeError("Mask evaluation created parameter gradients.")

    if not hooks_removed:
        raise RuntimeError("Mask evaluation leaked hooks.")

    if metrics.evaluated_example_count != EVALUATED_EXAMPLE_COUNT:
        raise RuntimeError("Mask evaluation did not use all examples.")

    row = {
        "stage14_run_id": context.run_id,
        "main_model_reference_phase_label": (context.checkpoint_phase),
        "phase_label_scope": "main_model_reference_only",
        "checkpoint_step": context.checkpoint_step,
        "checkpoint_path": relative_or_absolute(
            artifact_root,
            context.checkpoint_path,
        ),
        "checkpoint_sha256": context.checkpoint_sha256,
        "model_state_sha256": context.model_state_sha256,
        "mask_id": mask.mask_id,
        "mask_type": mask_type,
        "mask_path": relative_or_absolute(
            artifact_root,
            mask_path,
        ),
        "retained_attention_heads": (metrics.retained_attention_head_count),
        "retained_mlp_neurons": (metrics.retained_mlp_neuron_count),
        "retained_components": metrics.retained_component_count,
        "retained_proportion": (metrics.retained_component_proportion),
        "prediction_agreement_count": (metrics.prediction_agreement_count),
        "primary_fidelity": metrics.primary_fidelity,
        "full_accuracy": metrics.full_accuracy,
        "masked_accuracy": metrics.masked_accuracy,
        "accuracy_change": metrics.accuracy_change,
        "full_cross_entropy": metrics.full_cross_entropy,
        "masked_cross_entropy": metrics.masked_cross_entropy,
        "cross_entropy_change": metrics.cross_entropy_change,
        "mean_kl_divergence": metrics.mean_kl_divergence,
        "mean_jensen_shannon_divergence": (metrics.mean_jensen_shannon_divergence),
        "maximum_absolute_logit_difference": (metrics.maximum_absolute_logit_difference),
        "evaluated_examples": metrics.evaluated_example_count,
        "evaluation_batch_size": metrics.evaluation_batch_size,
        "output_class_minimum": 0,
        "output_class_maximum": 112,
        "output_class_count": OUTPUT_CLASS_COUNT,
        "equals_token_output_excluded": True,
        "evaluated_token_position": EVALUATED_TOKEN_POSITION,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": state_unchanged,
        "parameter_gradient_count_before": gradients_before,
        "parameter_gradient_count_after": gradients_after,
        "parameter_gradients_absent": gradients_absent,
        "hook_counts_before": json.dumps(
            hooks_before,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "hook_counts_after": json.dumps(
            hooks_after,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "hooks_removed": hooks_removed,
        "saved_reloaded_mask_match": (saved_reloaded_mask_match),
        "validation_status": "passed",
    }
    return row, metrics


def write_validation_table(
    file_path: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write the deterministic 11-row masking table."""

    if file_path.exists():
        raise FileExistsError(f"Masking-validation table already exists: {file_path}")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TABLE_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({column: row.get(column) for column in TABLE_COLUMNS})

    return file_path


def main() -> None:
    """Run the complete Stage 14 masking validation."""

    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    repository = args.repository_root.resolve()
    output_root = resolve_path(repository, args.output_root)

    implementation_commit = require_unchanged_implementation(
        repository,
        expected_commit=args.expected_implementation_commit,
    )

    checkpoint_manifest_path = resolve_recorded_path(
        repository=repository,
        output_root=output_root,
        value=args.stage14_checkpoint_manifest,
    )
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    validate_checkpoint_manifest(checkpoint_manifest)

    if checkpoint_manifest.get("implementation_commit") != implementation_commit:
        raise RuntimeError(
            "Checkpoint manifest does not trace to the current implementation commit."
        )

    source_training_record = checkpoint_manifest["source_training_manifest"]
    training_manifest_path = resolve_recorded_path(
        repository=repository,
        output_root=output_root,
        value=source_training_record["path"],
    )

    if file_sha256(training_manifest_path) != source_training_record["sha256"]:
        raise ValueError("Source training manifest hash mismatch.")

    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))

    if training_manifest.get("run_id") != checkpoint_manifest.get("stage14_run_id"):
        raise ValueError("Training and checkpoint run IDs differ.")

    stage8_manifest_path = repository / SOURCE_STAGE8_MANIFEST

    if file_sha256(stage8_manifest_path) != (SOURCE_STAGE8_MANIFEST_SHA256):
        raise ValueError("Source Stage 8 manifest hash mismatch.")

    stage8 = load_stage8_validator(repository)
    masks = stage8.representative_masks()
    filenames = stage8.representative_mask_filenames()

    stage14_run_id = str(checkpoint_manifest["stage14_run_id"])
    mask_directory = output_root / "results/raw" / f"{stage14_run_id}-masking" / "masks"

    if mask_directory.exists() and any(mask_directory.iterdir()):
        raise FileExistsError(f"Mask directory is not empty: {mask_directory}")

    mask_directory.mkdir(parents=True, exist_ok=True)

    mask_paths = {
        mask_type: save_component_mask(
            mask_directory / filename,
            masks[mask_type],
        )
        for mask_type, filename in filenames.items()
    }

    reloaded_arbitrary = load_component_mask(mask_paths["saved_arbitrary_reloaded"])
    saved_reloaded_mask_match = reloaded_arbitrary == masks["saved_arbitrary_reloaded"]

    if not saved_reloaded_mask_match:
        raise RuntimeError("Saved arbitrary mask did not reload exactly.")

    device = resolve_device(args.device)

    if device.type == "mps":
        raise RuntimeError("Stage 14 must not execute on MPS.")

    matched_records = checkpoint_manifest["matched_checkpoints"]

    contexts = {
        int(record["requested_step"]): load_context(
            repository=repository,
            output_root=output_root,
            checkpoint_manifest_path=checkpoint_manifest_path,
            checkpoint_manifest=checkpoint_manifest,
            training_manifest_path=training_manifest_path,
            training_manifest=training_manifest,
            matched_record=record,
            device=device,
        )
        for record in matched_records
    }

    rows: list[dict[str, Any]] = []

    for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS:
        row, metrics = evaluate_case(
            stage8=stage8,
            artifact_root=output_root,
            context=contexts[step],
            mask=masks["all_retained"],
            mask_type="all_retained",
            mask_path=mask_paths["all_retained"],
            batch_size=args.batch_size,
            saved_reloaded_mask_match=None,
        )
        stage8._validate_all_retained(
            metrics,
            absolute_tolerance=(DEFAULT_LOGIT_ABSOLUTE_TOLERANCE),
            relative_tolerance=(DEFAULT_LOGIT_RELATIVE_TOLERANCE),
        )
        rows.append(row)

    stable_context = contexts[FINAL_STEP]

    all_ablated_row, all_ablated_first = evaluate_case(
        stage8=stage8,
        artifact_root=output_root,
        context=stable_context,
        mask=masks["all_ablated"],
        mask_type="all_ablated",
        mask_path=mask_paths["all_ablated"],
        batch_size=args.batch_size,
        saved_reloaded_mask_match=None,
    )
    _, all_ablated_second = evaluate_case(
        stage8=stage8,
        artifact_root=output_root,
        context=stable_context,
        mask=masks["all_ablated"],
        mask_type="all_ablated",
        mask_path=mask_paths["all_ablated"],
        batch_size=args.batch_size,
        saved_reloaded_mask_match=None,
    )

    if all_ablated_first != all_ablated_second:
        raise RuntimeError("All-ablated evaluation was not deterministic.")

    if all_ablated_first.maximum_absolute_logit_difference <= 0.0:
        raise RuntimeError("All-ablated outputs did not differ from the full model.")

    rows.append(all_ablated_row)

    for mask_type in (
        "head_H0_ablated",
        "neuron_N0_ablated",
    ):
        row, _ = evaluate_case(
            stage8=stage8,
            artifact_root=output_root,
            context=stable_context,
            mask=masks[mask_type],
            mask_type=mask_type,
            mask_path=mask_paths[mask_type],
            batch_size=args.batch_size,
            saved_reloaded_mask_match=None,
        )
        rows.append(row)

    original_row, original_metrics = evaluate_case(
        stage8=stage8,
        artifact_root=output_root,
        context=stable_context,
        mask=masks["saved_arbitrary_reloaded"],
        mask_type="saved_arbitrary_reloaded",
        mask_path=mask_paths["saved_arbitrary_reloaded"],
        batch_size=args.batch_size,
        saved_reloaded_mask_match=True,
    )
    _, reloaded_metrics = evaluate_case(
        stage8=stage8,
        artifact_root=output_root,
        context=stable_context,
        mask=reloaded_arbitrary,
        mask_type="saved_arbitrary_reloaded",
        mask_path=mask_paths["saved_arbitrary_reloaded"],
        batch_size=args.batch_size,
        saved_reloaded_mask_match=True,
    )

    if original_metrics != reloaded_metrics:
        raise RuntimeError("Saved and reloaded arbitrary masks produced different metrics.")

    rows.append(original_row)

    actual_case_keys = tuple(
        (
            int(row["checkpoint_step"]),
            str(row["mask_type"]),
        )
        for row in rows
    )

    if actual_case_keys != validation_case_keys():
        raise RuntimeError("Generated masking cases differ from the frozen plan.")

    result_table_path = resolve_path(
        output_root,
        args.result_table,
    )
    write_validation_table(result_table_path, rows)

    print("===== STAGE 14 MASKING VALIDATION =====")
    print(f"stage14_run_id: {stage14_run_id}")
    print(f"implementation_commit: {implementation_commit}")
    print(f"device: {device.type}")
    print(f"evaluation_batch_size: {args.batch_size}")
    print("evaluated_examples_per_case: 12769")
    print(f"validation_case_count: {len(rows)}")
    print("all_retained_checkpoint_steps:")

    for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS:
        print(f"  {step}: passed")

    print("stable_step_interventions:")
    print("  all_ablated: passed")
    print("  H0_ablated: passed")
    print("  N0_ablated: passed")
    print("  saved_arbitrary_reloaded: passed")
    print("model_state_unchanged: passed")
    print("parameter_gradients_absent: passed")
    print("hooks_removed: passed")
    print("output_classes: 0-112")
    print("output_class_113_excluded: passed")
    print("evaluated_token_position: final")
    print(f"result_table: {result_table_path}")
    print(f"result_table_sha256: {file_sha256(result_table_path)}")
    print("scientific_interpretation: machinery_validation_only")
    print("random_label_sparse_search_started: false")
    print("diversity_search_started: false")
    print("stage15_started: false")
    print("stage14_masking_validation: passed")


if __name__ == "__main__":
    main()

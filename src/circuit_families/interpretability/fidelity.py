"""Deterministic behavioural-fidelity evaluation for component masks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformer_lens import HookedTransformer

from circuit_families.interpretability.component_ablation import (
    masked_model_logits,
    validate_mask_model,
)
from circuit_families.interpretability.masks import ComponentMask
from circuit_families.training.metrics import (
    OUTPUT_CLASS_COUNT,
    final_position_logits,
)

DEFAULT_LOGIT_ABSOLUTE_TOLERANCE = 1.0e-6
DEFAULT_LOGIT_RELATIVE_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class FullModelReference:
    """Frozen full-model final-position outputs for exact mask evaluation."""

    final_logits: torch.Tensor
    predictions: torch.Tensor
    evaluated_example_count: int
    inference_batch_size: int


def compute_full_model_reference(
    model: HookedTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
) -> FullModelReference:
    """Compute full-model final-position outputs once in fixed order."""

    _validate_evaluation_data(model, inputs, targets)
    batch_size = _validate_batch_size(batch_size)

    example_count = inputs.shape[0]
    final_logit_batches: list[torch.Tensor] = []

    was_training = model.training
    model.eval()

    try:
        for start in range(0, example_count, batch_size):
            stop = min(start + batch_size, example_count)

            with torch.inference_mode():
                sequence_logits = model(inputs[start:stop])

            final_logit_batches.append(final_position_logits(sequence_logits).detach().clone())
    finally:
        model.train(was_training)

    final_logits = torch.cat(final_logit_batches, dim=0)
    predictions = final_logits.argmax(dim=-1)

    if not bool(torch.isfinite(final_logits).all().item()):
        raise FloatingPointError("Full-model reference logits must all be finite.")

    return FullModelReference(
        final_logits=final_logits,
        predictions=predictions,
        evaluated_example_count=example_count,
        inference_batch_size=batch_size,
    )


def _validate_full_model_reference(
    reference: FullModelReference,
    inputs: torch.Tensor,
) -> None:
    if not isinstance(reference, FullModelReference):
        raise TypeError("full_model_reference must be a FullModelReference.")

    example_count = inputs.shape[0]

    if reference.evaluated_example_count != example_count:
        raise ValueError("Full-model reference example count does not match inputs.")

    if reference.final_logits.ndim != 2:
        raise ValueError("Full-model reference logits must have shape [example, class].")

    if reference.final_logits.shape[0] != example_count:
        raise ValueError("Full-model reference logit count does not match inputs.")

    if reference.predictions.shape != (example_count,):
        raise ValueError("Full-model reference predictions must have shape [example].")

    if reference.final_logits.device != inputs.device:
        raise ValueError("Full-model reference logits must be on the input device.")

    if reference.predictions.device != inputs.device:
        raise ValueError("Full-model reference predictions must be on the input device.")

    if reference.final_logits.requires_grad:
        raise ValueError("Full-model reference logits must be detached.")

    if not bool(torch.isfinite(reference.final_logits).all().item()):
        raise FloatingPointError("Full-model reference logits must all be finite.")


@dataclass(frozen=True)
class MaskEvaluationMetrics:
    """Complete Stage 8 behavioural-fidelity metrics for one mask."""

    primary_fidelity: float
    prediction_agreement_count: int
    full_accuracy: float
    masked_accuracy: float
    accuracy_change: float
    full_cross_entropy: float
    masked_cross_entropy: float
    cross_entropy_change: float
    mean_kl_divergence: float
    mean_jensen_shannon_divergence: float
    maximum_absolute_logit_difference: float
    retained_attention_head_count: int
    retained_mlp_neuron_count: int
    retained_component_count: int
    retained_component_proportion: float
    evaluated_example_count: int
    evaluation_batch_size: int

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-safe metric record."""

        return asdict(self)


def _validate_evaluation_data(
    model: HookedTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    validate_mask_model(model)

    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a PyTorch tensor.")

    if inputs.ndim != 2:
        raise ValueError("inputs must have shape (example, sequence_position).")

    if inputs.shape[0] == 0:
        raise ValueError("inputs must contain at least one example.")

    if inputs.shape[1] != model.cfg.n_ctx:
        raise ValueError(f"inputs must contain exactly {model.cfg.n_ctx} positions.")

    if inputs.dtype != torch.long:
        raise TypeError("inputs must have dtype torch.long.")

    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a PyTorch tensor.")

    if targets.ndim != 1:
        raise ValueError("targets must be one-dimensional.")

    if targets.shape[0] != inputs.shape[0]:
        raise ValueError("targets length must equal the number of input examples.")

    if targets.dtype != torch.long:
        raise TypeError("targets must have dtype torch.long.")

    minimum_target = int(targets.min().item())
    maximum_target = int(targets.max().item())

    if minimum_target < 0 or maximum_target >= OUTPUT_CLASS_COUNT:
        raise ValueError("targets must contain only output classes 0 through 112.")

    parameter = next(model.parameters(), None)

    if parameter is not None and inputs.device != parameter.device:
        raise ValueError("inputs and model parameters must be on the same device.")

    if targets.device != inputs.device:
        raise ValueError("targets and inputs must be on the same device.")


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("evaluation batch size must be an integer.")

    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive.")

    return batch_size


def _distribution_divergences(
    full_logits: torch.Tensor,
    masked_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-example KL(full || masked) and Jensen-Shannon values."""

    full_log_probabilities = functional.log_softmax(
        full_logits,
        dim=-1,
    )
    masked_log_probabilities = functional.log_softmax(
        masked_logits,
        dim=-1,
    )

    full_probabilities = full_log_probabilities.exp()
    masked_probabilities = masked_log_probabilities.exp()

    kl_divergence = (full_probabilities * (full_log_probabilities - masked_log_probabilities)).sum(
        dim=-1
    )

    log_mixture = torch.logaddexp(
        full_log_probabilities,
        masked_log_probabilities,
    ) - math.log(2.0)

    full_to_mixture = (full_probabilities * (full_log_probabilities - log_mixture)).sum(dim=-1)
    masked_to_mixture = (masked_probabilities * (masked_log_probabilities - log_mixture)).sum(
        dim=-1
    )

    jensen_shannon = 0.5 * (full_to_mixture + masked_to_mixture)

    return kl_divergence, jensen_shannon


def evaluate_component_mask(
    model: HookedTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: ComponentMask,
    *,
    batch_size: int,
    full_model_reference: FullModelReference | None = None,
) -> MaskEvaluationMetrics:
    """Compare masked and full models over all examples in fixed order.

    Cross-entropy, KL divergence, and Jensen-Shannon divergence are
    calculated per example and then reduced with one example-weighted mean.

    KL uses KL(full-model distribution || masked-model distribution).
    Jensen-Shannon uses the equal-weight mixture of the two distributions.
    Only logits at the final sequence position are evaluated.
    """

    if not isinstance(mask, ComponentMask):
        raise TypeError("mask must be a ComponentMask.")

    _validate_evaluation_data(model, inputs, targets)
    batch_size = _validate_batch_size(batch_size)

    if full_model_reference is None:
        full_model_reference = compute_full_model_reference(
            model,
            inputs,
            targets,
            batch_size=batch_size,
        )
    else:
        _validate_full_model_reference(
            full_model_reference,
            inputs,
        )

    example_count = inputs.shape[0]

    agreement_count = 0
    full_correct_count = 0
    masked_correct_count = 0

    full_cross_entropy_sum = 0.0
    masked_cross_entropy_sum = 0.0
    kl_divergence_sum = 0.0
    jensen_shannon_sum = 0.0
    maximum_absolute_logit_difference = 0.0

    was_training = model.training
    model.eval()

    try:
        for start in range(0, example_count, batch_size):
            stop = min(start + batch_size, example_count)
            batch_inputs = inputs[start:stop]
            batch_targets = targets[start:stop]

            full_logits = full_model_reference.final_logits[start:stop]
            full_predictions = full_model_reference.predictions[start:stop]

            masked_sequence_logits = masked_model_logits(
                model,
                batch_inputs,
                mask,
            )
            masked_logits = final_position_logits(masked_sequence_logits)

            masked_predictions = masked_logits.argmax(dim=-1)

            agreement_count += int((full_predictions == masked_predictions).sum().item())
            full_correct_count += int((full_predictions == batch_targets).sum().item())
            masked_correct_count += int((masked_predictions == batch_targets).sum().item())

            full_losses = functional.cross_entropy(
                full_logits,
                batch_targets,
                reduction="none",
            )
            masked_losses = functional.cross_entropy(
                masked_logits,
                batch_targets,
                reduction="none",
            )

            full_cross_entropy_sum += float(full_losses.to(torch.float64).sum().item())
            masked_cross_entropy_sum += float(masked_losses.to(torch.float64).sum().item())

            kl_values, jensen_shannon_values = _distribution_divergences(
                full_logits,
                masked_logits,
            )

            kl_divergence_sum += float(kl_values.to(torch.float64).sum().item())
            jensen_shannon_sum += float(jensen_shannon_values.to(torch.float64).sum().item())

            batch_maximum = float((masked_logits - full_logits).abs().max().item())
            maximum_absolute_logit_difference = max(
                maximum_absolute_logit_difference,
                batch_maximum,
            )
    finally:
        model.train(was_training)

    denominator = float(example_count)

    full_accuracy = full_correct_count / denominator
    masked_accuracy = masked_correct_count / denominator
    full_cross_entropy = full_cross_entropy_sum / denominator
    masked_cross_entropy = masked_cross_entropy_sum / denominator

    metrics = MaskEvaluationMetrics(
        primary_fidelity=agreement_count / denominator,
        prediction_agreement_count=agreement_count,
        full_accuracy=full_accuracy,
        masked_accuracy=masked_accuracy,
        accuracy_change=masked_accuracy - full_accuracy,
        full_cross_entropy=full_cross_entropy,
        masked_cross_entropy=masked_cross_entropy,
        cross_entropy_change=(masked_cross_entropy - full_cross_entropy),
        mean_kl_divergence=kl_divergence_sum / denominator,
        mean_jensen_shannon_divergence=(jensen_shannon_sum / denominator),
        maximum_absolute_logit_difference=(maximum_absolute_logit_difference),
        retained_attention_head_count=(mask.retained_attention_head_count),
        retained_mlp_neuron_count=(mask.retained_mlp_neuron_count),
        retained_component_count=mask.retained_component_count,
        retained_component_proportion=(mask.retained_component_proportion),
        evaluated_example_count=example_count,
        evaluation_batch_size=batch_size,
    )

    numeric_values = (
        metrics.primary_fidelity,
        metrics.full_accuracy,
        metrics.masked_accuracy,
        metrics.accuracy_change,
        metrics.full_cross_entropy,
        metrics.masked_cross_entropy,
        metrics.cross_entropy_change,
        metrics.mean_kl_divergence,
        metrics.mean_jensen_shannon_divergence,
        metrics.maximum_absolute_logit_difference,
        metrics.retained_component_proportion,
    )

    if not all(math.isfinite(value) for value in numeric_values):
        raise FloatingPointError("Mask-evaluation metrics must all be finite.")

    return metrics


@dataclass(frozen=True)
class CheckpointEvaluationContext:
    """Validated model, data, and provenance for one checkpoint."""

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
    model: HookedTransformer
    inputs: torch.Tensor
    targets: torch.Tensor
    device: torch.device


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")

    import json

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object.")

    return value


def _resolve_path(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _require_equal(
    actual: Any,
    expected: Any,
    message: str,
) -> None:
    if actual != expected:
        raise ValueError(f"{message}: expected {expected!r}, received {actual!r}.")


def _phase_checkpoint_records(
    checkpoint_manifest: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []

    pre = checkpoint_manifest.get("pre_checkpoint")
    if isinstance(pre, dict):
        records.append(pre)

    transitions = checkpoint_manifest.get("formal_transition_checkpoints")
    if not isinstance(transitions, dict):
        raise ValueError("Checkpoint manifest formal_transition_checkpoints must be a mapping.")

    for record in transitions.values():
        if not isinstance(record, dict):
            raise ValueError("Every formal transition checkpoint must be a mapping.")
        records.append(record)

    stable = checkpoint_manifest.get("selected_stable_post_checkpoint")
    if isinstance(stable, dict):
        records.append(stable)

    return tuple(records)


def load_checkpoint_evaluation_context(
    *,
    repository_root: str | Path,
    run_id: str,
    checkpoint_manifest_path: str | Path,
    checkpoint_step: int,
    device_override: str | None,
) -> CheckpointEvaluationContext:
    """Load and cross-validate one frozen Stage 8 checkpoint."""

    import json

    from circuit_families.config import (
        combined_config_hash,
        config_hash,
        load_config,
        load_model_config,
        load_training_config,
        mapping_hash,
    )
    from circuit_families.models import build_transformer
    from circuit_families.training import (
        canonical_state_hash,
        file_sha256,
        load_checkpoint_payload,
        load_training_data,
        resolve_device,
    )

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string.")

    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise ValueError("checkpoint_step must be a non-negative integer.")

    repository = Path(repository_root).resolve()
    checkpoint_manifest_file = _resolve_path(
        repository,
        checkpoint_manifest_path,
    )
    checkpoint_manifest = _load_json_object(
        checkpoint_manifest_file,
        "checkpoint manifest",
    )

    _require_equal(
        checkpoint_manifest.get("run_id"),
        run_id,
        "Checkpoint manifest run ID mismatch",
    )

    training_manifest_file = repository / "manifests" / f"training_{run_id}.json"
    training_manifest = _load_json_object(
        training_manifest_file,
        "training manifest",
    )

    _require_equal(
        training_manifest.get("run_id"),
        run_id,
        "Training manifest run ID mismatch",
    )

    recorded_training_manifest = checkpoint_manifest.get("source_training_manifest")
    expected_training_manifest = str(training_manifest_file.relative_to(repository))
    _require_equal(
        recorded_training_manifest,
        expected_training_manifest,
        "Checkpoint manifest training-manifest path mismatch",
    )

    phase_records = {
        int(record["training_step"]): record
        for record in _phase_checkpoint_records(checkpoint_manifest)
    }

    if checkpoint_step not in phase_records:
        raise ValueError(
            f"Checkpoint step {checkpoint_step} is not in the frozen Stage 7 phase grid."
        )

    phase_record = phase_records[checkpoint_step]

    training_records = {
        int(record["training_step"]): record
        for record in training_manifest.get("checkpoints", [])
        if isinstance(record, dict)
    }

    if checkpoint_step not in training_records:
        raise ValueError(f"Checkpoint step {checkpoint_step} is absent from the training manifest.")

    training_record = training_records[checkpoint_step]

    _require_equal(
        phase_record["checkpoint_path"],
        training_record["path"],
        "Checkpoint path mismatch between manifests",
    )
    _require_equal(
        phase_record["checkpoint_sha256"],
        training_record["file_sha256"],
        "Checkpoint hash mismatch between manifests",
    )
    _require_equal(
        phase_record["run_id"],
        run_id,
        "Checkpoint record run ID mismatch",
    )

    checkpoint_file = _resolve_path(
        repository,
        phase_record["checkpoint_path"],
    )
    actual_checkpoint_sha256 = file_sha256(checkpoint_file)

    _require_equal(
        actual_checkpoint_sha256,
        phase_record["checkpoint_sha256"],
        "Checkpoint physical hash mismatch",
    )

    configs = training_manifest["configs"]

    task_config = load_config(_resolve_path(repository, configs["task"]["path"]))
    model_config = load_model_config(_resolve_path(repository, configs["model"]["path"]))
    training_config = load_training_config(_resolve_path(repository, configs["training"]["path"]))

    actual_task_hash = config_hash(task_config)
    actual_model_hash = mapping_hash(model_config)
    actual_training_hash = mapping_hash(training_config)
    identity_inputs = {
        "task": task_config,
        "model": model_config,
        "training": training_config,
        "execution": training_manifest["execution"],
    }
    if "run_identity" in training_manifest:
        identity_inputs["run_identity"] = training_manifest["run_identity"]
    if "conditional_extension" in training_manifest:
        extension = training_manifest["conditional_extension"]
        identity_inputs["conditional_extension"] = {
            key: extension[key]
            for key in (
                "standard_horizon",
                "extension_increment",
                "absolute_max_steps",
                "decision",
            )
        }
    actual_combined_hash = combined_config_hash(identity_inputs)

    _require_equal(
        actual_task_hash,
        configs["task"]["sha256"],
        "Task-config hash mismatch",
    )
    _require_equal(
        actual_model_hash,
        configs["model"]["sha256"],
        "Model-config hash mismatch",
    )
    _require_equal(
        actual_training_hash,
        configs["training"]["sha256"],
        "Training-config hash mismatch",
    )
    _require_equal(
        actual_combined_hash,
        configs["combined_sha256"],
        "Combined-config hash mismatch",
    )

    device = resolve_device(device_override)

    dataset = training_manifest["dataset"]
    data = load_training_data(
        archive_path=_resolve_path(
            repository,
            dataset["archive_path"],
        ),
        metadata_path=_resolve_path(
            repository,
            dataset["metadata_path"],
        ),
        manifest_path=_resolve_path(
            repository,
            dataset["manifest_path"],
        ),
        task_config=task_config,
        device=device,
    )

    if data.full_inputs is None or data.full_targets is None:
        raise RuntimeError("Validated full lexicographic tensors are unavailable.")

    _require_equal(
        data.total_count,
        12_769,
        "Complete evaluation-example count mismatch",
    )
    _require_equal(
        data.archive_sha256,
        dataset["archive_sha256"],
        "Dataset archive hash mismatch",
    )
    _require_equal(
        data.metadata_sha256,
        dataset["metadata_sha256"],
        "Dataset metadata hash mismatch",
    )

    canonical_hashes = dataset["canonical_hashes"]

    for name in ("dataset_sha256", "split_sha256"):
        _require_equal(
            data.dataset_hashes[name],
            canonical_hashes[name],
            f"Dataset canonical {name} mismatch",
        )

    payload = load_checkpoint_payload(
        checkpoint_file,
        map_location=device,
    )

    _require_equal(
        payload["training_step"],
        checkpoint_step,
        "Checkpoint payload step mismatch",
    )
    _require_equal(
        payload["model_seed"],
        training_manifest["seed"]["value"],
        "Checkpoint model seed mismatch",
    )
    _require_equal(
        payload["model_config"],
        model_config,
        "Checkpoint model configuration mismatch",
    )
    _require_equal(
        payload["training_config"],
        training_config,
        "Checkpoint training configuration mismatch",
    )

    for name in ("dataset_sha256", "split_sha256"):
        _require_equal(
            payload["dataset_hashes"][name],
            data.dataset_hashes[name],
            f"Checkpoint dataset {name} mismatch",
        )

    _require_equal(
        payload["model_state_sha256"],
        training_record["model_state_sha256"],
        "Checkpoint model-state hash mismatch",
    )

    model = build_transformer(
        model_config,
        seed=payload["model_seed"],
        device=device,
    )
    model.load_state_dict(payload["model_state"], strict=True)

    _require_equal(
        canonical_state_hash(model.state_dict()),
        payload["model_state_sha256"],
        "Restored model-state hash mismatch",
    )

    expected_left = torch.arange(
        113,
        device=device,
    ).repeat_interleave(113)
    expected_right = torch.arange(
        113,
        device=device,
    ).repeat(113)

    if not torch.equal(data.full_inputs[:, 0], expected_left):
        raise ValueError("Complete inputs are not in lexicographic left-operand order.")

    if not torch.equal(data.full_inputs[:, 1], expected_right):
        raise ValueError("Complete inputs are not in lexicographic right-operand order.")

    if not torch.all(data.full_inputs[:, 2] == 113):
        raise ValueError("Complete inputs do not use the frozen equals token.")

    expected_targets = (expected_left + expected_right) % 113

    if not torch.equal(data.full_targets, expected_targets):
        raise ValueError("Complete targets do not match modular addition.")

    json.dumps(training_manifest, allow_nan=False)

    return CheckpointEvaluationContext(
        run_id=run_id,
        checkpoint_phase=str(phase_record["phase_label"]),
        checkpoint_step=checkpoint_step,
        checkpoint_path=checkpoint_file,
        checkpoint_sha256=actual_checkpoint_sha256,
        checkpoint_manifest_path=checkpoint_manifest_file,
        checkpoint_manifest_sha256=file_sha256(checkpoint_manifest_file),
        training_manifest_path=training_manifest_file,
        training_manifest_sha256=file_sha256(training_manifest_file),
        model_state_sha256=payload["model_state_sha256"],
        task_config_sha256=actual_task_hash,
        model_config_sha256=actual_model_hash,
        training_config_sha256=actual_training_hash,
        combined_config_sha256=actual_combined_hash,
        dataset_sha256=data.dataset_hashes["dataset_sha256"],
        split_sha256=data.dataset_hashes["split_sha256"],
        dataset_archive_sha256=data.archive_sha256,
        dataset_metadata_sha256=data.metadata_sha256,
        example_ordering="lexicographic",
        model=model,
        inputs=data.full_inputs,
        targets=data.full_targets,
        device=device,
    )

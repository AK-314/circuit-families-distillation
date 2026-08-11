"""Deterministic gate-gradient ranking for sparse-circuit search."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformer_lens import HookedTransformer

from circuit_families.interpretability.component_ablation import (
    validate_mask_model,
)
from circuit_families.interpretability.fidelity import (
    CheckpointEvaluationContext,
    MaskEvaluationMetrics,
    compute_full_model_reference,
    evaluate_component_mask,
)
from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_COUNT,
    ATTENTION_HEAD_HOOK_NAME,
    MLP_NEURON_COUNT,
    MLP_NEURON_HOOK_NAME,
    SEARCHABLE_COMPONENT_COUNT,
    SEARCHABLE_COMPONENT_IDS,
    ComponentMask,
    component_location,
    save_component_mask,
)
from circuit_families.training import (
    canonical_state_hash,
    file_sha256,
)
from circuit_families.training.metrics import (
    OUTPUT_CLASS_COUNT,
    final_position_logits,
)

CANDIDATE_BATCH_SIZE = 16
DEFAULT_EXACT_EVALUATION_BUDGET = 10_000
MEANINGFULLY_SPARSE_MAX_COMPONENTS = 258

RANKING_SCORE_DEFINITION = (
    "signed first-order predicted pseudo-target loss change under deletion: "
    "score_i = -g_i * dL/dg_i; retained gates have g_i=1"
)

_COMPONENT_INDEX_BY_ID = {
    identifier: index
    for index, identifier in enumerate(SEARCHABLE_COMPONENT_IDS)
}


@dataclass(frozen=True)
class ComponentRanking:
    """One retained component's first-order deletion ranking."""

    component_identifier: str
    component_index: int
    component_class: str
    gate_gradient: float
    estimated_removal_damage: float
    ranking_position: int


@dataclass(frozen=True)
class RankingResult:
    """Complete result of one full-dataset gate-gradient pass."""

    mean_pseudo_target_loss: float
    mean_gate_gradients: tuple[float, ...]
    ranked_components: tuple[ComponentRanking, ...]
    evaluated_example_count: int
    ranking_batch_size: int
    retained_component_count: int
    model_state_sha256_before: str
    model_state_sha256_after: str
    hook_counts_before: tuple[tuple[str, int], ...]
    hook_counts_after: tuple[tuple[str, int], ...]
    full_model_reference_sha256: str = ""
    full_model_reference_example_count: int = 0
    full_model_reference_batch_size: int = 0
    gradient_source: str = "component_gates"
    score_definition: str = RANKING_SCORE_DEFINITION


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("ranking batch size must be an integer.")

    if batch_size <= 0:
        raise ValueError("ranking batch size must be positive.")

    return batch_size


def _validate_inputs(
    model: HookedTransformer,
    inputs: torch.Tensor,
) -> None:
    validate_mask_model(model)

    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a PyTorch tensor.")

    if inputs.ndim != 2:
        raise ValueError(
            "inputs must have shape (example, sequence_position)."
        )

    if inputs.shape[0] == 0:
        raise ValueError("inputs must contain at least one example.")

    if inputs.shape[1] != model.cfg.n_ctx:
        raise ValueError(
            f"inputs must contain exactly {model.cfg.n_ctx} positions."
        )

    if inputs.dtype != torch.long:
        raise TypeError("inputs must have dtype torch.long.")

    parameter = next(model.parameters(), None)

    if parameter is not None and inputs.device != parameter.device:
        raise ValueError(
            "inputs and model parameters must be on the same device."
        )


def _validate_pseudo_targets(
    pseudo_targets: torch.Tensor,
    *,
    inputs: torch.Tensor,
) -> None:
    if not isinstance(pseudo_targets, torch.Tensor):
        raise TypeError("pseudo_targets must be a PyTorch tensor.")

    if pseudo_targets.ndim != 1:
        raise ValueError("pseudo_targets must be one-dimensional.")

    if pseudo_targets.shape[0] != inputs.shape[0]:
        raise ValueError(
            "pseudo_targets length must equal the example count."
        )

    if pseudo_targets.dtype != torch.long:
        raise TypeError("pseudo_targets must have dtype torch.long.")

    if pseudo_targets.device != inputs.device:
        raise ValueError(
            "pseudo_targets and inputs must be on the same device."
        )

    minimum = int(pseudo_targets.min().item())
    maximum = int(pseudo_targets.max().item())

    if minimum < 0 or maximum >= OUTPUT_CLASS_COUNT:
        raise ValueError(
            "pseudo_targets must contain only classes 0 through 112."
        )


def _mask_values(mask: ComponentMask) -> tuple[int, ...]:
    if not isinstance(mask, ComponentMask):
        raise TypeError("mask must be a ComponentMask.")

    values = mask.attention_head_mask + mask.mlp_neuron_mask

    if len(values) != SEARCHABLE_COMPONENT_COUNT:
        raise RuntimeError("Component-mask ordering or length changed.")

    return values


def _hook_counts(
    model: HookedTransformer,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            hook_name,
            len(model.hook_dict[hook_name]._forward_hooks),
        )
        for hook_name in (
            ATTENTION_HEAD_HOOK_NAME,
            MLP_NEURON_HOOK_NAME,
        )
    )


def freeze_full_model_pseudo_targets(
    model: HookedTransformer,
    inputs: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Freeze checkpoint-specific full-model final-position predictions."""

    _validate_inputs(model, inputs)
    batch_size = _validate_batch_size(batch_size)

    predictions: list[torch.Tensor] = []
    was_training = model.training
    model.eval()

    try:
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])

            with torch.inference_mode():
                sequence_logits = model(inputs[start:stop])
                logits = final_position_logits(sequence_logits)

            if logits.ndim != 2:
                raise ValueError(
                    "Final-position logits must be two-dimensional."
                )

            if logits.shape[1] != OUTPUT_CLASS_COUNT:
                raise ValueError(
                    f"Final logits must contain {OUTPUT_CLASS_COUNT} "
                    "output classes."
                )

            predictions.append(logits.argmax(dim=-1))
    finally:
        model.train(was_training)

    frozen = torch.cat(predictions, dim=0).detach()

    if frozen.shape != (inputs.shape[0],):
        raise RuntimeError(
            "Frozen pseudo-target shape does not match the dataset."
        )

    _validate_pseudo_targets(frozen, inputs=inputs)
    return frozen


def sort_component_rankings(
    rankings: Sequence[ComponentRanking],
) -> tuple[ComponentRanking, ...]:
    """Sort by ascending score, then stable component index."""

    values = tuple(rankings)

    if any(
        not isinstance(value, ComponentRanking)
        for value in values
    ):
        raise TypeError(
            "rankings must contain ComponentRanking values."
        )

    ordered = sorted(
        values,
        key=lambda value: (
            value.estimated_removal_damage,
            value.component_index,
        ),
    )

    return tuple(
        ComponentRanking(
            component_identifier=value.component_identifier,
            component_index=value.component_index,
            component_class=value.component_class,
            gate_gradient=value.gate_gradient,
            estimated_removal_damage=(
                value.estimated_removal_damage
            ),
            ranking_position=position,
        )
        for position, value in enumerate(ordered, start=1)
    )


def rank_retained_components(
    model: HookedTransformer,
    inputs: torch.Tensor,
    pseudo_targets: torch.Tensor,
    mask: ComponentMask,
    *,
    batch_size: int,
) -> RankingResult:
    """Rank retained components by signed first-order deletion damage.

    The global ranking loss is the example-weighted mean cross-entropy
    between gated-model final-position logits and frozen full-model
    top-one pseudo-targets.

    For retained component i, deletion changes its gate from one to zero.
    First-order Taylor expansion therefore gives:

        estimated_delta_loss_i = -g_i * dL/dg_i

    Retained gates have g_i = 1. Already removed components are multiplied
    by a fixed zero binary mask and therefore remain zero throughout the
    ranking pass. Model parameters are not ranking variables.
    """

    _validate_inputs(model, inputs)
    _validate_pseudo_targets(pseudo_targets, inputs=inputs)
    batch_size = _validate_batch_size(batch_size)
    mask_values = _mask_values(mask)

    model_state_before = canonical_state_hash(model.state_dict())
    hook_counts_before = _hook_counts(model)
    was_training = model.training

    active_mask = torch.tensor(
        mask_values,
        dtype=torch.float32,
        device=inputs.device,
    )
    gates = torch.ones(
        SEARCHABLE_COMPONENT_COUNT,
        dtype=torch.float32,
        device=inputs.device,
        requires_grad=True,
    )

    gradient_sum = torch.zeros(
        SEARCHABLE_COMPONENT_COUNT,
        dtype=torch.float64,
        device=inputs.device,
    )
    loss_sum = 0.0

    def attention_hook(
        activation: torch.Tensor,
        hook: Any,
    ) -> torch.Tensor:
        if hook.name != ATTENTION_HEAD_HOOK_NAME:
            raise RuntimeError("Attention-head hook name changed.")

        if activation.ndim != 4:
            raise ValueError(
                "Attention activation must be four-dimensional."
            )

        if activation.shape[2] != ATTENTION_HEAD_COUNT:
            raise ValueError(
                "Attention activation head count changed."
            )

        effective = (
            gates[:ATTENTION_HEAD_COUNT]
            * active_mask[:ATTENTION_HEAD_COUNT]
        ).to(dtype=activation.dtype)

        return activation * effective.view(
            1,
            1,
            ATTENTION_HEAD_COUNT,
            1,
        )

    def neuron_hook(
        activation: torch.Tensor,
        hook: Any,
    ) -> torch.Tensor:
        if hook.name != MLP_NEURON_HOOK_NAME:
            raise RuntimeError("MLP-neuron hook name changed.")

        if activation.ndim != 3:
            raise ValueError(
                "MLP activation must be three-dimensional."
            )

        if activation.shape[2] != MLP_NEURON_COUNT:
            raise ValueError("MLP activation width changed.")

        effective = (
            gates[ATTENTION_HEAD_COUNT:]
            * active_mask[ATTENTION_HEAD_COUNT:]
        ).to(dtype=activation.dtype)

        return activation * effective.view(
            1,
            1,
            MLP_NEURON_COUNT,
        )

    model.zero_grad(set_to_none=True)
    model.eval()

    try:
        with model.hooks(
            fwd_hooks=[
                (ATTENTION_HEAD_HOOK_NAME, attention_hook),
                (MLP_NEURON_HOOK_NAME, neuron_hook),
            ]
        ):
            for start in range(0, inputs.shape[0], batch_size):
                stop = min(start + batch_size, inputs.shape[0])
                batch_inputs = inputs[start:stop]
                batch_targets = pseudo_targets[start:stop]

                sequence_logits = model(batch_inputs)
                logits = final_position_logits(sequence_logits)

                if logits.shape != (
                    stop - start,
                    OUTPUT_CLASS_COUNT,
                ):
                    raise ValueError(
                        "Ranking logits have an invalid shape."
                    )

                batch_loss_sum = functional.cross_entropy(
                    logits,
                    batch_targets,
                    reduction="sum",
                )

                batch_gradient = torch.autograd.grad(
                    batch_loss_sum,
                    gates,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                loss_sum += float(
                    batch_loss_sum.detach().to(torch.float64).item()
                )
                gradient_sum += batch_gradient.detach().to(
                    torch.float64
                )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    example_count = inputs.shape[0]
    mean_loss = loss_sum / float(example_count)
    mean_gradient = gradient_sum / float(example_count)

    if not math.isfinite(mean_loss):
        raise FloatingPointError(
            "Mean pseudo-target ranking loss is not finite."
        )

    if not bool(torch.isfinite(mean_gradient).all().item()):
        raise FloatingPointError(
            "Mean component-gate gradients are not finite."
        )

    raw_rankings: list[ComponentRanking] = []

    for identifier in mask.retained_component_ids:
        index = _COMPONENT_INDEX_BY_ID[identifier]
        gradient = float(mean_gradient[index].item())
        gate_value = float(active_mask[index].item())
        score = -gate_value * gradient
        location = component_location(identifier)

        raw_rankings.append(
            ComponentRanking(
                component_identifier=identifier,
                component_index=index,
                component_class=location.component_class,
                gate_gradient=gradient,
                estimated_removal_damage=score,
                ranking_position=0,
            )
        )

    ranked = sort_component_rankings(raw_rankings)

    model_state_after = canonical_state_hash(model.state_dict())
    hook_counts_after = _hook_counts(model)

    if model_state_after != model_state_before:
        raise RuntimeError(
            "Gate-gradient ranking changed the model-state hash."
        )

    if hook_counts_after != hook_counts_before:
        raise RuntimeError(
            "Gate-gradient ranking leaked TransformerLens hooks."
        )

    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError(
            "Gate-gradient ranking populated parameter gradients."
        )

    return RankingResult(
        mean_pseudo_target_loss=mean_loss,
        mean_gate_gradients=tuple(
            float(value)
            for value in mean_gradient.detach().cpu().tolist()
        ),
        ranked_components=ranked,
        evaluated_example_count=example_count,
        ranking_batch_size=batch_size,
        retained_component_count=mask.retained_component_count,
        model_state_sha256_before=model_state_before,
        model_state_sha256_after=model_state_after,
        hook_counts_before=hook_counts_before,
        hook_counts_after=hook_counts_after,
    )


def partition_ranked_candidates(
    rankings: Sequence[ComponentRanking],
    *,
    candidate_batch_size: int = CANDIDATE_BATCH_SIZE,
) -> tuple[tuple[ComponentRanking, ...], ...]:
    """Divide an ordered ranking into consecutive candidate batches."""

    candidate_batch_size = _validate_batch_size(
        candidate_batch_size
    )
    values = tuple(rankings)

    if any(
        not isinstance(value, ComponentRanking)
        for value in values
    ):
        raise TypeError(
            "rankings must contain ComponentRanking values."
        )

    return tuple(
        values[start : start + candidate_batch_size]
        for start in range(0, len(values), candidate_batch_size)
    )


def remove_component(
    mask: ComponentMask,
    identifier: str,
) -> ComponentMask:
    """Return the binary mask after deleting one retained component."""

    if not isinstance(mask, ComponentMask):
        raise TypeError("mask must be a ComponentMask.")

    if identifier not in mask.retained_component_ids:
        raise ValueError(
            f"Component is not currently retained: {identifier}"
        )

    retained = tuple(
        component
        for component in mask.retained_component_ids
        if component != identifier
    )
    result = ComponentMask.from_retained_identifiers(retained)

    if result.retained_component_count != (
        mask.retained_component_count - 1
    ):
        raise RuntimeError(
            "Single-component deletion changed the count incorrectly."
        )

    return result


def is_meaningfully_sparse(mask: ComponentMask) -> bool:
    """Return whether a mask satisfies the frozen 258-component rule."""

    if not isinstance(mask, ComponentMask):
        raise TypeError("mask must be a ComponentMask.")

    return (
        mask.retained_component_count
        <= MEANINGFULLY_SPARSE_MAX_COMPONENTS
    )





@dataclass(frozen=True)
class SparseSearchArtifacts:
    """Paths and physical hashes for one serialized search cell."""

    output_directory: Path
    final_mask_path: Path
    final_mask_sha256: str
    accepted_mask_paths: tuple[Path, ...]
    accepted_mask_sha256s: tuple[str, ...]
    accepted_removal_trajectory_path: Path
    accepted_removal_trajectory_sha256: str
    candidate_evaluation_log_path: Path
    candidate_evaluation_log_sha256: str
    cell_summary_path: Path
    cell_summary_sha256: str
    hashes_path: Path
    hashes_sha256: str


def _stable_json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _write_stable_json(
    path: Path,
    value: Mapping[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _stable_json_text(value),
        encoding="utf-8",
    )
    return path


def _write_stable_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        for record in records
    ]

    content = "".join(f"{line}\n" for line in lines)
    path.write_text(content, encoding="utf-8")
    return path


def _relative_artifact_path(
    output_directory: Path,
    path: Path,
) -> str:
    return str(path.resolve().relative_to(output_directory.resolve()))


def _metric_record(
    metrics: MaskEvaluationMetrics,
) -> dict[str, Any]:
    if not isinstance(metrics, MaskEvaluationMetrics):
        raise TypeError("metrics must be MaskEvaluationMetrics.")

    return metrics.to_record()


def _accepted_removal_record(
    removal: AcceptedRemoval,
    *,
    output_directory: Path,
    accepted_mask_path: Path,
    accepted_mask_sha256: str,
) -> dict[str, Any]:
    return {
        "iteration": removal.iteration,
        "removed_component": removal.removed_component,
        "removed_component_type": removal.component_class,
        "component_index": removal.component_index,
        "retained_count_before": removal.retained_count_before,
        "retained_count_after": removal.retained_count_after,
        "exact_fidelity_after_removal": (
            removal.exact_fidelity_after_removal
        ),
        "masked_ground_truth_accuracy": (
            removal.metrics.masked_accuracy
        ),
        "masked_cross_entropy": (
            removal.metrics.masked_cross_entropy
        ),
        "mean_kl_divergence": (
            removal.metrics.mean_kl_divergence
        ),
        "mean_jensen_shannon_divergence": (
            removal.metrics.mean_jensen_shannon_divergence
        ),
        "ranking_score": removal.ranking_score,
        "ranking_position": removal.ranking_position,
        "candidate_batch_index": (
            removal.candidate_batch_index
        ),
        "candidates_exactly_tested_in_iteration": (
            removal.candidates_exactly_tested_in_iteration
        ),
        "cumulative_exact_evaluations": (
            removal.cumulative_exact_evaluations
        ),
        "cumulative_ranking_passes": (
            removal.cumulative_ranking_passes
        ),
        "accepted_mask_id": removal.accepted_mask.mask_id,
        "accepted_mask_path": _relative_artifact_path(
            output_directory,
            accepted_mask_path,
        ),
        "accepted_mask_sha256": accepted_mask_sha256,
        "metrics": _metric_record(removal.metrics),
    }


def _candidate_evaluation_record(
    evaluation: CandidateEvaluation,
) -> dict[str, Any]:
    return {
        "iteration": evaluation.iteration,
        "candidate_component": (
            evaluation.candidate_component
        ),
        "component_index": evaluation.component_index,
        "component_type": evaluation.component_class,
        "ranking_score": evaluation.ranking_score,
        "ranking_position": evaluation.ranking_position,
        "candidate_batch_index": (
            evaluation.candidate_batch_index
        ),
        "exact_fidelity": evaluation.exact_fidelity,
        "passed_threshold": evaluation.passed_threshold,
        "accepted": evaluation.accepted,
        "rejection_reason": evaluation.rejection_reason,
        "cumulative_exact_evaluations": (
            evaluation.cumulative_exact_evaluations
        ),
        "candidate_mask_id": evaluation.candidate_mask.mask_id,
        "retained_component_count": (
            evaluation.candidate_mask.retained_component_count
        ),
        "metrics": _metric_record(evaluation.metrics),
    }


def _prepare_empty_output_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)

    existing = tuple(path.iterdir())

    if existing:
        names = ", ".join(sorted(item.name for item in existing))
        raise FileExistsError(
            "Sparse-search output directory must be empty. "
            f"Existing entries: {names}"
        )

    return path


def write_sparse_search_artifacts(
    output_directory: str | Path,
    result: SparseSearchResult,
    *,
    cell_metadata: Mapping[str, Any],
) -> SparseSearchArtifacts:
    """Serialize one search cell into deterministic scientific artifacts.

    Wall-clock telemetry is intentionally not included here because it is
    nondeterministic. Runtime measurements are recorded separately by the
    execution scripts and are not part of these reproducibility hashes.
    """

    if not isinstance(result, SparseSearchResult):
        raise TypeError("result must be a SparseSearchResult.")

    if not isinstance(cell_metadata, Mapping):
        raise TypeError("cell_metadata must be a mapping.")

    output = _prepare_empty_output_directory(
        Path(output_directory)
    )

    accepted_mask_directory = output / "accepted_masks"
    accepted_mask_directory.mkdir(parents=True, exist_ok=True)

    accepted_mask_paths: list[Path] = []
    accepted_mask_sha256s: list[str] = []

    for removal in result.accepted_removals:
        mask_path = (
            accepted_mask_directory
            / f"iteration_{removal.iteration:04d}.json"
        )
        save_component_mask(mask_path, removal.accepted_mask)
        accepted_mask_paths.append(mask_path)
        accepted_mask_sha256s.append(file_sha256(mask_path))

    final_mask_path = output / "final_mask.json"
    save_component_mask(final_mask_path, result.final_mask)
    final_mask_sha256 = file_sha256(final_mask_path)

    trajectory_records = [
        _accepted_removal_record(
            removal,
            output_directory=output,
            accepted_mask_path=mask_path,
            accepted_mask_sha256=mask_sha256,
        )
        for removal, mask_path, mask_sha256 in zip(
            result.accepted_removals,
            accepted_mask_paths,
            accepted_mask_sha256s,
            strict=True,
        )
    ]

    candidate_records = [
        _candidate_evaluation_record(evaluation)
        for evaluation in result.candidate_evaluations
    ]

    trajectory_path = _write_stable_jsonl(
        output / "accepted_removals.jsonl",
        trajectory_records,
    )
    candidate_log_path = _write_stable_jsonl(
        output / "candidate_evaluations.jsonl",
        candidate_records,
    )

    trajectory_sha256 = file_sha256(trajectory_path)
    candidate_log_sha256 = file_sha256(candidate_log_path)

    summary = {
        "schema_version": 1,
        "cell_metadata": dict(cell_metadata),
        "search": {
            "status": result.status,
            "status_definition": SEARCH_STATUS_DEFINITIONS[
                result.status
            ],
            "fidelity_threshold": result.fidelity_threshold,
            "exact_evaluation_budget": (
                result.exact_evaluation_budget
            ),
            "exact_evaluations_used": (
                result.exact_evaluations_used
            ),
            "ranking_passes_used": (
                result.ranking_passes_used
            ),
            "accepted_removal_count": len(
                result.accepted_removals
            ),
            "rejected_candidate_count": (
                result.rejected_candidate_count
            ),
            "candidate_batches_tested": (
                result.candidate_batches_tested
            ),
            "budget_remaining": result.budget_remaining,
            "budget_exhausted": result.budget_exhausted,
            "locally_single_deletion_minimal": (
                result.locally_single_deletion_minimal
            ),
            "meaningfully_sparse": (
                result.meaningfully_sparse
            ),
            "reported_sparse_circuit": (
                result.status == "valid_sparse_circuit"
            ),
            "stopping_reason": result.stopping_reason,
            "failure_detail": result.failure_detail,
        },
        "initial_mask": {
            "mask_id": result.initial_mask.mask_id,
            "retained_component_count": (
                result.initial_mask.retained_component_count
            ),
        },
        "final_mask": {
            "mask_id": result.final_mask.mask_id,
            "path": _relative_artifact_path(
                output,
                final_mask_path,
            ),
            "sha256": final_mask_sha256,
            "retained_attention_head_count": (
                result.final_mask.retained_attention_head_count
            ),
            "retained_mlp_neuron_count": (
                result.final_mask.retained_mlp_neuron_count
            ),
            "retained_component_count": (
                result.final_mask.retained_component_count
            ),
            "retained_component_proportion": (
                result.final_mask.retained_component_proportion
            ),
        },
        "final_metrics": _metric_record(result.final_metrics),
        "outputs": {
            "accepted_removal_trajectory": {
                "path": _relative_artifact_path(
                    output,
                    trajectory_path,
                ),
                "sha256": trajectory_sha256,
                "record_count": len(trajectory_records),
            },
            "candidate_evaluation_log": {
                "path": _relative_artifact_path(
                    output,
                    candidate_log_path,
                ),
                "sha256": candidate_log_sha256,
                "record_count": len(candidate_records),
            },
            "accepted_masks": [
                {
                    "iteration": removal.iteration,
                    "mask_id": removal.accepted_mask.mask_id,
                    "path": _relative_artifact_path(
                        output,
                        mask_path,
                    ),
                    "sha256": mask_sha256,
                }
                for removal, mask_path, mask_sha256 in zip(
                    result.accepted_removals,
                    accepted_mask_paths,
                    accepted_mask_sha256s,
                    strict=True,
                )
            ],
        },
        "runtime_telemetry": {
            "included_in_deterministic_artifacts": False,
            "reason": (
                "Wall-clock measurements vary between reruns and are "
                "stored separately by execution scripts."
            ),
        },
    }

    summary_path = _write_stable_json(
        output / "cell_summary.json",
        summary,
    )
    summary_sha256 = file_sha256(summary_path)

    hashes = {
        "schema_version": 1,
        "final_mask": {
            "path": _relative_artifact_path(
                output,
                final_mask_path,
            ),
            "sha256": final_mask_sha256,
        },
        "accepted_removal_trajectory": {
            "path": _relative_artifact_path(
                output,
                trajectory_path,
            ),
            "sha256": trajectory_sha256,
        },
        "candidate_evaluation_log": {
            "path": _relative_artifact_path(
                output,
                candidate_log_path,
            ),
            "sha256": candidate_log_sha256,
        },
        "cell_summary": {
            "path": _relative_artifact_path(
                output,
                summary_path,
            ),
            "sha256": summary_sha256,
        },
        "accepted_masks": [
            {
                "path": _relative_artifact_path(
                    output,
                    mask_path,
                ),
                "sha256": mask_sha256,
            }
            for mask_path, mask_sha256 in zip(
                accepted_mask_paths,
                accepted_mask_sha256s,
                strict=True,
            )
        ],
    }

    hashes_path = _write_stable_json(
        output / "hashes.json",
        hashes,
    )
    hashes_sha256 = file_sha256(hashes_path)

    return SparseSearchArtifacts(
        output_directory=output,
        final_mask_path=final_mask_path,
        final_mask_sha256=final_mask_sha256,
        accepted_mask_paths=tuple(accepted_mask_paths),
        accepted_mask_sha256s=tuple(
            accepted_mask_sha256s
        ),
        accepted_removal_trajectory_path=trajectory_path,
        accepted_removal_trajectory_sha256=(
            trajectory_sha256
        ),
        candidate_evaluation_log_path=candidate_log_path,
        candidate_evaluation_log_sha256=(
            candidate_log_sha256
        ),
        cell_summary_path=summary_path,
        cell_summary_sha256=summary_sha256,
        hashes_path=hashes_path,
        hashes_sha256=hashes_sha256,
    )


@dataclass(frozen=True)
class CheckpointSearchExecution:
    """Integrity evidence for one checkpoint-backed sparse search."""

    result: SparseSearchResult
    pseudo_target_sha256: str
    pseudo_target_count: int
    ranking_batch_size: int
    evaluation_batch_size: int
    model_state_sha256_before: str
    model_state_sha256_after: str
    hook_counts_before: tuple[tuple[str, int], ...]
    hook_counts_after: tuple[tuple[str, int], ...]
    full_model_reference_sha256: str = ""
    full_model_reference_example_count: int = 0
    full_model_reference_batch_size: int = 0


def run_checkpoint_sparse_search(
    context: CheckpointEvaluationContext,
    *,
    fidelity_threshold: float,
    ranking_batch_size: int,
    evaluation_batch_size: int,
    exact_evaluation_budget: int = (
        DEFAULT_EXACT_EVALUATION_BUDGET
    ),
) -> CheckpointSearchExecution:
    """Run one sparse search through the validated Stage 8 machinery.

    The full checkpoint's final-position top-one predictions are frozen once
    and used only as pseudo-targets for gate-gradient ranking.

    Every candidate acceptance is delegated to the unchanged Stage 8 exact
    component-mask evaluator over the supplied complete evaluation tensors.
    """

    if not isinstance(context, CheckpointEvaluationContext):
        raise TypeError(
            "context must be a CheckpointEvaluationContext."
        )

    ranking_batch_size = _validate_batch_size(
        ranking_batch_size
    )
    evaluation_batch_size = _validate_batch_size(
        evaluation_batch_size
    )
    exact_evaluation_budget = _validate_exact_evaluation_budget(
        exact_evaluation_budget
    )

    model_state_before = canonical_state_hash(
        context.model.state_dict()
    )
    hook_counts_before = _hook_counts(context.model)

    if model_state_before != context.model_state_sha256:
        raise ValueError(
            "Checkpoint context model-state hash does not match "
            "the loaded model."
        )

    full_model_reference = compute_full_model_reference(
        context.model,
        context.inputs,
        context.targets,
        batch_size=evaluation_batch_size,
    )
    pseudo_targets = (
        full_model_reference.predictions.detach().clone()
    )
    pseudo_target_sha256 = canonical_state_hash(
        {"pseudo_targets": pseudo_targets}
    )
    full_model_reference_sha256 = canonical_state_hash(
        {
            "final_logits": full_model_reference.final_logits,
            "predictions": full_model_reference.predictions,
        }
    )

    initial_mask = ComponentMask.all_retained()
    initial_metrics = evaluate_component_mask(
        context.model,
        context.inputs,
        context.targets,
        initial_mask,
        batch_size=evaluation_batch_size,
        full_model_reference=full_model_reference,
    )

    def ranking_function(mask: ComponentMask) -> RankingResult:
        return rank_retained_components(
            context.model,
            context.inputs,
            pseudo_targets,
            mask,
            batch_size=ranking_batch_size,
        )

    def exact_evaluation_function(
        mask: ComponentMask,
    ) -> MaskEvaluationMetrics:
        return evaluate_component_mask(
            context.model,
            context.inputs,
            context.targets,
            mask,
            batch_size=evaluation_batch_size,
            full_model_reference=full_model_reference,
        )

    result = greedy_sparse_search(
        ranking_function=ranking_function,
        exact_evaluation_function=exact_evaluation_function,
        initial_metrics=initial_metrics,
        fidelity_threshold=fidelity_threshold,
        exact_evaluation_budget=exact_evaluation_budget,
    )

    model_state_after = canonical_state_hash(
        context.model.state_dict()
    )
    hook_counts_after = _hook_counts(context.model)

    if model_state_after != model_state_before:
        raise RuntimeError(
            "Checkpoint sparse search changed the model-state hash."
        )

    if hook_counts_after != hook_counts_before:
        raise RuntimeError(
            "Checkpoint sparse search leaked TransformerLens hooks."
        )

    if any(
        parameter.grad is not None
        for parameter in context.model.parameters()
    ):
        raise RuntimeError(
            "Checkpoint sparse search left parameter gradients populated."
        )

    return CheckpointSearchExecution(
        result=result,
        pseudo_target_sha256=pseudo_target_sha256,
        pseudo_target_count=int(pseudo_targets.shape[0]),
        ranking_batch_size=ranking_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        full_model_reference_sha256=(
            full_model_reference_sha256
        ),
        full_model_reference_example_count=(
            full_model_reference.evaluated_example_count
        ),
        full_model_reference_batch_size=(
            full_model_reference.inference_batch_size
        ),
        model_state_sha256_before=model_state_before,
        model_state_sha256_after=model_state_after,
        hook_counts_before=hook_counts_before,
        hook_counts_after=hook_counts_after,
    )


SEARCH_STATUS_DEFINITIONS = {
    "valid_sparse_circuit": (
        "Search proved local single-deletion termination and the final "
        "fidelity-valid mask retains at most 258 components."
    ),
    "valid_but_not_meaningfully_sparse": (
        "Search proved local single-deletion termination after accepting "
        "at least one removal, but the final fidelity-valid mask retains "
        "more than 258 components."
    ),
    "fidelity_failure": (
        "The all-retained starting state failed the active fidelity "
        "threshold, so no valid search trajectory could begin."
    ),
    "sparsity_failure": (
        "Reserved final-output validation status for an object claimed as "
        "a sparse circuit that violates the frozen sparsity definition. "
        "The primary greedy search uses the more specific terminal statuses."
    ),
    "budget_exhaustion": (
        "The exact-evaluation budget ended before local single-deletion "
        "termination could be established."
    ),
    "ranking_failure": (
        "Gate-gradient ranking raised an exception or returned a ranking "
        "inconsistent with the currently retained component universe."
    ),
    "invalid_masking_output": (
        "Exact mask evaluation failed or returned metrics inconsistent "
        "with the evaluated binary mask."
    ),
    "no_feasible_sparse_candidate_discovered_within_budget": (
        "Every one-component deletion from the all-retained mask was "
        "tested and failed the fidelity threshold, so no removal was "
        "accepted. This does not prove that no multi-component sparse "
        "candidate exists."
    ),
}

SUPPORTED_SEARCH_STATUSES = frozenset(SEARCH_STATUS_DEFINITIONS)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Auditable exact evaluation of one proposed deletion."""

    iteration: int
    candidate_component: str
    component_index: int
    component_class: str
    ranking_score: float
    ranking_position: int
    candidate_batch_index: int
    exact_fidelity: float
    passed_threshold: bool
    accepted: bool
    rejection_reason: str | None
    cumulative_exact_evaluations: int
    candidate_mask: ComponentMask
    metrics: MaskEvaluationMetrics


@dataclass(frozen=True)
class AcceptedRemoval:
    """One accepted greedy component deletion."""

    iteration: int
    removed_component: str
    component_index: int
    component_class: str
    retained_count_before: int
    retained_count_after: int
    exact_fidelity_after_removal: float
    ranking_score: float
    ranking_position: int
    candidate_batch_index: int
    candidates_exactly_tested_in_iteration: int
    cumulative_exact_evaluations: int
    cumulative_ranking_passes: int
    accepted_mask: ComponentMask
    metrics: MaskEvaluationMetrics


@dataclass(frozen=True)
class SparseSearchResult:
    """Complete deterministic result of one greedy sparse search."""

    status: str
    fidelity_threshold: float
    exact_evaluation_budget: int
    initial_mask: ComponentMask
    final_mask: ComponentMask
    final_metrics: MaskEvaluationMetrics
    accepted_removals: tuple[AcceptedRemoval, ...]
    candidate_evaluations: tuple[CandidateEvaluation, ...]
    exact_evaluations_used: int
    ranking_passes_used: int
    candidate_batches_tested: int
    rejected_candidate_count: int
    budget_remaining: int
    budget_exhausted: bool
    locally_single_deletion_minimal: bool
    meaningfully_sparse: bool
    stopping_reason: str
    failure_detail: str | None


@dataclass(frozen=True)
class _PendingCandidate:
    """Temporary complete evaluation before batch-level selection."""

    ranking: ComponentRanking
    candidate_mask: ComponentMask
    metrics: MaskEvaluationMetrics
    passed_threshold: bool
    cumulative_exact_evaluations: int


RankingFunction = Callable[[ComponentMask], RankingResult]
ExactEvaluationFunction = Callable[
    [ComponentMask],
    MaskEvaluationMetrics,
]


def _validate_fidelity_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("fidelity threshold must be numeric.")

    validated = float(value)

    if not math.isfinite(validated):
        raise ValueError("fidelity threshold must be finite.")

    if validated <= 0.0 or validated > 1.0:
        raise ValueError(
            "fidelity threshold must be greater than zero and at most one."
        )

    return validated


def _validate_exact_evaluation_budget(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("exact-evaluation budget must be an integer.")

    if value < 0:
        raise ValueError(
            "exact-evaluation budget must be non-negative."
        )

    return value


def _validate_exact_metrics(
    mask: ComponentMask,
    metrics: MaskEvaluationMetrics,
) -> None:
    if not isinstance(metrics, MaskEvaluationMetrics):
        raise TypeError(
            "exact evaluator must return MaskEvaluationMetrics."
        )

    if metrics.retained_attention_head_count != (
        mask.retained_attention_head_count
    ):
        raise ValueError(
            "Exact metrics retained-head count does not match the mask."
        )

    if metrics.retained_mlp_neuron_count != (
        mask.retained_mlp_neuron_count
    ):
        raise ValueError(
            "Exact metrics retained-neuron count does not match the mask."
        )

    if metrics.retained_component_count != (
        mask.retained_component_count
    ):
        raise ValueError(
            "Exact metrics retained-component count does not match "
            "the mask."
        )

    if metrics.evaluated_example_count <= 0:
        raise ValueError(
            "Exact metrics must evaluate at least one example."
        )

    if (
        not math.isfinite(metrics.primary_fidelity)
        or metrics.primary_fidelity < 0.0
        or metrics.primary_fidelity > 1.0
    ):
        raise ValueError(
            "Exact fidelity must be finite and between zero and one."
        )

    for name, value in metrics.to_record().items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"Exact metric {name} must be finite."
            )


def _validate_complete_ranking(
    mask: ComponentMask,
    result: RankingResult,
) -> None:
    if not isinstance(result, RankingResult):
        raise TypeError(
            "ranking function must return RankingResult."
        )

    if result.retained_component_count != (
        mask.retained_component_count
    ):
        raise ValueError(
            "Ranking retained-component count does not match the mask."
        )

    rankings = result.ranked_components
    expected_identifiers = mask.retained_component_ids
    returned_identifiers = tuple(
        value.component_identifier
        for value in rankings
    )

    if len(returned_identifiers) != len(
        set(returned_identifiers)
    ):
        raise ValueError(
            "Ranking contains duplicate component identifiers."
        )

    if set(returned_identifiers) != set(expected_identifiers):
        raise ValueError(
            "Ranking must contain every and only currently retained "
            "component."
        )

    expected_positions = tuple(
        range(1, len(rankings) + 1)
    )
    returned_positions = tuple(
        value.ranking_position
        for value in rankings
    )

    if returned_positions != expected_positions:
        raise ValueError(
            "Ranking positions must be consecutive and one-based."
        )

    for value in rankings:
        expected_index = _COMPONENT_INDEX_BY_ID[
            value.component_identifier
        ]

        if value.component_index != expected_index:
            raise ValueError(
                "Ranking component index does not match the frozen "
                "component universe."
            )

        expected_class = component_location(
            value.component_identifier
        ).component_class

        if value.component_class != expected_class:
            raise ValueError(
                "Ranking component class does not match the frozen "
                "component universe."
            )

        if not math.isfinite(value.gate_gradient):
            raise ValueError(
                "Ranking gate gradients must be finite."
            )

        if not math.isfinite(value.estimated_removal_damage):
            raise ValueError(
                "Ranking scores must be finite."
            )

    expected_order = tuple(
        sorted(
            rankings,
            key=lambda value: (
                value.estimated_removal_damage,
                value.component_index,
            ),
        )
    )

    if rankings != expected_order:
        raise ValueError(
            "Ranking is not ordered by score and stable component index."
        )


def _candidate_record(
    pending: _PendingCandidate,
    *,
    iteration: int,
    candidate_batch_index: int,
    accepted: bool,
    rejection_reason: str | None,
) -> CandidateEvaluation:
    ranking = pending.ranking

    return CandidateEvaluation(
        iteration=iteration,
        candidate_component=ranking.component_identifier,
        component_index=ranking.component_index,
        component_class=ranking.component_class,
        ranking_score=ranking.estimated_removal_damage,
        ranking_position=ranking.ranking_position,
        candidate_batch_index=candidate_batch_index,
        exact_fidelity=pending.metrics.primary_fidelity,
        passed_threshold=pending.passed_threshold,
        accepted=accepted,
        rejection_reason=rejection_reason,
        cumulative_exact_evaluations=(
            pending.cumulative_exact_evaluations
        ),
        candidate_mask=pending.candidate_mask,
        metrics=pending.metrics,
    )


def _build_search_result(
    *,
    status: str,
    fidelity_threshold: float,
    exact_evaluation_budget: int,
    initial_mask: ComponentMask,
    final_mask: ComponentMask,
    final_metrics: MaskEvaluationMetrics,
    accepted_removals: list[AcceptedRemoval],
    candidate_evaluations: list[CandidateEvaluation],
    exact_evaluations_used: int,
    ranking_passes_used: int,
    candidate_batches_tested: int,
    locally_single_deletion_minimal: bool,
    stopping_reason: str,
    failure_detail: str | None = None,
) -> SparseSearchResult:
    if status not in SUPPORTED_SEARCH_STATUSES:
        raise ValueError(f"Unsupported search status: {status}")

    if exact_evaluations_used > exact_evaluation_budget:
        raise RuntimeError(
            "Exact-evaluation budget was exceeded."
        )

    rejected_candidate_count = sum(
        not record.accepted
        for record in candidate_evaluations
    )

    return SparseSearchResult(
        status=status,
        fidelity_threshold=fidelity_threshold,
        exact_evaluation_budget=exact_evaluation_budget,
        initial_mask=initial_mask,
        final_mask=final_mask,
        final_metrics=final_metrics,
        accepted_removals=tuple(accepted_removals),
        candidate_evaluations=tuple(candidate_evaluations),
        exact_evaluations_used=exact_evaluations_used,
        ranking_passes_used=ranking_passes_used,
        candidate_batches_tested=candidate_batches_tested,
        rejected_candidate_count=rejected_candidate_count,
        budget_remaining=(
            exact_evaluation_budget - exact_evaluations_used
        ),
        budget_exhausted=(
            exact_evaluations_used
            >= exact_evaluation_budget
        ),
        locally_single_deletion_minimal=(
            locally_single_deletion_minimal
        ),
        meaningfully_sparse=is_meaningfully_sparse(final_mask),
        stopping_reason=stopping_reason,
        failure_detail=failure_detail,
    )


def _append_incomplete_batch_records(
    pending_candidates: Sequence[_PendingCandidate],
    *,
    iteration: int,
    candidate_batch_index: int,
    output: list[CandidateEvaluation],
    rejection_reason: str,
) -> None:
    for pending in pending_candidates:
        output.append(
            _candidate_record(
                pending,
                iteration=iteration,
                candidate_batch_index=candidate_batch_index,
                accepted=False,
                rejection_reason=rejection_reason,
            )
        )


def greedy_sparse_search(
    *,
    ranking_function: RankingFunction,
    exact_evaluation_function: ExactEvaluationFunction,
    initial_metrics: MaskEvaluationMetrics,
    fidelity_threshold: float,
    exact_evaluation_budget: int = (
        DEFAULT_EXACT_EVALUATION_BUDGET
    ),
) -> SparseSearchResult:
    """Run the frozen deterministic single-deletion greedy search.

    Ranking controls candidate test order only. Every candidate decision
    uses exact fidelity. Every complete candidate batch is evaluated before
    selecting the highest-fidelity valid deletion in that batch.

    If the budget ends partway through a candidate batch, no candidate from
    that incomplete batch is accepted because the required within-batch
    comparison is incomplete.
    """

    if not callable(ranking_function):
        raise TypeError("ranking_function must be callable.")

    if not callable(exact_evaluation_function):
        raise TypeError(
            "exact_evaluation_function must be callable."
        )

    threshold = _validate_fidelity_threshold(
        fidelity_threshold
    )
    budget = _validate_exact_evaluation_budget(
        exact_evaluation_budget
    )

    initial_mask = ComponentMask.all_retained()
    current_mask = initial_mask

    try:
        _validate_exact_metrics(initial_mask, initial_metrics)
    except (TypeError, ValueError) as exc:
        return _build_search_result(
            status="invalid_masking_output",
            fidelity_threshold=threshold,
            exact_evaluation_budget=budget,
            initial_mask=initial_mask,
            final_mask=current_mask,
            final_metrics=initial_metrics,
            accepted_removals=[],
            candidate_evaluations=[],
            exact_evaluations_used=0,
            ranking_passes_used=0,
            candidate_batches_tested=0,
            locally_single_deletion_minimal=False,
            stopping_reason="invalid_all_retained_metrics",
            failure_detail=str(exc),
        )

    current_metrics = initial_metrics
    accepted_removals: list[AcceptedRemoval] = []
    candidate_evaluations: list[CandidateEvaluation] = []

    exact_evaluations_used = 0
    ranking_passes_used = 0
    candidate_batches_tested = 0

    if current_metrics.primary_fidelity < threshold:
        return _build_search_result(
            status="fidelity_failure",
            fidelity_threshold=threshold,
            exact_evaluation_budget=budget,
            initial_mask=initial_mask,
            final_mask=current_mask,
            final_metrics=current_metrics,
            accepted_removals=accepted_removals,
            candidate_evaluations=candidate_evaluations,
            exact_evaluations_used=exact_evaluations_used,
            ranking_passes_used=ranking_passes_used,
            candidate_batches_tested=candidate_batches_tested,
            locally_single_deletion_minimal=False,
            stopping_reason=(
                "all_retained_mask_below_active_threshold"
            ),
        )

    while True:
        if exact_evaluations_used >= budget:
            return _build_search_result(
                status="budget_exhaustion",
                fidelity_threshold=threshold,
                exact_evaluation_budget=budget,
                initial_mask=initial_mask,
                final_mask=current_mask,
                final_metrics=current_metrics,
                accepted_removals=accepted_removals,
                candidate_evaluations=candidate_evaluations,
                exact_evaluations_used=exact_evaluations_used,
                ranking_passes_used=ranking_passes_used,
                candidate_batches_tested=(
                    candidate_batches_tested
                ),
                locally_single_deletion_minimal=False,
                stopping_reason=(
                    "budget_exhausted_before_next_ranking"
                ),
            )

        ranking_passes_used += 1

        try:
            ranking_result = ranking_function(current_mask)
            _validate_complete_ranking(
                current_mask,
                ranking_result,
            )
        except Exception as exc:
            return _build_search_result(
                status="ranking_failure",
                fidelity_threshold=threshold,
                exact_evaluation_budget=budget,
                initial_mask=initial_mask,
                final_mask=current_mask,
                final_metrics=current_metrics,
                accepted_removals=accepted_removals,
                candidate_evaluations=candidate_evaluations,
                exact_evaluations_used=exact_evaluations_used,
                ranking_passes_used=ranking_passes_used,
                candidate_batches_tested=(
                    candidate_batches_tested
                ),
                locally_single_deletion_minimal=False,
                stopping_reason="ranking_failed",
                failure_detail=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        iteration = len(accepted_removals) + 1
        candidates_tested_in_iteration = 0
        tested_component_ids: set[str] = set()
        accepted_in_iteration = False

        candidate_batches = partition_ranked_candidates(
            ranking_result.ranked_components
        )

        for batch_index, candidate_batch in enumerate(
            candidate_batches,
            start=1,
        ):
            candidate_batches_tested += 1
            pending_candidates: list[_PendingCandidate] = []

            for ranking in candidate_batch:
                if exact_evaluations_used >= budget:
                    _append_incomplete_batch_records(
                        pending_candidates,
                        iteration=iteration,
                        candidate_batch_index=batch_index,
                        output=candidate_evaluations,
                        rejection_reason=(
                            "incomplete_candidate_batch_due_to_budget"
                        ),
                    )

                    return _build_search_result(
                        status="budget_exhaustion",
                        fidelity_threshold=threshold,
                        exact_evaluation_budget=budget,
                        initial_mask=initial_mask,
                        final_mask=current_mask,
                        final_metrics=current_metrics,
                        accepted_removals=accepted_removals,
                        candidate_evaluations=(
                            candidate_evaluations
                        ),
                        exact_evaluations_used=(
                            exact_evaluations_used
                        ),
                        ranking_passes_used=ranking_passes_used,
                        candidate_batches_tested=(
                            candidate_batches_tested
                        ),
                        locally_single_deletion_minimal=False,
                        stopping_reason=(
                            "budget_exhausted_inside_candidate_batch"
                        ),
                    )

                candidate_mask = remove_component(
                    current_mask,
                    ranking.component_identifier,
                )

                try:
                    metrics = exact_evaluation_function(
                        candidate_mask
                    )
                    _validate_exact_metrics(
                        candidate_mask,
                        metrics,
                    )
                except Exception as exc:
                    _append_incomplete_batch_records(
                        pending_candidates,
                        iteration=iteration,
                        candidate_batch_index=batch_index,
                        output=candidate_evaluations,
                        rejection_reason=(
                            "candidate_batch_aborted_by_invalid_output"
                        ),
                    )

                    return _build_search_result(
                        status="invalid_masking_output",
                        fidelity_threshold=threshold,
                        exact_evaluation_budget=budget,
                        initial_mask=initial_mask,
                        final_mask=current_mask,
                        final_metrics=current_metrics,
                        accepted_removals=accepted_removals,
                        candidate_evaluations=(
                            candidate_evaluations
                        ),
                        exact_evaluations_used=(
                            exact_evaluations_used
                        ),
                        ranking_passes_used=ranking_passes_used,
                        candidate_batches_tested=(
                            candidate_batches_tested
                        ),
                        locally_single_deletion_minimal=False,
                        stopping_reason=(
                            "exact_candidate_evaluation_failed"
                        ),
                        failure_detail=(
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )

                exact_evaluations_used += 1
                candidates_tested_in_iteration += 1
                tested_component_ids.add(
                    ranking.component_identifier
                )

                pending_candidates.append(
                    _PendingCandidate(
                        ranking=ranking,
                        candidate_mask=candidate_mask,
                        metrics=metrics,
                        passed_threshold=(
                            metrics.primary_fidelity >= threshold
                        ),
                        cumulative_exact_evaluations=(
                            exact_evaluations_used
                        ),
                    )
                )

            valid_candidates = [
                pending
                for pending in pending_candidates
                if pending.passed_threshold
            ]

            if not valid_candidates:
                for pending in pending_candidates:
                    candidate_evaluations.append(
                        _candidate_record(
                            pending,
                            iteration=iteration,
                            candidate_batch_index=batch_index,
                            accepted=False,
                            rejection_reason=(
                                "below_fidelity_threshold"
                            ),
                        )
                    )
                continue

            selected = max(
                valid_candidates,
                key=lambda pending: (
                    pending.metrics.primary_fidelity,
                    -pending.ranking.component_index,
                ),
            )

            for pending in pending_candidates:
                if pending is selected:
                    candidate_evaluations.append(
                        _candidate_record(
                            pending,
                            iteration=iteration,
                            candidate_batch_index=batch_index,
                            accepted=True,
                            rejection_reason=None,
                        )
                    )
                    continue

                if not pending.passed_threshold:
                    reason = "below_fidelity_threshold"
                elif (
                    pending.metrics.primary_fidelity
                    < selected.metrics.primary_fidelity
                ):
                    reason = (
                        "lower_exact_fidelity_within_first_valid_batch"
                    )
                else:
                    reason = (
                        "exact_fidelity_tie_broken_by_component_index"
                    )

                candidate_evaluations.append(
                    _candidate_record(
                        pending,
                        iteration=iteration,
                        candidate_batch_index=batch_index,
                        accepted=False,
                        rejection_reason=reason,
                    )
                )

            retained_before = current_mask.retained_component_count
            current_mask = selected.candidate_mask
            current_metrics = selected.metrics

            accepted_removals.append(
                AcceptedRemoval(
                    iteration=iteration,
                    removed_component=(
                        selected.ranking.component_identifier
                    ),
                    component_index=(
                        selected.ranking.component_index
                    ),
                    component_class=(
                        selected.ranking.component_class
                    ),
                    retained_count_before=retained_before,
                    retained_count_after=(
                        current_mask.retained_component_count
                    ),
                    exact_fidelity_after_removal=(
                        current_metrics.primary_fidelity
                    ),
                    ranking_score=(
                        selected.ranking.estimated_removal_damage
                    ),
                    ranking_position=(
                        selected.ranking.ranking_position
                    ),
                    candidate_batch_index=batch_index,
                    candidates_exactly_tested_in_iteration=(
                        candidates_tested_in_iteration
                    ),
                    cumulative_exact_evaluations=(
                        exact_evaluations_used
                    ),
                    cumulative_ranking_passes=(
                        ranking_passes_used
                    ),
                    accepted_mask=current_mask,
                    metrics=current_metrics,
                )
            )

            accepted_in_iteration = True
            break

        if accepted_in_iteration:
            continue

        expected_tested = set(current_mask.retained_component_ids)

        if tested_component_ids != expected_tested:
            return _build_search_result(
                status="ranking_failure",
                fidelity_threshold=threshold,
                exact_evaluation_budget=budget,
                initial_mask=initial_mask,
                final_mask=current_mask,
                final_metrics=current_metrics,
                accepted_removals=accepted_removals,
                candidate_evaluations=candidate_evaluations,
                exact_evaluations_used=exact_evaluations_used,
                ranking_passes_used=ranking_passes_used,
                candidate_batches_tested=(
                    candidate_batches_tested
                ),
                locally_single_deletion_minimal=False,
                stopping_reason=(
                    "terminal_deletion_check_incomplete"
                ),
                failure_detail=(
                    "Not every retained component was exactly tested "
                    "under the terminal mask."
                ),
            )

        if not accepted_removals:
            terminal_status = (
                "no_feasible_sparse_candidate_discovered_within_budget"
            )
            stopping_reason = (
                "all_single_deletions_from_all_retained_failed"
            )
        elif is_meaningfully_sparse(current_mask):
            terminal_status = "valid_sparse_circuit"
            stopping_reason = (
                "locally_single_deletion_minimal_sparse_mask"
            )
        else:
            terminal_status = (
                "valid_but_not_meaningfully_sparse"
            )
            stopping_reason = (
                "locally_single_deletion_minimal_nonsparse_mask"
            )

        return _build_search_result(
            status=terminal_status,
            fidelity_threshold=threshold,
            exact_evaluation_budget=budget,
            initial_mask=initial_mask,
            final_mask=current_mask,
            final_metrics=current_metrics,
            accepted_removals=accepted_removals,
            candidate_evaluations=candidate_evaluations,
            exact_evaluations_used=exact_evaluations_used,
            ranking_passes_used=ranking_passes_used,
            candidate_batches_tested=candidate_batches_tested,
            locally_single_deletion_minimal=True,
            stopping_reason=stopping_reason,
        )

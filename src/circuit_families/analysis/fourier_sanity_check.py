"""Deterministic Fourier diagnostics for the Stage 10 pipeline check."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_IDS,
    MLP_NEURON_IDS,
    SEARCHABLE_COMPONENT_IDS,
    ComponentMask,
    component_location,
)
from circuit_families.interpretability.sparse_search import remove_component
from circuit_families.training import file_sha256

MODULUS = 113
OUTPUT_CLASS_COUNT = 113
CANONICAL_FREQUENCIES = tuple(range(1, 57))
STABLE_POST_CHECKPOINT_STEP = 9050
STABLE_POST_THRESHOLDS = (
    0.99,
    0.975,
    0.95,
    0.90,
    0.85,
    0.80,
)

FOURIER_NORMALIZATION = "ortho"
FOURIER_COMPUTATION_DTYPE = "complex128"
SCIENTIFIC_REAL_DTYPE = "float64"

POWER_ABSOLUTE_TOLERANCE = 1.0e-8
NORMALIZED_COMPARISON_TOLERANCE = 1.0e-10
NEAR_CONSTANT_VARIANCE_FLOOR = 1.0e-12


@dataclass(frozen=True)
class ShiftedRelationDiagnostics:
    """Power measurements over all deterministic shifted relations."""

    power_by_shift: tuple[float, ...]
    correct_shift_rank: int
    correct_to_incorrect_mean_ratio: float
    correct_to_largest_incorrect_ratio: float
    correct_family_fraction: float


@dataclass(frozen=True)
class FourierTensorDiagnostics:
    """Fourier summary of one centred [a, b, c] logit tensor."""

    total_power: float
    addition_manifold_power: float
    addition_manifold_fraction: float
    frequency_power: tuple[float, ...]
    canonical_pair_power: tuple[float, ...]
    normalized_canonical_pair_power: tuple[float, ...]
    shifted_relations: ShiftedRelationDiagnostics


@dataclass(frozen=True)
class WeightSpectrumDiagnostics:
    """One-dimensional token/class spectrum."""

    canonical_pair_power: tuple[float, ...]
    normalized_canonical_pair_power: tuple[float, ...]
    ranked_frequency_pairs: tuple[int, ...]


@dataclass(frozen=True)
class ActivationSpectrumDiagnostics:
    """Two-dimensional activation spectrum for one MLP neuron."""

    total_non_dc_power: float
    diagonal_power: float
    diagonal_power_fraction: float
    canonical_pair_power: tuple[float, ...]
    normalized_canonical_pair_power: tuple[float, ...]
    dominant_frequency_pair: int | None
    activation_variance: float
    activation_mean: float
    near_constant: bool


@dataclass(frozen=True)
class Stage9CircuitRecord:
    """One fully verified stable-post Stage 9 circuit."""

    fidelity_threshold: float
    checkpoint_step: int
    checkpoint_sha256: str
    retained_heads: int
    retained_neurons: int
    retained_components: int
    exact_fidelity: float
    final_mask_path: str
    final_mask_sha256: str
    cell_summary_path: str
    cell_summary_sha256: str
    mask: ComponentMask


def centre_logits(logits: np.ndarray) -> np.ndarray:
    """Centre one [a, b, c] tensor independently over output class."""

    values = np.asarray(logits, dtype=np.float64)

    if values.shape != (MODULUS, MODULUS, OUTPUT_CLASS_COUNT):
        raise ValueError(
            "logits must have shape (113, 113, 113)."
        )

    if not np.isfinite(values).all():
        raise FloatingPointError("logits must all be finite.")

    return values - values.mean(axis=2, keepdims=True)


def reshape_lexicographic_logits(
    inputs: torch.Tensor,
    final_logits: torch.Tensor,
) -> np.ndarray:
    """Validate lexicographic operands and reshape logits to [a, b, c]."""

    if inputs.ndim != 2:
        raise ValueError("inputs must be two-dimensional.")

    if final_logits.shape != (MODULUS * MODULUS, OUTPUT_CLASS_COUNT):
        raise ValueError(
            "final_logits must have shape (12769, 113)."
        )

    left = inputs[:, 0].detach().cpu()
    right = inputs[:, 1].detach().cpu()

    expected_left = torch.arange(MODULUS).repeat_interleave(MODULUS)
    expected_right = torch.arange(MODULUS).repeat(MODULUS)

    if not torch.equal(left, expected_left):
        raise ValueError(
            "First operands are not in frozen lexicographic order."
        )

    if not torch.equal(right, expected_right):
        raise ValueError(
            "Second operands are not in frozen lexicographic order."
        )

    values = (
        final_logits.detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )
    return values.reshape(MODULUS, MODULUS, OUTPUT_CLASS_COUNT)


def modular_addition_indices(
    *,
    shift: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices for Md={(k,k,-k+d mod p): k=1,...,p-1}."""

    if isinstance(shift, bool) or not isinstance(shift, int):
        raise TypeError("shift must be an integer.")

    shift %= MODULUS
    k = np.arange(1, MODULUS, dtype=np.int64)
    return k, k, (-k + shift) % MODULUS


def canonical_pair_power(
    frequency_power: Sequence[float],
) -> tuple[float, ...]:
    """Pair k and p-k exactly once for canonical k=1,...,56."""

    values = np.asarray(frequency_power, dtype=np.float64)

    if values.shape != (MODULUS,):
        raise ValueError(
            "frequency_power must contain exactly 113 values."
        )

    return tuple(
        float(values[k] + values[MODULUS - k])
        for k in CANONICAL_FREQUENCIES
    )


def normalize_nonnegative(
    values: Sequence[float],
) -> tuple[float, ...]:
    """Normalize a finite nonnegative vector, preserving an all-zero vector."""

    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty vector.")

    if not np.isfinite(array).all():
        raise FloatingPointError("values must be finite.")

    if np.any(array < -POWER_ABSOLUTE_TOLERANCE):
        raise ValueError("values must be nonnegative.")

    array = np.maximum(array, 0.0)
    total = float(array.sum())

    if total <= POWER_ABSOLUTE_TOLERANCE:
        return tuple(0.0 for _ in array)

    return tuple(float(value / total) for value in array)


def rank_descending_stable(
    values: Sequence[float],
) -> tuple[int, ...]:
    """Return zero-based descending ranks with lower index breaking ties."""

    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    if not np.isfinite(array).all():
        raise FloatingPointError("values must be finite.")

    return tuple(
        sorted(
            range(array.size),
            key=lambda index: (-float(array[index]), index),
        )
    )


def shifted_relation_diagnostics(
    power: np.ndarray,
) -> ShiftedRelationDiagnostics:
    """Measure every shifted frequency-index relation Md."""

    values = np.asarray(power, dtype=np.float64)

    if values.shape != (MODULUS, MODULUS, MODULUS):
        raise ValueError("power must have shape (113, 113, 113).")

    relation_powers = []

    for shift in range(MODULUS):
        indices = modular_addition_indices(shift=shift)
        relation_powers.append(float(values[indices].sum()))

    ranking = rank_descending_stable(relation_powers)
    correct_rank = ranking.index(0) + 1

    correct = relation_powers[0]
    incorrect = relation_powers[1:]
    incorrect_mean = float(np.mean(incorrect))
    incorrect_max = float(np.max(incorrect))
    family_total = float(np.sum(relation_powers))

    return ShiftedRelationDiagnostics(
        power_by_shift=tuple(relation_powers),
        correct_shift_rank=correct_rank,
        correct_to_incorrect_mean_ratio=(
            math.inf
            if incorrect_mean <= POWER_ABSOLUTE_TOLERANCE
            and correct > POWER_ABSOLUTE_TOLERANCE
            else (
                1.0
                if incorrect_mean <= POWER_ABSOLUTE_TOLERANCE
                else correct / incorrect_mean
            )
        ),
        correct_to_largest_incorrect_ratio=(
            math.inf
            if incorrect_max <= POWER_ABSOLUTE_TOLERANCE
            and correct > POWER_ABSOLUTE_TOLERANCE
            else (
                1.0
                if incorrect_max <= POWER_ABSOLUTE_TOLERANCE
                else correct / incorrect_max
            )
        ),
        correct_family_fraction=(
            0.0
            if family_total <= POWER_ABSOLUTE_TOLERANCE
            else correct / family_total
        ),
    )


def analyse_logit_tensor(
    logits: np.ndarray,
) -> FourierTensorDiagnostics:
    """Analyse one complete real final-logit tensor."""

    centred = centre_logits(logits)
    spectrum = np.fft.fftn(
        centred,
        axes=(0, 1, 2),
        norm=FOURIER_NORMALIZATION,
    )
    power = np.abs(spectrum) ** 2

    total_power = float(power.sum())
    manifold_indices = modular_addition_indices()
    frequency_power = np.zeros(MODULUS, dtype=np.float64)
    frequency_power[1:] = power[manifold_indices]

    manifold_power = float(frequency_power.sum())
    pair_power = canonical_pair_power(frequency_power)

    return FourierTensorDiagnostics(
        total_power=total_power,
        addition_manifold_power=manifold_power,
        addition_manifold_fraction=(
            0.0
            if total_power <= POWER_ABSOLUTE_TOLERANCE
            else manifold_power / total_power
        ),
        frequency_power=tuple(float(value) for value in frequency_power),
        canonical_pair_power=pair_power,
        normalized_canonical_pair_power=normalize_nonnegative(pair_power),
        shifted_relations=shifted_relation_diagnostics(power),
    )


def analyse_weight_spectrum(
    matrix: np.ndarray,
) -> WeightSpectrumDiagnostics:
    """Analyse token/class-index structure of a [113, feature] matrix."""

    values = np.asarray(matrix, dtype=np.float64)

    if values.ndim != 2 or values.shape[0] != MODULUS:
        raise ValueError("matrix must have shape (113, feature).")

    centred = values - values.mean(axis=0, keepdims=True)
    spectrum = np.fft.fft(
        centred,
        axis=0,
        norm=FOURIER_NORMALIZATION,
    )
    frequency_power = (np.abs(spectrum) ** 2).sum(axis=1)
    pair_power = canonical_pair_power(frequency_power)
    normalized = normalize_nonnegative(pair_power)
    ranking = tuple(
        index + 1
        for index in rank_descending_stable(pair_power)
    )

    return WeightSpectrumDiagnostics(
        canonical_pair_power=pair_power,
        normalized_canonical_pair_power=normalized,
        ranked_frequency_pairs=ranking,
    )


def analyse_activation_matrix(
    activation: np.ndarray,
    *,
    variance_floor: float = NEAR_CONSTANT_VARIANCE_FLOOR,
) -> ActivationSpectrumDiagnostics:
    """Analyse one [a,b] final-position post-ReLU activation."""

    values = np.asarray(activation, dtype=np.float64)

    if values.shape != (MODULUS, MODULUS):
        raise ValueError("activation must have shape (113, 113).")

    if not np.isfinite(values).all():
        raise FloatingPointError("activation must be finite.")

    mean = float(values.mean())
    variance = float(values.var())
    near_constant = variance <= variance_floor

    centred = values - mean
    spectrum = np.fft.fft2(
        centred,
        norm=FOURIER_NORMALIZATION,
    )
    power = np.abs(spectrum) ** 2
    power[0, 0] = 0.0

    total_non_dc = float(power.sum())
    frequency_power = np.zeros(MODULUS, dtype=np.float64)
    k = np.arange(1, MODULUS, dtype=np.int64)
    frequency_power[1:] = power[k, k]

    diagonal_power = float(frequency_power.sum())
    pair_power = canonical_pair_power(frequency_power)
    normalized = normalize_nonnegative(pair_power)

    dominant = None
    if not near_constant and total_non_dc > POWER_ABSOLUTE_TOLERANCE:
        dominant = rank_descending_stable(pair_power)[0] + 1

    return ActivationSpectrumDiagnostics(
        total_non_dc_power=total_non_dc,
        diagonal_power=diagonal_power,
        diagonal_power_fraction=(
            0.0
            if total_non_dc <= POWER_ABSOLUTE_TOLERANCE
            else diagonal_power / total_non_dc
        ),
        canonical_pair_power=pair_power,
        normalized_canonical_pair_power=normalized,
        dominant_frequency_pair=dominant,
        activation_variance=variance,
        activation_mean=mean,
        near_constant=near_constant,
    )


def cosine_similarity(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Return cosine similarity or NaN when either vector has zero norm."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)

    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("vectors must be one-dimensional and equal length.")

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))

    if (
        left_norm <= NORMALIZED_COMPARISON_TOLERANCE
        or right_norm <= NORMALIZED_COMPARISON_TOLERANCE
    ):
        return math.nan

    return float(np.dot(left, right) / (left_norm * right_norm))


def synthetic_relation_tensor(
    *,
    frequency: int,
    shift: int = 0,
    use_sine: bool = False,
) -> np.ndarray:
    """Construct a tensor on the declared shifted index relation.

    This uses frequency indices (k, k, -k+d). Adding a constant phase
    offset to cos(k(a+b-c)+phase) would not change the Fourier indices and
    therefore would not test Md for d != 0.
    """

    if frequency <= 0 or frequency >= MODULUS:
        raise ValueError("frequency must be between 1 and 112.")

    a, b, c = np.meshgrid(
        np.arange(MODULUS),
        np.arange(MODULUS),
        np.arange(MODULUS),
        indexing="ij",
    )
    output_frequency = (-frequency + shift) % MODULUS
    phase = (
        2.0
        * np.pi
        * (
            frequency * a
            + frequency * b
            + output_frequency * c
        )
        / MODULUS
    )
    return np.sin(phase) if use_sine else np.cos(phase)


def _safe_archive_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Unsafe Stage 9 archive member path: {name}"
        )

    return path


def _archive_member_for_loose_path(path: str) -> str:
    prefix = "results/raw/"

    if not path.startswith(prefix):
        raise ValueError(
            "Stage 9 raw artifact path must begin with results/raw/."
        )

    member = path[len(prefix):]
    _safe_archive_member_name(member)
    return member


def _read_verified_archive_member(
    archive: tarfile.TarFile,
    *,
    member_name: str,
    expected_sha256: str,
) -> bytes:
    _safe_archive_member_name(member_name)

    try:
        member = archive.getmember(member_name)
    except KeyError as exc:
        raise FileNotFoundError(
            f"Stage 9 archive member is missing: {member_name}"
        ) from exc

    if not member.isfile():
        raise ValueError(
            f"Stage 9 archive member is not a regular file: {member_name}"
        )

    stream = archive.extractfile(member)

    if stream is None:
        raise RuntimeError(
            f"Could not read Stage 9 archive member: {member_name}"
        )

    payload = stream.read()
    actual_sha256 = hashlib.sha256(payload).hexdigest()

    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Stage 9 archive member hash mismatch: {member_name}"
        )

    return payload


def load_stable_post_stage9_circuits(
    *,
    stage9_manifest_path: str | Path,
    stage9_table_path: str | Path,
    stage9_archive_path: str | Path,
) -> tuple[Stage9CircuitRecord, ...]:
    """Load and cross-validate all six stable-post circuits from the archive."""

    manifest_path = Path(stage9_manifest_path)
    table_path = Path(stage9_table_path)
    archive_path = Path(stage9_archive_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("primary_fidelity_threshold_selected") is not False:
        raise ValueError(
            "Stage 9 manifest unexpectedly selected a primary threshold."
        )

    expected_archive = manifest["outputs"]["raw_artifact_archive"]

    if file_sha256(archive_path) != expected_archive["sha256"]:
        raise ValueError("Stage 9 archive SHA-256 mismatch.")

    if file_sha256(table_path) != manifest["outputs"][
        "deterministic_result_table"
    ]["sha256"]:
        raise ValueError("Stage 9 result-table SHA-256 mismatch.")

    with table_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["phase"] == "stable post-grokking"
        ]

    if len(rows) != len(STABLE_POST_THRESHOLDS):
        raise ValueError(
            "Expected exactly six stable-post Stage 9 rows."
        )

    by_threshold: dict[float, dict[str, str]] = {}

    for row in rows:
        threshold = float(row["fidelity_threshold"])

        if threshold in by_threshold:
            raise ValueError(
                f"Duplicate stable-post threshold: {threshold}"
            )

        by_threshold[threshold] = row

    if set(by_threshold) != set(STABLE_POST_THRESHOLDS):
        raise ValueError(
            "Stable-post Stage 9 threshold set changed."
        )

    manifest_cells = {
        (
            int(cell["checkpoint_step"]),
            float(cell["fidelity_threshold"]),
        ): cell
        for cell in manifest["outputs"]["cells"]
    }

    records = []

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            _safe_archive_member_name(member.name)

        for threshold in STABLE_POST_THRESHOLDS:
            row = by_threshold[threshold]

            if int(row["checkpoint_step"]) != STABLE_POST_CHECKPOINT_STEP:
                raise ValueError(
                    "Stable-post circuit checkpoint step changed."
                )

            cell_key = (STABLE_POST_CHECKPOINT_STEP, threshold)

            if cell_key not in manifest_cells:
                raise ValueError(
                    f"Stage 9 manifest cell missing for {threshold}."
                )

            cell = manifest_cells[cell_key]

            for field, nested in (
                ("final_mask_path", cell["final_mask"]),
                ("cell_summary_path", cell["cell_summary"]),
            ):
                if row[field] != nested["path"]:
                    raise ValueError(
                        f"Stage 9 table/manifest path mismatch: {field}"
                    )

            mask_member = _archive_member_for_loose_path(
                row["final_mask_path"]
            )
            summary_member = _archive_member_for_loose_path(
                row["cell_summary_path"]
            )

            mask_payload = _read_verified_archive_member(
                archive,
                member_name=mask_member,
                expected_sha256=row["final_mask_sha256"],
            )
            summary_payload = _read_verified_archive_member(
                archive,
                member_name=summary_member,
                expected_sha256=row["cell_summary_sha256"],
            )

            mask_record = json.load(io.BytesIO(mask_payload))
            summary = json.load(io.BytesIO(summary_payload))
            mask = ComponentMask.from_record(mask_record)

            if mask.retained_component_count != int(
                row["total_retained_components"]
            ):
                raise ValueError(
                    "Stage 9 retained-component count mismatch."
                )

            if mask.retained_attention_head_count != int(
                row["retained_heads"]
            ):
                raise ValueError(
                    "Stage 9 retained-head count mismatch."
                )

            if mask.retained_mlp_neuron_count != int(
                row["retained_neurons"]
            ):
                raise ValueError(
                    "Stage 9 retained-neuron count mismatch."
                )

            metrics = summary["final_metrics"]

            if not math.isclose(
                float(metrics["primary_fidelity"]),
                float(row["final_exact_fidelity"]),
                abs_tol=0.0,
                rel_tol=0.0,
            ):
                raise ValueError(
                    "Stage 9 exact-fidelity mismatch."
                )

            if summary["cell_metadata"]["checkpoint_step"] != (
                STABLE_POST_CHECKPOINT_STEP
            ):
                raise ValueError(
                    "Stage 9 cell-summary checkpoint mismatch."
                )

            if float(
                summary["search"]["fidelity_threshold"]
            ) != threshold:
                raise ValueError(
                    "Stage 9 cell-summary threshold mismatch."
                )

            records.append(
                Stage9CircuitRecord(
                    fidelity_threshold=threshold,
                    checkpoint_step=STABLE_POST_CHECKPOINT_STEP,
                    checkpoint_sha256=row["checkpoint_sha256"],
                    retained_heads=mask.retained_attention_head_count,
                    retained_neurons=mask.retained_mlp_neuron_count,
                    retained_components=mask.retained_component_count,
                    exact_fidelity=float(row["final_exact_fidelity"]),
                    final_mask_path=row["final_mask_path"],
                    final_mask_sha256=row["final_mask_sha256"],
                    cell_summary_path=row["cell_summary_path"],
                    cell_summary_sha256=row["cell_summary_sha256"],
                    mask=mask,
                )
            )

    return tuple(records)


def one_component_ablation_mask(
    identifier: str,
) -> ComponentMask:
    """Return the frozen exact single-component-ablation mask."""

    if identifier in ATTENTION_HEAD_IDS:
        return ComponentMask.one_head_ablated(identifier)

    if identifier in MLP_NEURON_IDS:
        return ComponentMask.one_neuron_ablated(identifier)

    raise ValueError(f"Unknown component identifier: {identifier}")


def retained_flags(
    circuits: Sequence[Stage9CircuitRecord],
) -> dict[str, tuple[bool, ...]]:
    """Return per-component retention flags in frozen threshold order."""

    return {
        identifier: tuple(
            identifier in circuit.mask.retained_component_ids
            for circuit in circuits
        )
        for identifier in SEARCHABLE_COMPONENT_IDS
    }


@dataclass(frozen=True)
class CollectedModelOutputs:
    """Complete final-position outputs collected in frozen example order."""

    final_logits: torch.Tensor
    predictions: torch.Tensor
    source_dtype: str
    evaluated_example_count: int
    evaluation_batch_size: int


@dataclass(frozen=True)
class BehaviourMetrics:
    """Stage 8-compatible behavioural comparison from collected logits."""

    primary_fidelity: float
    prediction_agreement_count: int
    prediction_disagreement_count: int
    full_accuracy: float
    evaluated_accuracy: float
    accuracy_change: float
    full_cross_entropy: float
    evaluated_cross_entropy: float
    cross_entropy_change: float
    mean_kl_divergence: float
    mean_jensen_shannon_divergence: float
    maximum_absolute_logit_difference: float
    evaluated_example_count: int


@dataclass(frozen=True)
class ComponentAssociationRecord:
    """Exact causal Fourier-association result for one component."""

    component_identifier: str
    component_type: str
    component_index: int
    primary_fidelity: float
    prediction_agreement_count: int
    prediction_disagreement_count: int
    ground_truth_accuracy_change: float
    cross_entropy_change: float
    mean_kl_divergence: float
    mean_jensen_shannon_divergence: float
    maximum_absolute_logit_change: float
    total_delta_fourier_power: float
    addition_manifold_delta_power: float
    addition_manifold_delta_fraction: float
    correct_shift_rank: int
    correct_shift_selectivity: float
    dominant_canonical_frequency_pair: int | None
    activation_diagonal_power_fraction: float | None
    activation_near_constant: bool | None
    retained_flags: tuple[bool, ...]


@dataclass(frozen=True)
class ComponentAssociationExecution:
    """Complete component-association results and integrity evidence."""

    records: tuple[ComponentAssociationRecord, ...]
    model_state_sha256_before: str
    model_state_sha256_after: str
    hook_counts_before: tuple[tuple[str, int], ...]
    hook_counts_after: tuple[tuple[str, int], ...]
    gradients_absent_after: bool


def _validate_positive_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch size must be an integer.")

    if batch_size <= 0:
        raise ValueError("batch size must be positive.")

    return batch_size


def _analysis_hook_counts(model: Any) -> tuple[tuple[str, int], ...]:
    """Return stable forward-hook counts for Stage 10 hook locations."""

    from circuit_families.interpretability.masks import (
        ATTENTION_HEAD_HOOK_NAME,
        MLP_NEURON_HOOK_NAME,
    )

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


def collect_final_position_outputs(
    model: Any,
    inputs: torch.Tensor,
    *,
    batch_size: int,
    mask: ComponentMask | None = None,
) -> CollectedModelOutputs:
    """Collect final-position logits sequentially without retaining graphs."""

    from circuit_families.interpretability.component_ablation import (
        masked_model_logits,
        validate_mask_model,
    )
    from circuit_families.training.metrics import final_position_logits

    validate_mask_model(model)
    batch_size = _validate_positive_batch_size(batch_size)

    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a tensor.")

    if inputs.ndim != 2 or inputs.shape[0] == 0:
        raise ValueError(
            "inputs must have shape (example, sequence_position)."
        )

    if inputs.dtype != torch.long:
        raise TypeError("inputs must have dtype torch.long.")

    batches: list[torch.Tensor] = []
    was_training = model.training
    model.eval()

    try:
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])
            batch_inputs = inputs[start:stop]

            with torch.inference_mode():
                if mask is None:
                    sequence_logits = model(batch_inputs)
                else:
                    sequence_logits = masked_model_logits(
                        model,
                        batch_inputs,
                        mask,
                    )

            batch_final = (
                final_position_logits(sequence_logits)
                .detach()
                .clone()
            )
            batches.append(batch_final)
    finally:
        model.train(was_training)

    logits = torch.cat(batches, dim=0)

    if logits.shape != (inputs.shape[0], OUTPUT_CLASS_COUNT):
        raise ValueError(
            "Collected logits must have shape (example, 113)."
        )

    if logits.requires_grad:
        raise RuntimeError("Collected logits must be detached.")

    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("Collected logits must be finite.")

    return CollectedModelOutputs(
        final_logits=logits,
        predictions=logits.argmax(dim=-1),
        source_dtype=str(logits.dtype).replace("torch.", ""),
        evaluated_example_count=int(inputs.shape[0]),
        evaluation_batch_size=batch_size,
    )


def collect_final_position_mlp_activations(
    model: Any,
    inputs: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Collect [example, neuron] post-ReLU final-position activations."""

    from circuit_families.interpretability.component_ablation import (
        validate_mask_model,
    )
    from circuit_families.interpretability.masks import (
        MLP_NEURON_COUNT,
        MLP_NEURON_HOOK_NAME,
    )

    validate_mask_model(model)
    batch_size = _validate_positive_batch_size(batch_size)

    if inputs.ndim != 2 or inputs.dtype != torch.long:
        raise ValueError(
            "inputs must be a two-dimensional torch.long tensor."
        )

    collected: list[torch.Tensor] = []
    was_training = model.training
    model.eval()

    def capture_hook(
        activation: torch.Tensor,
        hook: Any,
    ) -> torch.Tensor:
        if hook.name != MLP_NEURON_HOOK_NAME:
            raise RuntimeError("MLP activation hook name changed.")

        if (
            activation.ndim != 3
            or activation.shape[2] != MLP_NEURON_COUNT
        ):
            raise ValueError(
                "MLP activation must have shape "
                "(batch, position, 512)."
            )

        collected.append(
            activation[:, -1, :]
            .detach()
            .clone()
        )
        return activation

    try:
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])

            with model.hooks(
                fwd_hooks=[
                    (MLP_NEURON_HOOK_NAME, capture_hook),
                ]
            ):
                with torch.inference_mode():
                    model(inputs[start:stop])
    finally:
        model.train(was_training)

    if not collected:
        raise RuntimeError("No MLP activations were captured.")

    activations = torch.cat(collected, dim=0)

    if activations.shape != (
        inputs.shape[0],
        MLP_NEURON_COUNT,
    ):
        raise ValueError(
            "Collected MLP activations must have shape "
            "(example, 512)."
        )

    if activations.requires_grad:
        raise RuntimeError("Collected activations must be detached.")

    if not bool(torch.isfinite(activations).all().item()):
        raise FloatingPointError(
            "Collected activations must be finite."
        )

    return activations


def behaviour_metrics_from_logits(
    full_logits: torch.Tensor,
    evaluated_logits: torch.Tensor,
    targets: torch.Tensor,
) -> BehaviourMetrics:
    """Calculate Stage 8-compatible metrics from final-position logits."""

    import torch.nn.functional as functional

    if full_logits.shape != evaluated_logits.shape:
        raise ValueError("Full and evaluated logits must have equal shape.")

    if (
        full_logits.ndim != 2
        or full_logits.shape[1] != OUTPUT_CLASS_COUNT
    ):
        raise ValueError("Logits must have shape (example, 113).")

    if targets.shape != (full_logits.shape[0],):
        raise ValueError("targets must match the logit example count.")

    full_predictions = full_logits.argmax(dim=-1)
    evaluated_predictions = evaluated_logits.argmax(dim=-1)
    agreement_count = int(
        (full_predictions == evaluated_predictions).sum().item()
    )
    example_count = int(full_logits.shape[0])

    full_accuracy = float(
        (full_predictions == targets)
        .to(torch.float64)
        .mean()
        .item()
    )
    evaluated_accuracy = float(
        (evaluated_predictions == targets)
        .to(torch.float64)
        .mean()
        .item()
    )

    full_losses = functional.cross_entropy(
        full_logits,
        targets,
        reduction="none",
    )
    evaluated_losses = functional.cross_entropy(
        evaluated_logits,
        targets,
        reduction="none",
    )
    full_cross_entropy = float(
        full_losses.to(torch.float64).mean().item()
    )
    evaluated_cross_entropy = float(
        evaluated_losses.to(torch.float64).mean().item()
    )

    full_log_probabilities = functional.log_softmax(
        full_logits,
        dim=-1,
    )
    evaluated_log_probabilities = functional.log_softmax(
        evaluated_logits,
        dim=-1,
    )
    full_probabilities = full_log_probabilities.exp()
    evaluated_probabilities = evaluated_log_probabilities.exp()

    kl_values = (
        full_probabilities
        * (
            full_log_probabilities
            - evaluated_log_probabilities
        )
    ).sum(dim=-1)

    log_mixture = torch.logaddexp(
        full_log_probabilities,
        evaluated_log_probabilities,
    ) - math.log(2.0)

    full_to_mixture = (
        full_probabilities
        * (full_log_probabilities - log_mixture)
    ).sum(dim=-1)
    evaluated_to_mixture = (
        evaluated_probabilities
        * (evaluated_log_probabilities - log_mixture)
    ).sum(dim=-1)

    js_values = 0.5 * (
        full_to_mixture + evaluated_to_mixture
    )

    return BehaviourMetrics(
        primary_fidelity=agreement_count / example_count,
        prediction_agreement_count=agreement_count,
        prediction_disagreement_count=(
            example_count - agreement_count
        ),
        full_accuracy=full_accuracy,
        evaluated_accuracy=evaluated_accuracy,
        accuracy_change=evaluated_accuracy - full_accuracy,
        full_cross_entropy=full_cross_entropy,
        evaluated_cross_entropy=evaluated_cross_entropy,
        cross_entropy_change=(
            evaluated_cross_entropy - full_cross_entropy
        ),
        mean_kl_divergence=float(
            kl_values.to(torch.float64).mean().item()
        ),
        mean_jensen_shannon_divergence=float(
            js_values.to(torch.float64).mean().item()
        ),
        maximum_absolute_logit_difference=float(
            (full_logits - evaluated_logits)
            .abs()
            .max()
            .item()
        ),
        evaluated_example_count=example_count,
    )


def activation_diagnostics_for_all_neurons(
    activations: torch.Tensor,
) -> tuple[ActivationSpectrumDiagnostics, ...]:
    """Analyse all 512 activation matrices in frozen neuron order."""

    from circuit_families.interpretability.masks import MLP_NEURON_COUNT

    if activations.shape != (
        MODULUS * MODULUS,
        MLP_NEURON_COUNT,
    ):
        raise ValueError(
            "activations must have shape (12769, 512)."
        )

    values = (
        activations.detach()
        .cpu()
        .to(torch.float64)
        .numpy()
        .reshape(MODULUS, MODULUS, MLP_NEURON_COUNT)
    )

    return tuple(
        analyse_activation_matrix(values[:, :, neuron_index])
        for neuron_index in range(MLP_NEURON_COUNT)
    )


def component_association_ranking(
    records: Sequence[ComponentAssociationRecord],
) -> tuple[ComponentAssociationRecord, ...]:
    """Rank by absolute M0 delta power, then frozen component index."""

    values = tuple(records)

    if len(values) != len(SEARCHABLE_COMPONENT_IDS):
        raise ValueError(
            "Component association requires exactly 516 records."
        )

    if {
        record.component_identifier for record in values
    } != set(SEARCHABLE_COMPONENT_IDS):
        raise ValueError(
            "Component-association identifier set changed."
        )

    return tuple(
        sorted(
            values,
            key=lambda record: (
                -record.addition_manifold_delta_power,
                record.component_index,
            ),
        )
    )


def select_retained_components_for_removal(
    mask: ComponentMask,
    ranking: Sequence[ComponentAssociationRecord],
) -> tuple[tuple[str, str], ...]:
    """Select deterministic relevant and comparison removals."""

    retained = set(mask.retained_component_ids)
    ranked = tuple(
        record
        for record in ranking
        if record.component_identifier in retained
    )

    if not ranked:
        raise ValueError("Circuit retains no searchable components.")

    selections: list[tuple[str, str]] = [
        (
            "highest_associated_retained_overall",
            ranked[0].component_identifier,
        )
    ]

    retained_heads = [
        record
        for record in ranked
        if record.component_type == "attention_head"
    ]

    if retained_heads:
        selections.append(
            (
                "highest_associated_retained_attention_head",
                retained_heads[0].component_identifier,
            )
        )

    retained_neurons = [
        record
        for record in ranked
        if record.component_type == "mlp_neuron"
    ]

    if retained_neurons:
        selections.append(
            (
                "highest_associated_retained_mlp_neuron",
                retained_neurons[0].component_identifier,
            )
        )

    selections.append(
        (
            "lowest_associated_retained_component",
            ranked[-1].component_identifier,
        )
    )

    result: list[tuple[str, str]] = []
    seen: set[str] = set()

    for role, identifier in selections:
        if identifier not in seen:
            result.append((role, identifier))
            seen.add(identifier)

    return tuple(result)


def run_component_association(
    *,
    model: Any,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    full_outputs: CollectedModelOutputs,
    circuits: Sequence[Stage9CircuitRecord],
    activation_diagnostics: Sequence[
        ActivationSpectrumDiagnostics
    ],
    batch_size: int,
) -> ComponentAssociationExecution:
    """Evaluate every exact single-component ablation sequentially."""

    from circuit_families.interpretability.masks import (
        COMPONENT_LOCATIONS,
    )
    from circuit_families.training import canonical_state_hash

    if len(circuits) != len(STABLE_POST_THRESHOLDS):
        raise ValueError("Expected six stable-post circuits.")

    if len(activation_diagnostics) != len(MLP_NEURON_IDS):
        raise ValueError(
            "Expected one activation diagnostic per MLP neuron."
        )

    model_state_before = canonical_state_hash(model.state_dict())
    hook_counts_before = _analysis_hook_counts(model)
    flags = retained_flags(circuits)
    records: list[ComponentAssociationRecord] = []

    for component_index, location in enumerate(COMPONENT_LOCATIONS):
        mask = one_component_ablation_mask(location.identifier)
        ablated_outputs = collect_final_position_outputs(
            model,
            inputs,
            batch_size=batch_size,
            mask=mask,
        )
        metrics = behaviour_metrics_from_logits(
            full_outputs.final_logits,
            ablated_outputs.final_logits,
            targets,
        )

        delta_logits = (
            full_outputs.final_logits
            - ablated_outputs.final_logits
        )
        delta_tensor = reshape_lexicographic_logits(
            inputs,
            delta_logits,
        )
        fourier = analyse_logit_tensor(delta_tensor)

        dominant_frequency = None

        if (
            sum(fourier.canonical_pair_power)
            > POWER_ABSOLUTE_TOLERANCE
        ):
            dominant_frequency = (
                rank_descending_stable(
                    fourier.canonical_pair_power
                )[0]
                + 1
            )

        activation_fraction: float | None = None
        activation_near_constant: bool | None = None

        if location.component_class == "mlp_neuron":
            activation = activation_diagnostics[location.index]
            activation_fraction = (
                activation.diagonal_power_fraction
            )
            activation_near_constant = activation.near_constant

        records.append(
            ComponentAssociationRecord(
                component_identifier=location.identifier,
                component_type=location.component_class,
                component_index=component_index,
                primary_fidelity=metrics.primary_fidelity,
                prediction_agreement_count=(
                    metrics.prediction_agreement_count
                ),
                prediction_disagreement_count=(
                    metrics.prediction_disagreement_count
                ),
                ground_truth_accuracy_change=(
                    metrics.accuracy_change
                ),
                cross_entropy_change=(
                    metrics.cross_entropy_change
                ),
                mean_kl_divergence=(
                    metrics.mean_kl_divergence
                ),
                mean_jensen_shannon_divergence=(
                    metrics.mean_jensen_shannon_divergence
                ),
                maximum_absolute_logit_change=(
                    metrics.maximum_absolute_logit_difference
                ),
                total_delta_fourier_power=(
                    fourier.total_power
                ),
                addition_manifold_delta_power=(
                    fourier.addition_manifold_power
                ),
                addition_manifold_delta_fraction=(
                    fourier.addition_manifold_fraction
                ),
                correct_shift_rank=(
                    fourier.shifted_relations.correct_shift_rank
                ),
                correct_shift_selectivity=(
                    fourier.shifted_relations
                    .correct_to_incorrect_mean_ratio
                ),
                dominant_canonical_frequency_pair=(
                    dominant_frequency
                ),
                activation_diagonal_power_fraction=(
                    activation_fraction
                ),
                activation_near_constant=(
                    activation_near_constant
                ),
                retained_flags=flags[location.identifier],
            )
        )

        del ablated_outputs
        del delta_logits
        del delta_tensor

    model_state_after = canonical_state_hash(model.state_dict())
    hook_counts_after = _analysis_hook_counts(model)
    gradients_absent = all(
        parameter.grad is None
        for parameter in model.parameters()
    )

    if model_state_after != model_state_before:
        raise RuntimeError(
            "Component association changed the model-state hash."
        )

    if hook_counts_after != hook_counts_before:
        raise RuntimeError(
            "Component association leaked TransformerLens hooks."
        )

    if not gradients_absent:
        raise RuntimeError(
            "Component association left parameter gradients populated."
        )

    return ComponentAssociationExecution(
        records=tuple(records),
        model_state_sha256_before=model_state_before,
        model_state_sha256_after=model_state_after,
        hook_counts_before=hook_counts_before,
        hook_counts_after=hook_counts_after,
        gradients_absent_after=gradients_absent,
    )


FOURIER_MATCH_FRACTION_THRESHOLD = 0.5
FOURIER_DEGENERATE_POWER_THRESHOLD = 1.0e-10

FOURIER_CLASSIFICATION_DEFINITIONS = {
    "clear_match": (
        "The correct modular-addition shifted relation ranks first and "
        "contains at least half of the complete non-DC Fourier power."
    ),
    "partial_match": (
        "The correct modular-addition shifted relation ranks first but "
        "contains less than half of the complete non-DC Fourier power."
    ),
    "mismatch": (
        "At least one incorrect shifted frequency relation contains more "
        "power than the correct modular-addition relation."
    ),
    "degenerate": (
        "The analysed delta or output tensor has effectively zero non-DC "
        "Fourier power, so relation selectivity is not interpretable."
    ),
}


@dataclass(frozen=True)
class CircuitFourierRecord:
    """Fourier and behavioural summary for one Stage 9 circuit."""

    fidelity_threshold: float
    mask_id: str
    retained_heads: int
    retained_neurons: int
    retained_components: int
    exact_fidelity: float
    ground_truth_accuracy: float
    cross_entropy: float
    addition_manifold_power: float
    total_fourier_power: float
    addition_manifold_fraction: float
    correct_shift_rank: int
    correct_shift_selectivity: float
    dominant_canonical_frequency_pair: int | None
    diagnostic_classification: str
    retained_association_power: float
    total_association_power: float
    retained_association_fraction: float
    highest_associated_retained_component: str
    lowest_associated_retained_component: str


@dataclass(frozen=True)
class RemovalFourierRecord:
    """Result of removing one selected retained component from a circuit."""

    fidelity_threshold: float
    selection_role: str
    removed_component: str
    removed_component_type: str
    original_retained_components: int
    remaining_retained_components: int
    original_primary_fidelity: float
    removal_primary_fidelity: float
    primary_fidelity_change: float
    original_accuracy: float
    removal_accuracy: float
    accuracy_change: float
    original_cross_entropy: float
    removal_cross_entropy: float
    cross_entropy_change: float
    addition_manifold_power: float
    total_fourier_power: float
    addition_manifold_fraction: float
    correct_shift_rank: int
    correct_shift_selectivity: float
    diagnostic_classification: str


@dataclass(frozen=True)
class Stage10ArtifactPaths:
    """Deterministic Stage 10 table, figure, and manifest paths."""

    component_table: Path
    circuit_table: Path
    removal_table: Path
    embedding_table: Path
    activation_table: Path
    manifest: Path
    output_directory: Path


def classify_fourier_diagnostic(
    diagnostics: FourierTensorDiagnostics,
) -> str:
    """Apply the frozen prospective Stage 10 diagnostic classification."""

    if not isinstance(diagnostics, FourierTensorDiagnostics):
        raise TypeError(
            "diagnostics must be a FourierTensorDiagnostics value."
        )

    if diagnostics.total_power <= FOURIER_DEGENERATE_POWER_THRESHOLD:
        return "degenerate"

    if diagnostics.shifted_relations.correct_shift_rank != 1:
        return "mismatch"

    if (
        diagnostics.addition_manifold_fraction
        >= FOURIER_MATCH_FRACTION_THRESHOLD
    ):
        return "clear_match"

    return "partial_match"


def dominant_frequency_pair(
    canonical_power: Sequence[float],
) -> int | None:
    """Return the strongest canonical frequency pair, or None if empty."""

    values = np.asarray(canonical_power, dtype=np.float64)

    if values.shape != (len(CANONICAL_FREQUENCIES),):
        raise ValueError(
            "canonical_power must contain exactly 56 values."
        )

    if float(values.sum()) <= POWER_ABSOLUTE_TOLERANCE:
        return None

    return rank_descending_stable(values)[0] + 1


def association_power_summary(
    mask: ComponentMask,
    ranking: Sequence[ComponentAssociationRecord],
) -> tuple[float, float, float, str, str]:
    """Summarize association mass retained by one Stage 9 circuit."""

    retained = set(mask.retained_component_ids)
    values = tuple(ranking)

    if not values:
        raise ValueError("ranking must not be empty.")

    total_power = float(
        sum(
            max(record.addition_manifold_delta_power, 0.0)
            for record in values
        )
    )
    retained_records = tuple(
        record
        for record in values
        if record.component_identifier in retained
    )

    if not retained_records:
        raise ValueError(
            "Circuit must retain at least one ranked component."
        )

    retained_power = float(
        sum(
            max(record.addition_manifold_delta_power, 0.0)
            for record in retained_records
        )
    )
    retained_fraction = (
        0.0
        if total_power <= POWER_ABSOLUTE_TOLERANCE
        else retained_power / total_power
    )

    return (
        retained_power,
        total_power,
        retained_fraction,
        retained_records[0].component_identifier,
        retained_records[-1].component_identifier,
    )


def build_circuit_fourier_record(
    *,
    circuit: Stage9CircuitRecord,
    metrics: BehaviourMetrics,
    diagnostics: FourierTensorDiagnostics,
    association_ranking: Sequence[ComponentAssociationRecord],
) -> CircuitFourierRecord:
    """Build one deterministic circuit-level Stage 10 record."""

    (
        retained_power,
        total_power,
        retained_fraction,
        highest_retained,
        lowest_retained,
    ) = association_power_summary(
        circuit.mask,
        association_ranking,
    )

    return CircuitFourierRecord(
        fidelity_threshold=circuit.fidelity_threshold,
        mask_id=circuit.mask.mask_id,
        retained_heads=circuit.retained_heads,
        retained_neurons=circuit.retained_neurons,
        retained_components=circuit.retained_components,
        exact_fidelity=metrics.primary_fidelity,
        ground_truth_accuracy=metrics.evaluated_accuracy,
        cross_entropy=metrics.evaluated_cross_entropy,
        addition_manifold_power=diagnostics.addition_manifold_power,
        total_fourier_power=diagnostics.total_power,
        addition_manifold_fraction=(
            diagnostics.addition_manifold_fraction
        ),
        correct_shift_rank=(
            diagnostics.shifted_relations.correct_shift_rank
        ),
        correct_shift_selectivity=(
            diagnostics.shifted_relations
            .correct_to_incorrect_mean_ratio
        ),
        dominant_canonical_frequency_pair=dominant_frequency_pair(
            diagnostics.canonical_pair_power
        ),
        diagnostic_classification=classify_fourier_diagnostic(
            diagnostics
        ),
        retained_association_power=retained_power,
        total_association_power=total_power,
        retained_association_fraction=retained_fraction,
        highest_associated_retained_component=highest_retained,
        lowest_associated_retained_component=lowest_retained,
    )


def build_removal_fourier_record(
    *,
    circuit: Stage9CircuitRecord,
    selection_role: str,
    removed_component: str,
    original_metrics: BehaviourMetrics,
    removal_metrics: BehaviourMetrics,
    diagnostics: FourierTensorDiagnostics,
) -> RemovalFourierRecord:
    """Build one deterministic selected-removal diagnostic record."""

    location = component_location(removed_component)

    return RemovalFourierRecord(
        fidelity_threshold=circuit.fidelity_threshold,
        selection_role=selection_role,
        removed_component=removed_component,
        removed_component_type=location.component_class,
        original_retained_components=circuit.retained_components,
        remaining_retained_components=(
            circuit.retained_components - 1
        ),
        original_primary_fidelity=original_metrics.primary_fidelity,
        removal_primary_fidelity=removal_metrics.primary_fidelity,
        primary_fidelity_change=(
            removal_metrics.primary_fidelity
            - original_metrics.primary_fidelity
        ),
        original_accuracy=original_metrics.evaluated_accuracy,
        removal_accuracy=removal_metrics.evaluated_accuracy,
        accuracy_change=(
            removal_metrics.evaluated_accuracy
            - original_metrics.evaluated_accuracy
        ),
        original_cross_entropy=(
            original_metrics.evaluated_cross_entropy
        ),
        removal_cross_entropy=(
            removal_metrics.evaluated_cross_entropy
        ),
        cross_entropy_change=(
            removal_metrics.evaluated_cross_entropy
            - original_metrics.evaluated_cross_entropy
        ),
        addition_manifold_power=diagnostics.addition_manifold_power,
        total_fourier_power=diagnostics.total_power,
        addition_manifold_fraction=(
            diagnostics.addition_manifold_fraction
        ),
        correct_shift_rank=(
            diagnostics.shifted_relations.correct_shift_rank
        ),
        correct_shift_selectivity=(
            diagnostics.shifted_relations
            .correct_to_incorrect_mean_ratio
        ),
        diagnostic_classification=classify_fourier_diagnostic(
            diagnostics
        ),
    )


def _json_safe_value(value: Any) -> Any:
    """Convert dataclass-table values into strict deterministic JSON."""

    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return (
                "positive_infinity"
                if value > 0
                else "negative_infinity"
            )
        return value

    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]

    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item in value.items()
        }

    return value


def _record_mapping(record: Any) -> dict[str, Any]:
    """Return a strict JSON-safe mapping from one frozen dataclass."""

    from dataclasses import asdict, is_dataclass

    if not is_dataclass(record):
        raise TypeError("record must be a dataclass value.")

    return {
        key: _json_safe_value(value)
        for key, value in asdict(record).items()
    }


def write_deterministic_csv(
    path: str | Path,
    records: Sequence[Any],
    *,
    fieldnames: Sequence[str],
) -> Path:
    """Write one stable LF-terminated CSV table."""

    output_path = Path(path)
    rows = [_record_mapping(record) for record in records]

    if not rows:
        raise ValueError("records must not be empty.")

    expected_fields = tuple(fieldnames)

    for row in rows:
        if tuple(row) != expected_fields:
            raise ValueError(
                "CSV record field order does not match fieldnames."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=expected_fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def write_deterministic_json(
    path: str | Path,
    value: Mapping[str, Any],
) -> Path:
    """Write one stable strict JSON object."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_safe_value(dict(value)),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    output_path.write_text(payload + "\n", encoding="utf-8")
    return output_path


def stage10_output_paths(
    repository: str | Path,
    *,
    stage10_run_id: str,
) -> Stage10ArtifactPaths:
    """Return deterministic Stage 10 artifact locations."""

    root = Path(repository)
    output_directory = (
        root / "results" / "raw" / stage10_run_id
    )

    return Stage10ArtifactPaths(
        component_table=(
            root
            / "results"
            / "tables"
            / "seed_1_stage10_component_fourier.csv"
        ),
        circuit_table=(
            root
            / "results"
            / "tables"
            / "seed_1_stage10_circuit_fourier.csv"
        ),
        removal_table=(
            root
            / "results"
            / "tables"
            / "seed_1_stage10_removal_fourier.csv"
        ),
        embedding_table=(
            root
            / "results"
            / "tables"
            / "seed_1_stage10_embedding_fourier.csv"
        ),
        activation_table=(
            root
            / "results"
            / "tables"
            / "seed_1_stage10_activation_fourier.csv"
        ),
        manifest=(
            root
            / "manifests"
            / f"stage10_fourier_{stage10_run_id}.json"
        ),
        output_directory=output_directory,
    )


def deterministic_stage10_run_id(
    configuration: Mapping[str, Any],
) -> str:
    """Return the Stage 10 run ID from its prospective configuration."""

    payload = json.dumps(
        dict(configuration),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"stage10-fourier-s1-{digest[:12]}"


def stage10_configuration_record(
    *,
    source_training_run_id: str,
    checkpoint_sha256: str,
    stage9_manifest_sha256: str,
    stage9_table_sha256: str,
    stage9_archive_sha256: str,
    implementation_git_commit: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Return the complete run-ID-defining Stage 10 configuration."""

    return {
        "source_training_run_id": source_training_run_id,
        "checkpoint_step": STABLE_POST_CHECKPOINT_STEP,
        "checkpoint_sha256": checkpoint_sha256,
        "stage9_manifest_sha256": stage9_manifest_sha256,
        "stage9_table_sha256": stage9_table_sha256,
        "stage9_archive_sha256": stage9_archive_sha256,
        "implementation_git_commit": implementation_git_commit,
        "device": device,
        "batch_size": _validate_positive_batch_size(batch_size),
        "modulus": MODULUS,
        "evaluated_example_count": MODULUS * MODULUS,
        "evaluated_output_classes": OUTPUT_CLASS_COUNT,
        "output_centering": "subtract_mean_over_output_class",
        "fft_normalization": FOURIER_NORMALIZATION,
        "scientific_real_dtype": SCIENTIFIC_REAL_DTYPE,
        "fourier_computation_dtype": FOURIER_COMPUTATION_DTYPE,
        "addition_manifold": (
            "M0={(k,k,-k mod 113): k=1,...,112}"
        ),
        "shifted_relations": (
            "Md={(k,k,-k+d mod 113): k=1,...,112}, d=0,...,112"
        ),
        "stable_post_thresholds": list(
            STABLE_POST_THRESHOLDS
        ),
        "component_association_method": (
            "exact single-component zero ablation; Fourier transform of "
            "full-model minus ablated-model centred final logits"
        ),
        "component_ranking": (
            "descending addition-manifold delta power; frozen component "
            "index breaks ties"
        ),
        "classification_fraction_threshold": (
            FOURIER_MATCH_FRACTION_THRESHOLD
        ),
        "classification_degenerate_power_threshold": (
            FOURIER_DEGENERATE_POWER_THRESHOLD
        ),
        "classification_definitions": (
            FOURIER_CLASSIFICATION_DEFINITIONS
        ),
        "primary_fidelity_threshold_selected": False,
        "stage11_calibration_performed": False,
    }


@dataclass(frozen=True)
class WeightFourierRecord:
    """One canonical frequency-pair record for a model weight matrix."""

    weight_name: str
    canonical_frequency_pair: int
    raw_pair_power: float
    normalized_pair_power: float
    descending_rank: int


@dataclass(frozen=True)
class ActivationFourierRecord:
    """One compact Fourier record for an MLP neuron activation."""

    component_identifier: str
    neuron_index: int
    activation_mean: float
    activation_variance: float
    near_constant: bool
    total_non_dc_power: float
    diagonal_power: float
    diagonal_power_fraction: float
    dominant_canonical_frequency_pair: int | None


@dataclass(frozen=True)
class CircuitEvaluation:
    """Outputs and diagnostics for one complete Stage 9 circuit mask."""

    circuit: Stage9CircuitRecord
    outputs: CollectedModelOutputs
    metrics: BehaviourMetrics
    diagnostics: FourierTensorDiagnostics
    record: CircuitFourierRecord


@dataclass(frozen=True)
class RemovalEvaluation:
    """Outputs and diagnostics for one selected circuit-component removal."""

    circuit_threshold: float
    selection_role: str
    removed_component: str
    outputs: CollectedModelOutputs
    metrics: BehaviourMetrics
    diagnostics: FourierTensorDiagnostics
    record: RemovalFourierRecord


@dataclass(frozen=True)
class Stage10AnalysisResult:
    """Complete deterministic in-memory Stage 10 scientific result."""

    full_outputs: CollectedModelOutputs
    full_diagnostics: FourierTensorDiagnostics
    embedding_records: tuple[WeightFourierRecord, ...]
    unembedding_records: tuple[WeightFourierRecord, ...]
    activation_diagnostics: tuple[
        ActivationSpectrumDiagnostics,
        ...
    ]
    activation_records: tuple[ActivationFourierRecord, ...]
    component_execution: ComponentAssociationExecution
    component_ranking: tuple[ComponentAssociationRecord, ...]
    circuit_evaluations: tuple[CircuitEvaluation, ...]
    removal_evaluations: tuple[RemovalEvaluation, ...]
    model_state_sha256_before: str
    model_state_sha256_after: str
    hook_counts_before: tuple[tuple[str, int], ...]
    hook_counts_after: tuple[tuple[str, int], ...]


def weight_fourier_records(
    *,
    weight_name: str,
    matrix: np.ndarray,
) -> tuple[WeightFourierRecord, ...]:
    """Return all 56 canonical frequency-pair records for one matrix."""

    if not isinstance(weight_name, str) or not weight_name:
        raise ValueError("weight_name must be a non-empty string.")

    diagnostics = analyse_weight_spectrum(matrix)
    rank_by_frequency = {
        frequency: rank
        for rank, frequency in enumerate(
            diagnostics.ranked_frequency_pairs,
            start=1,
        )
    }

    return tuple(
        WeightFourierRecord(
            weight_name=weight_name,
            canonical_frequency_pair=frequency,
            raw_pair_power=diagnostics.canonical_pair_power[
                frequency - 1
            ],
            normalized_pair_power=(
                diagnostics.normalized_canonical_pair_power[
                    frequency - 1
                ]
            ),
            descending_rank=rank_by_frequency[frequency],
        )
        for frequency in CANONICAL_FREQUENCIES
    )


def model_weight_fourier_records(
    model: Any,
) -> tuple[
    tuple[WeightFourierRecord, ...],
    tuple[WeightFourierRecord, ...],
]:
    """Analyse operand-token embeddings and valid-class unembeddings."""

    if not hasattr(model, "W_E") or not hasattr(model, "W_U"):
        raise ValueError(
            "Model must expose TransformerLens W_E and W_U tensors."
        )

    embedding = model.W_E.detach().cpu().to(torch.float64)

    if (
        embedding.ndim != 2
        or embedding.shape[0] < MODULUS
    ):
        raise ValueError(
            "W_E must have shape (vocabulary, d_model) with at least "
            "113 token rows."
        )

    unembedding = model.W_U.detach().cpu().to(torch.float64)

    if (
        unembedding.ndim != 2
        or unembedding.shape[1] < OUTPUT_CLASS_COUNT
    ):
        raise ValueError(
            "W_U must have shape (d_model, output vocabulary) with at "
            "least 113 output columns."
        )

    embedding_matrix = embedding[:MODULUS, :].numpy()
    unembedding_matrix = (
        unembedding[:, :OUTPUT_CLASS_COUNT]
        .transpose(0, 1)
        .contiguous()
        .numpy()
    )

    return (
        weight_fourier_records(
            weight_name="token_embedding_W_E",
            matrix=embedding_matrix,
        ),
        weight_fourier_records(
            weight_name="valid_class_unembedding_W_U",
            matrix=unembedding_matrix,
        ),
    )


def activation_fourier_records(
    diagnostics: Sequence[ActivationSpectrumDiagnostics],
) -> tuple[ActivationFourierRecord, ...]:
    """Build compact activation records in frozen N0-to-N511 order."""

    values = tuple(diagnostics)

    if len(values) != len(MLP_NEURON_IDS):
        raise ValueError(
            "Expected exactly 512 activation diagnostics."
        )

    return tuple(
        ActivationFourierRecord(
            component_identifier=identifier,
            neuron_index=index,
            activation_mean=diagnostic.activation_mean,
            activation_variance=diagnostic.activation_variance,
            near_constant=diagnostic.near_constant,
            total_non_dc_power=diagnostic.total_non_dc_power,
            diagonal_power=diagnostic.diagonal_power,
            diagonal_power_fraction=(
                diagnostic.diagonal_power_fraction
            ),
            dominant_canonical_frequency_pair=(
                diagnostic.dominant_frequency_pair
            ),
        )
        for index, (identifier, diagnostic) in enumerate(
            zip(
                MLP_NEURON_IDS,
                values,
                strict=True,
            )
        )
    )


def evaluate_stage9_circuits(
    *,
    model: Any,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    full_outputs: CollectedModelOutputs,
    circuits: Sequence[Stage9CircuitRecord],
    association_ranking: Sequence[ComponentAssociationRecord],
    batch_size: int,
) -> tuple[CircuitEvaluation, ...]:
    """Evaluate all six archived Stage 9 stable-post circuit masks."""

    circuit_values = tuple(circuits)

    if tuple(
        circuit.fidelity_threshold
        for circuit in circuit_values
    ) != STABLE_POST_THRESHOLDS:
        raise ValueError(
            "Circuits are not in the frozen threshold order."
        )

    results: list[CircuitEvaluation] = []

    for circuit in circuit_values:
        outputs = collect_final_position_outputs(
            model,
            inputs,
            batch_size=batch_size,
            mask=circuit.mask,
        )
        metrics = behaviour_metrics_from_logits(
            full_outputs.final_logits,
            outputs.final_logits,
            targets,
        )

        if not math.isclose(
            metrics.primary_fidelity,
            circuit.exact_fidelity,
            abs_tol=0.0,
            rel_tol=0.0,
        ):
            raise RuntimeError(
                "Stage 10 circuit fidelity does not reproduce the "
                f"Stage 9 value at threshold {circuit.fidelity_threshold}."
            )

        tensor = reshape_lexicographic_logits(
            inputs,
            outputs.final_logits,
        )
        diagnostics = analyse_logit_tensor(tensor)
        record = build_circuit_fourier_record(
            circuit=circuit,
            metrics=metrics,
            diagnostics=diagnostics,
            association_ranking=association_ranking,
        )

        results.append(
            CircuitEvaluation(
                circuit=circuit,
                outputs=outputs,
                metrics=metrics,
                diagnostics=diagnostics,
                record=record,
            )
        )

    return tuple(results)


def evaluate_selected_removals(
    *,
    model: Any,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    full_outputs: CollectedModelOutputs,
    circuit_evaluations: Sequence[CircuitEvaluation],
    association_ranking: Sequence[ComponentAssociationRecord],
    batch_size: int,
) -> tuple[RemovalEvaluation, ...]:
    """Evaluate the frozen deduplicated selected removals per circuit."""

    results: list[RemovalEvaluation] = []

    for circuit_evaluation in circuit_evaluations:
        circuit = circuit_evaluation.circuit
        selections = select_retained_components_for_removal(
            circuit.mask,
            association_ranking,
        )

        for selection_role, identifier in selections:
            removal_mask = remove_component(
                circuit.mask,
                identifier,
            )
            outputs = collect_final_position_outputs(
                model,
                inputs,
                batch_size=batch_size,
                mask=removal_mask,
            )
            metrics = behaviour_metrics_from_logits(
                full_outputs.final_logits,
                outputs.final_logits,
                targets,
            )
            tensor = reshape_lexicographic_logits(
                inputs,
                outputs.final_logits,
            )
            diagnostics = analyse_logit_tensor(tensor)
            record = build_removal_fourier_record(
                circuit=circuit,
                selection_role=selection_role,
                removed_component=identifier,
                original_metrics=circuit_evaluation.metrics,
                removal_metrics=metrics,
                diagnostics=diagnostics,
            )

            results.append(
                RemovalEvaluation(
                    circuit_threshold=(
                        circuit.fidelity_threshold
                    ),
                    selection_role=selection_role,
                    removed_component=identifier,
                    outputs=outputs,
                    metrics=metrics,
                    diagnostics=diagnostics,
                    record=record,
                )
            )

    return tuple(results)


def run_stage10_analysis(
    *,
    model: Any,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    circuits: Sequence[Stage9CircuitRecord],
    batch_size: int,
) -> Stage10AnalysisResult:
    """Execute the complete deterministic Stage 10 analysis in memory."""

    from circuit_families.training import canonical_state_hash

    batch_size = _validate_positive_batch_size(batch_size)
    model_state_before = canonical_state_hash(model.state_dict())
    hook_counts_before = _analysis_hook_counts(model)

    full_outputs = collect_final_position_outputs(
        model,
        inputs,
        batch_size=batch_size,
    )
    full_tensor = reshape_lexicographic_logits(
        inputs,
        full_outputs.final_logits,
    )
    full_diagnostics = analyse_logit_tensor(full_tensor)

    embedding_records, unembedding_records = (
        model_weight_fourier_records(model)
    )

    activation_tensor = collect_final_position_mlp_activations(
        model,
        inputs,
        batch_size=batch_size,
    )
    activation_diagnostics = (
        activation_diagnostics_for_all_neurons(
            activation_tensor
        )
    )
    activation_records = activation_fourier_records(
        activation_diagnostics
    )

    component_execution = run_component_association(
        model=model,
        inputs=inputs,
        targets=targets,
        full_outputs=full_outputs,
        circuits=circuits,
        activation_diagnostics=activation_diagnostics,
        batch_size=batch_size,
    )
    ranking = component_association_ranking(
        component_execution.records
    )

    circuit_evaluations = evaluate_stage9_circuits(
        model=model,
        inputs=inputs,
        targets=targets,
        full_outputs=full_outputs,
        circuits=circuits,
        association_ranking=ranking,
        batch_size=batch_size,
    )
    removal_evaluations = evaluate_selected_removals(
        model=model,
        inputs=inputs,
        targets=targets,
        full_outputs=full_outputs,
        circuit_evaluations=circuit_evaluations,
        association_ranking=ranking,
        batch_size=batch_size,
    )

    model_state_after = canonical_state_hash(model.state_dict())
    hook_counts_after = _analysis_hook_counts(model)

    if model_state_after != model_state_before:
        raise RuntimeError(
            "Stage 10 analysis changed the model-state hash."
        )

    if hook_counts_after != hook_counts_before:
        raise RuntimeError(
            "Stage 10 analysis leaked TransformerLens hooks."
        )

    if any(
        parameter.grad is not None
        for parameter in model.parameters()
    ):
        raise RuntimeError(
            "Stage 10 analysis left parameter gradients populated."
        )

    return Stage10AnalysisResult(
        full_outputs=full_outputs,
        full_diagnostics=full_diagnostics,
        embedding_records=embedding_records,
        unembedding_records=unembedding_records,
        activation_diagnostics=activation_diagnostics,
        activation_records=activation_records,
        component_execution=component_execution,
        component_ranking=ranking,
        circuit_evaluations=circuit_evaluations,
        removal_evaluations=removal_evaluations,
        model_state_sha256_before=model_state_before,
        model_state_sha256_after=model_state_after,
        hook_counts_before=hook_counts_before,
        hook_counts_after=hook_counts_after,
    )


@dataclass(frozen=True)
class Stage10WrittenArtifacts:
    """Paths and hashes of all deterministic Stage 10 outputs."""

    component_table_path: Path
    component_table_sha256: str
    circuit_table_path: Path
    circuit_table_sha256: str
    removal_table_path: Path
    removal_table_sha256: str
    embedding_table_path: Path
    embedding_table_sha256: str
    activation_table_path: Path
    activation_table_sha256: str
    full_model_summary_path: Path
    full_model_summary_sha256: str
    circuit_figure_path: Path
    circuit_figure_sha256: str
    component_figure_path: Path
    component_figure_sha256: str
    retention_figure_path: Path
    retention_figure_sha256: str
    manifest_path: Path
    manifest_sha256: str


def dataclass_fieldnames(record_type: type[Any]) -> tuple[str, ...]:
    """Return frozen dataclass field order."""

    from dataclasses import fields, is_dataclass

    if not is_dataclass(record_type):
        raise TypeError("record_type must be a dataclass type.")

    return tuple(field.name for field in fields(record_type))


def _relative_path(
    repository: Path,
    path: Path,
) -> str:
    """Return a repository-relative artifact path."""

    resolved_repository = repository.resolve()
    resolved_path = path.resolve()

    try:
        return str(resolved_path.relative_to(resolved_repository))
    except ValueError as exc:
        raise ValueError(
            f"Artifact is outside the repository: {resolved_path}"
        ) from exc


def _prepare_empty_stage10_directory(path: Path) -> Path:
    """Create and validate an empty Stage 10 raw directory."""

    path.mkdir(parents=True, exist_ok=True)
    existing = tuple(path.iterdir())

    if existing:
        raise FileExistsError(
            "Stage 10 output directory must be empty. Existing entries: "
            + ", ".join(sorted(item.name for item in existing))
        )

    return path


def _configure_deterministic_svg() -> None:
    """Configure Matplotlib SVG output without timestamps or random IDs."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.hashsalt"] = (
        "circuit-families-stage10-fourier"
    )


def write_stage10_figures(
    *,
    repository: Path,
    stage10_run_id: str,
    result: Stage10AnalysisResult,
) -> tuple[Path, Path, Path]:
    """Write three deterministic Stage 10 SVG figures."""

    _configure_deterministic_svg()

    import matplotlib.pyplot as plt

    figure_directory = repository / "figures" / stage10_run_id
    figure_directory.mkdir(parents=True, exist_ok=True)

    circuit_figure = (
        figure_directory / "circuit_fourier_fraction.svg"
    )
    component_figure = (
        figure_directory / "component_association_ranking.svg"
    )
    retention_figure = (
        figure_directory / "retained_association_fraction.svg"
    )

    thresholds = [
        evaluation.record.fidelity_threshold
        for evaluation in result.circuit_evaluations
    ]
    circuit_fractions = [
        evaluation.record.addition_manifold_fraction
        for evaluation in result.circuit_evaluations
    ]

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(
        thresholds,
        circuit_fractions,
        marker="o",
    )
    axis.axhline(
        FOURIER_MATCH_FRACTION_THRESHOLD,
        linestyle="--",
    )
    axis.set_xlabel("Stage 9 fidelity threshold")
    axis.set_ylabel("Fraction of Fourier power on M0")
    axis.set_title("Stable-post circuit Fourier alignment")
    axis.set_xlim(max(thresholds) + 0.01, min(thresholds) - 0.01)
    axis.set_ylim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(
        circuit_figure,
        format="svg",
        metadata={"Date": None},
    )
    plt.close(figure)

    ranked_power = [
        record.addition_manifold_delta_power
        for record in result.component_ranking
    ]

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(
        range(1, len(ranked_power) + 1),
        ranked_power,
    )
    axis.set_xlabel("Exact component-association rank")
    axis.set_ylabel("M0 power in full-minus-ablated logits")
    axis.set_title("Component association with modular addition")
    axis.set_yscale("symlog", linthresh=1.0e-12)
    figure.tight_layout()
    figure.savefig(
        component_figure,
        format="svg",
        metadata={"Date": None},
    )
    plt.close(figure)

    retained_fractions = [
        evaluation.record.retained_association_fraction
        for evaluation in result.circuit_evaluations
    ]

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(
        thresholds,
        retained_fractions,
        marker="o",
    )
    axis.set_xlabel("Stage 9 fidelity threshold")
    axis.set_ylabel("Retained fraction of total M0 association power")
    axis.set_title("Association mass retained by sparse circuits")
    axis.set_xlim(max(thresholds) + 0.01, min(thresholds) - 0.01)
    axis.set_ylim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(
        retention_figure,
        format="svg",
        metadata={"Date": None},
    )
    plt.close(figure)

    return (
        circuit_figure,
        component_figure,
        retention_figure,
    )


def full_model_summary_record(
    result: Stage10AnalysisResult,
) -> dict[str, Any]:
    """Return the deterministic full-model Stage 10 summary."""

    diagnostics = result.full_diagnostics

    return {
        "schema_version": 1,
        "evaluated_example_count": (
            result.full_outputs.evaluated_example_count
        ),
        "evaluation_batch_size": (
            result.full_outputs.evaluation_batch_size
        ),
        "source_logit_dtype": result.full_outputs.source_dtype,
        "scientific_real_dtype": SCIENTIFIC_REAL_DTYPE,
        "fourier_computation_dtype": FOURIER_COMPUTATION_DTYPE,
        "total_fourier_power": diagnostics.total_power,
        "addition_manifold_power": (
            diagnostics.addition_manifold_power
        ),
        "addition_manifold_fraction": (
            diagnostics.addition_manifold_fraction
        ),
        "correct_shift_rank": (
            diagnostics.shifted_relations.correct_shift_rank
        ),
        "correct_to_incorrect_mean_ratio": (
            diagnostics.shifted_relations
            .correct_to_incorrect_mean_ratio
        ),
        "correct_to_largest_incorrect_ratio": (
            diagnostics.shifted_relations
            .correct_to_largest_incorrect_ratio
        ),
        "correct_shift_family_fraction": (
            diagnostics.shifted_relations.correct_family_fraction
        ),
        "dominant_canonical_frequency_pair": (
            dominant_frequency_pair(
                diagnostics.canonical_pair_power
            )
        ),
        "diagnostic_classification": (
            classify_fourier_diagnostic(diagnostics)
        ),
        "canonical_frequency_pair_power": list(
            diagnostics.canonical_pair_power
        ),
        "normalized_canonical_frequency_pair_power": list(
            diagnostics.normalized_canonical_pair_power
        ),
        "shifted_relation_power": list(
            diagnostics.shifted_relations.power_by_shift
        ),
    }


def component_table_records(
    execution: ComponentAssociationExecution,
) -> tuple[dict[str, Any], ...]:
    """Flatten component records for a transparent deterministic CSV."""

    records = []

    for record in execution.records:
        value = _record_mapping(record)
        flags = value.pop("retained_flags")

        for threshold, retained in zip(
            STABLE_POST_THRESHOLDS,
            flags,
            strict=True,
        ):
            label = str(threshold).replace(".", "_")
            value[f"retained_at_threshold_{label}"] = retained

        records.append(value)

    return tuple(records)


def write_mapping_csv(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> Path:
    """Write stable CSV rows supplied as ordered mappings."""

    output_path = Path(path)
    rows = [dict(record) for record in records]

    if not rows:
        raise ValueError("records must not be empty.")

    fieldnames = tuple(rows[0])

    for row in rows:
        if tuple(row) != fieldnames:
            raise ValueError(
                "Mapping CSV field order changed between records."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def build_stage10_manifest(
    *,
    repository: Path,
    stage10_run_id: str,
    configuration: Mapping[str, Any],
    context: Any,
    circuits: Sequence[Stage9CircuitRecord],
    result: Stage10AnalysisResult,
    outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the deterministic provenance-bearing Stage 10 manifest."""

    circuit_classifications = {
        str(evaluation.record.fidelity_threshold): (
            evaluation.record.diagnostic_classification
        )
        for evaluation in result.circuit_evaluations
    }

    removal_counts = {
        str(threshold): sum(
            evaluation.circuit_threshold == threshold
            for evaluation in result.removal_evaluations
        )
        for threshold in STABLE_POST_THRESHOLDS
    }

    return {
        "schema_version": 1,
        "experiment_type": "fourier_pipeline_sanity_check",
        "stage10_run_id": stage10_run_id,
        "source_training_run_id": context.run_id,
        "stage10_implementation_git_commit": (
            configuration["implementation_git_commit"]
        ),
        "scientific_interpretation": (
            "Pipeline diagnostic only. Fourier correspondence supports "
            "mechanistic plausibility but does not establish circuit "
            "uniqueness, and no primary fidelity threshold is selected."
        ),
        "configuration": dict(configuration),
        "checkpoint": {
            "phase": context.checkpoint_phase,
            "training_step": context.checkpoint_step,
            "path": _relative_path(
                repository,
                context.checkpoint_path,
            ),
            "checkpoint_sha256": context.checkpoint_sha256,
            "model_state_sha256": context.model_state_sha256,
        },
        "dataset": {
            "example_count": int(context.inputs.shape[0]),
            "example_ordering": context.example_ordering,
            "dataset_sha256": context.dataset_sha256,
            "split_sha256": context.split_sha256,
            "dataset_archive_sha256": (
                context.dataset_archive_sha256
            ),
            "dataset_metadata_sha256": (
                context.dataset_metadata_sha256
            ),
            "includes_training_and_test_examples": True,
        },
        "fourier_definition": {
            "output_tensor_shape": [113, 113, 113],
            "output_centering": (
                "subtract per-input mean over valid output classes"
            ),
            "fft_normalization": FOURIER_NORMALIZATION,
            "addition_manifold": (
                "M0={(k,k,-k mod 113): k=1,...,112}"
            ),
            "shifted_relations": (
                "Md={(k,k,-k+d mod 113): "
                "k=1,...,112}, d=0,...,112"
            ),
            "canonical_frequency_pairs": (
                "pair k with 113-k for k=1,...,56"
            ),
            "real_dtype": SCIENTIFIC_REAL_DTYPE,
            "complex_dtype": FOURIER_COMPUTATION_DTYPE,
        },
        "classification": {
            "fraction_threshold": (
                FOURIER_MATCH_FRACTION_THRESHOLD
            ),
            "degenerate_power_threshold": (
                FOURIER_DEGENERATE_POWER_THRESHOLD
            ),
            "definitions": FOURIER_CLASSIFICATION_DEFINITIONS,
            "full_model": classify_fourier_diagnostic(
                result.full_diagnostics
            ),
            "circuits": circuit_classifications,
        },
        "component_association": {
            "component_count": len(
                result.component_execution.records
            ),
            "method": (
                "Exact single-component zero ablation followed by the "
                "Fourier transform of full-model minus ablated-model "
                "centred final logits."
            ),
            "ranking": (
                "Descending M0 delta power with frozen component index "
                "breaking ties."
            ),
            "model_state_sha256_before": (
                result.component_execution
                .model_state_sha256_before
            ),
            "model_state_sha256_after": (
                result.component_execution
                .model_state_sha256_after
            ),
            "gradients_absent_after": (
                result.component_execution.gradients_absent_after
            ),
        },
        "stage9_circuits": [
            {
                "fidelity_threshold": circuit.fidelity_threshold,
                "mask_id": circuit.mask.mask_id,
                "retained_heads": circuit.retained_heads,
                "retained_neurons": circuit.retained_neurons,
                "retained_components": circuit.retained_components,
                "stage9_exact_fidelity": circuit.exact_fidelity,
                "final_mask_path": circuit.final_mask_path,
                "final_mask_sha256": circuit.final_mask_sha256,
            }
            for circuit in circuits
        ],
        "selected_removals": {
            "selection_rules": [
                "highest associated retained component overall",
                "highest associated retained attention head if present",
                "highest associated retained MLP neuron if present",
                "lowest associated retained component",
                "duplicate component identifiers are evaluated once",
            ],
            "total_evaluation_count": len(
                result.removal_evaluations
            ),
            "counts_by_threshold": removal_counts,
        },
        "integrity": {
            "model_state_sha256_before": (
                result.model_state_sha256_before
            ),
            "model_state_sha256_after": (
                result.model_state_sha256_after
            ),
            "model_state_unchanged": (
                result.model_state_sha256_before
                == result.model_state_sha256_after
            ),
            "hook_counts_before": list(
                result.hook_counts_before
            ),
            "hook_counts_after": list(
                result.hook_counts_after
            ),
            "hook_counts_unchanged": (
                result.hook_counts_before
                == result.hook_counts_after
            ),
        },
        "threshold_calibration": {
            "primary_fidelity_threshold_selected": False,
            "stage11_calibration_performed": False,
            "all_six_stable_post_thresholds_reported": True,
        },
        "outputs": dict(outputs),
    }


def write_stage10_artifacts(
    *,
    repository: str | Path,
    stage10_run_id: str,
    configuration: Mapping[str, Any],
    context: Any,
    circuits: Sequence[Stage9CircuitRecord],
    result: Stage10AnalysisResult,
) -> Stage10WrittenArtifacts:
    """Write all deterministic Stage 10 scientific artifacts."""

    root = Path(repository).resolve()
    paths = stage10_output_paths(
        root,
        stage10_run_id=stage10_run_id,
    )
    _prepare_empty_stage10_directory(paths.output_directory)

    component_path = write_mapping_csv(
        paths.component_table,
        component_table_records(result.component_execution),
    )
    circuit_path = write_deterministic_csv(
        paths.circuit_table,
        [
            evaluation.record
            for evaluation in result.circuit_evaluations
        ],
        fieldnames=dataclass_fieldnames(CircuitFourierRecord),
    )
    removal_path = write_deterministic_csv(
        paths.removal_table,
        [
            evaluation.record
            for evaluation in result.removal_evaluations
        ],
        fieldnames=dataclass_fieldnames(RemovalFourierRecord),
    )
    embedding_path = write_deterministic_csv(
        paths.embedding_table,
        (
            *result.embedding_records,
            *result.unembedding_records,
        ),
        fieldnames=dataclass_fieldnames(WeightFourierRecord),
    )
    activation_path = write_deterministic_csv(
        paths.activation_table,
        result.activation_records,
        fieldnames=dataclass_fieldnames(
            ActivationFourierRecord
        ),
    )

    full_model_path = write_deterministic_json(
        paths.output_directory / "full_model_fourier.json",
        full_model_summary_record(result),
    )

    (
        circuit_figure,
        component_figure,
        retention_figure,
    ) = write_stage10_figures(
        repository=root,
        stage10_run_id=stage10_run_id,
        result=result,
    )

    output_records = {
        "component_fourier_table": {
            "path": _relative_path(root, component_path),
            "sha256": file_sha256(component_path),
            "record_count": len(
                result.component_execution.records
            ),
        },
        "circuit_fourier_table": {
            "path": _relative_path(root, circuit_path),
            "sha256": file_sha256(circuit_path),
            "record_count": len(result.circuit_evaluations),
        },
        "selected_removal_table": {
            "path": _relative_path(root, removal_path),
            "sha256": file_sha256(removal_path),
            "record_count": len(result.removal_evaluations),
        },
        "embedding_unembedding_table": {
            "path": _relative_path(root, embedding_path),
            "sha256": file_sha256(embedding_path),
            "record_count": (
                len(result.embedding_records)
                + len(result.unembedding_records)
            ),
        },
        "activation_fourier_table": {
            "path": _relative_path(root, activation_path),
            "sha256": file_sha256(activation_path),
            "record_count": len(result.activation_records),
        },
        "full_model_fourier_summary": {
            "path": _relative_path(root, full_model_path),
            "sha256": file_sha256(full_model_path),
        },
        "circuit_fourier_figure": {
            "path": _relative_path(root, circuit_figure),
            "sha256": file_sha256(circuit_figure),
            "format": "svg",
        },
        "component_association_figure": {
            "path": _relative_path(root, component_figure),
            "sha256": file_sha256(component_figure),
            "format": "svg",
        },
        "retained_association_figure": {
            "path": _relative_path(root, retention_figure),
            "sha256": file_sha256(retention_figure),
            "format": "svg",
        },
    }

    manifest = build_stage10_manifest(
        repository=root,
        stage10_run_id=stage10_run_id,
        configuration=configuration,
        context=context,
        circuits=circuits,
        result=result,
        outputs=output_records,
    )
    manifest_path = write_deterministic_json(
        paths.manifest,
        manifest,
    )

    return Stage10WrittenArtifacts(
        component_table_path=component_path,
        component_table_sha256=file_sha256(component_path),
        circuit_table_path=circuit_path,
        circuit_table_sha256=file_sha256(circuit_path),
        removal_table_path=removal_path,
        removal_table_sha256=file_sha256(removal_path),
        embedding_table_path=embedding_path,
        embedding_table_sha256=file_sha256(embedding_path),
        activation_table_path=activation_path,
        activation_table_sha256=file_sha256(activation_path),
        full_model_summary_path=full_model_path,
        full_model_summary_sha256=file_sha256(full_model_path),
        circuit_figure_path=circuit_figure,
        circuit_figure_sha256=file_sha256(circuit_figure),
        component_figure_path=component_figure,
        component_figure_sha256=file_sha256(component_figure),
        retention_figure_path=retention_figure,
        retention_figure_sha256=file_sha256(retention_figure),
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
    )

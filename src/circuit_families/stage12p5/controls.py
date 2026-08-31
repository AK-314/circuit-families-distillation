"""Complete aligned-plus-five-controls construction with explicit invariants."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from circuit_families.seeds import MAX_SEED, numpy_generator
from circuit_families.stage12p3.records import canonical_json_bytes

from .contracts import CONDITIONS, Stage12P5ContractError, TrialContract
from .fourier import (
    AlignmentPlan,
    ExtractedCoordinateState,
    FourierTransformAdapter,
    InterventionPayload,
)


class ControlUnavailableError(RuntimeError):
    """Raised when a required control cannot be constructed without substitution."""


def derive_condition_seed(trial: TrialContract, domain: str) -> tuple[int, str]:
    """Derive a disjoint PCG64-compatible sub-seed from the declared root seed."""
    material = {
        "schema_version": "stage12p5-derived-seed/v1",
        "seed_namespace_ref": trial.seed_namespace_ref,
        "trial_id": trial.trial_id,
        "root_seed": trial.root_seed,
        "domain": domain,
        "scientific_data": False,
        "production_eligible": False,
    }
    digest = hashlib.sha256(canonical_json_bytes(material)).digest()
    return int.from_bytes(digest[:8], "big") % (MAX_SEED + 1), digest.hex()


def deterministic_derangement(input_ids: Sequence[str], *, seed: int) -> dict[str, str]:
    ordered = tuple(sorted(input_ids))
    if len(ordered) < 2:
        raise ControlUnavailableError("mismatched-input control requires at least two inputs")
    generator = numpy_generator(seed)
    offset = int(generator.integers(1, len(ordered)))
    mapping = {
        value: ordered[(index + offset) % len(ordered)]
        for index, value in enumerate(ordered)
    }
    if any(source == target for source, target in mapping.items()):
        raise ControlUnavailableError("deterministic mismatch mapping is not a derangement")
    return mapping


def _select(values: np.ndarray, indices: tuple[int, ...], *, label: str) -> np.ndarray:
    flattened = np.asarray(values).reshape(-1)
    if not indices or len(set(indices)) != len(indices):
        raise Stage12P5ContractError(f"{label} indices must be non-empty and unique")
    if any(index < 0 or index >= flattened.size for index in indices):
        raise Stage12P5ContractError(f"{label} index is outside the coordinate universe")
    return np.array(flattened[list(indices)], copy=True)


def _ordinary_from_selected(
    selected: np.ndarray,
    *,
    target_indices: tuple[int, ...],
    target_shape: tuple[int, ...],
    trial: TrialContract,
    fourier_adapter: FourierTransformAdapter,
    ordinary_dtype: str,
) -> np.ndarray:
    if selected.size != len(target_indices):
        raise Stage12P5ContractError("selected coordinate count does not match target mode")
    full = np.zeros(int(np.prod(target_shape)), dtype=np.complex128)
    full[list(target_indices)] = np.asarray(selected).reshape(-1)
    return fourier_adapter.inverse(
        full.reshape(target_shape),
        trial.fourier,
        ordinary_dtype=ordinary_dtype,
    )


def _payload(
    *,
    condition: str,
    trial: TrialContract,
    coordinate_values: np.ndarray,
    ordinary_state: np.ndarray,
    source_input_id: str,
    mode_id: str,
    plan: AlignmentPlan | None,
    evidence: Mapping[str, Any],
) -> InterventionPayload:
    return InterventionPayload(
        condition=condition,
        trial_id=trial.trial_id,
        pair_id=trial.pair.pair_id,
        recipient_input_id=trial.recipient_input_id,
        location_ref=trial.location.location_ref,
        coordinate_ids=trial.capacity.allowed_coordinate_ids,
        coordinate_values=np.asarray(coordinate_values),
        ordinary_state=np.asarray(ordinary_state),
        source_input_id=source_input_id,
        mode_id=mode_id,
        side_information=trial.capacity.allowed_side_information,
        external_identifiers=trial.capacity.external_identifier_budget,
        alignment_plan_sha256=None if plan is None else plan.plan_sha256,
        construction_evidence=dict(evidence),
    )


def build_comparison_payloads(
    *,
    trial: TrialContract,
    plan: AlignmentPlan,
    aligned_source: ExtractedCoordinateState,
    wrong_mode_source: ExtractedCoordinateState,
    mismatched_source: ExtractedCoordinateState,
    ordinary_patch_state: np.ndarray,
    source_mode_indices: tuple[int, ...],
    recipient_mode_indices: tuple[int, ...],
    wrong_mode_indices: tuple[int, ...],
    fourier_adapter: FourierTransformAdapter,
    ordinary_shape_adapter: Callable[[np.ndarray, tuple[int, ...]], np.ndarray],
    mismatch_input_roster: Sequence[str],
    norm_tolerance: float = 1e-12,
) -> dict[str, InterventionPayload]:
    """Build every required condition before execution may begin."""
    if plan.pair_id != trial.pair.pair_id:
        raise Stage12P5ContractError("alignment plan pair mismatch")
    if aligned_source.input_id != trial.source_input_id:
        raise Stage12P5ContractError("aligned source input identity mismatch")
    if aligned_source.mode_id != trial.source_mode_id:
        raise Stage12P5ContractError("aligned source mode identity mismatch")
    if wrong_mode_source.input_id != trial.source_input_id:
        raise Stage12P5ContractError("wrong-mode control changed input identity")
    if wrong_mode_source.mode_id == trial.source_mode_id:
        raise ControlUnavailableError("wrong Fourier mode is not distinct")
    if set(wrong_mode_indices).intersection(source_mode_indices):
        raise ControlUnavailableError("wrong Fourier mode overlaps the aligned mode")
    mismatch_seed, mismatch_digest = derive_condition_seed(trial, "mismatch")
    derangement = deterministic_derangement(mismatch_input_roster, seed=mismatch_seed)
    expected_mismatch = derangement.get(trial.recipient_input_id)
    if expected_mismatch is None or mismatched_source.input_id != expected_mismatch:
        raise Stage12P5ContractError("mismatched-input source does not follow the derangement")

    selected = _select(aligned_source.values, source_mode_indices, label="aligned mode")
    aligned = plan.apply(selected, pair_id=trial.pair.pair_id)
    aligned_ordinary = _ordinary_from_selected(
        aligned,
        target_indices=recipient_mode_indices,
        target_shape=trial.location.recipient_shape,
        trial=trial,
        fourier_adapter=fourier_adapter,
        ordinary_dtype="<f8",
    )
    payloads: dict[str, InterventionPayload] = {}
    payloads[CONDITIONS[0]] = _payload(
        condition=CONDITIONS[0],
        trial=trial,
        coordinate_values=aligned,
        ordinary_state=aligned_ordinary,
        source_input_id=trial.source_input_id,
        mode_id=trial.recipient_mode_id,
        plan=plan,
        evidence={"mapping_applied": True, "condition_difference": "aligned_mapping"},
    )

    wrong_selected = _select(wrong_mode_source.values, wrong_mode_indices, label="wrong mode")
    wrong_mapped = plan.apply(wrong_selected, pair_id=trial.pair.pair_id)
    payloads[CONDITIONS[1]] = _payload(
        condition=CONDITIONS[1],
        trial=trial,
        coordinate_values=wrong_mapped,
        ordinary_state=_ordinary_from_selected(
            wrong_mapped,
            target_indices=wrong_mode_indices,
            target_shape=trial.location.recipient_shape,
            trial=trial,
            fourier_adapter=fourier_adapter,
            ordinary_dtype="<f8",
        ),
        source_input_id=trial.source_input_id,
        mode_id=wrong_mode_source.mode_id,
        plan=plan,
        evidence={"distinct_mode": True, "condition_difference": "source_mode"},
    )

    shuffle_seed, shuffle_digest = derive_condition_seed(trial, "shuffle")
    permutation = numpy_generator(shuffle_seed).permutation(selected.size)
    if selected.size > 1 and np.array_equal(permutation, np.arange(selected.size)):
        permutation = np.roll(permutation, 1)
    shuffled = aligned[permutation]
    payloads[CONDITIONS[2]] = _payload(
        condition=CONDITIONS[2],
        trial=trial,
        coordinate_values=shuffled,
        ordinary_state=_ordinary_from_selected(
            shuffled,
            target_indices=recipient_mode_indices,
            target_shape=trial.location.recipient_shape,
            trial=trial,
            fourier_adapter=fourier_adapter,
            ordinary_dtype="<f8",
        ),
        source_input_id=trial.source_input_id,
        mode_id=trial.recipient_mode_id,
        plan=plan,
        evidence={
            "permutation": permutation.tolist(),
            "permutation_seed_digest": shuffle_digest,
            "marginal_values_preserved": True,
            "norm_preserved": bool(np.isclose(np.linalg.norm(shuffled), np.linalg.norm(aligned))),
            "condition_difference": "coefficient_permutation",
        },
    )

    mismatch_selected = _select(
        mismatched_source.values, source_mode_indices, label="mismatched-input mode"
    )
    mismatch_mapped = plan.apply(mismatch_selected, pair_id=trial.pair.pair_id)
    payloads[CONDITIONS[3]] = _payload(
        condition=CONDITIONS[3],
        trial=trial,
        coordinate_values=mismatch_mapped,
        ordinary_state=_ordinary_from_selected(
            mismatch_mapped,
            target_indices=recipient_mode_indices,
            target_shape=trial.location.recipient_shape,
            trial=trial,
            fourier_adapter=fourier_adapter,
            ordinary_dtype="<f8",
        ),
        source_input_id=mismatched_source.input_id,
        mode_id=trial.recipient_mode_id,
        plan=plan,
        evidence={
            "derangement_seed_digest": mismatch_digest,
            "derangement": derangement,
            "no_source_recipient_match": mismatched_source.input_id != trial.recipient_input_id,
            "condition_difference": "source_input",
        },
    )

    random_seed, random_digest = derive_condition_seed(trial, "random_state")
    target_norm = float(np.linalg.norm(aligned))
    generator = numpy_generator(random_seed)
    if target_norm == 0.0:
        random_values = np.zeros_like(aligned)
        zero_behavior = "deterministic_zero"
    else:
        raw = generator.normal(size=aligned.size) + 1j * generator.normal(size=aligned.size)
        if aligned.size == 2:
            raw[1] = np.conjugate(raw[0])
        raw_norm = float(np.linalg.norm(raw))
        if raw_norm == 0.0:
            raise ControlUnavailableError("equal-norm random draw is rank-deficient")
        random_values = raw * (target_norm / raw_norm)
        zero_behavior = "not_applicable"
    if not np.isclose(np.linalg.norm(random_values), target_norm, atol=norm_tolerance, rtol=0.0):
        raise ControlUnavailableError("equal-norm random state missed declared tolerance")
    payloads[CONDITIONS[4]] = _payload(
        condition=CONDITIONS[4],
        trial=trial,
        coordinate_values=random_values,
        ordinary_state=_ordinary_from_selected(
            random_values,
            target_indices=recipient_mode_indices,
            target_shape=trial.location.recipient_shape,
            trial=trial,
            fourier_adapter=fourier_adapter,
            ordinary_dtype="<f8",
        ),
        source_input_id=trial.source_input_id,
        mode_id=trial.recipient_mode_id,
        plan=None,
        evidence={
            "random_seed_digest": random_digest,
            "target_norm": target_norm,
            "observed_norm": float(np.linalg.norm(random_values)),
            "norm_tolerance": norm_tolerance,
            "zero_norm_behavior": zero_behavior,
            "condition_difference": "random_values",
        },
    )

    ordinary = np.asarray(
        ordinary_shape_adapter(ordinary_patch_state, trial.location.recipient_shape)
    )
    if ordinary.shape != trial.location.recipient_shape:
        raise ControlUnavailableError("ordinary patching shape adapter did not meet write budget")
    flattened = ordinary.reshape(-1).astype(np.float64)
    if flattened.size < 2 * len(trial.capacity.allowed_coordinate_ids):
        raise ControlUnavailableError("ordinary patch lacks capacity-matched scalar payload")
    ordinary_capacity = flattened[: 2 * len(trial.capacity.allowed_coordinate_ids)].reshape(-1, 2)
    ordinary_capacity = ordinary_capacity[:, 0] + 1j * ordinary_capacity[:, 1]
    payloads[CONDITIONS[5]] = _payload(
        condition=CONDITIONS[5],
        trial=trial,
        coordinate_values=ordinary_capacity,
        ordinary_state=ordinary,
        source_input_id=trial.source_input_id,
        mode_id="ordinary-activation/v1",
        plan=None,
        evidence={
            "shape_adapter_ref": getattr(
                ordinary_shape_adapter, "adapter_ref", "injected-shape-adapter/v1"
            ),
            "Fourier_alignment_applied": False,
            "condition_difference": "ordinary_unaligned_patch",
        },
    )
    if tuple(payloads) != CONDITIONS:
        raise Stage12P5ContractError("control construction did not close the full inventory")
    return payloads

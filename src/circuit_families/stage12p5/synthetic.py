"""Portable synthetic-only Stage 12-P5 fixture construction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .capacity import CapacityAccounting, account_payload
from .contracts import (
    AlignmentProfile,
    CapacityContract,
    FourierCoordinateContract,
    LocationContract,
    ModelReference,
    PairContract,
    TrialContract,
)
from .controls import build_comparison_payloads
from .fourier import (
    ArrayActivationAdapter,
    InterventionPayload,
    NumpyFourierAdapter,
    extract_coordinates,
    fit_linear_alignment,
)
from .runner import CensoredOutcome, UnavailableCondition

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
LOCATION_REF = "location/synthetic-residual/v1"
SOURCE_ARCH = "architecture/synthetic-source/v1"
RECIPIENT_ARCH = "architecture/synthetic-recipient/v1"


class IdentityShapeAdapter:
    adapter_ref = "shape-adapter/synthetic-identity/v1"

    def __call__(self, state: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(state).reshape(shape)


class SyntheticOutcomeAdapter:
    adapter_ref = "outcome/synthetic-raw-sum/v1"

    def __init__(self, *, exercise_terminal_paths: bool = True) -> None:
        self.exercise_terminal_paths = exercise_terminal_paths

    def observe(self, model: Any, *, input_id: str, condition: str) -> float:
        if self.exercise_terminal_paths:
            if condition == "wrong_fourier_mode":
                return float("nan")
            if condition == "shuffled_coefficients":
                raise CensoredOutcome("synthetic policy-censored observation")
            if condition == "mismatched_input":
                raise UnavailableCondition("synthetic outcome adapter unavailable")
            if condition == "equal_norm_random_state":
                raise RuntimeError("synthetic forced outcome failure")
        return float(np.sum(np.asarray(model[input_id], dtype=np.float64)))


@dataclass(frozen=True)
class SyntheticFixture:
    trial: TrialContract
    payloads: dict[str, InterventionPayload]
    accounting: dict[str, CapacityAccounting]
    recipient_adapter: ArrayActivationAdapter
    recipient_model_template: dict[str, np.ndarray]
    outcome_adapter: SyntheticOutcomeAdapter
    alignment_plan_sha256: str

    def model_factory(self) -> dict[str, np.ndarray]:
        return copy.deepcopy(self.recipient_model_template)


def build_synthetic_fixture(
    *,
    terminal_paths: bool = True,
    source_a: np.ndarray | None = None,
    source_b: np.ndarray | None = None,
) -> SyntheticFixture:
    source = ModelReference(
        "source_teacher",
        "model/synthetic-teacher/v1",
        SOURCE_ARCH,
        SHA_A,
        "checkpoint/synthetic-teacher/v1",
        SHA_B,
        "direct_teacher",
    )
    recipient = ModelReference(
        "recipient_student",
        "model/synthetic-student/v1",
        RECIPIENT_ARCH,
        SHA_C,
        "checkpoint/synthetic-student/v1",
        SHA_D,
        "hard_target",
    )
    pair = PairContract(
        source,
        recipient,
        "pair-selection/synthetic-roster/v1",
        SHA_A,
        "pair-selection/outcome-independent/v1",
    )
    location = LocationContract(
        LOCATION_REF,
        SOURCE_ARCH,
        RECIPIENT_ARCH,
        "hook/source-residual/v1",
        "hook/recipient-residual/v1",
        "layout/source-feature-major/v1",
        "layout/recipient-token-major/v1",
        (4,),
        (4,),
    )
    fourier = FourierCoordinateContract(
        "fourier/synthetic-vector/v1",
        "coordinates/synthetic-four/v1",
        "numpy-fftn-negative-exponent/v1",
        "ortho",
        (0,),
        "complex",
        "full_spectrum",
        "gauge/additive-mean-centred/v1",
    )
    alignment = AlignmentProfile(
        "alignment/synthetic-linear/v1",
        SHA_B,
        SOURCE_ARCH,
        RECIPIENT_ARCH,
        2,
        2,
        "alignment-data/synthetic-excluded/v1",
        SHA_C,
        "boundary/no-trial-outcomes/v1",
    )
    capacity = CapacityContract(
        "capacity/synthetic-two-complex/v1",
        fourier.coordinate_universe_ref,
        ("slot-0", "slot-1"),
        4,
        2,
        2,
        64,
        "quantization/float64-complex128/v1",
        ("condition-kind/v1",),
        ("input-slot/v1", "mode-slot/v1"),
        LOCATION_REF,
        (4,),
        4,
        "padding/fixed-zero/v1",
    )
    trial = TrialContract(
        pair=pair,
        input_set_ref="input-set/synthetic-two/v1",
        input_set_sha256=SHA_D,
        source_input_id="input-a",
        recipient_input_id="input-a",
        location=location,
        fourier=fourier,
        source_mode_id="mode/source-pair-1-3/v1",
        recipient_mode_id="mode/recipient-pair-1-3/v1",
        alignment=alignment,
        capacity=capacity,
        outcome_adapter_ref="outcome/synthetic-raw-sum/v1",
        outcome_adapter_sha256=SHA_A,
        comparison_set_ref="comparison/synthetic-complete/v1",
        root_seed=314159,
        seed_namespace_ref="seed-derivation/v1",
    )
    source_model = {
        "input-a": np.array(
            [1.0, 2.0, 3.0, 4.0] if source_a is None else source_a,
            dtype=np.float64,
        ),
        "input-b": np.array(
            [4.0, 1.0, 0.0, 3.0] if source_b is None else source_b,
            dtype=np.float64,
        ),
    }
    recipient_template = {"input-a": np.zeros(4, dtype=np.float64)}
    source_adapter = ArrayActivationAdapter(
        architecture_ref=SOURCE_ARCH,
        supported_locations=(LOCATION_REF,),
        external_layout_ref="layout/source-feature-major/v1",
        canonical_axes=(0,),
    )
    recipient_adapter = ArrayActivationAdapter(
        architecture_ref=RECIPIENT_ARCH,
        supported_locations=(LOCATION_REF,),
        external_layout_ref="layout/recipient-token-major/v1",
        canonical_axes=(0,),
    )
    fft = NumpyFourierAdapter()
    aligned_state = extract_coordinates(
        model=source_model,
        model_ref=source.model_ref,
        architecture_ref=SOURCE_ARCH,
        input_id="input-a",
        location=location,
        fourier=fourier,
        mode_id=trial.source_mode_id,
        activation_adapter=source_adapter,
        fourier_adapter=fft,
    )
    wrong_state = extract_coordinates(
        model=source_model,
        model_ref=source.model_ref,
        architecture_ref=SOURCE_ARCH,
        input_id="input-a",
        location=location,
        fourier=fourier,
        mode_id="mode/source-pair-0-2/v1",
        activation_adapter=source_adapter,
        fourier_adapter=fft,
    )
    mismatch_state = extract_coordinates(
        model=source_model,
        model_ref=source.model_ref,
        architecture_ref=SOURCE_ARCH,
        input_id="input-b",
        location=location,
        fourier=fourier,
        mode_id=trial.source_mode_id,
        activation_adapter=source_adapter,
        fourier_adapter=fft,
    )
    fit_source = np.array(
        [[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j], [1 + 1j, 1 - 1j]],
        dtype=np.complex128,
    )
    plan = fit_linear_alignment(
        profile=alignment,
        pair_id=pair.pair_id,
        source_fit=fit_source,
        recipient_fit=fit_source.copy(),
    )
    payloads = build_comparison_payloads(
        trial=trial,
        plan=plan,
        aligned_source=aligned_state,
        wrong_mode_source=wrong_state,
        mismatched_source=mismatch_state,
        ordinary_patch_state=source_model["input-a"],
        source_mode_indices=(1, 3),
        recipient_mode_indices=(1, 3),
        wrong_mode_indices=(0, 2),
        fourier_adapter=fft,
        ordinary_shape_adapter=IdentityShapeAdapter(),
        mismatch_input_roster=("input-a", "input-b"),
    )
    accounting = {
        condition: account_payload(payload, capacity, scalar_precision_bits=64)
        for condition, payload in payloads.items()
    }
    return SyntheticFixture(
        trial,
        payloads,
        accounting,
        recipient_adapter,
        recipient_template,
        SyntheticOutcomeAdapter(exercise_terminal_paths=terminal_paths),
        plan.plan_sha256,
    )

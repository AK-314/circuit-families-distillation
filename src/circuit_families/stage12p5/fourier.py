"""Injected activation, Fourier-coordinate, alignment, and write adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from circuit_families.stage12p3.records import canonical_sha256

from .contracts import (
    AlignmentProfile,
    FourierCoordinateContract,
    LocationContract,
    Stage12P5ContractError,
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite(value: np.ndarray, *, label: str) -> None:
    if not np.all(np.isfinite(value)):
        raise Stage12P5ContractError(f"{label} contains nonfinite values")


class ActivationAdapter(Protocol):
    architecture_ref: str
    supported_locations: tuple[str, ...]

    def read(self, model: Any, *, input_id: str, location: LocationContract) -> np.ndarray: ...

    def write(
        self,
        model: Any,
        *,
        input_id: str,
        location: LocationContract,
        state: np.ndarray,
    ) -> Mapping[str, Any]: ...


class FourierTransformAdapter(Protocol):
    implementation_ref: str

    def forward(
        self, state: np.ndarray, contract: FourierCoordinateContract
    ) -> np.ndarray: ...

    def inverse(
        self,
        coordinates: np.ndarray,
        contract: FourierCoordinateContract,
        *,
        ordinary_dtype: str,
    ) -> np.ndarray: ...


class ArrayActivationAdapter:
    """Tiny technical adapter with explicit layout conversions and locations."""

    def __init__(
        self,
        *,
        architecture_ref: str,
        supported_locations: tuple[str, ...],
        external_layout_ref: str,
        canonical_axes: tuple[int, ...],
    ) -> None:
        self.architecture_ref = architecture_ref
        self.supported_locations = supported_locations
        self.external_layout_ref = external_layout_ref
        self.canonical_axes = canonical_axes

    def _validate(self, location: LocationContract, *, source: bool) -> tuple[int, ...]:
        architecture = (
            location.source_architecture_ref if source else location.recipient_architecture_ref
        )
        layout = location.source_layout_ref if source else location.recipient_layout_ref
        shape = location.source_shape if source else location.recipient_shape
        if architecture != self.architecture_ref:
            raise Stage12P5ContractError("activation adapter architecture mismatch")
        if location.location_ref not in self.supported_locations:
            raise Stage12P5ContractError("unsupported typed activation location")
        if layout != self.external_layout_ref:
            raise Stage12P5ContractError("activation layout reference mismatch")
        if sorted(self.canonical_axes) != list(range(len(shape))):
            raise Stage12P5ContractError("layout adapter axes are not a permutation")
        return shape

    def read(self, model: Any, *, input_id: str, location: LocationContract) -> np.ndarray:
        expected = self._validate(location, source=True)
        if not isinstance(model, Mapping) or input_id not in model:
            raise Stage12P5ContractError("source activation input identity is unavailable")
        external = np.asarray(model[input_id])
        if external.shape != expected:
            raise Stage12P5ContractError("source activation shape mismatch")
        canonical = np.transpose(external, self.canonical_axes)
        _finite(canonical, label="ordinary activation")
        return np.array(canonical, copy=True)

    def write(
        self,
        model: Any,
        *,
        input_id: str,
        location: LocationContract,
        state: np.ndarray,
    ) -> Mapping[str, Any]:
        expected = self._validate(location, source=False)
        if not isinstance(model, dict) or input_id not in model:
            raise Stage12P5ContractError("recipient activation input identity is unavailable")
        state = np.asarray(state)
        canonical_shape = tuple(expected[index] for index in self.canonical_axes)
        if state.shape != canonical_shape:
            raise Stage12P5ContractError("recipient canonical intervention shape mismatch")
        _finite(state, label="intervention state")
        inverse_axes = tuple(int(index) for index in np.argsort(self.canonical_axes))
        external = np.transpose(state, inverse_axes)
        if external.shape != expected:
            raise Stage12P5ContractError("recipient external write shape mismatch")
        model[input_id] = np.array(external, copy=True)
        return {
            "adapter_architecture_ref": self.architecture_ref,
            "location_ref": location.location_ref,
            "input_id": input_id,
            "canonical_shape": list(state.shape),
            "external_shape": list(external.shape),
            "written_sha256": _array_sha256(external),
            "hook_invocations": 1,
            "scientific_data": False,
            "production_eligible": False,
        }


class NumpyFourierAdapter:
    implementation_ref = "stage12p5-numpy-fftn/v1"

    def forward(
        self, state: np.ndarray, contract: FourierCoordinateContract
    ) -> np.ndarray:
        state = np.asarray(state)
        _finite(state, label="ordinary activation")
        coordinates = np.fft.fftn(
            state.astype(np.float64, copy=False),
            axes=contract.transform_axes,
            norm=contract.normalization,
        )
        _finite(coordinates, label="Fourier coordinates")
        return coordinates

    def inverse(
        self,
        coordinates: np.ndarray,
        contract: FourierCoordinateContract,
        *,
        ordinary_dtype: str,
    ) -> np.ndarray:
        coordinates = np.asarray(coordinates)
        _finite(coordinates, label="Fourier coordinates")
        reconstructed = np.fft.ifftn(
            coordinates,
            axes=contract.transform_axes,
            norm=contract.normalization,
        )
        imaginary_max = float(np.max(np.abs(reconstructed.imag)))
        if imaginary_max > 1e-10:
            raise Stage12P5ContractError("reconstruction has unrecorded complex residue")
        return reconstructed.real.astype(np.dtype(ordinary_dtype), copy=False)


@dataclass(frozen=True)
class ExtractedCoordinateState:
    model_ref: str
    architecture_ref: str
    input_id: str
    location_ref: str
    representation_ref: str
    mode_id: str
    ordinary_shape: tuple[int, ...]
    ordinary_dtype: str
    coordinate_shape: tuple[int, ...]
    coordinate_dtype: str
    ordinary_sha256: str
    coordinate_sha256: str
    norm: float
    values: np.ndarray
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P5ContractError("extracted states must remain technical-only")
        if tuple(self.values.shape) != self.coordinate_shape:
            raise Stage12P5ContractError("extracted coordinate shape mismatch")
        if self.values.dtype.str != self.coordinate_dtype:
            raise Stage12P5ContractError("extracted coordinate dtype mismatch")
        if _array_sha256(self.values) != self.coordinate_sha256:
            raise Stage12P5ContractError("extracted coordinate hash mismatch")
        _finite(self.values, label="extracted coordinates")

    def provenance_mapping(self) -> dict[str, Any]:
        return {
            "model_ref": self.model_ref,
            "architecture_ref": self.architecture_ref,
            "input_id": self.input_id,
            "location_ref": self.location_ref,
            "representation_ref": self.representation_ref,
            "mode_id": self.mode_id,
            "ordinary_shape": list(self.ordinary_shape),
            "ordinary_dtype": self.ordinary_dtype,
            "coordinate_shape": list(self.coordinate_shape),
            "coordinate_dtype": self.coordinate_dtype,
            "ordinary_sha256": self.ordinary_sha256,
            "coordinate_sha256": self.coordinate_sha256,
            "norm": self.norm,
            "scientific_data": False,
            "production_eligible": False,
        }


def extract_coordinates(
    *,
    model: Any,
    model_ref: str,
    architecture_ref: str,
    input_id: str,
    location: LocationContract,
    fourier: FourierCoordinateContract,
    mode_id: str,
    activation_adapter: ActivationAdapter,
    fourier_adapter: FourierTransformAdapter,
) -> ExtractedCoordinateState:
    ordinary = activation_adapter.read(model, input_id=input_id, location=location)
    coordinates = fourier_adapter.forward(ordinary, fourier)
    return ExtractedCoordinateState(
        model_ref=model_ref,
        architecture_ref=architecture_ref,
        input_id=input_id,
        location_ref=location.location_ref,
        representation_ref=fourier.representation_ref,
        mode_id=mode_id,
        ordinary_shape=tuple(ordinary.shape),
        ordinary_dtype=ordinary.dtype.str,
        coordinate_shape=tuple(coordinates.shape),
        coordinate_dtype=coordinates.dtype.str,
        ordinary_sha256=_array_sha256(ordinary),
        coordinate_sha256=_array_sha256(coordinates),
        norm=float(np.linalg.norm(coordinates)),
        values=np.array(coordinates, copy=True),
    )


@dataclass(frozen=True)
class AlignmentPlan:
    profile: AlignmentProfile
    pair_id: str
    matrix: np.ndarray
    source_rank: int
    mapped_rank: int
    fit_residual_norm: float
    matrix_sha256: str
    plan_sha256: str
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        expected = (
            self.profile.recipient_coordinate_count,
            self.profile.source_coordinate_count,
        )
        if self.matrix.shape != expected:
            raise Stage12P5ContractError("alignment matrix dimension mismatch")
        _finite(self.matrix, label="alignment matrix")
        if _array_sha256(self.matrix) != self.matrix_sha256:
            raise Stage12P5ContractError("alignment matrix hash mismatch")
        if self.plan_sha256 != canonical_sha256(self.identity_mapping()):
            raise Stage12P5ContractError("alignment plan hash mismatch")
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P5ContractError("alignment plan must remain technical-only")

    def identity_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "stage12p5-alignment-plan/v1",
            "alignment_profile_sha256": self.profile.profile_sha256,
            "pair_id": self.pair_id,
            "matrix_shape": list(self.matrix.shape),
            "matrix_dtype": self.matrix.dtype.str,
            "matrix_sha256": self.matrix_sha256,
            "source_rank": self.source_rank,
            "mapped_rank": self.mapped_rank,
            "fit_residual_norm": self.fit_residual_norm,
            "scientific_data": False,
            "production_eligible": False,
        }

    def apply(self, source: np.ndarray, *, pair_id: str) -> np.ndarray:
        if pair_id != self.pair_id:
            raise Stage12P5ContractError("alignment plan reused across incompatible pair")
        flattened = np.asarray(source).reshape(-1)
        if flattened.size != self.profile.source_coordinate_count:
            raise Stage12P5ContractError("alignment source rank/shape mismatch")
        mapped = self.matrix @ flattened
        _finite(mapped, label="aligned coordinates")
        return mapped


def fit_linear_alignment(
    *,
    profile: AlignmentProfile,
    pair_id: str,
    source_fit: np.ndarray,
    recipient_fit: np.ndarray,
) -> AlignmentPlan:
    """Fit an injected technical least-squares plan without trial outcomes."""
    source = np.asarray(source_fit)
    recipient = np.asarray(recipient_fit)
    if source.ndim != 2 or recipient.ndim != 2 or source.shape[0] != recipient.shape[0]:
        raise Stage12P5ContractError("alignment fit arrays require matched example rows")
    if source.shape[1] != profile.source_coordinate_count:
        raise Stage12P5ContractError("alignment source coordinate count mismatch")
    if recipient.shape[1] != profile.recipient_coordinate_count:
        raise Stage12P5ContractError("alignment recipient coordinate count mismatch")
    _finite(source, label="alignment source fit data")
    _finite(recipient, label="alignment recipient fit data")
    solution, _, source_rank, _ = np.linalg.lstsq(source, recipient, rcond=None)
    matrix = np.ascontiguousarray(solution.T)
    prediction = source @ solution
    residual = float(np.linalg.norm(prediction - recipient))
    matrix_sha = _array_sha256(matrix)
    material = {
        "schema_version": "stage12p5-alignment-plan/v1",
        "alignment_profile_sha256": profile.profile_sha256,
        "pair_id": pair_id,
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": matrix.dtype.str,
        "matrix_sha256": matrix_sha,
        "source_rank": int(source_rank),
        "mapped_rank": int(np.linalg.matrix_rank(matrix)),
        "fit_residual_norm": residual,
        "scientific_data": False,
        "production_eligible": False,
    }
    return AlignmentPlan(
        profile=profile,
        pair_id=pair_id,
        matrix=matrix,
        source_rank=int(source_rank),
        mapped_rank=int(np.linalg.matrix_rank(matrix)),
        fit_residual_norm=residual,
        matrix_sha256=matrix_sha,
        plan_sha256=canonical_sha256(material),
    )


@dataclass(frozen=True)
class InterventionPayload:
    condition: str
    trial_id: str
    pair_id: str
    recipient_input_id: str
    location_ref: str
    coordinate_ids: tuple[str, ...]
    coordinate_values: np.ndarray
    ordinary_state: np.ndarray
    source_input_id: str
    mode_id: str
    side_information: tuple[str, ...]
    external_identifiers: tuple[str, ...]
    alignment_plan_sha256: str | None
    construction_evidence: Mapping[str, Any]
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        _finite(self.coordinate_values, label="intervention coordinates")
        _finite(self.ordinary_state, label="intervention ordinary state")
        if self.scientific_data is not False or self.production_eligible is not False:
            raise Stage12P5ContractError("intervention payload must remain technical-only")

    @property
    def payload_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema_version": "stage12p5-intervention-payload/v1",
                "condition": self.condition,
                "trial_id": self.trial_id,
                "pair_id": self.pair_id,
                "recipient_input_id": self.recipient_input_id,
                "location_ref": self.location_ref,
                "coordinate_ids": list(self.coordinate_ids),
                "coordinate_shape": list(self.coordinate_values.shape),
                "coordinate_dtype": self.coordinate_values.dtype.str,
                "coordinate_sha256": _array_sha256(self.coordinate_values),
                "ordinary_shape": list(self.ordinary_state.shape),
                "ordinary_dtype": self.ordinary_state.dtype.str,
                "ordinary_sha256": _array_sha256(self.ordinary_state),
                "source_input_id": self.source_input_id,
                "mode_id": self.mode_id,
                "side_information": list(self.side_information),
                "external_identifiers": list(self.external_identifiers),
                "alignment_plan_sha256": self.alignment_plan_sha256,
                "construction_evidence": dict(self.construction_evidence),
                "scientific_data": False,
                "production_eligible": False,
            }
        )

"""Closed, policy-neutral contracts for Stage 12-P5 Fourier interchange."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final

from circuit_families.stage12p3.records import canonical_sha256, require_reference, require_sha256

CONTRACT_VERSION: Final = "stage12p5-trial-contract/v1"
PAIR_VERSION: Final = "stage12p5-pair/v1"
LOCATION_VERSION: Final = "stage12p5-location/v1"
FOURIER_VERSION: Final = "stage12p5-fourier-coordinate/v1"
ALIGNMENT_VERSION: Final = "stage12p5-alignment-profile/v1"
CAPACITY_VERSION: Final = "stage12p5-capacity/v1"

CONDITIONS: Final = (
    "aligned_fourier_interchange",
    "wrong_fourier_mode",
    "shuffled_coefficients",
    "mismatched_input",
    "equal_norm_random_state",
    "unaligned_ordinary_activation_patching",
)
LIFECYCLE_STATES: Final = frozenset(
    {"planned", "running", "complete", "failed", "unavailable", "censored"}
)
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+%-]*\Z")


class Stage12P5ContractError(ValueError):
    """Raised when a Stage 12-P5 record violates its closed contract."""


def _false_boundary(scientific_data: bool, production_eligible: bool) -> None:
    if scientific_data is not False or production_eligible is not False:
        raise Stage12P5ContractError(
            "Stage 12-P5 records require scientific_data=false and production_eligible=false"
        )


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise Stage12P5ContractError(f"{label} must be a non-empty canonical identifier")
    return value


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise Stage12P5ContractError(f"{label} must be a {qualifier} integer")
    return value


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise Stage12P5ContractError(f"{label} must be a non-empty tuple")
    for dimension in value:
        _positive_int(dimension, label=f"{label} dimension")
    return value


def _closed(mapping: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if not isinstance(mapping, Mapping) or set(mapping) != expected:
        raise Stage12P5ContractError(f"{label} fields do not match the closed schema")


@dataclass(frozen=True)
class ModelReference:
    role: str
    model_ref: str
    architecture_ref: str
    architecture_sha256: str
    checkpoint_ref: str
    checkpoint_sha256: str
    distillation_condition: str

    def __post_init__(self) -> None:
        if self.role not in {"source_teacher", "recipient_student"}:
            raise Stage12P5ContractError("model role must be source_teacher or recipient_student")
        for label in ("model_ref", "architecture_ref", "checkpoint_ref"):
            require_reference(getattr(self, label), label=label)
        require_sha256(self.architecture_sha256, label="architecture_sha256")
        require_sha256(self.checkpoint_sha256, label="checkpoint_sha256")
        if self.role == "source_teacher" and self.distillation_condition != "direct_teacher":
            raise Stage12P5ContractError("source teacher must use direct_teacher condition")
        if self.role == "recipient_student" and self.distillation_condition not in {
            "hard_target",
            "soft_target",
        }:
            raise Stage12P5ContractError("recipient student must retain hard/soft estimand")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairContract:
    source: ModelReference
    recipient: ModelReference
    selection_evidence_ref: str
    selection_evidence_sha256: str
    selection_rule_ref: str
    outcome_independent: bool = True
    candidate_outcomes_consulted: bool = False
    schema_version: str = PAIR_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != PAIR_VERSION:
            raise Stage12P5ContractError("unsupported pair schema")
        if self.source.role != "source_teacher" or self.recipient.role != "recipient_student":
            raise Stage12P5ContractError("pair source/recipient roles are reversed or ambiguous")
        if self.source.model_ref == self.recipient.model_ref:
            raise Stage12P5ContractError("source and recipient model identities must be distinct")
        require_reference(self.selection_evidence_ref, label="selection_evidence_ref")
        require_reference(self.selection_rule_ref, label="selection_rule_ref")
        require_sha256(self.selection_evidence_sha256, label="selection_evidence_sha256")
        if self.outcome_independent is not True or self.candidate_outcomes_consulted is not False:
            raise Stage12P5ContractError(
                "pair selection must be created without candidate outcomes"
            )
        _false_boundary(self.scientific_data, self.production_eligible)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_mapping(),
            "recipient": self.recipient.to_mapping(),
            "selection_evidence_ref": self.selection_evidence_ref,
            "selection_evidence_sha256": self.selection_evidence_sha256,
            "selection_rule_ref": self.selection_rule_ref,
            "outcome_independent": True,
            "candidate_outcomes_consulted": False,
            "scientific_data": False,
            "production_eligible": False,
        }

    @property
    def pair_id(self) -> str:
        return canonical_sha256(self.identity_payload())

    def to_mapping(self) -> dict[str, Any]:
        return {"pair_id": self.pair_id, **self.identity_payload()}


@dataclass(frozen=True)
class LocationContract:
    location_ref: str
    source_architecture_ref: str
    recipient_architecture_ref: str
    source_hook_name: str
    recipient_hook_name: str
    source_layout_ref: str
    recipient_layout_ref: str
    source_shape: tuple[int, ...]
    recipient_shape: tuple[int, ...]
    schema_version: str = LOCATION_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != LOCATION_VERSION:
            raise Stage12P5ContractError("unsupported location schema")
        for label in (
            "location_ref",
            "source_architecture_ref",
            "recipient_architecture_ref",
            "source_hook_name",
            "recipient_hook_name",
            "source_layout_ref",
            "recipient_layout_ref",
        ):
            require_reference(getattr(self, label), label=label)
        _shape(self.source_shape, label="source_shape")
        _shape(self.recipient_shape, label="recipient_shape")
        _false_boundary(self.scientific_data, self.production_eligible)

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_shape"] = list(self.source_shape)
        value["recipient_shape"] = list(self.recipient_shape)
        return value


@dataclass(frozen=True)
class FourierCoordinateContract:
    representation_ref: str
    coordinate_universe_ref: str
    convention_ref: str
    normalization: str
    transform_axes: tuple[int, ...]
    complex_encoding: str
    conjugate_handling: str
    gauge_ref: str
    schema_version: str = FOURIER_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != FOURIER_VERSION:
            raise Stage12P5ContractError("unsupported Fourier-coordinate schema")
        for label in (
            "representation_ref",
            "coordinate_universe_ref",
            "convention_ref",
            "gauge_ref",
        ):
            require_reference(getattr(self, label), label=label)
        if self.normalization not in {"backward", "forward", "ortho"}:
            raise Stage12P5ContractError("ambiguous Fourier normalization")
        if not self.transform_axes or len(set(self.transform_axes)) != len(self.transform_axes):
            raise Stage12P5ContractError("transform axes must be non-empty and unique")
        if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in self.transform_axes):
            raise Stage12P5ContractError("transform axes must be integers")
        if self.complex_encoding not in {"complex", "real_imag_pairs"}:
            raise Stage12P5ContractError("complex encoding must be explicit")
        if self.conjugate_handling not in {"full_spectrum", "declared_half_spectrum"}:
            raise Stage12P5ContractError("conjugate handling must be explicit")
        _false_boundary(self.scientific_data, self.production_eligible)

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["transform_axes"] = list(self.transform_axes)
        return value


@dataclass(frozen=True)
class AlignmentProfile:
    alignment_ref: str
    implementation_sha256: str
    source_architecture_ref: str
    recipient_architecture_ref: str
    source_coordinate_count: int
    recipient_coordinate_count: int
    fit_data_ref: str
    fit_data_sha256: str
    fit_boundary_ref: str
    outcome_independent: bool = True
    schema_version: str = ALIGNMENT_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != ALIGNMENT_VERSION:
            raise Stage12P5ContractError("unsupported alignment profile schema")
        for label in (
            "alignment_ref",
            "source_architecture_ref",
            "recipient_architecture_ref",
            "fit_data_ref",
            "fit_boundary_ref",
        ):
            require_reference(getattr(self, label), label=label)
        require_sha256(self.implementation_sha256, label="implementation_sha256")
        require_sha256(self.fit_data_sha256, label="fit_data_sha256")
        _positive_int(self.source_coordinate_count, label="source_coordinate_count")
        _positive_int(self.recipient_coordinate_count, label="recipient_coordinate_count")
        if self.outcome_independent is not True:
            raise Stage12P5ContractError("alignment fit must be outcome-independent")
        _false_boundary(self.scientific_data, self.production_eligible)

    @property
    def profile_sha256(self) -> str:
        return canonical_sha256(self.to_mapping())

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapacityContract:
    capacity_ref: str
    coordinate_universe_ref: str
    allowed_coordinate_ids: tuple[str, ...]
    real_degrees_of_freedom: int
    maximum_rank: int
    maximum_support: int
    scalar_precision_bits: int
    quantization_ref: str
    allowed_side_information: tuple[str, ...]
    external_identifier_budget: tuple[str, ...]
    recipient_location_ref: str
    recipient_shape: tuple[int, ...]
    write_budget_scalars: int
    padding_rule: str
    variable_length_payloads: bool = False
    limitation: str = (
        "Prespecified operational information allowance; not a universal "
        "information-theoretic channel-capacity claim."
    )
    schema_version: str = CAPACITY_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CAPACITY_VERSION:
            raise Stage12P5ContractError("unsupported capacity schema")
        for label in (
            "capacity_ref",
            "coordinate_universe_ref",
            "quantization_ref",
            "recipient_location_ref",
            "padding_rule",
        ):
            require_reference(getattr(self, label), label=label)
        coordinates = tuple(self.allowed_coordinate_ids)
        if not coordinates or len(set(coordinates)) != len(coordinates):
            raise Stage12P5ContractError("allowed coordinates must be non-empty and unique")
        for coordinate in coordinates:
            _identifier(coordinate, label="coordinate identifier")
        if coordinates != tuple(sorted(coordinates)):
            raise Stage12P5ContractError("allowed coordinates must use canonical sorted order")
        for label in (
            "real_degrees_of_freedom",
            "maximum_rank",
            "maximum_support",
            "scalar_precision_bits",
            "write_budget_scalars",
        ):
            _positive_int(getattr(self, label), label=label)
        if self.maximum_support > len(coordinates):
            raise Stage12P5ContractError("maximum support exceeds coordinate universe")
        _shape(self.recipient_shape, label="recipient_shape")
        for label in ("allowed_side_information", "external_identifier_budget"):
            values = tuple(getattr(self, label))
            if len(set(values)) != len(values) or values != tuple(sorted(values)):
                raise Stage12P5ContractError(f"{label} must be unique and sorted")
            for item in values:
                _identifier(item, label=label)
        if self.variable_length_payloads is not False:
            raise Stage12P5ContractError("state-dependent variable-length payloads are forbidden")
        if not isinstance(self.limitation, str) or "not a universal" not in self.limitation:
            raise Stage12P5ContractError("capacity limitation must remain explicit")
        _false_boundary(self.scientific_data, self.production_eligible)

    @property
    def capacity_sha256(self) -> str:
        return canonical_sha256(self.to_mapping())

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        for label in (
            "allowed_coordinate_ids",
            "allowed_side_information",
            "external_identifier_budget",
            "recipient_shape",
        ):
            value[label] = list(getattr(self, label))
        return value


@dataclass(frozen=True)
class TrialContract:
    pair: PairContract
    input_set_ref: str
    input_set_sha256: str
    source_input_id: str
    recipient_input_id: str
    location: LocationContract
    fourier: FourierCoordinateContract
    source_mode_id: str
    recipient_mode_id: str
    alignment: AlignmentProfile
    capacity: CapacityContract
    outcome_adapter_ref: str
    outcome_adapter_sha256: str
    comparison_set_ref: str
    root_seed: int
    seed_namespace_ref: str
    conditions: tuple[str, ...] = CONDITIONS
    lifecycle_state: str = "planned"
    schema_version: str = CONTRACT_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise Stage12P5ContractError("unsupported trial schema")
        for label in (
            "input_set_ref",
            "outcome_adapter_ref",
            "comparison_set_ref",
            "seed_namespace_ref",
        ):
            require_reference(getattr(self, label), label=label)
        require_sha256(self.input_set_sha256, label="input_set_sha256")
        require_sha256(self.outcome_adapter_sha256, label="outcome_adapter_sha256")
        _identifier(self.source_input_id, label="source_input_id")
        _identifier(self.recipient_input_id, label="recipient_input_id")
        _identifier(self.source_mode_id, label="source_mode_id")
        _identifier(self.recipient_mode_id, label="recipient_mode_id")
        _positive_int(self.root_seed, label="root_seed", allow_zero=True)
        if self.conditions != CONDITIONS:
            raise Stage12P5ContractError("trial must contain exactly aligned plus five controls")
        if self.lifecycle_state not in LIFECYCLE_STATES:
            raise Stage12P5ContractError("unknown trial lifecycle state")
        if self.location.source_architecture_ref != self.pair.source.architecture_ref:
            raise Stage12P5ContractError("source location architecture mismatch")
        if self.location.recipient_architecture_ref != self.pair.recipient.architecture_ref:
            raise Stage12P5ContractError("recipient location architecture mismatch")
        if self.alignment.source_architecture_ref != self.pair.source.architecture_ref:
            raise Stage12P5ContractError("alignment source architecture mismatch")
        if self.alignment.recipient_architecture_ref != self.pair.recipient.architecture_ref:
            raise Stage12P5ContractError("alignment recipient architecture mismatch")
        if self.capacity.coordinate_universe_ref != self.fourier.coordinate_universe_ref:
            raise Stage12P5ContractError("capacity/Fourier coordinate universe mismatch")
        if self.capacity.recipient_location_ref != self.location.location_ref:
            raise Stage12P5ContractError("capacity recipient location mismatch")
        if self.capacity.recipient_shape != self.location.recipient_shape:
            raise Stage12P5ContractError("capacity recipient shape mismatch")
        _false_boundary(self.scientific_data, self.production_eligible)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair": self.pair.to_mapping(),
            "input_set_ref": self.input_set_ref,
            "input_set_sha256": self.input_set_sha256,
            "source_input_id": self.source_input_id,
            "recipient_input_id": self.recipient_input_id,
            "location": self.location.to_mapping(),
            "fourier": self.fourier.to_mapping(),
            "source_mode_id": self.source_mode_id,
            "recipient_mode_id": self.recipient_mode_id,
            "alignment": self.alignment.to_mapping(),
            "alignment_profile_sha256": self.alignment.profile_sha256,
            "capacity": self.capacity.to_mapping(),
            "capacity_sha256": self.capacity.capacity_sha256,
            "outcome_adapter_ref": self.outcome_adapter_ref,
            "outcome_adapter_sha256": self.outcome_adapter_sha256,
            "comparison_set_ref": self.comparison_set_ref,
            "root_seed": self.root_seed,
            "seed_namespace_ref": self.seed_namespace_ref,
            "conditions": list(self.conditions),
            "scientific_data": False,
            "production_eligible": False,
        }

    @property
    def trial_id(self) -> str:
        return canonical_sha256(self.identity_payload())

    @property
    def comparison_set_id(self) -> str:
        return canonical_sha256(
            {
                "schema_version": "stage12p5-comparison-set/v1",
                "trial_id": self.trial_id,
                "comparison_set_ref": self.comparison_set_ref,
                "conditions": list(CONDITIONS),
            }
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "comparison_set_id": self.comparison_set_id,
            "lifecycle_state": self.lifecycle_state,
            **self.identity_payload(),
        }


def _model_from_mapping(value: Mapping[str, Any]) -> ModelReference:
    expected = {
        "role",
        "model_ref",
        "architecture_ref",
        "architecture_sha256",
        "checkpoint_ref",
        "checkpoint_sha256",
        "distillation_condition",
    }
    _closed(value, expected, label="model reference")
    return ModelReference(**dict(value))


def trial_from_mapping(value: Mapping[str, Any]) -> TrialContract:
    """Parse the closed outer contract and verify all nested hashes and identities."""
    expected = {
        "trial_id",
        "comparison_set_id",
        "lifecycle_state",
        "schema_version",
        "pair",
        "input_set_ref",
        "input_set_sha256",
        "source_input_id",
        "recipient_input_id",
        "location",
        "fourier",
        "source_mode_id",
        "recipient_mode_id",
        "alignment",
        "alignment_profile_sha256",
        "capacity",
        "capacity_sha256",
        "outcome_adapter_ref",
        "outcome_adapter_sha256",
        "comparison_set_ref",
        "root_seed",
        "seed_namespace_ref",
        "conditions",
        "scientific_data",
        "production_eligible",
    }
    _closed(value, expected, label="trial")
    data = copy.deepcopy(dict(value))
    pair_data = data["pair"]
    pair_expected = {
        "pair_id",
        "schema_version",
        "source",
        "recipient",
        "selection_evidence_ref",
        "selection_evidence_sha256",
        "selection_rule_ref",
        "outcome_independent",
        "candidate_outcomes_consulted",
        "scientific_data",
        "production_eligible",
    }
    _closed(pair_data, pair_expected, label="pair")
    pair = PairContract(
        source=_model_from_mapping(pair_data["source"]),
        recipient=_model_from_mapping(pair_data["recipient"]),
        **{key: pair_data[key] for key in pair_expected - {"pair_id", "source", "recipient"}},
    )
    if pair_data["pair_id"] != pair.pair_id:
        raise Stage12P5ContractError("pair identity/hash mismatch")
    location_data = data["location"]
    location_expected = {
        "location_ref",
        "source_architecture_ref",
        "recipient_architecture_ref",
        "source_hook_name",
        "recipient_hook_name",
        "source_layout_ref",
        "recipient_layout_ref",
        "source_shape",
        "recipient_shape",
        "schema_version",
        "scientific_data",
        "production_eligible",
    }
    _closed(location_data, location_expected, label="location")
    location_data["source_shape"] = tuple(location_data["source_shape"])
    location_data["recipient_shape"] = tuple(location_data["recipient_shape"])
    fourier_data = data["fourier"]
    fourier_expected = {
        "representation_ref",
        "coordinate_universe_ref",
        "convention_ref",
        "normalization",
        "transform_axes",
        "complex_encoding",
        "conjugate_handling",
        "gauge_ref",
        "schema_version",
        "scientific_data",
        "production_eligible",
    }
    _closed(fourier_data, fourier_expected, label="Fourier coordinate")
    fourier_data["transform_axes"] = tuple(fourier_data["transform_axes"])
    alignment_data = data["alignment"]
    _closed(alignment_data, set(AlignmentProfile.__dataclass_fields__), label="alignment")
    capacity_data = data["capacity"]
    _closed(capacity_data, set(CapacityContract.__dataclass_fields__), label="capacity")
    for label in (
        "allowed_coordinate_ids",
        "allowed_side_information",
        "external_identifier_budget",
        "recipient_shape",
    ):
        capacity_data[label] = tuple(capacity_data[label])
    trial = TrialContract(
        pair=pair,
        location=LocationContract(**location_data),
        fourier=FourierCoordinateContract(**fourier_data),
        alignment=AlignmentProfile(**alignment_data),
        capacity=CapacityContract(**capacity_data),
        input_set_ref=data["input_set_ref"],
        input_set_sha256=data["input_set_sha256"],
        source_input_id=data["source_input_id"],
        recipient_input_id=data["recipient_input_id"],
        source_mode_id=data["source_mode_id"],
        recipient_mode_id=data["recipient_mode_id"],
        outcome_adapter_ref=data["outcome_adapter_ref"],
        outcome_adapter_sha256=data["outcome_adapter_sha256"],
        comparison_set_ref=data["comparison_set_ref"],
        root_seed=data["root_seed"],
        seed_namespace_ref=data["seed_namespace_ref"],
        conditions=tuple(data["conditions"]),
        lifecycle_state=data["lifecycle_state"],
        schema_version=data["schema_version"],
        scientific_data=data["scientific_data"],
        production_eligible=data["production_eligible"],
    )
    if data["alignment_profile_sha256"] != trial.alignment.profile_sha256:
        raise Stage12P5ContractError("alignment profile hash mismatch")
    if data["capacity_sha256"] != trial.capacity.capacity_sha256:
        raise Stage12P5ContractError("capacity hash mismatch")
    if data["trial_id"] != trial.trial_id or data["comparison_set_id"] != trial.comparison_set_id:
        raise Stage12P5ContractError("trial or comparison-set identity/hash mismatch")
    return trial


def finite_json_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage12P5ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Stage12P5ContractError(f"{label} must be finite")
    return result

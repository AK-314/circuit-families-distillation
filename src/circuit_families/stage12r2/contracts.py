"""Stage 12-R2 versioned basis identity and technical comparison contracts.

Technical-only machinery.  These records must never be marked as scientific
data or production eligible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Literal

BASIS_CONTRACT_VERSION = "stage12r2-basis/v1"
BASIS_MASK_VERSION = "stage12r2-basis-mask/v1"

BasisRelationshipKind = Literal[
    "same_basis",
    "refinement",
    "coarsening",
    "rotated_view",
]


def canonical_json_bytes(payload: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for compact identity payloads."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    """Hash a canonical compact JSON payload."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_hash(name: str, value: str) -> str:
    _require_nonempty(name, value)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class BasisComponentDescriptor:
    component_id: str
    component_type: str
    source_subspace: str
    intervention_location: str
    parameter_weight: int
    coordinate_identity: str
    parent_component_identity: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "component_id",
            "component_type",
            "source_subspace",
            "intervention_location",
            "coordinate_identity",
        ):
            _require_nonempty(name, getattr(self, name))
        if self.parent_component_identity is not None:
            _require_nonempty(
                "parent_component_identity",
                self.parent_component_identity,
            )
        if not isinstance(self.parameter_weight, int) or self.parameter_weight < 0:
            raise ValueError("parameter_weight must be a non-negative integer")

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "source_subspace": self.source_subspace,
            "intervention_location": self.intervention_location,
            "parameter_weight": self.parameter_weight,
            "coordinate_identity": self.coordinate_identity,
            "parent_component_identity": self.parent_component_identity,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BasisComponentDescriptor:
        expected = {
            "component_id",
            "component_type",
            "source_subspace",
            "intervention_location",
            "parameter_weight",
            "coordinate_identity",
            "parent_component_identity",
        }
        if set(record) != expected:
            raise ValueError("component descriptor fields do not match contract")
        parent = record["parent_component_identity"]
        return cls(
            component_id=str(record["component_id"]),
            component_type=str(record["component_type"]),
            source_subspace=str(record["source_subspace"]),
            intervention_location=str(record["intervention_location"]),
            parameter_weight=int(record["parameter_weight"]),
            coordinate_identity=str(record["coordinate_identity"]),
            parent_component_identity=None if parent is None else str(parent),
        )


@dataclass(frozen=True)
class BasisRelationship:
    kind: BasisRelationshipKind
    parent_basis_hash: str
    mapping_identity: str

    def __post_init__(self) -> None:
        if self.kind not in {
            "same_basis",
            "refinement",
            "coarsening",
            "rotated_view",
        }:
            raise ValueError("unsupported basis relationship")
        _require_hash("parent_basis_hash", self.parent_basis_hash)
        _require_nonempty("mapping_identity", self.mapping_identity)

    def to_record(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "parent_basis_hash": self.parent_basis_hash,
            "mapping_identity": self.mapping_identity,
        }


@dataclass(frozen=True)
class BasisContract:
    parent_model_identity: str
    parent_component_basis_identity: str
    basis_family: str
    coordinate_definition: str
    components: tuple[BasisComponentDescriptor, ...]
    intervention_location: str
    intervention_semantics: str
    parameter_weight_denominator_definition: str
    raw_component_denominator_definition: str
    relationship: BasisRelationship | None = None
    grouping_partition_identity: str | None = None
    rotation_subspace_identity: str | None = None
    display_label: str | None = None
    version: str = BASIS_CONTRACT_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.version != BASIS_CONTRACT_VERSION:
            raise ValueError(f"version must equal {BASIS_CONTRACT_VERSION!r}")
        for name in (
            "parent_model_identity",
            "parent_component_basis_identity",
            "basis_family",
            "coordinate_definition",
            "intervention_location",
            "intervention_semantics",
            "parameter_weight_denominator_definition",
            "raw_component_denominator_definition",
        ):
            _require_nonempty(name, getattr(self, name))
        if not self.components:
            raise ValueError("basis must contain at least one component")
        ids = [component.component_id for component in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("component identities must be unique")
        coordinates = [component.coordinate_identity for component in self.components]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("coordinate identities must be unique")
        if self.scientific_data:
            raise ValueError("Stage 12-R2 requires scientific_data=false")
        if self.production_eligible:
            raise ValueError("Stage 12-R2 requires production_eligible=false")
        if self.grouping_partition_identity is not None:
            _require_nonempty(
                "grouping_partition_identity",
                self.grouping_partition_identity,
            )
        if self.rotation_subspace_identity is not None:
            _require_nonempty(
                "rotation_subspace_identity",
                self.rotation_subspace_identity,
            )

        relationship = self.relationship
        if relationship is not None:
            if relationship.kind == "same_basis":
                raise ValueError(
                    "same_basis must use identical basis hash, not a parent relationship"
                )
            if relationship.kind in {"refinement", "coarsening"}:
                if self.grouping_partition_identity is None:
                    raise ValueError(
                        "refinement/coarsening requires grouping_partition_identity"
                    )
            if relationship.kind == "rotated_view":
                if self.rotation_subspace_identity is None:
                    raise ValueError(
                        "rotated_view requires rotation_subspace_identity"
                    )

    def identity_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "parent_model_identity": self.parent_model_identity,
            "parent_component_basis_identity": self.parent_component_basis_identity,
            "basis_family": self.basis_family,
            "coordinate_definition": self.coordinate_definition,
            "components": [component.to_record() for component in self.components],
            "intervention_location": self.intervention_location,
            "intervention_semantics": self.intervention_semantics,
            "parameter_weight_denominator_definition":
                self.parameter_weight_denominator_definition,
            "raw_component_denominator_definition":
                self.raw_component_denominator_definition,
            "relationship": (
                None if self.relationship is None else self.relationship.to_record()
            ),
            "grouping_partition_identity": self.grouping_partition_identity,
            "rotation_subspace_identity": self.rotation_subspace_identity,
            "scientific_data": self.scientific_data,
            "production_eligible": self.production_eligible,
        }

    @property
    def basis_hash(self) -> str:
        return canonical_sha256(self.identity_payload())

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def parameter_weight_denominator(self) -> int:
        return sum(component.parameter_weight for component in self.components)

    def to_record(self) -> dict[str, object]:
        record = self.identity_payload()
        record["display_label"] = self.display_label
        record["basis_hash"] = self.basis_hash
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BasisContract:
        expected = {
            "version",
            "parent_model_identity",
            "parent_component_basis_identity",
            "basis_family",
            "coordinate_definition",
            "components",
            "intervention_location",
            "intervention_semantics",
            "parameter_weight_denominator_definition",
            "raw_component_denominator_definition",
            "relationship",
            "grouping_partition_identity",
            "rotation_subspace_identity",
            "display_label",
            "scientific_data",
            "production_eligible",
            "basis_hash",
        }
        if set(record) != expected:
            raise ValueError("basis record fields do not match contract")

        components_raw = record["components"]
        if not isinstance(components_raw, Sequence) or isinstance(
            components_raw, (str, bytes)
        ):
            raise ValueError("components must be a sequence")

        relationship_raw = record["relationship"]
        relationship = None
        if relationship_raw is not None:
            if not isinstance(relationship_raw, Mapping):
                raise ValueError("relationship must be a mapping or null")
            relationship = BasisRelationship(
                kind=str(relationship_raw["kind"]),  # type: ignore[arg-type]
                parent_basis_hash=str(relationship_raw["parent_basis_hash"]),
                mapping_identity=str(relationship_raw["mapping_identity"]),
            )

        obj = cls(
            version=str(record["version"]),
            parent_model_identity=str(record["parent_model_identity"]),
            parent_component_basis_identity=str(
                record["parent_component_basis_identity"]
            ),
            basis_family=str(record["basis_family"]),
            coordinate_definition=str(record["coordinate_definition"]),
            components=tuple(
                BasisComponentDescriptor.from_record(component)
                for component in components_raw
                if isinstance(component, Mapping)
            ),
            intervention_location=str(record["intervention_location"]),
            intervention_semantics=str(record["intervention_semantics"]),
            parameter_weight_denominator_definition=str(
                record["parameter_weight_denominator_definition"]
            ),
            raw_component_denominator_definition=str(
                record["raw_component_denominator_definition"]
            ),
            relationship=relationship,
            grouping_partition_identity=(
                None
                if record["grouping_partition_identity"] is None
                else str(record["grouping_partition_identity"])
            ),
            rotation_subspace_identity=(
                None
                if record["rotation_subspace_identity"] is None
                else str(record["rotation_subspace_identity"])
            ),
            display_label=(
                None if record["display_label"] is None else str(record["display_label"])
            ),
            scientific_data=bool(record["scientific_data"]),
            production_eligible=bool(record["production_eligible"]),
        )
        supplied_hash = str(record["basis_hash"])
        _require_hash("basis_hash", supplied_hash)
        if supplied_hash != obj.basis_hash:
            raise ValueError("basis hash does not match canonical identity payload")
        return obj


@dataclass(frozen=True)
class BasisMask:
    basis_hash: str
    values: tuple[int, ...]
    version: str = BASIS_MASK_VERSION

    def __post_init__(self) -> None:
        if self.version != BASIS_MASK_VERSION:
            raise ValueError(f"version must equal {BASIS_MASK_VERSION!r}")
        _require_hash("basis_hash", self.basis_hash)
        if any(value not in (0, 1) for value in self.values):
            raise ValueError("mask values must be binary")

    def validate_for(self, basis: BasisContract) -> None:
        if self.basis_hash != basis.basis_hash:
            raise ValueError("mask basis identity does not match target basis")
        if len(self.values) != basis.component_count:
            raise ValueError("mask length does not match target basis component count")


def validate_relationship(
    child: BasisContract,
    parent: BasisContract,
) -> None:
    relationship = child.relationship
    if relationship is None:
        raise ValueError("child basis does not declare a parent relationship")
    if relationship.parent_basis_hash != parent.basis_hash:
        raise ValueError("declared parent basis hash does not match supplied parent")
    if child.parent_model_identity != parent.parent_model_identity:
        raise ValueError("basis relationship crosses model identity")
    if relationship.kind == "refinement":
        valid_parent_ids = {component.component_id for component in parent.components}
        if any(
            component.parent_component_identity not in valid_parent_ids
            for component in child.components
        ):
            raise ValueError("refinement component has invalid parent identity")


def validate_technical_record_payload(
    payload: Any,
    *,
    max_sequence_items: int = 4096,
) -> None:
    """Reject unsafe/non-technical record content before serialization."""

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            if value.get("scientific_data") is True:
                raise ValueError("scientific_data=true is prohibited")
            if value.get("production_eligible") is True:
                raise ValueError("production_eligible=true is prohibited")
            for key, item in value.items():
                walk(item, f"{path}.{key}")
            return

        if isinstance(value, (list, tuple)):
            if len(value) > max_sequence_items:
                raise ValueError("large tensor-like payloads are prohibited")
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
            return

        if isinstance(value, str):
            if value.startswith(("/Users/", "/home/", "/private/", "/tmp/")):
                raise ValueError("absolute private filesystem paths are prohibited")
            if PurePath(value).is_absolute():
                raise ValueError("absolute filesystem paths are prohibited")

    walk(payload, "$")

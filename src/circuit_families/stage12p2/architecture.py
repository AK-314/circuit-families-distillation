"""Policy-neutral Stage 12-P2 architecture records and registry."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

ARCHITECTURE_SCHEMA_VERSION = "stage12p2-architecture/v1"

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_VERSION_RE = re.compile(r"^v[1-9][0-9]*$")
_VERSION_REFERENCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/v[1-9][0-9]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArchitectureContractError(ValueError):
    """Raised when a Stage 12-P2 architecture contract is invalid."""


def _require_slug(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SLUG_RE.fullmatch(value) is None:
        raise ArchitectureContractError(f"{name} must be a lowercase architecture slug")
    return value


def _require_version(value: Any) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise ArchitectureContractError("version must have form vN")
    return value


def _require_version_reference(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _VERSION_REFERENCE_RE.fullmatch(value) is None:
        raise ArchitectureContractError(f"{name} must be an explicitly versioned reference")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ArchitectureContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArchitectureContractError(f"{name} must be a positive integer")
    return value


def _canonical_json_mapping(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ArchitectureContractError(f"{name} must be a non-empty mapping")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ArchitectureContractError(f"{name} must contain only finite JSON values") from exc

    if not isinstance(normalized, dict):
        raise ArchitectureContractError(f"{name} must normalize to an object")
    return normalized


def canonical_architecture_sha256(value: Mapping[str, Any]) -> str:
    """Hash one canonical JSON-safe architecture mapping."""
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ArchitectureRecord:
    """One technical, explicitly versioned architecture definition."""

    family: str
    name: str
    version: str
    compatibility: Mapping[str, Any]
    dimensions: Mapping[str, int]
    activation: str
    normalization: str | None
    positional_embedding_type: str | None
    parameter_count: int
    searchable_component_count: int
    component_type_counts: Mapping[str, int]
    initialization_ref: str
    builder_ref: str
    builder_sha256: str
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        family = _require_slug(self.family, name="family")
        name = _require_slug(self.name, name="name")
        version = _require_version(self.version)

        compatibility = _canonical_json_mapping(
            self.compatibility,
            name="compatibility",
        )

        if not isinstance(self.dimensions, Mapping) or not self.dimensions:
            raise ArchitectureContractError("dimensions must be a non-empty mapping")

        dimensions: dict[str, int] = {}
        for key, value in self.dimensions.items():
            if not isinstance(key, str) or _SLUG_RE.fullmatch(key) is None:
                raise ArchitectureContractError("dimension names must be lowercase slugs")
            dimensions[key] = _require_positive_int(
                value,
                name=f"dimensions.{key}",
            )

        for required in ("n_layers", "d_model"):
            if required not in dimensions:
                raise ArchitectureContractError(f"dimensions must include {required}")

        head_keys = {"n_heads", "d_head"}
        present_head_keys = head_keys.intersection(dimensions)
        if present_head_keys and present_head_keys != head_keys:
            raise ArchitectureContractError("n_heads and d_head must be supplied together")
        if head_keys.issubset(dimensions):
            if dimensions["n_heads"] * dimensions["d_head"] != dimensions["d_model"]:
                raise ArchitectureContractError("n_heads multiplied by d_head must equal d_model")

        if not isinstance(self.activation, str) or not self.activation:
            raise ArchitectureContractError("activation must be a non-empty string")
        for field_name, value in (
            ("normalization", self.normalization),
            ("positional_embedding_type", self.positional_embedding_type),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ArchitectureContractError(f"{field_name} must be null or a non-empty string")

        parameter_count = _require_positive_int(
            self.parameter_count,
            name="parameter_count",
        )
        searchable_count = _require_positive_int(
            self.searchable_component_count,
            name="searchable_component_count",
        )

        if not isinstance(self.component_type_counts, Mapping) or not self.component_type_counts:
            raise ArchitectureContractError("component_type_counts must be a non-empty mapping")

        component_counts: dict[str, int] = {}
        for key, value in self.component_type_counts.items():
            if not isinstance(key, str) or _SLUG_RE.fullmatch(key) is None:
                raise ArchitectureContractError("component type names must be lowercase slugs")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ArchitectureContractError(
                    f"component_type_counts.{key} must be a non-negative integer"
                )
            component_counts[key] = value

        if sum(component_counts.values()) != searchable_count:
            raise ArchitectureContractError(
                "component type counts must sum to searchable_component_count"
            )

        initialization_ref = _require_version_reference(
            self.initialization_ref,
            name="initialization_ref",
        )
        builder_ref = _require_version_reference(
            self.builder_ref,
            name="builder_ref",
        )
        builder_sha256 = _require_sha256(
            self.builder_sha256,
            name="builder_sha256",
        )

        if self.scientific_data is not False:
            raise ArchitectureContractError(
                "Stage 12-P2 architecture records must declare scientific_data=false"
            )
        if self.production_eligible is not False:
            raise ArchitectureContractError(
                "Stage 12-P2 architecture records must declare production_eligible=false"
            )

        object.__setattr__(self, "family", family)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(
            self,
            "compatibility",
            copy.deepcopy(compatibility),
        )
        object.__setattr__(
            self,
            "dimensions",
            copy.deepcopy(dimensions),
        )
        object.__setattr__(
            self,
            "component_type_counts",
            copy.deepcopy(component_counts),
        )
        object.__setattr__(
            self,
            "initialization_ref",
            initialization_ref,
        )
        object.__setattr__(self, "builder_ref", builder_ref)
        object.__setattr__(self, "builder_sha256", builder_sha256)
        object.__setattr__(self, "parameter_count", parameter_count)
        object.__setattr__(
            self,
            "searchable_component_count",
            searchable_count,
        )

    @property
    def architecture_ref(self) -> str:
        return f"{self.family}-{self.name}/{self.version}"

    def to_mapping(self) -> dict[str, Any]:
        """Return a stable JSON-safe record without choosing a roster."""
        body = {
            "schema_version": ARCHITECTURE_SCHEMA_VERSION,
            "architecture_ref": self.architecture_ref,
            "family": self.family,
            "name": self.name,
            "version": self.version,
            "compatibility": copy.deepcopy(dict(self.compatibility)),
            "dimensions": copy.deepcopy(dict(self.dimensions)),
            "activation": self.activation,
            "normalization": self.normalization,
            "positional_embedding_type": self.positional_embedding_type,
            "parameter_count": self.parameter_count,
            "searchable_component_count": self.searchable_component_count,
            "component_type_counts": copy.deepcopy(dict(self.component_type_counts)),
            "initialization_ref": self.initialization_ref,
            "builder": {
                "builder_ref": self.builder_ref,
                "builder_sha256": self.builder_sha256,
            },
            "scientific_data": False,
            "production_eligible": False,
        }
        body["record_sha256"] = canonical_architecture_sha256(body)
        return body


@dataclass(frozen=True)
class BuilderDescriptor:
    """Identity of one injected architecture builder implementation."""

    builder_ref: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "builder_ref",
            _require_version_reference(
                self.builder_ref,
                name="builder_ref",
            ),
        )
        object.__setattr__(
            self,
            "implementation_sha256",
            _require_sha256(
                self.implementation_sha256,
                name="implementation_sha256",
            ),
        )


class ArchitectureBuilder(Protocol):
    """Injected builder contract; concrete model classes stay outside discovery."""

    descriptor: BuilderDescriptor

    def validate_record(self, record: ArchitectureRecord) -> None:
        """Reject architecture combinations unsupported by this builder."""
        ...

    def build(
        self,
        *,
        record: ArchitectureRecord,
        seed: int,
        device: Any,
    ) -> Any:
        """Construct one model bound to the validated architecture record."""
        ...


def validate_task_architecture_compatibility(
    record: ArchitectureRecord,
    requirements: Mapping[str, Any],
) -> None:
    """Require every explicit task-side compatibility field to match literally.

    Stage 12-P2 does not infer architecture policy from a task. The caller
    supplies the task's already-frozen compatibility mapping, and this function
    only verifies that the architecture advertises every required field with an
    exactly equal canonical JSON value.
    """
    if not isinstance(record, ArchitectureRecord):
        raise ArchitectureContractError("record must be ArchitectureRecord")

    normalized = _canonical_json_mapping(
        requirements,
        name="architecture compatibility requirements",
    )

    missing = sorted(key for key in normalized if key not in record.compatibility)
    if missing:
        raise ArchitectureContractError(
            "architecture is missing required compatibility fields: " + ", ".join(missing)
        )

    mismatched = sorted(
        key for key, expected in normalized.items() if record.compatibility[key] != expected
    )
    if mismatched:
        raise ArchitectureContractError(
            "architecture compatibility mismatch for fields: " + ", ".join(mismatched)
        )


class ArchitectureRegistry:
    """Registry of technical architecture records bound to injected builders."""

    def __init__(
        self,
        *,
        builders: Mapping[str, ArchitectureBuilder],
    ) -> None:
        if not isinstance(builders, Mapping) or not builders:
            raise ArchitectureContractError("builders must be a non-empty mapping")

        validated_builders: dict[str, ArchitectureBuilder] = {}
        for key, builder in builders.items():
            if not isinstance(key, str) or not key:
                raise ArchitectureContractError("builder registry keys must be non-empty strings")
            descriptor = getattr(builder, "descriptor", None)
            if not isinstance(descriptor, BuilderDescriptor):
                raise ArchitectureContractError("each builder must expose BuilderDescriptor")
            if key != descriptor.builder_ref:
                raise ArchitectureContractError(
                    "builder registry key must match descriptor.builder_ref"
                )
            validator = getattr(builder, "validate_record", None)
            if not callable(validator):
                raise ArchitectureContractError("each builder must expose callable validate_record")
            constructor = getattr(builder, "build", None)
            if not callable(constructor):
                raise ArchitectureContractError("each builder must expose callable build")
            if key in validated_builders:
                raise ArchitectureContractError(f"duplicate builder reference: {key}")
            validated_builders[key] = builder

        self._builders = dict(validated_builders)
        self._records: dict[str, ArchitectureRecord] = {}

    def register(self, record: ArchitectureRecord) -> ArchitectureRecord:
        if not isinstance(record, ArchitectureRecord):
            raise ArchitectureContractError("record must be ArchitectureRecord")
        if record.architecture_ref in self._records:
            raise ArchitectureContractError(
                f"duplicate architecture reference: {record.architecture_ref}"
            )

        try:
            builder = self._builders[record.builder_ref]
        except KeyError as exc:
            raise ArchitectureContractError(
                f"unregistered architecture builder: {record.builder_ref}"
            ) from exc

        if builder.descriptor.implementation_sha256 != record.builder_sha256:
            raise ArchitectureContractError("architecture builder hash does not match record")

        try:
            builder.validate_record(record)
        except ArchitectureContractError:
            raise
        except (TypeError, ValueError) as exc:
            raise ArchitectureContractError(f"builder rejected architecture record: {exc}") from exc

        self._records[record.architecture_ref] = record
        return record

    def architecture(self, architecture_ref: str) -> ArchitectureRecord:
        try:
            return self._records[architecture_ref]
        except KeyError as exc:
            raise ArchitectureContractError(
                f"unknown architecture reference: {architecture_ref}"
            ) from exc

    def build(
        self,
        architecture_ref: str,
        *,
        seed: int,
        device: Any,
    ) -> Any:
        record = self.architecture(architecture_ref)
        builder = self._builders[record.builder_ref]
        builder.validate_record(record)
        return builder.build(
            record=record,
            seed=seed,
            device=device,
        )

    def records(self) -> tuple[ArchitectureRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

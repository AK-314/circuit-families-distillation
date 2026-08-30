"""Policy-neutral immutable records for Stage 12-P3 orchestration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Final

CONTRACT_VERSION: Final = "stage12p3-contract/v1"
CAMPAIGN_VERSION: Final = "stage12p3-campaign/v1"
JOB_VERSION: Final = "stage12p3-logical-job/v1"
OUTPUT_CONTRACT_VERSION: Final = "stage12p3-output-contract/v1"
RESOURCE_VERSION: Final = "stage12p3-resource-class/v1"
PRIORITY_VERSION: Final = "stage12p3-priority-class/v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REFERENCE_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/+-]*\Z")


class Stage12P3ContractError(ValueError):
    """Raised when immutable orchestration input is unsafe or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON value deterministically and reject non-finite numbers."""
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Stage12P3ContractError("record is not canonical JSON") from exc
    return (text + "\n").encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_reference(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _REFERENCE_RE.fullmatch(value):
        raise Stage12P3ContractError(f"{label} must be a non-empty versioned reference")
    return value


def require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise Stage12P3ContractError(f"{label} must be lowercase SHA-256")
    return value


def safe_relative_path(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage12P3ContractError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Stage12P3ContractError(f"{label} must not escape its portable root")
    return path.as_posix()


def _false_boundary(scientific_data: bool, production_eligible: bool) -> None:
    if scientific_data is not False or production_eligible is not False:
        raise Stage12P3ContractError(
            "Stage 12-P3 records require scientific_data=false and production_eligible=false"
        )


@dataclass(frozen=True)
class HashBoundReference:
    """Opaque portable reference to producer input/config evidence."""

    reference: str
    sha256: str
    interface_version: str

    def __post_init__(self) -> None:
        require_reference(self.reference, label="reference")
        require_sha256(self.sha256, label="reference.sha256")
        require_reference(self.interface_version, label="interface_version")

    def to_mapping(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ExpectedArtifact:
    relative_path: str
    media_type: str

    def __post_init__(self) -> None:
        safe_relative_path(self.relative_path, label="artifact.relative_path")
        require_reference(self.media_type, label="artifact.media_type")

    def to_mapping(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class OutputContract:
    manifest_relative_path: str
    manifest_schema_version: str
    artifacts: tuple[ExpectedArtifact, ...]
    schema_version: str = OUTPUT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OUTPUT_CONTRACT_VERSION:
            raise Stage12P3ContractError("unsupported output contract schema")
        safe_relative_path(self.manifest_relative_path, label="manifest_relative_path")
        require_reference(self.manifest_schema_version, label="manifest_schema_version")
        if not self.artifacts:
            raise Stage12P3ContractError("output contract requires at least one artifact")
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(set(paths)) != len(paths):
            raise Stage12P3ContractError("duplicate expected output paths are forbidden")
        object.__setattr__(
            self,
            "artifacts",
            tuple(sorted(self.artifacts, key=lambda artifact: artifact.relative_path)),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_relative_path": self.manifest_relative_path,
            "manifest_schema_version": self.manifest_schema_version,
            "artifacts": [artifact.to_mapping() for artifact in self.artifacts],
        }


@dataclass(frozen=True)
class ResourceClass:
    reference: str
    cpu_units: int
    accelerator_capability: str | None
    memory_bytes: int
    scratch_bytes: int
    walltime_seconds: int
    affinity_labels: tuple[str, ...] = ()
    schema_version: str = RESOURCE_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_VERSION:
            raise Stage12P3ContractError("unsupported resource class schema")
        require_reference(self.reference, label="resource reference")
        for name in ("cpu_units", "memory_bytes", "scratch_bytes", "walltime_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Stage12P3ContractError(f"{name} must be a positive integer")
        if self.accelerator_capability is not None:
            require_reference(self.accelerator_capability, label="accelerator_capability")
        labels = tuple(sorted(self.affinity_labels))
        if len(set(labels)) != len(labels):
            raise Stage12P3ContractError("affinity labels must be unique")
        for label in labels:
            require_reference(label, label="affinity label")
        object.__setattr__(self, "affinity_labels", labels)
        _false_boundary(self.scientific_data, self.production_eligible)

    def to_mapping(self) -> dict[str, object]:
        mapping = asdict(self)
        mapping["affinity_labels"] = list(self.affinity_labels)
        return mapping


@dataclass(frozen=True)
class PriorityClass:
    reference: str
    dispatch_rank: int
    schema_version: str = PRIORITY_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != PRIORITY_VERSION:
            raise Stage12P3ContractError("unsupported priority class schema")
        require_reference(self.reference, label="priority reference")
        if isinstance(self.dispatch_rank, bool) or not isinstance(self.dispatch_rank, int):
            raise Stage12P3ContractError("dispatch_rank must be an integer")
        _false_boundary(self.scientific_data, self.production_eligible)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LogicalJobSpec:
    """Immutable logical job; execution coordinates are intentionally absent."""

    family: str
    producer_interface_version: str
    dependencies: tuple[str, ...]
    expected_inputs: tuple[HashBoundReference, ...]
    payload_reference: HashBoundReference
    config_reference: HashBoundReference
    output_contract: OutputContract
    resource_class_reference: str
    priority_class_reference: str
    protected_tier: str
    retry_seed_namespace_reference: str
    schema_version: str = JOB_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != JOB_VERSION:
            raise Stage12P3ContractError("unsupported logical job schema")
        for name in (
            "family",
            "producer_interface_version",
            "resource_class_reference",
            "priority_class_reference",
            "protected_tier",
            "retry_seed_namespace_reference",
        ):
            require_reference(getattr(self, name), label=name)
        dependencies = tuple(sorted(self.dependencies))
        if len(set(dependencies)) != len(dependencies):
            raise Stage12P3ContractError("duplicate dependencies are forbidden")
        for dependency in dependencies:
            require_sha256(dependency, label="dependency identity")
        inputs = tuple(sorted(self.expected_inputs, key=lambda item: item.reference))
        if len({item.reference for item in inputs}) != len(inputs):
            raise Stage12P3ContractError("duplicate expected input references are forbidden")
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "expected_inputs", inputs)
        _false_boundary(self.scientific_data, self.production_eligible)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "producer_interface_version": self.producer_interface_version,
            "dependencies": list(self.dependencies),
            "expected_inputs": [item.to_mapping() for item in self.expected_inputs],
            "payload_reference": self.payload_reference.to_mapping(),
            "config_reference": self.config_reference.to_mapping(),
            "output_contract": self.output_contract.to_mapping(),
            "resource_class_reference": self.resource_class_reference,
            "priority_class_reference": self.priority_class_reference,
            "protected_tier": self.protected_tier,
            "retry_seed_namespace_reference": self.retry_seed_namespace_reference,
            "scientific_data": self.scientific_data,
            "production_eligible": self.production_eligible,
        }

    @property
    def job_id(self) -> str:
        return canonical_sha256(self.identity_payload())

    def to_mapping(self) -> dict[str, object]:
        return {"job_id": self.job_id, **self.identity_payload()}


@dataclass(frozen=True)
class CampaignManifest:
    manifest_reference: HashBoundReference
    jobs: tuple[LogicalJobSpec, ...]
    resource_classes: tuple[ResourceClass, ...]
    priority_classes: tuple[PriorityClass, ...]
    schema_version: str = CAMPAIGN_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_VERSION:
            raise Stage12P3ContractError("unsupported campaign schema")
        if not self.jobs:
            raise Stage12P3ContractError("campaign requires at least one job")
        object.__setattr__(self, "jobs", tuple(sorted(self.jobs, key=lambda job: job.job_id)))
        object.__setattr__(
            self,
            "resource_classes",
            tuple(sorted(self.resource_classes, key=lambda item: item.reference)),
        )
        object.__setattr__(
            self,
            "priority_classes",
            tuple(sorted(self.priority_classes, key=lambda item: item.reference)),
        )
        _false_boundary(self.scientific_data, self.production_eligible)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_reference": self.manifest_reference.to_mapping(),
            "jobs": [job.to_mapping() for job in self.jobs],
            "resource_classes": [item.to_mapping() for item in self.resource_classes],
            "priority_classes": [item.to_mapping() for item in self.priority_classes],
            "scientific_data": self.scientific_data,
            "production_eligible": self.production_eligible,
        }

    @property
    def campaign_id(self) -> str:
        return canonical_sha256(self.identity_payload())

    def to_mapping(self) -> dict[str, object]:
        return {"campaign_id": self.campaign_id, **self.identity_payload()}


def logical_job_from_mapping(value: Any) -> LogicalJobSpec:
    """Parse a serialized logical job and verify its declared identity hash."""
    if not isinstance(value, dict):
        raise Stage12P3ContractError("serialized logical job must be a mapping")
    required = {
        "job_id",
        "schema_version",
        "family",
        "producer_interface_version",
        "dependencies",
        "expected_inputs",
        "payload_reference",
        "config_reference",
        "output_contract",
        "resource_class_reference",
        "priority_class_reference",
        "protected_tier",
        "retry_seed_namespace_reference",
        "scientific_data",
        "production_eligible",
    }
    if set(value) != required:
        raise Stage12P3ContractError("serialized logical job keys mismatch")

    def bound(item: Any, label: str) -> HashBoundReference:
        if not isinstance(item, dict) or set(item) != {
            "reference",
            "sha256",
            "interface_version",
        }:
            raise Stage12P3ContractError(f"{label} keys mismatch")
        return HashBoundReference(**item)

    output = value["output_contract"]
    if not isinstance(output, dict) or set(output) != {
        "schema_version",
        "manifest_relative_path",
        "manifest_schema_version",
        "artifacts",
    }:
        raise Stage12P3ContractError("serialized output contract keys mismatch")
    artifacts = output["artifacts"]
    if not isinstance(artifacts, list):
        raise Stage12P3ContractError("serialized output artifacts must be a list")
    expected_artifacts = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {
            "relative_path",
            "media_type",
        }:
            raise Stage12P3ContractError("serialized expected artifact keys mismatch")
        expected_artifacts.append(ExpectedArtifact(**artifact))
    inputs = value["expected_inputs"]
    dependencies = value["dependencies"]
    if not isinstance(inputs, list) or not isinstance(dependencies, list):
        raise Stage12P3ContractError("serialized dependencies and inputs must be lists")
    job = LogicalJobSpec(
        schema_version=value["schema_version"],
        family=value["family"],
        producer_interface_version=value["producer_interface_version"],
        dependencies=tuple(dependencies),
        expected_inputs=tuple(bound(item, "expected input") for item in inputs),
        payload_reference=bound(value["payload_reference"], "payload reference"),
        config_reference=bound(value["config_reference"], "config reference"),
        output_contract=OutputContract(
            manifest_relative_path=output["manifest_relative_path"],
            manifest_schema_version=output["manifest_schema_version"],
            artifacts=tuple(expected_artifacts),
            schema_version=output["schema_version"],
        ),
        resource_class_reference=value["resource_class_reference"],
        priority_class_reference=value["priority_class_reference"],
        protected_tier=value["protected_tier"],
        retry_seed_namespace_reference=value["retry_seed_namespace_reference"],
        scientific_data=value["scientific_data"],
        production_eligible=value["production_eligible"],
    )
    if value["job_id"] != job.job_id:
        raise Stage12P3ContractError("serialized logical job identity/hash mismatch")
    return job

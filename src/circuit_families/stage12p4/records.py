"""Closed, policy-neutral Stage 12-P4 storage metadata contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Final

from circuit_families.stage12p3.records import (
    LogicalJobSpec,
    canonical_json_bytes,
    require_reference,
    require_sha256,
    safe_relative_path,
)

STORAGE_CONTRACT_VERSION: Final = "stage12p4-storage-object/v1"
CODEC_PROFILE_VERSION: Final = "stage12p4-codec-profile/v1"
QUOTA_PROFILE_VERSION: Final = "stage12p4-quota-profile/v1"
RETENTION_PROFILE_VERSION: Final = "stage12p4-retention-profile/v1"

LIFECYCLE_STATES: Final = frozenset(
    {
        "planned",
        "partial",
        "complete",
        "sealed",
        "exporting",
        "exported",
        "verified",
        "conflict",
        "failed",
    }
)
ARTIFACT_CLASSES: Final = frozenset(
    {
        "mask-ledger",
        "metric-ledger",
        "checkpoint",
        "failure-ledger",
        "merged-ledger",
        "bundle",
        "inventory",
        "technical-report",
    }
)
ENCODINGS: Final = frozenset(
    {"bitpack-msb0/v1", "canonical-jsonl-row-array/v1", "opaque-bytes/v1", "ustar/v1"}
)
CODECS: Final = frozenset({"none", "gzip"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class Stage12P4Error(ValueError):
    """Raised when compact-storage evidence is unsafe or inconsistent."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Any, *, chunk_bytes: int = 1024 * 1024) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _false_boundary(scientific_data: bool, production_eligible: bool) -> None:
    if scientific_data is not False or production_eligible is not False:
        raise Stage12P4Error(
            "Stage 12-P4 records require scientific_data=false and production_eligible=false"
        )


def _positive_int(value: int, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise Stage12P4Error(f"{label} must be a {qualifier} integer")
    return value


@dataclass(frozen=True)
class CodecProfile:
    reference: str
    codec: str
    compression_level: int | None
    chunk_bytes: int | None = None
    schema_version: str = CODEC_PROFILE_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CODEC_PROFILE_VERSION:
            raise Stage12P4Error("unsupported codec profile schema")
        require_reference(self.reference, label="codec profile reference")
        if self.codec not in CODECS:
            raise Stage12P4Error("unsupported or ambiguous codec")
        if self.codec == "none":
            if self.compression_level is not None:
                raise Stage12P4Error("uncompressed profile cannot set a compression level")
        elif (
            isinstance(self.compression_level, bool)
            or not isinstance(self.compression_level, int)
            or not 0 <= self.compression_level <= 9
        ):
            raise Stage12P4Error("gzip compression level must be an integer from 0 to 9")
        if self.chunk_bytes is not None:
            _positive_int(self.chunk_bytes, label="chunk_bytes")
        _false_boundary(self.scientific_data, self.production_eligible)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QuotaProfile:
    reference: str
    hard_bytes: int
    warning_bytes: int
    atomic_reserve_bytes: int
    schema_version: str = QUOTA_PROFILE_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != QUOTA_PROFILE_VERSION:
            raise Stage12P4Error("unsupported quota profile schema")
        require_reference(self.reference, label="quota profile reference")
        _positive_int(self.hard_bytes, label="hard_bytes")
        _positive_int(self.warning_bytes, label="warning_bytes", allow_zero=True)
        _positive_int(self.atomic_reserve_bytes, label="atomic_reserve_bytes", allow_zero=True)
        if self.warning_bytes > self.hard_bytes:
            raise Stage12P4Error("warning threshold exceeds hard quota")
        if self.atomic_reserve_bytes > self.hard_bytes:
            raise Stage12P4Error("atomic reserve exceeds hard quota")
        _false_boundary(self.scientific_data, self.production_eligible)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetentionProfile:
    reference: str
    checkpoint_cadence: int
    maximum_retained_generations: int
    protected_artifact_classes: tuple[str, ...]
    partial_cleanup_eligible: bool
    schema_version: str = RETENTION_PROFILE_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != RETENTION_PROFILE_VERSION:
            raise Stage12P4Error("unsupported retention profile schema")
        require_reference(self.reference, label="retention profile reference")
        _positive_int(self.checkpoint_cadence, label="checkpoint_cadence")
        _positive_int(
            self.maximum_retained_generations,
            label="maximum_retained_generations",
        )
        classes = tuple(sorted(self.protected_artifact_classes))
        if len(set(classes)) != len(classes):
            raise Stage12P4Error("protected artifact classes must be unique")
        for artifact_class in classes:
            require_reference(artifact_class, label="protected artifact class")
        object.__setattr__(self, "protected_artifact_classes", classes)
        _false_boundary(self.scientific_data, self.production_eligible)

    def to_mapping(self) -> dict[str, object]:
        mapping = asdict(self)
        mapping["protected_artifact_classes"] = list(self.protected_artifact_classes)
        return mapping


@dataclass(frozen=True)
class ProducerEvidence:
    """References P3 evidence without importing execution metadata into identity."""

    logical_job_id: str
    attempt_index: int
    output_contract_sha256: str
    sealed_manifest_sha256: str
    source_relative_path: str

    def __post_init__(self) -> None:
        require_sha256(self.logical_job_id, label="logical_job_id")
        _positive_int(self.attempt_index, label="attempt_index", allow_zero=True)
        require_sha256(self.output_contract_sha256, label="output_contract_sha256")
        require_sha256(self.sealed_manifest_sha256, label="sealed_manifest_sha256")
        safe_relative_path(self.source_relative_path, label="source_relative_path")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StorageObjectContract:
    artifact_class: str
    producer_interface_version: str
    producer_evidence: ProducerEvidence
    logical_schema_version: str
    ordered_fields: tuple[str, ...]
    source_byte_length: int
    source_sha256: str
    storage_encoding: str
    codec_profile_reference: str
    chunking_reference: str
    compact_byte_length: int
    compact_sha256: str
    scratch_quota_profile_reference: str
    retention_profile_reference: str
    lifecycle_state: str
    schema_version: str = STORAGE_CONTRACT_VERSION
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != STORAGE_CONTRACT_VERSION:
            raise Stage12P4Error("unsupported storage contract schema")
        if self.artifact_class not in ARTIFACT_CLASSES:
            raise Stage12P4Error("unsupported artifact class")
        require_reference(self.producer_interface_version, label="producer_interface_version")
        require_reference(self.logical_schema_version, label="logical_schema_version")
        fields = tuple(self.ordered_fields)
        if len(set(fields)) != len(fields):
            raise Stage12P4Error("ordered fields must be unique")
        for field in fields:
            require_reference(field, label="ordered field")
        object.__setattr__(self, "ordered_fields", fields)
        _positive_int(self.source_byte_length, label="source_byte_length", allow_zero=True)
        _positive_int(self.compact_byte_length, label="compact_byte_length", allow_zero=True)
        require_sha256(self.source_sha256, label="source_sha256")
        require_sha256(self.compact_sha256, label="compact_sha256")
        if self.storage_encoding not in ENCODINGS:
            raise Stage12P4Error("unsupported or ambiguous storage encoding")
        for label in (
            "codec_profile_reference",
            "chunking_reference",
            "scratch_quota_profile_reference",
            "retention_profile_reference",
        ):
            require_reference(getattr(self, label), label=label)
        if self.lifecycle_state not in LIFECYCLE_STATES:
            raise Stage12P4Error("unsupported lifecycle state")
        _false_boundary(self.scientific_data, self.production_eligible)

    def to_mapping(self) -> dict[str, object]:
        mapping = asdict(self)
        mapping["ordered_fields"] = list(self.ordered_fields)
        return mapping

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_mapping())).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> StorageObjectContract:
        expected = {
            "artifact_class",
            "producer_interface_version",
            "producer_evidence",
            "logical_schema_version",
            "ordered_fields",
            "source_byte_length",
            "source_sha256",
            "storage_encoding",
            "codec_profile_reference",
            "chunking_reference",
            "compact_byte_length",
            "compact_sha256",
            "scratch_quota_profile_reference",
            "retention_profile_reference",
            "lifecycle_state",
            "schema_version",
            "scientific_data",
            "production_eligible",
        }
        if set(value) != expected:
            raise Stage12P4Error("storage contract fields do not match closed schema")
        evidence = value["producer_evidence"]
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "logical_job_id",
            "attempt_index",
            "output_contract_sha256",
            "sealed_manifest_sha256",
            "source_relative_path",
        }:
            raise Stage12P4Error("producer evidence fields do not match closed schema")
        fields = value["ordered_fields"]
        if not isinstance(fields, list):
            raise Stage12P4Error("ordered_fields must be a list")
        return cls(
            artifact_class=str(value["artifact_class"]),
            producer_interface_version=str(value["producer_interface_version"]),
            producer_evidence=ProducerEvidence(**dict(evidence)),
            logical_schema_version=str(value["logical_schema_version"]),
            ordered_fields=tuple(str(item) for item in fields),
            source_byte_length=int(value["source_byte_length"]),
            source_sha256=str(value["source_sha256"]),
            storage_encoding=str(value["storage_encoding"]),
            codec_profile_reference=str(value["codec_profile_reference"]),
            chunking_reference=str(value["chunking_reference"]),
            compact_byte_length=int(value["compact_byte_length"]),
            compact_sha256=str(value["compact_sha256"]),
            scratch_quota_profile_reference=str(value["scratch_quota_profile_reference"]),
            retention_profile_reference=str(value["retention_profile_reference"]),
            lifecycle_state=str(value["lifecycle_state"]),
            schema_version=str(value["schema_version"]),
            scientific_data=value["scientific_data"],
            production_eligible=value["production_eligible"],
        )


def producer_evidence_from_p3(
    job: LogicalJobSpec,
    *,
    attempt_index: int,
    sealed_manifest: Mapping[str, Any],
    sealed_manifest_bytes: bytes,
    source_relative_path: str,
) -> ProducerEvidence:
    """Validate and bind exact P3 job/output evidence for a storage wrapper."""
    required = {
        "schema_version",
        "declared_output_schema_version",
        "campaign_id",
        "job_id",
        "attempt_index",
        "retry_index",
        "artifacts",
        "scientific_data",
        "production_eligible",
    }
    if set(sealed_manifest) != required:
        raise Stage12P4Error("sealed P3 manifest fields mismatch")
    if sealed_manifest["job_id"] != job.job_id:
        raise Stage12P4Error("cross-job sealed output reuse is forbidden")
    if sealed_manifest["attempt_index"] != attempt_index:
        raise Stage12P4Error("sealed output attempt mismatch")
    _false_boundary(sealed_manifest["scientific_data"], sealed_manifest["production_eligible"])
    source_relative_path = safe_relative_path(source_relative_path, label="source_relative_path")
    declared_paths = {artifact.relative_path for artifact in job.output_contract.artifacts}
    if source_relative_path not in declared_paths:
        raise Stage12P4Error("source is absent from producer output contract")
    artifacts = sealed_manifest["artifacts"]
    if not isinstance(artifacts, list) or source_relative_path not in {
        item.get("relative_path") for item in artifacts if isinstance(item, Mapping)
    }:
        raise Stage12P4Error("source lacks sealed producer evidence")
    return ProducerEvidence(
        logical_job_id=job.job_id,
        attempt_index=attempt_index,
        output_contract_sha256=hashlib.sha256(
            canonical_json_bytes(job.output_contract.to_mapping())
        ).hexdigest(),
        sealed_manifest_sha256=hashlib.sha256(sealed_manifest_bytes).hexdigest(),
        source_relative_path=source_relative_path,
    )


def validate_technical_payload(value: Any) -> None:
    """Reject scientific relabeling, private paths, non-finite data, and binaries."""

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            if item.get("scientific_data") is True:
                raise Stage12P4Error("scientific payload relabeling is forbidden")
            if item.get("production_eligible") is True:
                raise Stage12P4Error("production_eligible=true is forbidden")
            for key, child in item.items():
                if not isinstance(key, str):
                    raise Stage12P4Error("record keys must be strings")
                walk(child)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                walk(child)
            return
        if isinstance(item, float) and not math.isfinite(item):
            raise Stage12P4Error("non-finite numeric payload is forbidden")
        if isinstance(item, (bytes, bytearray)):
            raise Stage12P4Error("embedded binary payload is forbidden")
        if isinstance(item, str):
            path = PurePosixPath(item)
            if path.is_absolute() or item.startswith(("/Users/", "/home/", "/private/")):
                raise Stage12P4Error("absolute private path is forbidden")

    walk(value)


def canonical_mapping_bytes(value: Mapping[str, Any]) -> bytes:
    validate_technical_payload(value)
    try:
        return canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise Stage12P4Error("value is not canonical JSON") from exc


def strict_json_mapping(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage12P4Error("invalid JSON record") from exc
    if not isinstance(value, dict):
        raise Stage12P4Error("JSON record must be a mapping")
    validate_technical_payload(value)
    return value


def require_hex64(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise Stage12P4Error(f"{label} must be lowercase SHA-256")
    return value

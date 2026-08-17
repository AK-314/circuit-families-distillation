"""Shared Stage 4 schema-envelope contract.

Record-specific payload schemas are intentionally not defined here. Parts M,
O, and Q define those bodies while reusing this common envelope.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from circuit_families.stage4_condition_identity import (
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    parse_condition_id,
)

_CONDITION_ID_RE = re.compile(r"cfdid:v1:d[2-8]\|.+\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CREATION_STAGE_RE = re.compile(r"stage[0-9]+[a-z]?\Z")

_PRODUCER_LANES = frozenset(
    {"lane_a", "lane_b", "lane_c", "lane_d", "joint"}
)

_COMMON_FIELDS = frozenset(
    {
        "namespace",
        "vocabulary_version",
        "schema_version",
        "record_type",
        "record_status",
        "condition_id",
        "identity_depth",
        "payload",
        "provenance",
    }
)

_PROVENANCE_FIELDS = frozenset(
    {
        "producer_lane",
        "creation_stage",
        "source_records",
    }
)

_REFERENCE_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "condition_id",
        "record_sha256",
    }
)


class Stage4SchemaError(ValueError):
    """Raised when a Stage 4 common record contract is violated."""


@dataclass(frozen=True)
class CommonSchemaContract:
    """Frozen structural data read from the Part F/G authority files."""

    namespace: str
    vocabulary_version: str
    record_types: tuple[str, ...]
    schema_versions: Mapping[str, str]
    record_status_values: tuple[str, ...]
    record_type_required_depths: Mapping[str, int]

    @classmethod
    def from_specs(
        cls,
        vocabulary: Mapping[str, Any],
        identity_spec: Mapping[str, Any],
    ) -> CommonSchemaContract:
        record_types = tuple(vocabulary["record_types"])
        schema_versions = dict(vocabulary["schema_versions"])
        record_status_values = tuple(vocabulary["record_status_values"])
        depths = dict(identity_spec["record_type_required_depths"])

        if len(record_types) != 14 or len(set(record_types)) != 14:
            raise Stage4SchemaError(
                "common contract requires exactly 14 unique record types"
            )

        if set(schema_versions) != set(record_types):
            raise Stage4SchemaError(
                "schema-version keys must equal record-type inventory"
            )

        if set(depths) != set(record_types):
            raise Stage4SchemaError(
                "record-depth keys must equal record-type inventory"
            )

        if vocabulary["identity_version"] != identity_spec["identity_version"]:
            raise Stage4SchemaError(
                "vocabulary and identity specification versions disagree"
            )

        if vocabulary["identity_version"] != "condition-identity/v1":
            raise Stage4SchemaError(
                "unsupported identity version"
            )

        return cls(
            namespace=vocabulary["namespace"],
            vocabulary_version=vocabulary["vocabulary_version"],
            record_types=record_types,
            schema_versions=schema_versions,
            record_status_values=record_status_values,
            record_type_required_depths=depths,
        )


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing or extra:
        raise Stage4SchemaError(
            f"{label} keys mismatch: missing={missing!r} extra={extra!r}"
        )


def _validate_record_reference(
    reference: Any,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    if not isinstance(reference, Mapping):
        raise Stage4SchemaError(
            "source-record reference must be an object"
        )

    actual = set(reference)
    required = {"record_type", "schema_version", "condition_id"}
    optional = {"record_sha256"}

    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)

    if missing or extra:
        raise Stage4SchemaError(
            "source-record reference keys mismatch: "
            f"missing={missing!r} extra={extra!r}"
        )

    record_type = reference["record_type"]

    if record_type not in contract.record_types:
        raise Stage4SchemaError(
            f"unknown referenced record_type: {record_type!r}"
        )

    expected_schema = contract.schema_versions[record_type]

    if reference["schema_version"] != expected_schema:
        raise Stage4SchemaError(
            "referenced schema_version does not match record_type"
        )

    condition_id = reference["condition_id"]

    if (
        not isinstance(condition_id, str)
        or not _CONDITION_ID_RE.fullmatch(condition_id)
    ):
        raise Stage4SchemaError(
            "referenced condition_id has invalid wire shape"
        )

    try:
        identity = parse_condition_id(condition_id, stage3)
    except ConditionIdentityError as exc:
        raise Stage4SchemaError(
            f"invalid referenced condition_id: {exc}"
        ) from exc

    expected_depth = contract.record_type_required_depths[record_type]

    if identity.depth != expected_depth:
        raise Stage4SchemaError(
            "referenced condition_id depth does not match referenced "
            f"record_type: actual={identity.depth} expected={expected_depth}"
        )

    if "record_sha256" in reference:
        digest = reference["record_sha256"]
        if (
            not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
        ):
            raise Stage4SchemaError(
                "record_sha256 must be 64 lowercase hexadecimal characters"
            )


def validate_common_envelope(
    record: Any,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    """Validate the shared envelope; payload semantics are validated later."""
    if not isinstance(record, Mapping):
        raise Stage4SchemaError("record must be an object")

    _require_exact_keys(
        record,
        _COMMON_FIELDS,
        label="common envelope",
    )

    if record["namespace"] != contract.namespace:
        raise Stage4SchemaError(
            f"namespace must be {contract.namespace!r}"
        )

    if record["vocabulary_version"] != contract.vocabulary_version:
        raise Stage4SchemaError(
            f"vocabulary_version must be {contract.vocabulary_version!r}"
        )

    record_type = record["record_type"]

    if record_type not in contract.record_types:
        raise Stage4SchemaError(
            f"unknown record_type: {record_type!r}"
        )

    expected_schema = contract.schema_versions[record_type]

    if record["schema_version"] != expected_schema:
        raise Stage4SchemaError(
            f"schema_version for {record_type!r} must be "
            f"{expected_schema!r}"
        )

    if record["record_status"] not in contract.record_status_values:
        raise Stage4SchemaError(
            f"invalid record_status: {record['record_status']!r}"
        )

    condition_id = record["condition_id"]

    if (
        not isinstance(condition_id, str)
        or not _CONDITION_ID_RE.fullmatch(condition_id)
    ):
        raise Stage4SchemaError(
            "condition_id has invalid wire shape"
        )

    try:
        identity = parse_condition_id(condition_id, stage3)
    except ConditionIdentityError as exc:
        raise Stage4SchemaError(
            f"invalid condition_id: {exc}"
        ) from exc

    identity_depth = record["identity_depth"]

    if (
        isinstance(identity_depth, bool)
        or not isinstance(identity_depth, int)
    ):
        raise Stage4SchemaError(
            "identity_depth must be an integer"
        )

    if identity_depth != identity.depth:
        raise Stage4SchemaError(
            "identity_depth does not match parsed condition_id depth"
        )

    expected_depth = contract.record_type_required_depths[record_type]

    if identity_depth != expected_depth:
        raise Stage4SchemaError(
            f"record_type {record_type!r} requires identity depth "
            f"{expected_depth}, found {identity_depth}"
        )

    if not isinstance(record["payload"], Mapping):
        raise Stage4SchemaError(
            "payload must be an object"
        )

    provenance = record["provenance"]

    if not isinstance(provenance, Mapping):
        raise Stage4SchemaError(
            "provenance must be an object"
        )

    _require_exact_keys(
        provenance,
        _PROVENANCE_FIELDS,
        label="provenance",
    )

    producer_lane = provenance["producer_lane"]

    if producer_lane not in _PRODUCER_LANES:
        raise Stage4SchemaError(
            f"invalid producer_lane: {producer_lane!r}"
        )

    creation_stage = provenance["creation_stage"]

    if (
        not isinstance(creation_stage, str)
        or not _CREATION_STAGE_RE.fullmatch(creation_stage)
    ):
        raise Stage4SchemaError(
            "creation_stage must match stage<integer>[optional-lowercase-suffix]"
        )

    source_records = provenance["source_records"]

    if not isinstance(source_records, list):
        raise Stage4SchemaError(
            "source_records must be an array"
        )

    seen: set[tuple[str, str, str, str | None]] = set()

    for reference in source_records:
        _validate_record_reference(
            reference,
            contract=contract,
            stage3=stage3,
        )

        key = (
            reference["record_type"],
            reference["schema_version"],
            reference["condition_id"],
            reference.get("record_sha256"),
        )

        if key in seen:
            raise Stage4SchemaError(
                "duplicate source-record reference"
            )

        seen.add(key)

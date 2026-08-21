from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import (
    SYNTHETIC_CLASSIFICATION,
    SyntheticRecordError,
    SyntheticUniverse,
)
from .synthetic_universe import (
    synthetic_universe_from_mapping,
    validate_required_synthetic_coverage,
)

INGESTION_ENVELOPE_SCHEMA_VERSION = "stage5d_ingestion_envelope_v1"
NORMALIZED_UNIVERSE_SCHEMA_VERSION = "stage5d_normalized_universe_v1"

COLLECTION_NAMES = (
    "method_budgets",
    "teacher_inventories",
    "student_attempts",
    "eligibility_records",
    "direct_teacher_endpoints",
    "student_endpoints",
    "cell_expectations",
)


class SyntheticIngestionError(ValueError):
    pass


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SyntheticIngestionError(f"{label} must be an object")
    return value


def _expect_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SyntheticIngestionError(f"{label} must be a non-empty string")
    return value


def _expect_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise SyntheticIngestionError(f"{label} must be a boolean")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise SyntheticIngestionError(
            f"{label} keys mismatch: "
            f"missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _unwrap_collection(
    raw: Any,
    collection_name: str,
    provenance_id: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(raw, list):
        raise SyntheticIngestionError(
            f"{collection_name} must be a list"
        )

    records: list[Mapping[str, Any]] = []

    for index, item_raw in enumerate(raw):
        item = _expect_mapping(
            item_raw,
            f"{collection_name}[{index}]",
        )
        _expect_exact_keys(
            item,
            {"provenance_id", "record"},
            f"{collection_name}[{index}]",
        )

        item_provenance = _expect_str(
            item["provenance_id"],
            f"{collection_name}[{index}].provenance_id",
        )
        if item_provenance != provenance_id:
            raise SyntheticIngestionError(
                f"mixed provenance in {collection_name}[{index}]: "
                f"expected={provenance_id} observed={item_provenance}"
            )

        records.append(
            _expect_mapping(
                item["record"],
                f"{collection_name}[{index}].record",
            )
        )

    return records


def _identity_key(identity: Any) -> tuple[Any, ...]:
    return (
        identity.teacher_seed,
        identity.phase,
        identity.distillation_condition or "",
        -1
        if identity.student_initialization is None
        else identity.student_initialization,
        identity.method_id,
        identity.endpoint_id,
        identity.protocol_id,
        identity.fidelity_id,
        identity.budget_id,
    )


def _cell_identity_key(identity: Any) -> tuple[Any, ...]:
    return (
        identity.teacher_seed,
        identity.phase,
        identity.distillation_condition,
        identity.method_id,
        identity.endpoint_id,
        identity.protocol_id,
        identity.fidelity_id,
        identity.budget_id,
    )


def normalized_mapping(
    universe: SyntheticUniverse,
    *,
    provenance_id: str,
    source_sha256: str,
) -> dict[str, Any]:
    method_budgets = sorted(
        universe.method_budgets,
        key=lambda record: (
            record.method_id,
            record.budget_id,
        ),
    )

    teacher_inventories = sorted(
        universe.teacher_inventories,
        key=lambda record: record.teacher_seed,
    )

    student_attempts = sorted(
        universe.student_attempts,
        key=lambda record: (
            record.teacher_seed,
            record.phase,
            record.distillation_condition,
            record.student_initialization,
            record.attempt_index,
            record.attempt_id,
        ),
    )

    eligibility_records = sorted(
        universe.eligibility_records,
        key=lambda record: (
            record.attempt_id,
            record.eligibility_id,
        ),
    )

    direct_teacher_endpoints = sorted(
        universe.direct_teacher_endpoints,
        key=lambda record: (
            _identity_key(record.identity),
            record.record_id,
        ),
    )

    student_endpoints = sorted(
        universe.student_endpoints,
        key=lambda record: (
            _identity_key(record.identity),
            record.attempt_id,
            record.record_id,
        ),
    )

    cell_expectations = sorted(
        universe.cell_expectations,
        key=lambda record: (
            _cell_identity_key(record.identity),
            record.cell_id,
        ),
    )

    return {
        "schema_version": NORMALIZED_UNIVERSE_SCHEMA_VERSION,
        "classification": SYNTHETIC_CLASSIFICATION,
        "scientific_data": False,
        "production_eligible": False,
        "source_provenance": {
            "provenance_id": provenance_id,
            "source_sha256": source_sha256,
        },
        "method_budgets": [
            asdict(record)
            for record in method_budgets
        ],
        "teacher_inventories": [
            asdict(record)
            for record in teacher_inventories
        ],
        "student_attempts": [
            asdict(record)
            for record in student_attempts
        ],
        "eligibility_records": [
            asdict(record)
            for record in eligibility_records
        ],
        "direct_teacher_endpoints": [
            asdict(record)
            for record in direct_teacher_endpoints
        ],
        "student_endpoints": [
            asdict(record)
            for record in student_endpoints
        ],
        "cell_expectations": [
            asdict(record)
            for record in cell_expectations
        ],
    }


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SyntheticIngestionError(
            f"value cannot be canonically serialized: {exc}"
        ) from exc

    return (text + "\n").encode("utf-8")


def normalized_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalize_ingestion_mapping(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "classification",
        "scientific_data",
        "production_eligible",
        "provenance",
        *COLLECTION_NAMES,
    }
    _expect_exact_keys(raw, expected, "ingestion envelope")

    schema_version = _expect_str(
        raw["schema_version"],
        "schema_version",
    )
    if schema_version != INGESTION_ENVELOPE_SCHEMA_VERSION:
        raise SyntheticIngestionError(
            f"unsupported ingestion schema version: {schema_version}"
        )

    classification = _expect_str(
        raw["classification"],
        "classification",
    )
    if classification != SYNTHETIC_CLASSIFICATION:
        raise SyntheticIngestionError(
            "ingestion envelope must be synthetic-only"
        )

    scientific_data = _expect_bool(
        raw["scientific_data"],
        "scientific_data",
    )
    if scientific_data:
        raise SyntheticIngestionError(
            "Stage 5D ingestion cannot contain scientific data"
        )

    production_eligible = _expect_bool(
        raw["production_eligible"],
        "production_eligible",
    )
    if production_eligible:
        raise SyntheticIngestionError(
            "Stage 5D ingestion cannot be production eligible"
        )

    provenance = _expect_mapping(
        raw["provenance"],
        "provenance",
    )
    _expect_exact_keys(
        provenance,
        {
            "provenance_id",
            "classification",
            "scientific_data",
            "production_eligible",
        },
        "provenance",
    )

    provenance_id = _expect_str(
        provenance["provenance_id"],
        "provenance.provenance_id",
    )

    if (
        provenance["classification"] != SYNTHETIC_CLASSIFICATION
        or provenance["scientific_data"] is not False
        or provenance["production_eligible"] is not False
    ):
        raise SyntheticIngestionError(
            "provenance metadata violates Stage 5D synthetic firewall"
        )

    unwrapped: dict[str, Any] = {
        "schema_version": "stage5d_synthetic_universe_v1",
        "classification": SYNTHETIC_CLASSIFICATION,
    }

    for collection_name in COLLECTION_NAMES:
        unwrapped[collection_name] = _unwrap_collection(
            raw[collection_name],
            collection_name,
            provenance_id,
        )

    try:
        universe = synthetic_universe_from_mapping(unwrapped)
        validate_required_synthetic_coverage(universe)
    except SyntheticRecordError as exc:
        raise SyntheticIngestionError(str(exc)) from exc

    semantic_source = normalized_mapping(
        universe,
        provenance_id=provenance_id,
        source_sha256="0" * 64,
    )
    semantic_source["schema_version"] = INGESTION_ENVELOPE_SCHEMA_VERSION
    semantic_source.pop("source_provenance")
    semantic_source["provenance"] = {
        "provenance_id": provenance_id,
        "classification": classification,
        "scientific_data": scientific_data,
        "production_eligible": production_eligible,
    }

    source_sha256 = hashlib.sha256(
        canonical_json_bytes(semantic_source)
    ).hexdigest()

    return normalized_mapping(
        universe,
        provenance_id=provenance_id,
        source_sha256=source_sha256,
    )


def load_and_normalize_ingestion(
    path: str | Path,
) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return normalize_ingestion_mapping(
        _expect_mapping(raw, "ingestion envelope")
    )


def reorder_ingestion_collections(
    raw: Mapping[str, Any],
    ordering: Sequence[int],
) -> dict[str, Any]:
    result = dict(raw)

    for collection_name in COLLECTION_NAMES:
        collection = raw[collection_name]
        if not isinstance(collection, list):
            raise SyntheticIngestionError(
                f"{collection_name} must be a list"
            )

        if len(collection) == len(ordering):
            result[collection_name] = [
                collection[index]
                for index in ordering
            ]

    return result

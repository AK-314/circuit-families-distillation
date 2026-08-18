"""Stage 4 cross-record graph validation.

The mapping keys supplied to ``validate_stage4_record_graph`` are externally
computed SHA-256 identities for record artifacts. Stage 4 deliberately does
not introduce a second canonical record-serialization/hash definition here.
The graph validator verifies that references resolve to those supplied hashes,
that referenced record identity matches the reference, and that cross-record
lifecycle/provenance invariants hold.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from circuit_families.stage4_condition_identity import Stage3AvailabilityIndex
from circuit_families.stage4_schema_analysis import validate_part_q_record
from circuit_families.stage4_schema_common import (
    CommonSchemaContract,
)
from circuit_families.stage4_schema_discovery import validate_part_o_record
from circuit_families.stage4_schema_records import (
    _error,
    validate_part_m_record,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_PART_M = frozenset(
    {
        "teacher_reference",
        "teacher_output_cache",
        "student_attempt",
        "student_eligibility",
        "sealed_dense_model",
    }
)

_PART_O = frozenset(
    {
        "discovery_run",
        "native_budget_ledger",
        "exact_mask_evaluation_ledger",
        "endpoint_record",
    }
)

_PART_Q = frozenset(
    {
        "student_cell_summary",
        "teacher_seed_inventory",
        "excluded_development_output",
        "reproduction_comparison",
        "analysis_freeze",
    }
)


def _explicit_references(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payload = record["payload"]
    record_type = record["record_type"]

    if record_type == "teacher_reference":
        return []

    if record_type == "teacher_output_cache":
        return [payload["teacher_reference"]]

    if record_type == "student_attempt":
        return [payload["target_cache"]]

    if record_type == "student_eligibility":
        return [payload["attempt_reference"]]

    if record_type == "sealed_dense_model":
        return [payload["eligibility_reference"]]

    if record_type == "discovery_run":
        return [payload["sealed_dense_model"]]

    if record_type == "native_budget_ledger":
        return [payload["discovery_run"]]

    if record_type == "exact_mask_evaluation_ledger":
        return [
            payload["sealed_dense_model"],
            payload["discovery_run"],
        ]

    if record_type == "endpoint_record":
        return [payload["exact_ledger"]]

    if record_type == "student_cell_summary":
        return [
            *[
                item["attempt_reference"]
                for item in payload["failed_attempts"]
            ],
            *payload["endpoint_records"],
        ]

    if record_type == "teacher_seed_inventory":
        return list(payload["student_cell_summaries"])

    if record_type == "excluded_development_output":
        return []

    if record_type == "reproduction_comparison":
        return [
            payload["source_record"],
            payload["reproduced_record"],
        ]

    if record_type == "analysis_freeze":
        return [
            *payload["primary_input_records"],
            *payload["excluded_development_records"],
        ]

    _error(f"unsupported graph record_type: {record_type!r}")


def _reference_key(reference: Mapping[str, Any]) -> tuple[str, str, str, str]:
    if "record_sha256" not in reference:
        _error("graph record reference requires record_sha256")

    digest = reference["record_sha256"]

    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        _error("graph record reference SHA-256 is invalid")

    return (
        reference["record_type"],
        reference["schema_version"],
        reference["condition_id"],
        digest,
    )


def _resolve(
    reference: Mapping[str, Any],
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    record_type, schema_version, condition_id, digest = _reference_key(
        reference
    )

    if digest not in records_by_sha256:
        _error(f"dangling record reference: {digest}")

    target = records_by_sha256[digest]

    if target["record_type"] != record_type:
        _error("resolved record_type does not match reference")

    if target["schema_version"] != schema_version:
        _error("resolved schema_version does not match reference")

    if target["condition_id"] != condition_id:
        _error("resolved condition_id does not match reference")

    return target


def _require_status(
    record: Mapping[str, Any],
    expected: str,
    *,
    label: str,
) -> None:
    if record["record_status"] != expected:
        _error(
            f"{label} must have record_status={expected!r}"
        )


def _validate_individual_record(
    record: Mapping[str, Any],
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
    stage3_registry: Mapping[str, Any],
    stage3_registry_sha256: str,
    current_unresolved_decision_ids: set[str],
) -> None:
    record_type = record["record_type"]

    if record_type in _PART_M:
        validate_part_m_record(
            record,
            contract=contract,
            stage3=stage3,
            stage3_registry=stage3_registry,
            stage3_registry_sha256=stage3_registry_sha256,
        )
        return

    if record_type in _PART_O:
        validate_part_o_record(
            record,
            contract=contract,
            stage3=stage3,
        )
        return

    if record_type in _PART_Q:
        validate_part_q_record(
            record,
            contract=contract,
            stage3=stage3,
            stage3_registry=stage3_registry,
            stage3_registry_sha256=stage3_registry_sha256,
            current_unresolved_decision_ids=current_unresolved_decision_ids,
        )
        return

    _error(f"unknown Stage 4 record_type: {record_type!r}")


def _validate_provenance_dependencies(
    record: Mapping[str, Any],
    *,
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    explicit = _explicit_references(record)
    provenance = record["provenance"]["source_records"]

    explicit_keys = {_reference_key(ref) for ref in explicit}
    provenance_keys = {_reference_key(ref) for ref in provenance}

    missing = sorted(explicit_keys - provenance_keys)

    if missing:
        _error(
            "explicit payload dependency missing from provenance "
            f"source_records: {missing[0]}"
        )

    target_hashes: list[str] = []

    for reference in provenance:
        _resolve(reference, records_by_sha256)
        target_hashes.append(reference["record_sha256"])

    for reference in explicit:
        _resolve(reference, records_by_sha256)

    return target_hashes


def _validate_student_eligibility_edge(
    record: Mapping[str, Any],
    *,
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> None:
    payload = record["payload"]
    attempt = _resolve(
        payload["attempt_reference"],
        records_by_sha256,
    )

    _require_status(
        attempt,
        "sealed",
        label="eligibility source student_attempt",
    )

    if attempt["payload"]["attempt_outcome"] != "succeeded":
        _error(
            "student_eligibility cannot derive from failed student_attempt"
        )

    if payload["attempt_index"] != attempt["payload"]["attempt_index"]:
        _error(
            "student_eligibility attempt_index must match student_attempt"
        )

    if payload["retry_index"] != attempt["payload"]["retry_index"]:
        _error(
            "student_eligibility retry_index must match student_attempt"
        )


def _validate_sealed_model_edge(
    record: Mapping[str, Any],
    *,
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> None:
    payload = record["payload"]
    eligibility = _resolve(
        payload["eligibility_reference"],
        records_by_sha256,
    )

    _require_status(
        eligibility,
        "sealed",
        label="sealed model eligibility source",
    )

    if eligibility["payload"]["eligibility_status"] != "passed":
        _error("sealed dense model requires actually passing eligibility")

    attempt = _resolve(
        eligibility["payload"]["attempt_reference"],
        records_by_sha256,
    )

    _require_status(
        attempt,
        "sealed",
        label="sealed model student_attempt source",
    )

    if attempt["payload"]["attempt_outcome"] != "succeeded":
        _error("sealed dense model requires succeeded student_attempt")

    if (
        payload["architecture_ref"]
        != attempt["payload"]["student_architecture_ref"]
    ):
        _error(
            "sealed dense model architecture_ref must match student_attempt"
        )

    if payload["model_checkpoint"] != attempt["payload"]["model_checkpoint"]:
        _error(
            "sealed dense model checkpoint must match student_attempt "
            "checkpoint"
        )


def _validate_discovery_edge(
    record: Mapping[str, Any],
    *,
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> None:
    sealed_model = _resolve(
        record["payload"]["sealed_dense_model"],
        records_by_sha256,
    )

    _require_status(
        sealed_model,
        "sealed",
        label="discovery sealed_dense_model source",
    )


def _validate_native_edge(
    record: Mapping[str, Any],
    *,
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> None:
    discovery = _resolve(
        record["payload"]["discovery_run"],
        records_by_sha256,
    )

    _require_status(
        discovery,
        "sealed",
        label="native ledger discovery_run source",
    )

    if (
        record["payload"]["method_budget_ref"]
        != discovery["payload"]["method_budget_ref"]
    ):
        _error(
            "native ledger method_budget_ref must match discovery_run"
        )


def _validate_exact_edge(
    record: Mapping[str, Any],
    *,
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> None:
    sealed_model = _resolve(
        record["payload"]["sealed_dense_model"],
        records_by_sha256,
    )
    discovery = _resolve(
        record["payload"]["discovery_run"],
        records_by_sha256,
    )

    _require_status(
        sealed_model,
        "sealed",
        label="exact ledger sealed_dense_model source",
    )
    _require_status(
        discovery,
        "sealed",
        label="exact ledger discovery_run source",
    )


def _validate_endpoint_edge(
    record: Mapping[str, Any],
    *,
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> None:
    payload = record["payload"]
    ledger = _resolve(
        payload["exact_ledger"],
        records_by_sha256,
    )

    _require_status(
        ledger,
        "sealed",
        label="endpoint exact ledger source",
    )

    entries = ledger["payload"]["entries"]
    qualifying = [
        entry
        for entry in entries
        if entry["qualifies"]
    ]

    if not qualifying:
        _error(
            "endpoint exact ledger must contain at least one qualifying mask"
        )

    minimum = min(
        float(entry["retained_proportion"])
        for entry in qualifying
    )

    endpoint_1 = payload["endpoint_1"]
    observed = float(
        endpoint_1["smallest_recovered_component_proportion"]
    )

    if abs(observed - minimum) > 1e-12:
        _error(
            "endpoint_1 does not reconstruct as the smallest qualifying "
            "proportion in exact ledger"
        )

    matching_endpoint_1 = [
        entry
        for entry in qualifying
        if (
            entry["mask_sha256"]
            == endpoint_1["qualifying_mask_sha256"]
            and abs(
                float(entry["retained_proportion"]) - minimum
            )
            <= 1e-12
        )
    ]

    if not matching_endpoint_1:
        _error(
            "endpoint_1 qualifying mask is not a minimum qualifying "
            "exact-ledger mask"
        )

    by_sha = {
        entry["mask_sha256"]: entry
        for entry in entries
    }

    for digest in payload["endpoint_2"]["packed_mask_sha256s"]:
        if digest not in by_sha:
            _error(
                "endpoint_2 packed mask is absent from exact ledger"
            )

        if not by_sha[digest]["qualifies"]:
            _error(
                "endpoint_2 packed mask must be a qualifying exact "
                "evaluation"
            )


def _validate_summary_edges(
    record: Mapping[str, Any],
    *,
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> None:
    payload = record["payload"]

    for item in payload["failed_attempts"]:
        attempt = _resolve(
            item["attempt_reference"],
            records_by_sha256,
        )

        _require_status(
            attempt,
            "sealed",
            label="summary failed student_attempt",
        )

        if attempt["payload"]["attempt_outcome"] != "failed":
            _error(
                "summary failed_attempt reference must target failed "
                "student_attempt"
            )

        if item["attempt_index"] != attempt["payload"]["attempt_index"]:
            _error(
                "summary failed attempt_index must match student_attempt"
            )

        if item["retry_index"] != attempt["payload"]["retry_index"]:
            _error(
                "summary failed retry_index must match student_attempt"
            )

    for reference in payload["endpoint_records"]:
        endpoint = _resolve(reference, records_by_sha256)

        _require_status(
            endpoint,
            "sealed",
            label="summary endpoint_record",
        )


def _validate_inventory_edges(
    record: Mapping[str, Any],
    *,
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> None:
    for reference in record["payload"]["student_cell_summaries"]:
        summary = _resolve(reference, records_by_sha256)

        _require_status(
            summary,
            "sealed",
            label="inventory student_cell_summary",
        )


def _validate_analysis_freeze_edges(
    record: Mapping[str, Any],
    *,
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> None:
    payload = record["payload"]

    for reference in payload["primary_input_records"]:
        target = _resolve(reference, records_by_sha256)

        _require_status(
            target,
            "sealed",
            label="analysis_freeze primary input",
        )

    for reference in payload["excluded_development_records"]:
        target = _resolve(reference, records_by_sha256)

        _require_status(
            target,
            "sealed",
            label="analysis_freeze excluded-development record",
        )


def _validate_cross_record_edges(
    record: Mapping[str, Any],
    *,
    records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> None:
    record_type = record["record_type"]

    if record_type == "student_eligibility":
        _validate_student_eligibility_edge(
            record,
            records_by_sha256=records_by_sha256,
        )
    elif record_type == "sealed_dense_model":
        _validate_sealed_model_edge(
            record,
            records_by_sha256=records_by_sha256,
        )
    elif record_type == "discovery_run":
        _validate_discovery_edge(
            record,
            records_by_sha256=records_by_sha256,
        )
    elif record_type == "native_budget_ledger":
        _validate_native_edge(
            record,
            records_by_sha256=records_by_sha256,
        )
    elif record_type == "exact_mask_evaluation_ledger":
        _validate_exact_edge(
            record,
            records_by_sha256=records_by_sha256,
        )
    elif record_type == "endpoint_record":
        _validate_endpoint_edge(
            record,
            records_by_sha256=records_by_sha256,
        )
    elif record_type == "student_cell_summary":
        _validate_summary_edges(
            record,
            records_by_sha256=records_by_sha256,
        )
    elif record_type == "teacher_seed_inventory":
        _validate_inventory_edges(
            record,
            records_by_sha256=records_by_sha256,
        )
    elif record_type == "analysis_freeze":
        _validate_analysis_freeze_edges(
            record,
            records_by_sha256=records_by_sha256,
        )


def _validate_acyclic(edges: Mapping[str, list[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return

        if node in visiting:
            _error("record graph contains a dependency cycle")

        visiting.add(node)

        for target in edges.get(node, []):
            visit(target)

        visiting.remove(node)
        visited.add(node)

    for node in edges:
        visit(node)


def validate_stage4_record_graph(
    records_by_sha256: Mapping[str, Mapping[str, Any]],
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
    stage3_registry: Mapping[str, Any],
    stage3_registry_sha256: str,
    current_unresolved_decision_ids: set[str],
) -> None:
    """Validate a complete supplied Stage 4 record dependency graph."""
    if not isinstance(records_by_sha256, Mapping):
        _error("records_by_sha256 must be a mapping")

    if not records_by_sha256:
        _error("record graph must contain at least one record")

    for digest, record in records_by_sha256.items():
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            _error("record graph key must be lowercase SHA-256")

        if not isinstance(record, Mapping):
            _error("record graph values must be record objects")

        _validate_individual_record(
            record,
            contract=contract,
            stage3=stage3,
            stage3_registry=stage3_registry,
            stage3_registry_sha256=stage3_registry_sha256,
            current_unresolved_decision_ids=current_unresolved_decision_ids,
        )

    edges: dict[str, list[str]] = {}

    for digest, record in records_by_sha256.items():
        targets = _validate_provenance_dependencies(
            record,
            records_by_sha256=records_by_sha256,
        )

        if digest in targets:
            _error("record cannot reference itself")

        edges[digest] = targets

        _validate_cross_record_edges(
            record,
            records_by_sha256=records_by_sha256,
        )

    _validate_acyclic(edges)

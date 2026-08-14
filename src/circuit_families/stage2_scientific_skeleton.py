from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ScientificSkeletonError(ValueError):
    pass


REQUIRED = {
    "FS-002": (
        "teacher seed -> phase within teacher -> distillation condition -> student initialization -> discovery method -> thresholds/circuits",
        "phase repeated within teacher",
    ),
    "FS-003": (
        "Teacher seed is the population-level unit.",
    ),
    "FS-006": (
        "separate estimands",
        "never pooled",
    ),
    "FS-011": (
        "smallest recovered qualifying component proportion",
        "intact mask at proportion 1.0",
        "fallback",
    ),
    "FS-012": (
        "procedure-relative",
        "only an upper bound",
        "unknown globally minimal sufficient proportion",
    ),
    "FS-013": (
        "procedure-relative packing lower bound",
        "zero when no qualifying circuit is recovered",
    ),
    "FS-015": (
        "six stated outcome categories are interpretations, not predictions",
    ),
    "FS-016": (
        "gated secondary experiment",
        "all five required controls",
    ),
    "FS-017": (
        "gated extensions outside the core critical path",
    ),
}

CLAIM_LIMITS = {
    "FS-003": ("not independent population replicates",),
    "FS-006": ("Pooling hard and soft students", "prohibited"),
    "FS-012": ("must not be described as the true or globally minimal",),
    "FS-013": ("must not be described as the true number of distinct circuits",),
    "FS-015": ("does not prospectively predict",),
    "FS-016": ("does not establish uniqueness",),
    "FS-017": ("cannot silently become required primary analyses",),
}


def validate_scientific_skeleton(record: dict[str, Any]) -> None:
    expected_root = {
        "schema_version",
        "namespace_version",
        "metadata",
        "authority",
        "freeze_scope",
        "frozen_items",
        "unresolved_register",
        "firewall",
        "claims_boundary",
    }

    if set(record) != expected_root:
        raise ScientificSkeletonError(
            f"root fields mismatch: expected={sorted(expected_root)} actual={sorted(record)}"
        )

    if record["schema_version"] != 1:
        raise ScientificSkeletonError("schema_version must equal 1")

    if record["namespace_version"] != 1:
        raise ScientificSkeletonError("namespace_version must equal 1")

    metadata = record["metadata"]
    if metadata != {
        "record_type": "stage2_scientific_skeleton_freeze",
        "stage": 2,
        "status": "partial_scientific_skeleton_frozen",
        "scientific_execution": False,
        "created_from_commit": "9118ecd239753c54fa5c66766e5d80b54d2a6259",
    }:
        raise ScientificSkeletonError("metadata violates Stage 2 freeze identity")

    if record["freeze_scope"] != {
        "is_partial_freeze": True,
        "numeric_protocol_fully_frozen": False,
        "teacher_registry_frozen": False,
        "stage3_started": False,
    }:
        raise ScientificSkeletonError("freeze_scope violates partial-freeze boundary")

    items = record["frozen_items"]
    ids = [x["item_id"] for x in items]

    expected_ids = [f"FS-{i:03d}" for i in range(1, 19)]

    if ids != expected_ids:
        raise ScientificSkeletonError(
            f"frozen item IDs must be exactly FS-001..FS-018; actual={ids}"
        )

    by_id = {x["item_id"]: x for x in items}

    for item_id, fragments in REQUIRED.items():
        statement = by_id[item_id]["normative_statement"]
        for fragment in fragments:
            if fragment not in statement:
                raise ScientificSkeletonError(
                    f"{item_id} lost required frozen meaning: {fragment}"
                )

    for item_id, fragments in CLAIM_LIMITS.items():
        claim = by_id[item_id]["claim_limit"]
        for fragment in fragments:
            if fragment not in claim:
                raise ScientificSkeletonError(
                    f"{item_id} lost required claim boundary: {fragment}"
                )

    if record["claims_boundary"]["scientific_results_present"] is not False:
        raise ScientificSkeletonError("scientific_results_present must remain false")

    if record["claims_boundary"]["endpoint_outputs_present"] is not False:
        raise ScientificSkeletonError("endpoint_outputs_present must remain false")

    if record["claims_boundary"]["full_numeric_protocol_frozen"] is not False:
        raise ScientificSkeletonError("full_numeric_protocol_frozen must remain false")

    if record["claims_boundary"]["stage3_authorized"] is not False:
        raise ScientificSkeletonError("stage3_authorized must remain false")


def load_scientific_skeleton(path: str | Path) -> dict[str, Any]:
    source = Path(path)

    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScientificSkeletonError(f"invalid JSON: {exc}") from exc

    if not isinstance(record, dict):
        raise ScientificSkeletonError("record root must be an object")

    validate_scientific_skeleton(record)
    return record

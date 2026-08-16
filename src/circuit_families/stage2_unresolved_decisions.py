from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class UnresolvedDecisionError(ValueError):
    pass


ROOT_KEYS = {
    "schema_version",
    "namespace_version",
    "metadata",
    "authority",
    "decisions",
    "coverage",
}

METADATA_EXPECTED = {
    "record_type": "stage2_unresolved_decision_register",
    "stage": 2,
    "status": "open_decisions_registered",
    "scientific_execution": False,
    "created_from_commit": "9118ecd239753c54fa5c66766e5d80b54d2a6259",
}

DECISION_KEYS = {
    "decision_id",
    "decision_family",
    "status",
    "section15_rows",
    "additional_freeze_items",
    "owner",
    "lane",
    "resolution_stage",
    "dependency",
    "forbidden_premature_selection",
}

SECTION15_ROWS = [
    "Teacher registry",
    "Primary phase grid",
    "Student architecture",
    "Student replication",
    "Hard training",
    "Soft target",
    "Soft training",
    "Soft eligibility",
    "Primary fidelity",
    "Endpoint 2 size cap",
    "Endpoint 2 overlap",
    "Discovery roster",
    "Method budgets",
    "Exact evaluation allowance",
    "Phase contrasts",
    "Student cell summary",
    "Missing-cell rule",
    "Production concurrency",
    "Analysis outputs",
]

EXPECTED_DECISIONS = {
    "UD-001": {
        "family": "Teacher registry",
        "rows": ["Teacher registry"],
        "owner": "Alex",
        "lane": "Lane A",
        "stage": "Stage 3",
    },
    "UD-002": {
        "family": "Exact primary phase grid",
        "rows": ["Primary phase grid"],
        "owner": "Alex",
        "lane": "Lane A",
        "stage": "Stage 3",
    },
    "UD-003": {
        "family": "Student architecture and initialization rule",
        "rows": ["Student architecture"],
        "owner": "Austin",
        "lane": "Lane B",
        "stage": "Stage 11",
    },
    "UD-004": {
        "family": "Student replication, attempt cap, replacement and minimum eligibility",
        "rows": ["Student replication"],
        "owner": "Austin",
        "lane": "Lane B",
        "stage": "Stage 11",
    },
    "UD-005": {
        "family": "Hard training configuration",
        "rows": ["Hard training"],
        "owner": "Austin",
        "lane": "Lane B",
        "stage": "Stage 11",
    },
    "UD-006": {
        "family": "Soft target, loss, temperature, training and eligibility",
        "rows": ["Soft target", "Soft training", "Soft eligibility"],
        "owner": "Austin",
        "lane": "Lane B",
        "stage": "Stage 11",
    },
    "UD-007": {
        "family": "Fidelity implementation details, threshold and precision",
        "rows": ["Primary fidelity"],
        "owner": "Alex",
        "lane": "Lane C",
        "stage": "Stage 12",
    },
    "UD-008": {
        "family": "Endpoint 2 component cap, overlap rule and cutoff",
        "rows": ["Endpoint 2 size cap", "Endpoint 2 overlap"],
        "owner": "Alex",
        "lane": "Lane C",
        "stage": "Stage 12",
    },
    "UD-009": {
        "family": "Discovery roster, method versions and native budgets",
        "rows": ["Discovery roster", "Method budgets"],
        "owner": "Alex",
        "lane": "Lane C",
        "stage": "Stage 12",
    },
    "UD-010": {
        "family": "Common exact-evaluation allowance",
        "rows": ["Exact evaluation allowance"],
        "owner": "Alex",
        "lane": "Lane C",
        "stage": "Stage 12",
    },
    "UD-011": {
        "family": "Primary phase contrasts, cell summary and missing-cell rule",
        "rows": ["Phase contrasts", "Student cell summary", "Missing-cell rule"],
        "owner": "Alex",
        "lane": "Lane D",
        "stage": "Stage 13",
    },
    "UD-012": {
        "family": "Required analysis tables, figures and manifests",
        "rows": ["Analysis outputs"],
        "owner": "Alex",
        "lane": "Lane D",
        "stage": "Stage 13",
    },
    "UD-013": {
        "family": "Production concurrency, isolation and merge rule",
        "rows": ["Production concurrency"],
        "owner": "Austin",
        "lane": "Lane D",
        "stage": "Stage 14",
    },
    "UD-014": {
        "family": "Definitive production scope",
        "rows": [],
        "owner": "Joint",
        "lane": "Joint",
        "stage": "Stage 14 / Barrier 3",
    },
}

EXPECTED_ADDITIONAL_FREEZE_ITEMS = {
    "UD-007": [
        {
            "item": "Fidelity sensitivity grid",
            "owner": "Alex",
            "resolution_stage": "Stage 12",
            "status": "unresolved",
        },
    ],
    "UD-008": [
        {
            "item": "Component-cap and overlap sensitivity settings",
            "owner": "Alex",
            "resolution_stage": "Stage 12",
            "status": "unresolved",
        },
        {
            "item": "Packing subset algorithm",
            "owner": "Alex",
            "resolution_stage": "Stage 12",
            "status": "unresolved",
        },
    ],
    "UD-009": [
        {
            "item": "Restart and termination rules",
            "owner": "Alex",
            "resolution_stage": "Stage 12",
            "status": "unresolved",
        },
    ],
    "UD-011": [
        {
            "item": "Direct teacher–student contrast",
            "owner": "Alex",
            "resolution_stage": "Stage 13",
            "status": "unresolved",
        },
        {
            "item": "Realization-dispersion summaries",
            "owner": "Alex",
            "resolution_stage": "Stage 13",
            "status": "unresolved",
        },
    ],
    "UD-012": [
        {
            "item": "Student-attempt failure summaries",
            "owner": "Alex",
            "resolution_stage": "Stage 13",
            "status": "unresolved",
        },
        {
            "item": "Sensitivity interpretation",
            "owner": "Alex",
            "resolution_stage": "Stage 13",
            "status": "unresolved",
        },
        {
            "item": "Outcome-category resolution rules",
            "owner": "Alex",
            "resolution_stage": "Stage 13",
            "status": "unresolved",
        },
    ],
}

IMPLEMENTATION_MASTER_OPEN_ITEMS = {
    entry["item"]
    for entries in EXPECTED_ADDITIONAL_FREEZE_ITEMS.values()
    for entry in entries
}


SETTLED_STAGE2_CONCEPTS = {
    "Exact research question",
    "Experimental hierarchy",
    "Population-level unit",
    "Student-initialization interpretation",
    "Repeated-measurement layers",
    "Hard and soft estimands",
    "Direct teacher evaluation",
    "Hard-target eligibility",
    "Student circuit-fidelity reference",
    "Proposed primary fidelity family",
    "Endpoint 1 definition",
    "Endpoint 1 interpretation limit",
    "Endpoint 2 definition and interpretation",
    "Method-budget interpretation",
    "Outcome-category status",
    "Fourier interchange scope",
    "Gated extensions",
    "Method-development firewall",
}

ALLOWED_OWNERS = {"Alex", "Austin", "Joint"}
ALLOWED_LANES = {"Lane A", "Lane B", "Lane C", "Lane D", "Joint"}
ALLOWED_STAGES = {
    "Stage 3",
    "Stage 11",
    "Stage 12",
    "Stage 13",
    "Stage 14",
    "Stage 14 / Barrier 3",
}


def _portable_path(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise UnresolvedDecisionError(f"{field} must be a non-empty string")
    if Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value):
        raise UnresolvedDecisionError(
            f"{field} must be a portable repository-relative path"
        )
    if ".." in Path(value).parts:
        raise UnresolvedDecisionError(f"{field} must not contain parent traversal")


def validate_unresolved_decisions(record: dict[str, Any]) -> None:
    if set(record) != ROOT_KEYS:
        raise UnresolvedDecisionError(
            f"root fields mismatch: expected={sorted(ROOT_KEYS)} actual={sorted(record)}"
        )

    if record["schema_version"] != 1:
        raise UnresolvedDecisionError("schema_version must equal 1")
    if record["namespace_version"] != 1:
        raise UnresolvedDecisionError("namespace_version must equal 1")
    if record["metadata"] != METADATA_EXPECTED:
        raise UnresolvedDecisionError("metadata violates Stage 2 unresolved-register identity")

    authority = record["authority"]
    if not isinstance(authority, list) or not authority:
        raise UnresolvedDecisionError("authority must be a non-empty array")
    for index, entry in enumerate(authority):
        if not isinstance(entry, dict):
            raise UnresolvedDecisionError(f"authority[{index}] must be an object")
        expected_keys = {
            "authority_id",
            "precedence",
            "repository_path",
            "git_blob",
            "sha256",
        }
        if set(entry) != expected_keys:
            raise UnresolvedDecisionError(
                f"authority[{index}] fields mismatch"
            )
        _portable_path(
            entry["repository_path"],
            f"authority[{index}].repository_path",
        )

    decisions = record["decisions"]
    if not isinstance(decisions, list):
        raise UnresolvedDecisionError("decisions must be an array")

    ids = [d.get("decision_id") for d in decisions if isinstance(d, dict)]
    if len(ids) != len(set(ids)):
        raise UnresolvedDecisionError("duplicate unresolved decision ID")

    expected_ids = set(EXPECTED_DECISIONS)
    actual_ids = set(ids)

    missing_ids = sorted(expected_ids - actual_ids)
    unknown_ids = sorted(actual_ids - expected_ids)

    if unknown_ids:
        raise UnresolvedDecisionError(
            f"unknown unresolved decision IDs: {unknown_ids}"
        )
    if missing_ids:
        raise UnresolvedDecisionError(
            f"missing unresolved decision IDs: {missing_ids}"
        )

    if len(decisions) != len(EXPECTED_DECISIONS):
        raise UnresolvedDecisionError(
            f"decisions must contain exactly {len(EXPECTED_DECISIONS)} entries"
        )

    by_id = {}

    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise UnresolvedDecisionError(f"decisions[{index}] must be an object")
        if set(decision) != DECISION_KEYS:
            raise UnresolvedDecisionError(
                f"decisions[{index}] fields mismatch"
            )

        decision_id = decision["decision_id"]
        expected = EXPECTED_DECISIONS[decision_id]
        by_id[decision_id] = decision

        if decision["status"] != "unresolved":
            raise UnresolvedDecisionError(
                f"{decision_id} is prematurely resolved; status must remain unresolved"
            )

        if decision["owner"] not in ALLOWED_OWNERS:
            raise UnresolvedDecisionError(
                f"{decision_id} has missing or unknown owner: {decision['owner']!r}"
            )

        if decision["lane"] not in ALLOWED_LANES:
            raise UnresolvedDecisionError(
                f"{decision_id} has missing or unknown lane: {decision['lane']!r}"
            )

        if decision["resolution_stage"] not in ALLOWED_STAGES:
            raise UnresolvedDecisionError(
                f"{decision_id} has missing or unknown resolution stage: "
                f"{decision['resolution_stage']!r}"
            )

        if decision["decision_family"] != expected["family"]:
            if decision["decision_family"] in SETTLED_STAGE2_CONCEPTS:
                raise UnresolvedDecisionError(
                    f"{decision_id} attempts to demote settled Stage 2 concept "
                    f"{decision['decision_family']!r} to later-lane discretion"
                )
            raise UnresolvedDecisionError(
                f"{decision_id} decision family mismatch"
            )

        if decision["section15_rows"] != expected["rows"]:
            raise UnresolvedDecisionError(
                f"{decision_id} Section 15 coverage mismatch"
            )

        expected_additional = EXPECTED_ADDITIONAL_FREEZE_ITEMS.get(
            decision_id,
            [],
        )
        if decision["additional_freeze_items"] != expected_additional:
            raise UnresolvedDecisionError(
                f"{decision_id} additional freeze-item coverage mismatch"
            )

        if decision["owner"] != expected["owner"]:
            raise UnresolvedDecisionError(
                f"{decision_id} owner mismatch; expected {expected['owner']}"
            )

        if decision["lane"] != expected["lane"]:
            raise UnresolvedDecisionError(
                f"{decision_id} lane mismatch; expected {expected['lane']}"
            )

        if decision["resolution_stage"] != expected["stage"]:
            raise UnresolvedDecisionError(
                f"{decision_id} resolution stage mismatch; expected {expected['stage']}"
            )

        if not isinstance(decision["dependency"], str) or not decision["dependency"].strip():
            raise UnresolvedDecisionError(
                f"{decision_id} dependency must be non-empty"
            )

        if (
            not isinstance(decision["forbidden_premature_selection"], str)
            or not decision["forbidden_premature_selection"].strip()
        ):
            raise UnresolvedDecisionError(
                f"{decision_id} forbidden_premature_selection must be non-empty"
            )

    additional_seen: dict[str, list[str]] = {}
    for decision_id, decision in by_id.items():
        for entry in decision["additional_freeze_items"]:
            if not isinstance(entry, dict):
                raise UnresolvedDecisionError(
                    f"{decision_id} additional freeze item must be an object"
                )
            expected_entry_keys = {
                "item",
                "owner",
                "resolution_stage",
                "status",
            }
            if set(entry) != expected_entry_keys:
                raise UnresolvedDecisionError(
                    f"{decision_id} additional freeze-item fields mismatch"
                )
            item = entry["item"]
            if item not in IMPLEMENTATION_MASTER_OPEN_ITEMS:
                raise UnresolvedDecisionError(
                    f"{decision_id} contains unknown implementation-master "
                    f"open item: {item!r}"
                )
            if entry["status"] != "unresolved":
                raise UnresolvedDecisionError(
                    f"{decision_id} additional freeze item {item!r} "
                    "must remain unresolved"
                )
            if entry["owner"] != decision["owner"]:
                raise UnresolvedDecisionError(
                    f"{decision_id} additional freeze item {item!r} "
                    "owner does not match accountable decision owner"
                )
            if entry["resolution_stage"] != decision["resolution_stage"]:
                raise UnresolvedDecisionError(
                    f"{decision_id} additional freeze item {item!r} "
                    "resolution stage does not match governing decision stage"
                )
            additional_seen.setdefault(item, []).append(decision_id)

    missing_additional = sorted(
        IMPLEMENTATION_MASTER_OPEN_ITEMS - set(additional_seen)
    )
    duplicate_additional = {
        item: decision_ids
        for item, decision_ids in additional_seen.items()
        if len(decision_ids) != 1
    }

    if missing_additional:
        raise UnresolvedDecisionError(
            "missing implementation-master open-item coverage: "
            f"{missing_additional}"
        )
    if duplicate_additional:
        raise UnresolvedDecisionError(
            "duplicate implementation-master open-item coverage: "
            f"{duplicate_additional}"
        )

    coverage = record["coverage"]
    expected_coverage_keys = {
        "protocol_section15_expected_rows",
        "protocol_section15_expected_count",
        "additional_open_choices",
        "all_decisions_unresolved",
        "recommended_values_are_nonbinding",
    }
    if not isinstance(coverage, dict) or set(coverage) != expected_coverage_keys:
        raise UnresolvedDecisionError("coverage fields mismatch")

    if coverage["protocol_section15_expected_rows"] != SECTION15_ROWS:
        raise UnresolvedDecisionError(
            "coverage.protocol_section15_expected_rows does not exactly match protocol Section 15"
        )
    if coverage["protocol_section15_expected_count"] != 19:
        raise UnresolvedDecisionError(
            "coverage.protocol_section15_expected_count must equal 19"
        )
    if coverage["additional_open_choices"] != ["Definitive production scope"]:
        raise UnresolvedDecisionError(
            "coverage.additional_open_choices must contain only Definitive production scope"
        )
    if coverage["all_decisions_unresolved"] is not True:
        raise UnresolvedDecisionError(
            "coverage.all_decisions_unresolved must remain true"
        )
    if coverage["recommended_values_are_nonbinding"] is not True:
        raise UnresolvedDecisionError(
            "coverage.recommended_values_are_nonbinding must remain true"
        )

    seen: dict[str, list[str]] = {}
    for decision_id, decision in by_id.items():
        for row in decision["section15_rows"]:
            if row not in SECTION15_ROWS:
                raise UnresolvedDecisionError(
                    f"{decision_id} contains unknown Section 15 row: {row}"
                )
            seen.setdefault(row, []).append(decision_id)

    missing_rows = [row for row in SECTION15_ROWS if row not in seen]
    duplicate_rows = {
        row: row_ids
        for row, row_ids in seen.items()
        if len(row_ids) != 1
    }

    if missing_rows:
        raise UnresolvedDecisionError(
            f"missing protocol Section 15 rows: {missing_rows}"
        )
    if duplicate_rows:
        raise UnresolvedDecisionError(
            f"duplicate protocol Section 15 coverage: {duplicate_rows}"
        )


def load_unresolved_decisions(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UnresolvedDecisionError(f"invalid JSON: {exc}") from exc

    if not isinstance(record, dict):
        raise UnresolvedDecisionError("register root must be an object")

    validate_unresolved_decisions(record)
    return record

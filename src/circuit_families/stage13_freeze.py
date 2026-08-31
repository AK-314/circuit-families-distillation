"""Deterministic Stage 13 freeze, manifest expansion, and synthetic reporting.

This module is deliberately planning-only.  It never imports model runners and
never reads checkpoints, registered outputs, credentials, or scheduler state.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any


class Stage13FreezeError(ValueError):
    """Raised when the prospective freeze is incomplete or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return newline-terminated, ASCII, finite canonical JSON."""
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Stage13FreezeError("value is not finite canonical JSON") from exc
    return (payload + "\n").encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_exact_fields(value: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    if set(value) != fields:
        missing = sorted(fields - set(value))
        unknown = sorted(set(value) - fields)
        raise Stage13FreezeError(f"{label} fields mismatch; missing={missing}, unknown={unknown}")


def require_safe_relative_path(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage13FreezeError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Stage13FreezeError(f"{label} escapes its portable root")


def _member_identity(
    *, array_id: str, member_index: int, array_sha256: str, scientific_profile_sha256: str
) -> str:
    return canonical_sha256(
        {
            "array_id": array_id,
            "array_sha256": array_sha256,
            "member_index": member_index,
            "scientific_profile_sha256": scientific_profile_sha256,
        }
    )


def expand_job_arrays(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Independently materialize the closed array specification in memory.

    Array dependencies bind to the canonical ordered identity seal of every
    predecessor array.  This retains exact dependency identity without a huge
    repeated list in every member record.
    """
    require_exact_fields(
        spec,
        {
            "schema_version",
            "scientific_data",
            "production_eligible",
            "definitive_execution_started",
            "approval_sha256",
            "scientific_profile_sha256",
            "analysis_plan_sha256",
            "resource_projection_sha256",
            "seed_derivation",
            "resource_classes",
            "retry_policy",
            "terminal_states",
            "human_gates",
            "arrays",
        },
        label="job-array specification",
    )
    if spec["schema_version"] != "stage13-job-array-spec/v1":
        raise Stage13FreezeError("unsupported job-array schema")
    for boundary in ("scientific_data", "production_eligible", "definitive_execution_started"):
        if spec[boundary] is not False:
            raise Stage13FreezeError(f"planning boundary violated: {boundary}")
    if spec["seed_derivation"] != "stage4-domain-separated-seed/v1":
        raise Stage13FreezeError("seed derivation authority changed")
    if spec["terminal_states"] != [
        "sealed_success",
        "failed",
        "unavailable",
        "censored",
        "not_admitted",
    ]:
        raise Stage13FreezeError("terminal-state roster changed")

    arrays = spec["arrays"]
    if not isinstance(arrays, list) or not arrays:
        raise Stage13FreezeError("job arrays must be a non-empty list")
    ids = [item.get("array_id") for item in arrays if isinstance(item, dict)]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates or len(ids) != len(arrays):
        raise Stage13FreezeError(f"duplicate or invalid array identities: {duplicates}")

    seals: dict[str, str] = {}
    ordered_members: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    scope_operation_counts: dict[str, Counter[str]] = {}
    for array in arrays:
        require_exact_fields(
            array,
            {
                "array_id",
                "family",
                "scope",
                "tier",
                "protected",
                "count",
                "operation_counts_per_member",
                "dimensions",
                "dependencies",
                "producer_interface",
                "config_sha256",
                "resource_class",
                "priority",
                "concurrency_group",
                "shedding",
                "output_template",
                "retention",
                "retry_class",
                "unavailable_consequence",
            },
            label=f"array {array.get('array_id')!r}",
        )
        count = array["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise Stage13FreezeError("array count must be a positive integer")
        dependencies = array["dependencies"]
        if not isinstance(dependencies, list) or len(set(dependencies)) != len(dependencies):
            raise Stage13FreezeError("array dependencies must be a unique list")
        missing = [item for item in dependencies if item not in seals]
        if missing:
            raise Stage13FreezeError(
                f"array {array['array_id']} has cyclic, forward, or missing dependencies: {missing}"
            )
        require_safe_relative_path(array["output_template"], label="output_template")
        array_payload = dict(array)
        array_sha256 = canonical_sha256(array_payload)
        dependency_seals = {item: seals[item] for item in dependencies}
        member_ids: list[str] = []
        for member_index in range(count):
            logical_job_id = _member_identity(
                array_id=array["array_id"],
                member_index=member_index,
                array_sha256=array_sha256,
                scientific_profile_sha256=spec["scientific_profile_sha256"],
            )
            seed = int(
                canonical_sha256(
                    {
                        "domain": "stage13/job-member/v1",
                        "logical_job_id": logical_job_id,
                        "retry_attempt_excluded": True,
                    }
                )[:16],
                16,
            )
            record = {
                "logical_job_id": logical_job_id,
                "array_id": array["array_id"],
                "member_index": member_index,
                "family": array["family"],
                "scope": array["scope"],
                "tier": array["tier"],
                "protected": array["protected"],
                "dimensions": array["dimensions"],
                "dependency_array_seals": dependency_seals,
                "producer_interface": array["producer_interface"],
                "config_sha256": array["config_sha256"],
                "resource_class": array["resource_class"],
                "priority": array["priority"],
                "concurrency_group": array["concurrency_group"],
                "output_relative_path": array["output_template"].format(index=member_index),
                "seed": seed,
                "retry_class": array["retry_class"],
                "scientific_data": False,
                "production_eligible": False,
            }
            ordered_members.append(record)
            member_ids.append(logical_job_id)
            family_counts[array["family"]] += 1
            tier_counts[array["tier"]] += 1
            scope_counts[array["scope"]] += 1
            scope_operations = scope_operation_counts.setdefault(array["scope"], Counter())
            for operation, amount in array["operation_counts_per_member"].items():
                if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                    raise Stage13FreezeError("operation counts must be non-negative integers")
                operation_counts[operation] += amount
                scope_operations[operation] += amount
        seals[array["array_id"]] = canonical_sha256(member_ids)

    identity_order = [item["logical_job_id"] for item in ordered_members]
    return {
        "schema_version": "stage13-expanded-manifest-seal/v1",
        "scientific_data": False,
        "production_eligible": False,
        "logical_job_count": len(ordered_members),
        "ordered_identity_sha256": canonical_sha256(identity_order),
        "canonical_members_sha256": canonical_sha256(ordered_members),
        "array_seals": seals,
        "counts_by_family": dict(sorted(family_counts.items())),
        "counts_by_tier": dict(sorted(tier_counts.items())),
        "counts_by_scope": dict(sorted(scope_counts.items())),
        "scientific_operation_counts": dict(sorted(operation_counts.items())),
        "scientific_operation_counts_by_scope": {
            scope: dict(sorted(counts.items()))
            for scope, counts in sorted(scope_operation_counts.items())
        },
        "members": ordered_members,
    }


def expansion_seal(expanded: Mapping[str, Any]) -> dict[str, Any]:
    """Strip materialized members while retaining independently verifiable seals."""
    return {key: value for key, value in expanded.items() if key != "members"}


def generate_synthetic_report(
    fixture: Mapping[str, Any], analysis_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Generate all frozen report surfaces from non-scientific topology fixtures."""
    if fixture.get("schema_version") != "stage13-synthetic-complete-fixture/v1":
        raise Stage13FreezeError("unsupported synthetic fixture")
    if analysis_plan.get("schema_version") != "stage13-analysis-report-plan/v1":
        raise Stage13FreezeError("unsupported analysis plan")
    if fixture.get("scientific_data") is not False:
        raise Stage13FreezeError("synthetic fixture is mislabeled")
    required_states = {
        "eligible",
        "failed_attempt",
        "insufficient_eligible",
        "teacher_unavailable",
        "phase_unavailable",
        "budget_exhausted",
        "search_failure",
        "packing_zero",
        "packing_lower_bound",
        "packing_censored",
        "nonfinite_rejected",
        "corrupted_rejected",
        "duplicate_rejected",
        "stale_rejected",
        "conflicting_rejected",
    }
    observed = {item["state"] for item in fixture["terminal_cases"]}
    if not required_states <= observed:
        raise Stage13FreezeError(
            f"synthetic terminal coverage missing {sorted(required_states - observed)}"
        )
    fourier = {item["outcome"] for item in fixture["fourier_cases"]}
    if fourier != {"winning", "tying", "losing", "failing", "incomplete"}:
        raise Stage13FreezeError("Fourier synthetic outcome coverage is incomplete")

    table_rows = []
    for order, surface in enumerate(analysis_plan["report_surfaces"], start=1):
        table_rows.append(
            {
                "order": order,
                "surface_id": surface["surface_id"],
                "kind": surface["kind"],
                "dimensions": surface["dimensions"],
                "denominators": surface["denominators"],
                "missingness": surface["missingness"],
                "synthetic_case_ids": [item["case_id"] for item in fixture["terminal_cases"]],
            }
        )
    statuses = ["met", "not_met", "indeterminate", "unavailable", "not_met", "indeterminate"]
    claim_rows = [
        {
            "category": category,
            "synthetic_status": statuses[index],
            "scientific_conclusion": None,
            "reason": "synthetic topology exercise only",
        }
        for index, category in enumerate(analysis_plan["outcome_categories"])
    ]
    return {
        "schema_version": "stage13-synthetic-complete-report/v1",
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
        "fixture_sha256": canonical_sha256(fixture),
        "analysis_plan_sha256": canonical_sha256(analysis_plan),
        "coverage": fixture["coverage"],
        "terminal_case_inventory": fixture["terminal_cases"],
        "fourier_case_inventory": fixture["fourier_cases"],
        "report_surfaces": table_rows,
        "claim_resolution_categories": claim_rows,
        "rejections_are_excluded_from_estimands": True,
        "manual_editing_required": False,
        "scientific_claim": None,
    }

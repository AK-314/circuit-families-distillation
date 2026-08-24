"""Injected synthetic technical discovery profiles for Stage 6D."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    RESOURCE_WARNING,
    UNRESOLVED_DECISIONS,
    TechnicalDiscoveryProfile,
)


def _require_exact_keys(
    record: dict[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(record)
    if actual != expected:
        raise ValueError(
            f"{label} keys mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def technical_profile_from_record(
    record: dict[str, Any],
) -> TechnicalDiscoveryProfile:
    expected = {
        "profile_id",
        "profile_version",
        "method_name",
        "method_version",
        "configuration_reference",
        "native_budget_unit",
        "native_budget_allowance",
        "exact_evaluation_allowance",
        "maximum_restarts",
        "technical_only",
        "production_eligible",
        "unresolved_decisions",
        "resource_warning",
    }
    _require_exact_keys(record, expected, label="technical discovery profile")

    unresolved = record["unresolved_decisions"]
    if not isinstance(unresolved, list) or not all(
        isinstance(item, str) for item in unresolved
    ):
        raise ValueError("unresolved_decisions must be a list of strings")

    return TechnicalDiscoveryProfile(
        profile_id=record["profile_id"],
        profile_version=record["profile_version"],
        method_name=record["method_name"],
        method_version=record["method_version"],
        configuration_reference=record["configuration_reference"],
        native_budget_unit=record["native_budget_unit"],
        native_budget_allowance=record["native_budget_allowance"],
        exact_evaluation_allowance=record["exact_evaluation_allowance"],
        maximum_restarts=record["maximum_restarts"],
        technical_only=record["technical_only"],
        production_eligible=record["production_eligible"],
        unresolved_decisions=tuple(unresolved),
        resource_warning=record["resource_warning"],
    )


def load_technical_profiles(
    path: str | Path,
) -> tuple[TechnicalDiscoveryProfile, ...]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("technical discovery profile set must be an object")

    _require_exact_keys(
        payload,
        {
            "schema_version",
            "technical_only",
            "production_eligible",
            "unresolved_decisions",
            "resource_warning",
            "profiles",
        },
        label="technical discovery profile set",
    )

    if payload["schema_version"] != "stage6d-technical-discovery-profiles/v1":
        raise ValueError("unsupported Stage 6D profile schema version")
    if payload["technical_only"] is not True:
        raise ValueError("profile set must be technical_only")
    if payload["production_eligible"] is not False:
        raise ValueError("profile set must not be production eligible")
    if tuple(payload["unresolved_decisions"]) != UNRESOLVED_DECISIONS:
        raise ValueError("profile set must preserve unresolved decisions")
    if payload["resource_warning"] != RESOURCE_WARNING:
        raise ValueError("profile set has non-canonical resource warning")

    records = payload["profiles"]
    if not isinstance(records, list) or not records:
        raise ValueError("profiles must be a non-empty list")

    profiles = tuple(technical_profile_from_record(item) for item in records)

    identifiers = [profile.profile_id for profile in profiles]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("profile_id values must be unique")

    return profiles

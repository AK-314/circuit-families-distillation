from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_families.stage2_method_development_firewall import (
    MethodDevelopmentFirewallError,
    assert_pilot_evidence_may_not_select_protocol_values,
    load_firewall_register,
    require_registered_development_output,
    validate_firewall_register,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTER = (
    ROOT
    / "followup"
    / "manifests"
    / "stage2_excluded_development_register_v1.json"
)


def canonical() -> dict:
    return json.loads(REGISTER.read_text(encoding="utf-8"))


def valid_entry() -> dict:
    return {
        "exclusion_id": "EXCL-001",
        "artifact_identity": "followup/excluded_development/pilot_endpoint.json",
        "development_context": "Stage 2 technical pilot",
        "exclusion_reason": "Pre-freeze endpoint-producing development output",
        "endpoint_values_emitted": True,
        "primary_analysis_eligible": False,
        "scientific_selection_eligible": False,
        "regeneration_required": True,
        "regenerate_after": "Relevant later freeze and production authorization",
        "disposition": "registered_excluded",
        "promotion_in_place_permitted": False,
    }


def test_canonical_empty_register_passes() -> None:
    record = load_firewall_register(REGISTER)
    assert record["entries"] == []
    assert record["firewall"]["primary_analysis_eligible"] is False


def test_valid_excluded_entry_passes() -> None:
    record = canonical()
    record["entries"] = [valid_entry()]
    validate_firewall_register(record)


def test_unregistered_development_output_is_rejected() -> None:
    record = canonical()
    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="unregistered development output",
    ):
        require_registered_development_output(
            record,
            "followup/excluded_development/pilot_endpoint.json",
        )


def test_attempted_primary_output_classification_is_rejected() -> None:
    record = canonical()
    entry = valid_entry()
    entry["primary_analysis_eligible"] = True
    record["entries"] = [entry]

    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="attempted primary-output classification",
    ):
        validate_firewall_register(record)


def test_attempted_scientific_selection_classification_is_rejected() -> None:
    record = canonical()
    entry = valid_entry()
    entry["scientific_selection_eligible"] = True
    record["entries"] = [entry]

    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="attempted scientific-selection classification",
    ):
        validate_firewall_register(record)


def test_missing_regeneration_requirement_is_rejected() -> None:
    record = canonical()
    entry = valid_entry()
    entry["regeneration_required"] = False
    record["entries"] = [entry]

    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="missing mandatory post-freeze regeneration",
    ):
        validate_firewall_register(record)


def test_predecessor_output_style_root_is_rejected() -> None:
    record = canonical()
    entry = valid_entry()
    entry["artifact_identity"] = "outputs/pilot_endpoint.json"
    record["entries"] = [entry]

    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="forbidden predecessor/output-style root",
    ):
        validate_firewall_register(record)


def test_absolute_private_canonical_path_is_rejected() -> None:
    record = canonical()
    entry = valid_entry()
    entry["artifact_identity"] = (
        "/Users/example/private-repository/outputs/pilot.json"
    )
    record["entries"] = [entry]

    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="absolute private canonical path",
    ):
        validate_firewall_register(record)


def test_followup_output_outside_excluded_root_is_rejected() -> None:
    record = canonical()
    entry = valid_entry()
    entry["artifact_identity"] = "followup/reviewed/tables/pilot_endpoint.json"
    record["entries"] = [entry]

    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="outside followup/excluded_development",
    ):
        validate_firewall_register(record)


def test_pilot_phase_effect_cannot_choose_protocol_value() -> None:
    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="forbidden evidence",
    ):
        assert_pilot_evidence_may_not_select_protocol_values(
            evidence_kind="pilot_phase_effect",
            intended_use="choose_protocol_value",
        )


def test_pilot_condition_effect_cannot_choose_threshold() -> None:
    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="forbidden evidence",
    ):
        assert_pilot_evidence_may_not_select_protocol_values(
            evidence_kind="pilot_condition_effect",
            intended_use="choose_threshold",
        )


def test_non_effect_runtime_evidence_may_support_technical_work() -> None:
    assert_pilot_evidence_may_not_select_protocol_values(
        evidence_kind="runtime_measurement",
        intended_use="technical_feasibility",
    )


def test_promotion_in_place_is_rejected() -> None:
    record = canonical()
    entry = valid_entry()
    entry["promotion_in_place_permitted"] = True
    record["entries"] = [entry]

    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="cannot be promoted in place",
    ):
        validate_firewall_register(record)

def test_parent_traversal_artifact_identity_is_rejected() -> None:
    record = canonical()
    entry = valid_entry()
    entry["artifact_identity"] = (
        "followup/excluded_development/../pilot_endpoint.json"
    )
    record["entries"] = [entry]

    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="parent traversal",
    ):
        validate_firewall_register(record)

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from circuit_families.stage2_method_development_firewall import (
    MethodDevelopmentFirewallError,
    validate_firewall_register,
)
from circuit_families.stage2_scientific_skeleton import (
    ScientificSkeletonError,
    validate_scientific_skeleton,
)
from circuit_families.stage2_unresolved_decisions import (
    UnresolvedDecisionError,
    validate_unresolved_decisions,
)


ROOT = Path(__file__).resolve().parents[1]
SKELETON = ROOT / "followup/manifests/stage2_scientific_skeleton_freeze_v1.json"
UNRESOLVED = ROOT / "followup/configs/stage2_unresolved_decisions_v1.json"
FIREWALL = ROOT / "followup/manifests/stage2_excluded_development_register_v1.json"

spec = importlib.util.spec_from_file_location(
    "validate_followup_stage2",
    ROOT / "scripts/validate_followup_stage2.py",
)
assert spec is not None and spec.loader is not None
portable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portable)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonical_records_pass() -> None:
    validate_scientific_skeleton(load(SKELETON))
    validate_unresolved_decisions(load(UNRESOLVED))
    validate_firewall_register(load(FIREWALL))


def test_stage3_started_rejected() -> None:
    record = load(SKELETON)
    record["freeze_scope"]["stage3_started"] = True
    with pytest.raises(ScientificSkeletonError, match="partial-freeze boundary"):
        validate_scientific_skeleton(record)


def test_stage3_authorized_rejected() -> None:
    record = load(SKELETON)
    record["claims_boundary"]["stage3_authorized"] = True
    with pytest.raises(ScientificSkeletonError, match="stage3_authorized"):
        validate_scientific_skeleton(record)


def test_numeric_full_freeze_rejected() -> None:
    record = load(SKELETON)
    record["freeze_scope"]["numeric_protocol_fully_frozen"] = True
    with pytest.raises(ScientificSkeletonError, match="partial-freeze boundary"):
        validate_scientific_skeleton(record)


def test_production_ready_rejected_by_portable_boundary() -> None:
    skeleton = load(SKELETON)
    unresolved = load(UNRESOLVED)
    firewall = load(FIREWALL)
    skeleton["claims_boundary"]["production_ready"] = True

    with pytest.raises(
        portable.PortableValidationError,
        match="production_ready",
    ):
        portable.validate_boundary(skeleton, unresolved, firewall)


def test_resolved_decision_rejected() -> None:
    record = load(UNRESOLVED)
    record["decisions"][0]["status"] = "resolved"
    with pytest.raises(UnresolvedDecisionError, match="prematurely resolved"):
        validate_unresolved_decisions(record)


def test_absolute_authority_path_rejected() -> None:
    record = load(UNRESOLVED)
    record["authority"][0]["repository_path"] = "/Users/example/private/file.md"
    with pytest.raises(
        UnresolvedDecisionError,
        match="portable repository-relative path",
    ):
        validate_unresolved_decisions(record)


def test_parent_traversal_authority_path_rejected() -> None:
    record = load(UNRESOLVED)
    record["authority"][0]["repository_path"] = "../outside/file.md"
    with pytest.raises(UnresolvedDecisionError, match="parent traversal"):
        validate_unresolved_decisions(record)


def test_stale_authority_hash_rejected() -> None:
    skeleton = load(SKELETON)
    unresolved = load(UNRESOLVED)
    firewall = load(FIREWALL)
    skeleton["authority"][0]["sha256"] = "0" * 64

    with pytest.raises(
        portable.PortableValidationError,
        match="stale authority",
    ):
        portable.validate_authority([skeleton, unresolved, firewall])


def test_firewall_primary_promotion_rejected() -> None:
    record = load(FIREWALL)
    record["firewall"]["primary_analysis_eligible"] = True
    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="firewall semantics changed",
    ):
        validate_firewall_register(record)


def test_firewall_selection_promotion_rejected() -> None:
    record = load(FIREWALL)
    record["firewall"]["scientific_selection_eligible"] = True
    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="firewall semantics changed",
    ):
        validate_firewall_register(record)


def test_firewall_regeneration_removal_rejected() -> None:
    record = load(FIREWALL)
    record["firewall"]["post_freeze_regeneration_required"] = False
    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="firewall semantics changed",
    ):
        validate_firewall_register(record)


def test_excluded_output_promotion_in_place_rejected() -> None:
    record = load(FIREWALL)
    record["entries"] = [{
        "exclusion_id": "EXCL-001",
        "artifact_identity": "followup/excluded_development/pilot.json",
        "development_context": "Stage 2 technical pilot",
        "exclusion_reason": "Pre-freeze endpoint-producing development output",
        "endpoint_values_emitted": True,
        "primary_analysis_eligible": False,
        "scientific_selection_eligible": False,
        "regeneration_required": True,
        "regenerate_after": "Relevant later freeze",
        "disposition": "registered_excluded",
        "promotion_in_place_permitted": True,
    }]

    with pytest.raises(
        MethodDevelopmentFirewallError,
        match="cannot be promoted in place",
    ):
        validate_firewall_register(record)

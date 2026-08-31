#!/usr/bin/env python3
"""Read-only validation for the pre-approval Stage 13 decision dossier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOSSIER = REPOSITORY_ROOT / "followup/decisions/stage13_decision_dossier_v2.json"
SCHEMA = REPOSITORY_ROOT / "followup/schemas/stage13/decision_dossier_v2.schema.json"
INVENTORY = REPOSITORY_ROOT / "followup/manifests/stage13_compatibility_inventory_v1.json"
PART_A = REPOSITORY_ROOT / "followup/manifests/stage13_part_a_evidence_v1.json"
BENCHMARK = REPOSITORY_ROOT / "followup/benchmarks/stage13_search_profile_benchmark_v1.json"
PROJECTION = REPOSITORY_ROOT / "followup/manifests/stage13_package_resource_projection_v2.json"

EXPECTED_DECISIONS = tuple([f"RD-{index:03d}" for index in range(1, 13)] + ["RD-014"])
EXPECTED_HEADS = {
    "6e027a8c22e5228dadad8707f3a262e78028f855",
    "0ba93daece3328d9bc2aa5ad2296f200a92f26fb",
    "b11c32aa0ef2328a86f41096f188a7a055352fd8",
    "c4065c2977f2d4e0cd09a54014f60f993f08aceb",
    "21d80d72376f004656d827069fc53bf55864ce61",
    "955d357609f8916797d2ea929f924e8f120e2334",
    "464904c19c3f4bafdd0fc424aa7be4fb3350a831",
    "ed4d9f3c8dc334a8dd80f8439dbd75826fd75e0f",
    "528ee302b5659f7848d098d27bb60d1d26c397eb",
}
ROOT_FIELDS = {
    "schema_version",
    "status",
    "created_date",
    "implementation_base",
    "authority_owner",
    "scientific_data",
    "production_eligible",
    "definitive_execution_started",
    "evidence_firewall",
    "approval",
    "recommended_package_id",
    "decisions",
    "deferred_decisions",
    "package_alternatives",
    "supersedes",
    "benchmark_evidence",
    "resource_projection",
}
DECISION_FIELDS = {
    "decision_id",
    "question",
    "downstream_consequences",
    "stage11_direction",
    "stage12_evidence",
    "viable_options",
    "invalid_options",
    "protected_tier_consequences",
    "recommendation",
    "rationale",
    "uncertainty",
    "amendment_consequence",
    "requires_alex_choice",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_dossier_mapping(dossier: dict[str, Any]) -> None:
    if set(dossier) != ROOT_FIELDS:
        raise ValueError("dossier root fields do not match the closed schema")
    if dossier["schema_version"] != "stage13-decision-dossier/v2":
        raise ValueError("unsupported dossier schema version")
    if dossier["status"] != "recommended_package_pending_alex_approval":
        raise ValueError("pre-approval dossier has an invalid status")
    if dossier["implementation_base"] != "7976c98cc83a6df098ae0ef8c59b56027a7f4899":
        raise ValueError("implementation base changed")
    if dossier["authority_owner"] != "Alex":
        raise ValueError("authority owner changed")
    if dossier["recommended_package_id"] != "stage13-package-a-conservative-protected/v2":
        raise ValueError("recommended package identifier changed")
    firewall = dossier["evidence_firewall"]
    if set(firewall) != {
        "permitted_evidence_used",
        "excluded_evidence",
        "registered_or_private_artifacts_accessed",
    }:
        raise ValueError("evidence firewall fields do not match the closed schema")
    if not firewall["permitted_evidence_used"] or not firewall["excluded_evidence"]:
        raise ValueError("evidence firewall inventories must not be empty")
    if firewall["registered_or_private_artifacts_accessed"] is not False:
        raise ValueError("registered/private artifact boundary violated")
    approval = dossier["approval"]
    if set(approval) != {
        "status",
        "required_approver",
        "approved_package_id",
        "approval_date",
        "approval_text",
        "dossier_sha256",
    }:
        raise ValueError("approval fields do not match the closed schema")
    if approval != {
        "status": "pending",
        "required_approver": "Alex",
        "approved_package_id": None,
        "approval_date": None,
        "approval_text": None,
        "dossier_sha256": None,
    }:
        raise ValueError("pre-approval dossier must not contain fabricated approval")
    decisions = dossier["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(EXPECTED_DECISIONS):
        raise ValueError("decision roster length mismatch")
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != DECISION_FIELDS:
            raise ValueError("decision fields do not match the closed schema")
        if decision["requires_alex_choice"] is not True:
            raise ValueError("genuinely underdetermined decision lacks Alex gate")
        if len(decision["viable_options"]) < 2 or not decision["invalid_options"]:
            raise ValueError("decision options are incomplete")
        if set(decision["protected_tier_consequences"]) != {"tier1", "tier2", "tier3"}:
            raise ValueError("decision tier consequences are incomplete")
        if not decision["recommendation"]:
            raise ValueError("decision recommendation is empty")
    if dossier["deferred_decisions"][0]["decision_id"] != "RD-013":
        raise ValueError("RD-013 must remain deferred")
    if len(dossier["package_alternatives"]) < 3:
        raise ValueError("complete package alternatives are missing")


def validate() -> dict[str, Any]:
    dossier = _load(DOSSIER)
    _load(SCHEMA)
    validate_dossier_mapping(dossier)

    decision_ids = tuple(item["decision_id"] for item in dossier["decisions"])
    if decision_ids != EXPECTED_DECISIONS:
        raise ValueError(f"decision roster/order mismatch: {decision_ids!r}")
    if dossier["approval"]["status"] != "pending":
        raise ValueError("the dossier itself must remain immutable pre-approval evidence")
    if dossier["deferred_decisions"] != [
        {
            "decision_id": "RD-013",
            "status": "pending_stage14",
            "reason": dossier["deferred_decisions"][0]["reason"],
        }
    ]:
        raise ValueError("RD-013 must be the sole fully deferred decision")

    serialized = json.dumps(
        dossier, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    canonical_sha256 = hashlib.sha256(serialized).hexdigest()

    inventory = _load(INVENTORY)
    part_a = _load(PART_A)
    benchmark = _load(BENCHMARK)
    projection = _load(PROJECTION)
    if dossier["benchmark_evidence"]["file_sha256"] != _sha256(BENCHMARK):
        raise ValueError("dossier benchmark binding mismatch")
    if dossier["resource_projection"]["file_sha256"] != _sha256(PROJECTION):
        raise ValueError("dossier projection binding mismatch")
    if benchmark["scientific_data"] is not False or benchmark["production_eligible"] is not False:
        raise ValueError("benchmark evidence crossed the scientific boundary")
    if benchmark["registered_or_private_artifacts_accessed"] is not False:
        raise ValueError("benchmark accessed prohibited artifacts")
    if projection["envelope"] != {
        "total_hours": 96,
        "science_execution_hours": 84,
        "final_audit_reserve_hours": 12,
        "cpu_cores": 256,
        "cuda_gpus": 16,
        "scratch_tib": 4,
        "persistent_tib": 1,
        "grant_verified": False,
    }:
        raise ValueError("resource envelope changed")
    if set(projection["packages"]) != {"A", "B", "C"}:
        raise ValueError("resource projection package roster changed")
    for package in projection["packages"].values():
        if [item["scenario"] for item in package["scenarios"]] != [
            "lower", "central", "conservative"
        ]:
            raise ValueError("resource scenarios are incomplete")
        if package["job_counts"]["exact_calibration_masks"] != 2**18:
            raise ValueError("exact calibration count changed")
    decision_map = {item["decision_id"]: item for item in dossier["decisions"]}
    architecture_anchors = decision_map["RD-002"]["recommendation"][
        "tier2_assignment"
    ]["anchor_teacher_seeds"]
    if architecture_anchors != [0, 1, 2, 3, 4]:
        raise ValueError("architecture anchor panel changed")
    basis_anchors = decision_map["RD-004"]["recommendation"]["assignment"][
        "anchor_teacher_seeds"
    ]
    if basis_anchors != [0, 1, 2, 3, 4]:
        raise ValueError("basis anchor panel changed")
    grid = decision_map["RD-006"]["recommendation"]["sensitivity_grid"]
    if grid["cartesian_cell_count"] != 12 or len(grid["cells"]) != 12:
        raise ValueError("packing sensitivity grid is incomplete")
    rd11 = decision_map["RD-011"]["recommendation"]
    if rd11["primary_task_modes"] != [[1, 1], [2, 2], [3, 3], [4, 4]]:
        raise ValueError("Fourier task-mode support changed")
    if len(rd11["conditions"]) != 6 or rd11["uniqueness_claim"] is not False:
        raise ValueError("Fourier controls or claim boundary changed")
    if decision_map["RD-014"]["recommendation"]["final_audit_reserve_hours"] != 12:
        raise ValueError("audit reserve changed")
    if inventory["implementation_base"] != dossier["implementation_base"]:
        raise ValueError("inventory and dossier implementation bases differ")
    if part_a["repository"]["implementation_base"] != dossier["implementation_base"]:
        raise ValueError("Part A and dossier implementation bases differ")
    heads = {item["sha"] for item in inventory["accepted_heads"]}
    if heads != EXPECTED_HEADS:
        raise ValueError("accepted Stage 11/12 head inventory mismatch")
    if any(item["compatibility"] != "closed" for item in inventory["contracts"]):
        raise ValueError("compatibility inventory contains an open contract")
    if inventory["collision_audit"]["blocking_incompatibilities"]:
        raise ValueError("compatibility inventory contains a blocking incompatibility")
    if part_a["blocking_incompatibilities"]:
        raise ValueError("Part A contains a blocking incompatibility")
    if any(
        lane["validate_only"] != "PASS" or lane["missing_exports"]
        for lane in part_a["stage12_import_audit"].values()
    ):
        raise ValueError("Part A Stage 12 import/validation audit is incomplete")
    for record in (dossier, inventory, part_a):
        if record["scientific_data"] is not False:
            raise ValueError("scientific_data boundary violated")
        if record["production_eligible"] is not False:
            raise ValueError("production_eligible boundary violated")
        if record["definitive_execution_started"] is not False:
            raise ValueError("definitive execution boundary violated")

    return {
        "schema_version": "stage13-decision-dossier-validation/v2",
        "implementation_base": dossier["implementation_base"],
        "decision_ids": list(decision_ids),
        "dossier_file_sha256": _sha256(DOSSIER),
        "dossier_canonical_sha256": canonical_sha256,
        "compatibility_inventory_sha256": _sha256(INVENTORY),
        "part_a_evidence_sha256": _sha256(PART_A),
        "search_profile_benchmark_sha256": _sha256(BENCHMARK),
        "package_resource_projection_sha256": _sha256(PROJECTION),
        "compatibility_contract_count": len(inventory["contracts"]),
        "approval_status": "pending",
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
        "validation": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(validate(), allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

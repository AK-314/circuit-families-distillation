#!/usr/bin/env python3
"""Read-only, cwd-independent validation of the complete Stage 13 freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from circuit_families.stage13_freeze import (
    Stage13FreezeError,
    canonical_json_bytes,
    expand_job_arrays,
    expansion_seal,
    generate_synthetic_report,
    require_exact_fields,
)
from scripts.validate_stage13_decision_dossier import validate as validate_dossier

ROOT = Path(__file__).resolve().parents[1]
DOSSIER_SHA = "642118fe30e1c435fc05c656ac6167446c76a445c9a1efd376d5a7c23410b1f4"
PACKAGE_ID = "stage13-package-a-protected-core-optional-five-task/v3"
APPROVAL_TEXT = (
    "I approve `stage13-package-a-protected-core-optional-five-task/v3` exactly as "
    "described in the Stage 13 v3 decision dossier with SHA-256 "
    "`642118fe30e1c435fc05c656ac6167446c76a445c9a1efd376d5a7c23410b1f4`; "
    "Parts C–H may proceed subject to the Stage 14 protected-core launch gate."
)

PATHS = {
    "approval": ROOT / "followup/decisions/stage13_approval_v1.json",
    "dossier": ROOT / "followup/decisions/stage13_decision_dossier_v3.json",
    "protocol": ROOT / "followup/configs/stage13/frozen_scientific_protocol_v1.json",
    "profiles": ROOT / "followup/configs/stage13/production_profiles_v1.json",
    "analysis": ROOT / "followup/configs/stage13/analysis_report_plan_v1.json",
    "arrays": ROOT / "followup/manifests/stage13_job_array_spec_v1.json",
    "seal": ROOT / "followup/manifests/stage13_expanded_manifest_seal_v1.json",
    "root": ROOT / "followup/manifests/stage13_campaign_root_v1.json",
    "excluded": ROOT / "followup/manifests/stage13_excluded_evidence_v1.json",
    "resource": ROOT / "followup/manifests/stage13_scope_resource_projection_v3.json",
    "fixture": ROOT / "followup/fixtures/stage13/synthetic_complete_fixture_v1.json",
    "report": ROOT / "followup/reports/stage13_synthetic_complete_report_v1.json",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage13FreezeError(f"{path} must contain one object")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boundary(record: dict[str, Any], *, label: str) -> None:
    for key in ("scientific_data", "production_eligible", "definitive_execution_started"):
        if record.get(key) is not False:
            raise Stage13FreezeError(f"{label} violates {key}=false boundary")


def validate_approval(value: dict[str, Any]) -> None:
    require_exact_fields(
        value,
        {
            "schema_version",
            "authority",
            "approval_date",
            "approval_text",
            "approved_package_id",
            "dossier_path",
            "dossier_file_sha256",
            "scope",
            "scientific_data",
            "production_eligible",
            "definitive_execution_started",
        },
        label="approval",
    )
    if value["schema_version"] != "stage13-approval/v1" or value["authority"] != "Alex":
        raise Stage13FreezeError("approval authority changed")
    if value["approval_date"] != "2026-08-31" or value["approval_text"] != APPROVAL_TEXT:
        raise Stage13FreezeError("approval is not the exact verbatim authority record")
    if value["approved_package_id"] != PACKAGE_ID or value["dossier_file_sha256"] != DOSSIER_SHA:
        raise Stage13FreezeError("approval identity binding changed")
    _boundary(value, label="approval")


def validate_protocol(value: dict[str, Any], dossier: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "approval_sha256",
        "approved_dossier_sha256",
        "package_id",
        "resolution_status",
        "population_and_tasks",
        "architecture",
        "students_and_training",
        "basis",
        "fidelity",
        "endpoints",
        "discovery",
        "nulls",
        "calibration",
        "interpretation_units",
        "optional_tier3_attempt_policy",
        "scientific_data",
        "production_eligible",
        "definitive_execution_started",
    }
    require_exact_fields(value, expected, label="scientific protocol")
    if value["schema_version"] != "stage13-frozen-scientific-protocol/v1":
        raise Stage13FreezeError("scientific protocol version changed")
    decisions = {item["decision_id"]: item["recommendation"] for item in dossier["decisions"]}
    bindings = {
        "population_and_tasks": "RD-001",
        "architecture": "RD-002",
        "students_and_training": "RD-003",
        "basis": "RD-004",
        "fidelity": "RD-005",
        "endpoints": "RD-006",
        "discovery": "RD-007",
        "nulls": "RD-008",
        "calibration": "RD-009",
    }
    for field, decision in bindings.items():
        if value[field] != decisions[decision]:
            raise Stage13FreezeError(f"post-approval decision edit: {decision}")
    statuses = value["resolution_status"]
    if any(statuses[f"RD-{index:03d}"] != "resolved" for index in range(1, 13)):
        raise Stage13FreezeError("RD-001 through RD-012 must be resolved")
    if statuses["RD-013"] != "pending_stage14":
        raise Stage13FreezeError("RD-013 boundary changed")
    if value["population_and_tasks"]["teacher_seeds"] != list(range(15)):
        raise Stage13FreezeError("Task 1 fixed seed roster changed")
    if value["population_and_tasks"]["replacement_rule"].startswith("no replacement") is False:
        raise Stage13FreezeError("no-replacement rule changed")
    units = value["interpretation_units"]
    if units != {
        "population_unit": "teacher_seed",
        "student_initialization": "conditional_realization_sensitivity",
        "hard_soft_pooling": False,
        "tasks_as_population_replicates": False,
    }:
        raise Stage13FreezeError("population hierarchy or hard/soft separation changed")
    if value["fidelity"]["primary_threshold"] != 0.99:
        raise Stage13FreezeError("primary fidelity changed")
    if value["endpoints"]["sensitivity_grid"]["hierarchy"]["primary"] != "Endpoint 1 only":
        raise Stage13FreezeError("Endpoint 1 primary status changed")
    if "lower bound" not in value["endpoints"]["claim"]:
        raise Stage13FreezeError("packing lower-bound boundary changed")
    if len(value["endpoints"]["sensitivity_grid"]["cells"]) != 12:
        raise Stage13FreezeError("packing sensitivity grid changed")
    if value["discovery"]["common_exact_evaluation_allowance"] != 256:
        raise Stage13FreezeError("common exact allowance changed")
    if len(value["discovery"]["methods"]) != 2:
        raise Stage13FreezeError("two-family discovery roster changed")
    if value["calibration"]["mask_count"] != 2**18:
        raise Stage13FreezeError("exact calibration size changed")
    _boundary(value, label="scientific protocol")


def validate_analysis(value: dict[str, Any], dossier: dict[str, Any]) -> None:
    require_exact_fields(
        value,
        {
            "schema_version",
            "approval_sha256",
            "scientific_protocol_sha256",
            "production_profiles_sha256",
            "hierarchical_analysis",
            "fourier",
            "tiering",
            "orchestration_boundary",
            "outcome_categories",
            "claim_rules",
            "report_surfaces",
            "every_terminal_dataset_maps_to_report",
            "manual_setting_selection",
            "scientific_data",
            "production_eligible",
            "definitive_execution_started",
        },
        label="analysis plan",
    )
    rd = {item["decision_id"]: item["recommendation"] for item in dossier["decisions"]}
    for field, decision in (
        ("hierarchical_analysis", "RD-010"),
        ("fourier", "RD-011"),
        ("tiering", "RD-012"),
        ("orchestration_boundary", "RD-014"),
    ):
        if value[field] != rd[decision]:
            raise Stage13FreezeError(f"post-approval analysis edit: {decision}")
    if value["hierarchical_analysis"]["population_unit"] != "teacher_seed":
        raise Stage13FreezeError("wrong analysis population unit")
    fourier = value["fourier"]
    if len(fourier["conditions"]) != 6 or fourier["uniqueness_claim"] is not False:
        raise Stage13FreezeError("Fourier control or uniqueness boundary changed")
    if fourier["conditions"][0] != "aligned_fourier_interchange":
        raise Stage13FreezeError("aligned Fourier condition changed")
    if "centered_logit" not in fourier["primary_outcome"]["name"]:
        raise Stage13FreezeError("Fourier primary outcome is not gauge invariant")
    if value["tiering"]["protected_launch_scope"]["definition"] != "Tier 1 plus protected Tier 2":
        raise Stage13FreezeError("protected launch scope changed")
    if value["tiering"]["tier3_order"] != [
        "Task 3 symmetric quadratic increment",
        "Task 4 asymmetric separable mixed-degree increment",
        "Task 5 coupled homogeneous cubic increment",
    ]:
        raise Stage13FreezeError("optional order changed")
    if len(value["outcome_categories"]) != 6 or not value["every_terminal_dataset_maps_to_report"]:
        raise Stage13FreezeError("terminal outcome mapping is incomplete")
    expected_claim_rules = {
        "endpoint1": "procedure-relative primary; never global minimum",
        "packing": "procedure-relative lower bound and key secondary",
        "nulls": "calibration diagnostics only",
        "tractable": "small-instance procedure calibration only",
        "basis": "paired conditional sensitivity",
        "method": "within-method primary; cross-method descriptive",
        "fourier": (
            "shared abstraction only under full six-condition success rule; uniqueness prohibited"
        ),
    }
    if value["claim_rules"] != expected_claim_rules:
        raise Stage13FreezeError("claim boundary changed")
    _boundary(value, label="analysis plan")


def validate_array_spec(value: dict[str, Any], expected_seal: dict[str, Any]) -> dict[str, Any]:
    first = expand_job_arrays(value)
    second = expand_job_arrays(json.loads(canonical_json_bytes(value)))
    if canonical_json_bytes(first) != canonical_json_bytes(second):
        raise Stage13FreezeError("independent full manifest expansions differ")
    compact = expansion_seal(first)
    if compact != expected_seal:
        raise Stage13FreezeError("expanded manifest seal mismatch")
    members = first["members"]
    ids = [item["logical_job_id"] for item in members]
    seeds = [item["seed"] for item in members]
    outputs = [item["output_relative_path"] for item in members]
    if len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
        raise Stage13FreezeError("job identity or seed collision")
    if len(outputs) != len(set(outputs)):
        raise Stage13FreezeError("unreachable or duplicate expected output")
    protected = compact["scientific_operation_counts_by_scope"]["protected_core"]
    expected = {
        "teacher_training": 21,
        "student_attempts": 940,
        "greedy_discovery_ledgers": 645,
        "hard_concrete_discovery_ledgers": 645,
        "hard_concrete_restart_runs": 2580,
        "ordinary_restart_subjobs": 960,
        "exact_evaluations": 615424,
        "fourier_input_condition_trials": 92160,
        "merge_export": 1,
    }
    for key, count in expected.items():
        if protected.get(key) != count:
            raise Stage13FreezeError(f"protected operation count changed: {key}")
    if protected["hard_concrete_optimizer_steps"] != 15_300_000:
        raise Stage13FreezeError("hard-concrete workload changed")
    for task in (3, 4, 5):
        scope = compact["scientific_operation_counts_by_scope"][f"optional_task_{task}_increment"]
        if {
            key: scope[key]
            for key in (
                "teacher_training",
                "student_attempts",
                "greedy_discovery_ledgers",
                "hard_concrete_discovery_ledgers",
                "hard_concrete_restart_runs",
                "exact_evaluations",
                "merge_export",
            )
        } != {
            "teacher_training": 5,
            "student_attempts": 40,
            "greedy_discovery_ledgers": 30,
            "hard_concrete_discovery_ledgers": 30,
            "hard_concrete_restart_runs": 120,
            "exact_evaluations": 15360,
            "merge_export": 1,
        }:
            raise Stage13FreezeError(f"optional Task {task} workload changed")
    arrays = {item["array_id"]: item for item in value["arrays"]}
    if "task1-student-attempts" in arrays["greedy-task1-direct"]["dependencies"]:
        raise Stage13FreezeError("direct teacher discovery incorrectly waits for students")
    if any(
        "packing" in item["family"] and "discovery" in item["family"] for item in value["arrays"]
    ):
        raise Stage13FreezeError("packing sensitivity duplicated discovery")
    if any(resource["provider"] is not None for resource in value["resource_classes"]):
        raise Stage13FreezeError("unverified provider embedded in manifest")
    return compact


def _validate_schemas(records: dict[str, dict[str, Any]]) -> None:
    schema_dir = ROOT / "followup/schemas/stage13"
    for name in (
        "approval",
        "protocol",
        "profiles",
        "analysis",
        "arrays",
        "seal",
        "root",
        "excluded",
        "fixture",
        "report",
    ):
        schema = load(schema_dir / f"{name}_v1.schema.json")
        if schema.get("additionalProperties") is not False:
            raise Stage13FreezeError(f"{name} schema is open to unknown fields")
        if set(schema.get("required", [])) != set(records[name]):
            raise Stage13FreezeError(f"{name} schema does not close the current root fields")


def _scan_changed_surface() -> None:
    roots = [
        ROOT / "followup/configs/stage13",
        ROOT / "followup/decisions",
        ROOT / "followup/fixtures/stage13",
        ROOT / "followup/manifests",
        ROOT / "followup/reports",
        ROOT / "followup/schemas/stage13",
    ]
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".bin", ".zip", ".tar", ".gz", ".7z", ".lfs"}
    forbidden_text = (
        "/users/",
        "\\users\\",
        "/home/",
        ".ssh/",
        "s3://",
        "aws_secret",
        "private_key",
    )
    for base in roots:
        for path in base.rglob("*"):
            if not path.is_file() or "stage13" not in path.as_posix().lower():
                continue
            if path.suffix.lower() in forbidden_suffixes:
                raise Stage13FreezeError(f"forbidden binary/archive/checkpoint: {path}")
            if path.stat().st_size > 2_000_000:
                raise Stage13FreezeError(f"oversized Stage 13 artifact: {path}")
            text = path.read_text(encoding="utf-8").lower()
            if text.startswith("version https://git-lfs.github.com/spec"):
                raise Stage13FreezeError(f"LFS pointer forbidden: {path}")
            if any(token in text for token in forbidden_text):
                raise Stage13FreezeError(f"private/provider-sensitive token in {path}")


def validate() -> dict[str, Any]:
    preapproval = validate_dossier()
    if preapproval["approval_status"] != "pending":
        raise Stage13FreezeError("approved dossier was mutated instead of additively approved")
    if file_sha256(PATHS["dossier"]) != DOSSIER_SHA:
        raise Stage13FreezeError("approved dossier file hash changed")
    records = {
        name: load(path) for name, path in PATHS.items() if name not in {"dossier", "resource"}
    }
    approval = records["approval"]
    protocol = records["protocol"]
    profiles = records["profiles"]
    analysis = records["analysis"]
    validate_approval(approval)
    dossier = load(PATHS["dossier"])
    validate_protocol(protocol, dossier)
    validate_analysis(analysis, dossier)
    if profiles["hidden_defaults_permitted"] is not False:
        raise Stage13FreezeError("production profiles permit hidden defaults")
    if profiles["scientific_protocol_sha256"] != file_sha256(PATHS["protocol"]):
        raise Stage13FreezeError("production profile protocol hash mismatch")
    _boundary(profiles, label="production profiles")
    compact = validate_array_spec(records["arrays"], records["seal"])
    report = generate_synthetic_report(records["fixture"], analysis)
    if canonical_json_bytes(report) != canonical_json_bytes(records["report"]):
        raise Stage13FreezeError("synthetic report is not byte-deterministic")
    if report["scientific_claim"] is not None or report["manual_editing_required"] is not False:
        raise Stage13FreezeError("synthetic report crossed its claim boundary")
    root = records["root"]
    for name in (
        "approval",
        "scientific_protocol",
        "production_profiles",
        "analysis_report_plan",
        "job_array_spec",
        "expanded_manifest_seal",
        "excluded_evidence",
        "resource_projection",
        "synthetic_fixture",
        "synthetic_report",
    ):
        binding = root[name]
        path = ROOT / binding["path"]
        if file_sha256(path) != binding["sha256"]:
            raise Stage13FreezeError(f"campaign root hash mismatch: {name}")
    if (
        root["expanded_manifest_seal"]["canonical_members_sha256"]
        != compact["canonical_members_sha256"]
    ):
        raise Stage13FreezeError("campaign root expansion hash mismatch")
    if (
        root["registered_or_private_artifacts_accessed"] is not False
        or root["scientific_jobs_executed"] is not False
    ):
        raise Stage13FreezeError("scientific/private execution boundary violated")
    if any(root[key] is not False for key in ("stage14_started", "stage15_started")):
        raise Stage13FreezeError("Stage 14 or Stage 15 started")
    excluded = records["excluded"]
    if (
        excluded["registered_or_private_artifacts_accessed"] is not False
        or excluded["scientific_jobs_executed"] is not False
    ):
        raise Stage13FreezeError("excluded evidence register reports prohibited access")
    _validate_schemas(records)
    _scan_changed_surface()
    return {
        "validation": "PASS",
        "approval_sha256": file_sha256(PATHS["approval"]),
        "dossier_sha256": DOSSIER_SHA,
        "protocol_sha256": file_sha256(PATHS["protocol"]),
        "profiles_sha256": file_sha256(PATHS["profiles"]),
        "analysis_sha256": file_sha256(PATHS["analysis"]),
        "manifest_sha256": compact["canonical_members_sha256"],
        "logical_job_count": compact["logical_job_count"],
        "report_sha256": file_sha256(PATHS["report"]),
        "registered_or_private_artifacts_accessed": False,
        "scientific_jobs_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", required=True)
    parser.parse_args()
    result = validate()
    for key, value in result.items():
        print(f"{key}={str(value).lower() if isinstance(value, bool) else value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the approved prospective Stage 13 freeze artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from circuit_families.stage13_freeze import (
    canonical_json_bytes,
    canonical_sha256,
    expand_job_arrays,
    expansion_seal,
    generate_synthetic_report,
)

ROOT = Path(__file__).resolve().parents[1]
DOSSIER_PATH = ROOT / "followup/decisions/stage13_decision_dossier_v3.json"
DOSSIER_SHA256 = "642118fe30e1c435fc05c656ac6167446c76a445c9a1efd376d5a7c23410b1f4"
PACKAGE_ID = "stage13-package-a-protected-core-optional-five-task/v3"
APPROVAL_TEXT = (
    "I approve `stage13-package-a-protected-core-optional-five-task/v3` exactly as "
    "described in the Stage 13 v3 decision dossier with SHA-256 "
    "`642118fe30e1c435fc05c656ac6167446c76a445c9a1efd376d5a7c23410b1f4`; "
    "Parts C–H may proceed subject to the Stage 14 protected-core launch gate."
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_map(dossier: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["decision_id"]: item for item in dossier["decisions"]}


def array(
    array_id: str,
    family: str,
    count: int,
    *,
    scope: str = "protected_core",
    tier: str = "tier1",
    protected: bool = True,
    dependencies: list[str] | None = None,
    operations: dict[str, int] | None = None,
    dimensions: dict[str, Any] | None = None,
    resource: str = "qualified-cpu/v1",
    priority: int = 10,
    producer: str = "stage12p3-contract/v1",
    config_sha256: str,
    concurrency: str = "cpu-independent/v1",
    shedding: str = "never",
) -> dict[str, Any]:
    return {
        "array_id": array_id,
        "family": family,
        "scope": scope,
        "tier": tier,
        "protected": protected,
        "count": count,
        "operation_counts_per_member": operations or {},
        "dimensions": dimensions or {},
        "dependencies": dependencies or [],
        "producer_interface": producer,
        "config_sha256": config_sha256,
        "resource_class": resource,
        "priority": priority,
        "concurrency_group": concurrency,
        "shedding": shedding,
        "output_template": f"stage15/{scope}/{array_id}/{{index}}.json",
        "retention": "final-plus-latest-valid-recovery/v1",
        "retry_class": "stage13-fixed-logical-identity-max-three-attempts/v1",
        "unavailable_consequence": "retain-terminal-state-no-imputation-no-replacement",
    }


def build_arrays(profile_sha: str) -> list[dict[str, Any]]:
    a: list[dict[str, Any]] = []
    a.append(array("gate-approval", "human-gate", 1, config_sha256=profile_sha))
    a.append(
        array(
            "stage14-qualification",
            "qualification-prerequisite",
            1,
            dependencies=["gate-approval"],
            config_sha256=profile_sha,
        )
    )
    a.append(
        array(
            "gate-launch",
            "human-gate",
            1,
            dependencies=["stage14-qualification"],
            config_sha256=profile_sha,
        )
    )
    for name, count, dims in (
        ("task1-teachers", 15, {"task": 1, "seeds": list(range(15))}),
        ("task2-teachers", 5, {"task": 2, "seeds": list(range(5))}),
        ("tractable-teacher", 1, {"task": "mod7-exact-calibration", "seeds": [0]}),
    ):
        a.append(
            array(
                name,
                "teacher-training",
                count,
                dependencies=["gate-launch"],
                operations={"teacher_training": 1},
                dimensions=dims,
                resource="qualified-accelerator-or-cpu/v1",
                config_sha256=profile_sha,
                concurrency="training/v1",
            )
        )
    a.extend(
        [
            array(
                "task1-student-attempts",
                "student-attempt",
                540,
                dependencies=["task1-teachers"],
                operations={"student_attempts": 1},
                dimensions={"task": 1, "seeds": 15, "phases": 3, "conditions": 2, "slots": 6},
                resource="qualified-accelerator-or-cpu/v1",
                config_sha256=profile_sha,
                concurrency="training/v1",
            ),
            array(
                "task2-student-attempts",
                "student-attempt",
                80,
                tier="tier2-protected",
                dependencies=["task2-teachers"],
                operations={"student_attempts": 1},
                dimensions={"task": 2, "seeds": 5, "phases": 2, "conditions": 2, "slots": 4},
                resource="qualified-accelerator-or-cpu/v1",
                config_sha256=profile_sha,
                concurrency="training/v1",
            ),
            array(
                "architecture-student-attempts",
                "student-attempt",
                320,
                tier="tier2-protected",
                dependencies=["task1-teachers"],
                operations={"student_attempts": 1},
                dimensions={
                    "seeds": 5,
                    "phases": 2,
                    "conditions": 2,
                    "architectures": 4,
                    "slots": 4,
                },
                resource="qualified-accelerator-or-cpu/v1",
                config_sha256=profile_sha,
                concurrency="training/v1",
            ),
        ]
    )
    strata = [
        ("task1-direct", 45, ["task1-teachers"], "tier1", {"models": "15 seeds x 3 phases"}),
        (
            "task1-student",
            270,
            ["task1-student-attempts"],
            "tier1",
            {"models": "15 x 3 x 2 conditions x 3 eligible ranks"},
        ),
        ("task2-direct", 10, ["task2-teachers"], "tier2-protected", {"models": "5 x 2 phases"}),
        (
            "task2-student",
            40,
            ["task2-student-attempts"],
            "tier2-protected",
            {"models": "5 x 2 x 2 conditions x 2 eligible ranks"},
        ),
        (
            "architecture",
            160,
            ["architecture-student-attempts"],
            "tier2-protected",
            {"models": "5 x 2 x 2 x 4 alternates x 2 eligible ranks"},
        ),
        (
            "basis-direct",
            40,
            ["task1-teachers"],
            "tier2-protected",
            {"models": "5 x 2 x direct x 4 alternate bases"},
        ),
        (
            "basis-student",
            80,
            ["task1-student-attempts"],
            "tier2-protected",
            {"models": "5 x 2 x hard/soft x 4 alternate bases"},
        ),
    ]
    greedy_arrays: list[str] = []
    hard_restart_arrays: list[str] = []
    hard_ledger_arrays: list[str] = []
    for label, count, deps, tier, dims in strata:
        greedy_id = f"greedy-{label}"
        hard_restart_id = f"hard-restarts-{label}"
        hard_ledger_id = f"hard-ledgers-{label}"
        greedy_arrays.append(greedy_id)
        hard_restart_arrays.append(hard_restart_id)
        hard_ledger_arrays.append(hard_ledger_id)
        a.append(
            array(
                greedy_id,
                "greedy-discovery-ledger",
                count,
                tier=tier,
                dependencies=deps,
                operations={"greedy_discovery_ledgers": 1, "exact_evaluations": 256},
                dimensions=dims,
                resource="qualified-cpu/v1",
                config_sha256=profile_sha,
                producer="stage12r1-independent-discovery/v1",
                concurrency="discovery-greedy/v1",
            )
        )
        a.append(
            array(
                hard_restart_id,
                "hard-concrete-restart",
                count * 4,
                tier=tier,
                dependencies=deps,
                operations={"hard_concrete_restart_runs": 1, "hard_concrete_optimizer_steps": 5000},
                dimensions={**dims, "restarts_per_ledger": 4},
                resource="qualified-accelerator-or-cpu/v1",
                config_sha256=profile_sha,
                producer="stage12r1-independent-discovery/v1",
                concurrency="discovery-hard-concrete/v1",
            )
        )
        a.append(
            array(
                hard_ledger_id,
                "hard-concrete-discovery-ledger",
                count,
                tier=tier,
                dependencies=[hard_restart_id],
                operations={"hard_concrete_discovery_ledgers": 1, "exact_evaluations": 256},
                dimensions=dims,
                resource="qualified-cpu/v1",
                config_sha256=profile_sha,
                producer="stage12r1-independent-discovery/v1",
            )
        )
    a.extend(
        [
            array(
                "endpoint-reducers",
                "sealed-ledger-endpoint-reducer",
                1290,
                dependencies=greedy_arrays + hard_ledger_arrays,
                operations={
                    "endpoint1_reductions": 1,
                    "frontier_reductions": 5,
                    "packing_grid_reductions": 12,
                },
                dimensions={"methods": 2, "ledgers_per_method": 645},
                config_sha256=profile_sha,
                producer="stage6a-stage6e-reducer/v1",
            ),
            array(
                "ordinary-restart-greedy-subjobs",
                "ordinary-restart",
                480,
                tier="tier2-protected",
                dependencies=["task1-teachers", "task1-student-attempts"],
                operations={"ordinary_restart_subjobs": 1},
                dimensions={"profiles": 30, "method": "greedy", "restarts": 16},
                resource="qualified-cpu/v1",
                config_sha256=profile_sha,
                producer="stage12r3-ordinary-restart/v1",
                concurrency="calibration-ordinary/v1",
            ),
            array(
                "ordinary-restart-hard-subjobs",
                "ordinary-restart",
                480,
                tier="tier2-protected",
                dependencies=["task1-teachers", "task1-student-attempts"],
                operations={"ordinary_restart_subjobs": 1, "hard_concrete_optimizer_steps": 5000},
                dimensions={"profiles": 30, "method": "hard-concrete", "restarts": 16},
                resource="qualified-accelerator-or-cpu/v1",
                config_sha256=profile_sha,
                producer="stage12r3-ordinary-restart/v1",
                concurrency="calibration-ordinary/v1",
            ),
            array(
                "ordinary-restart-ledgers",
                "ordinary-restart-ledger",
                60,
                tier="tier2-protected",
                dependencies=["ordinary-restart-greedy-subjobs", "ordinary-restart-hard-subjobs"],
                operations={"exact_evaluations": 256},
                dimensions={"profiles": 60},
                config_sha256=profile_sha,
                producer="stage12r3-ordinary-restart/v1",
            ),
            array(
                "combinatorial-nulls",
                "combinatorial-null",
                60,
                tier="tier2-protected",
                dependencies=["endpoint-reducers"],
                operations={"combinatorial_null_draws": 10000},
                dimensions={"profiles": 60},
                config_sha256=profile_sha,
                producer="stage12r3-combinatorial/v1",
            ),
            array(
                "local-perturbation-nulls",
                "local-perturbation-null",
                60,
                tier="tier2-protected",
                dependencies=["endpoint-reducers"],
                operations={"exact_evaluations": 128},
                dimensions={"profiles": 60, "maximum_radius": 2},
                config_sha256=profile_sha,
                producer="stage12r3-local-exact/v1",
            ),
            array(
                "tractable-calibration-shards",
                "tractable-exact-calibration",
                256,
                tier="tier2-protected",
                dependencies=["tractable-teacher"],
                operations={"exact_evaluations": 1024},
                dimensions={"total_masks": 262144, "shard_size": 1024},
                config_sha256=profile_sha,
                producer="stage12r3-tractable/v1",
                concurrency="calibration-exact/v1",
            ),
            array(
                "fourier-condition-jobs",
                "fourier-interchange-condition",
                360,
                tier="tier2-protected",
                dependencies=["task1-teachers", "task1-student-attempts"],
                operations={"fourier_condition_jobs": 1, "fourier_input_condition_trials": 256},
                dimensions={"comparison_sets": 60, "conditions": 6, "trials": 256},
                config_sha256=profile_sha,
                producer="stage12p5-fourier-runner/v1",
                concurrency="fourier/v1",
            ),
            array(
                "protected-report-reduction",
                "analysis-report-reduction",
                1,
                dependencies=[
                    "endpoint-reducers",
                    "ordinary-restart-ledgers",
                    "combinatorial-nulls",
                    "local-perturbation-nulls",
                    "tractable-calibration-shards",
                    "fourier-condition-jobs",
                ],
                operations={"report_reductions": 1},
                config_sha256=profile_sha,
                producer="stage5d-hierarchical-output/v1",
            ),
            array(
                "gate-primary-completeness",
                "human-gate",
                1,
                dependencies=["protected-report-reduction"],
                config_sha256=profile_sha,
            ),
            array(
                "protected-merge-export",
                "compact-merge-export",
                1,
                dependencies=["gate-primary-completeness"],
                operations={"merge_export": 1},
                config_sha256=profile_sha,
                producer="stage12p4-compact-export/v1",
                concurrency="serial-merge/v1",
            ),
            array(
                "gate-exit",
                "human-gate",
                1,
                dependencies=["protected-merge-export"],
                config_sha256=profile_sha,
            ),
        ]
    )
    previous = "gate-primary-completeness"
    for task in (3, 4, 5):
        scope = f"optional_task_{task}_increment"
        tier = f"tier3-task{task}"
        shedding = f"not-admitted-before-task-{task}"
        admission = f"task{task}-admission"
        teacher = f"task{task}-teachers"
        students = f"task{task}-student-attempts"
        greedy_direct = f"task{task}-greedy-direct"
        greedy_student = f"task{task}-greedy-student"
        hard_direct = f"task{task}-hard-restarts-direct"
        hard_student = f"task{task}-hard-restarts-student"
        hard_ledger_direct = f"task{task}-hard-ledgers-direct"
        hard_ledger_student = f"task{task}-hard-ledgers-student"
        a.append(
            array(
                admission,
                "optional-capacity-gate",
                1,
                scope=scope,
                tier=tier,
                protected=False,
                dependencies=[previous],
                config_sha256=profile_sha,
                shedding=shedding,
            )
        )
        a.append(
            array(
                teacher,
                "teacher-training",
                5,
                scope=scope,
                tier=tier,
                protected=False,
                dependencies=[admission],
                operations={"teacher_training": 1},
                dimensions={"task": task, "seeds": 5},
                resource="qualified-accelerator-or-cpu/v1",
                config_sha256=profile_sha,
                concurrency="training/v1",
                shedding=shedding,
            )
        )
        a.append(
            array(
                students,
                "student-attempt",
                40,
                scope=scope,
                tier=tier,
                protected=False,
                dependencies=[teacher],
                operations={"student_attempts": 1},
                dimensions={"task": task, "seeds": 5, "phases": 2, "conditions": 2, "slots": 2},
                resource="qualified-accelerator-or-cpu/v1",
                config_sha256=profile_sha,
                concurrency="training/v1",
                shedding=shedding,
            )
        )
        a.append(
            array(
                greedy_direct,
                "greedy-discovery-ledger",
                10,
                scope=scope,
                tier=tier,
                protected=False,
                dependencies=[teacher],
                operations={"greedy_discovery_ledgers": 1, "exact_evaluations": 256},
                dimensions={"task": task, "role": "direct"},
                config_sha256=profile_sha,
                producer="stage12r1-independent-discovery/v1",
                shedding=shedding,
            )
        )
        a.append(
            array(
                greedy_student,
                "greedy-discovery-ledger",
                20,
                scope=scope,
                tier=tier,
                protected=False,
                dependencies=[students],
                operations={"greedy_discovery_ledgers": 1, "exact_evaluations": 256},
                dimensions={"task": task, "roles": ["lowest-hard", "lowest-soft"]},
                config_sha256=profile_sha,
                producer="stage12r1-independent-discovery/v1",
                shedding=shedding,
            )
        )
        a.append(
            array(
                hard_direct,
                "hard-concrete-restart",
                40,
                scope=scope,
                tier=tier,
                protected=False,
                dependencies=[teacher],
                operations={"hard_concrete_restart_runs": 1, "hard_concrete_optimizer_steps": 5000},
                dimensions={"task": task, "role": "direct", "restarts": 4},
                resource="qualified-accelerator-or-cpu/v1",
                config_sha256=profile_sha,
                producer="stage12r1-independent-discovery/v1",
                shedding=shedding,
            )
        )
        a.append(
            array(
                hard_student,
                "hard-concrete-restart",
                80,
                scope=scope,
                tier=tier,
                protected=False,
                dependencies=[students],
                operations={"hard_concrete_restart_runs": 1, "hard_concrete_optimizer_steps": 5000},
                dimensions={"task": task, "roles": ["lowest-hard", "lowest-soft"], "restarts": 4},
                resource="qualified-accelerator-or-cpu/v1",
                config_sha256=profile_sha,
                producer="stage12r1-independent-discovery/v1",
                shedding=shedding,
            )
        )
        a.append(
            array(
                hard_ledger_direct,
                "hard-concrete-discovery-ledger",
                10,
                scope=scope,
                tier=tier,
                protected=False,
                dependencies=[hard_direct],
                operations={"hard_concrete_discovery_ledgers": 1, "exact_evaluations": 256},
                dimensions={"task": task, "role": "direct"},
                config_sha256=profile_sha,
                producer="stage12r1-independent-discovery/v1",
                shedding=shedding,
            )
        )
        a.append(
            array(
                hard_ledger_student,
                "hard-concrete-discovery-ledger",
                20,
                scope=scope,
                tier=tier,
                protected=False,
                dependencies=[hard_student],
                operations={"hard_concrete_discovery_ledgers": 1, "exact_evaluations": 256},
                dimensions={"task": task, "roles": ["lowest-hard", "lowest-soft"]},
                config_sha256=profile_sha,
                producer="stage12r1-independent-discovery/v1",
                shedding=shedding,
            )
        )
        reducers = f"task{task}-endpoint-reducers"
        a.append(
            array(
                reducers,
                "sealed-ledger-endpoint-reducer",
                60,
                scope=scope,
                tier=tier,
                protected=False,
                dependencies=[
                    greedy_direct,
                    greedy_student,
                    hard_ledger_direct,
                    hard_ledger_student,
                ],
                operations={
                    "endpoint1_reductions": 1,
                    "frontier_reductions": 5,
                    "packing_grid_reductions": 12,
                },
                dimensions={"task": task, "methods": 2, "models": 30},
                config_sha256=profile_sha,
                producer="stage6a-stage6e-reducer/v1",
                shedding=shedding,
            )
        )
        terminal = f"task{task}-merge-export"
        a.append(
            array(
                terminal,
                "compact-merge-export",
                1,
                scope=scope,
                tier=tier,
                protected=False,
                dependencies=[reducers],
                operations={"merge_export": 1},
                dimensions={"task": task},
                config_sha256=profile_sha,
                producer="stage12p4-compact-export/v1",
                concurrency="serial-merge/v1",
                shedding=shedding,
            )
        )
        previous = terminal
    return a


def report_surfaces() -> list[dict[str, Any]]:
    specs = [
        (
            "population-inventory",
            "table",
            ["task", "seed", "phase", "availability"],
            ["planned", "available"],
        ),
        (
            "raw-teacher-seed",
            "table",
            ["seed", "phase", "condition", "method", "endpoint"],
            ["teacher_seed"],
        ),
        (
            "student-attempt-inventory",
            "table",
            ["condition", "slot", "terminal_state"],
            ["all_fixed_attempts"],
        ),
        (
            "student-cell-summary",
            "table",
            ["seed", "phase", "condition", "architecture"],
            ["eligible_initializations"],
        ),
        (
            "endpoint1-primary",
            "table",
            ["seed", "phase", "condition", "method"],
            ["matched_teacher_seeds"],
        ),
        (
            "packing-key-secondary",
            "table",
            ["cap", "overlap", "method", "censoring"],
            ["sealed_ledger_masks"],
        ),
        (
            "fidelity-frontier",
            "figure_data",
            ["threshold", "phase", "condition", "method"],
            ["sealed_ledger_masks"],
        ),
        (
            "architecture-contrast",
            "table",
            ["anchor_seed", "architecture", "phase", "condition"],
            ["complete_checkpoint_pairs"],
        ),
        (
            "basis-contrast",
            "table",
            ["anchor_seed", "basis", "phase", "condition"],
            ["complete_checkpoint_pairs"],
        ),
        (
            "method-disagreement",
            "figure_data",
            ["seed", "method", "endpoint"],
            ["within_method_first"],
        ),
        (
            "null-calibration",
            "table",
            ["layer", "coverage", "terminal_state"],
            ["predeclared_profiles"],
        ),
        (
            "tractable-calibration",
            "table",
            ["mask_identity", "fidelity", "proof"],
            ["262144_masks"],
        ),
        (
            "fourier-comparison",
            "table",
            ["seed", "phase", "condition", "control", "outcome"],
            ["complete_comparison_sets"],
        ),
        (
            "failure-and-unavailable",
            "table",
            ["family", "reason", "tier", "count"],
            ["all_planned_jobs"],
        ),
        (
            "resource-accounting",
            "table",
            ["scope", "scenario", "resource_kind"],
            ["planned_operations"],
        ),
        (
            "claim-resolution",
            "table",
            ["category", "status", "reason"],
            ["complete_predeclared_inputs"],
        ),
        (
            "tidy-machine-output",
            "tidy_output",
            ["estimand_id", "unit_id", "value", "status"],
            ["declared_units"],
        ),
    ]
    return [
        {
            "surface_id": sid,
            "kind": kind,
            "dimensions": dims,
            "denominators": denoms,
            "missingness": "explicit unavailable/indeterminate; never imputed",
        }
        for sid, kind, dims, denoms in specs
    ]


def main() -> int:
    dossier = load(DOSSIER_PATH)
    if file_sha256(DOSSIER_PATH) != DOSSIER_SHA256:
        raise ValueError("approved dossier bytes changed")
    rd = decision_map(dossier)
    approval = {
        "schema_version": "stage13-approval/v1",
        "authority": "Alex",
        "approval_date": "2026-08-31",
        "approval_text": APPROVAL_TEXT,
        "approved_package_id": PACKAGE_ID,
        "dossier_path": "followup/decisions/stage13_decision_dossier_v3.json",
        "dossier_file_sha256": DOSSIER_SHA256,
        "scope": "Parts C-H may proceed subject to the Stage 14 protected-core launch gate",
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
    }
    approval_path = ROOT / "followup/decisions/stage13_approval_v1.json"
    write(approval_path, approval)
    approval_sha = file_sha256(approval_path)

    protocol = {
        "schema_version": "stage13-frozen-scientific-protocol/v1",
        "approval_sha256": approval_sha,
        "approved_dossier_sha256": DOSSIER_SHA256,
        "package_id": PACKAGE_ID,
        "resolution_status": {
            **{f"RD-{i:03d}": "resolved" for i in range(1, 13)},
            "RD-013": "pending_stage14",
            "RD-014": "stage13_portion_resolved_stage14_bindings_pending",
        },
        "population_and_tasks": rd["RD-001"]["recommendation"],
        "architecture": rd["RD-002"]["recommendation"],
        "students_and_training": rd["RD-003"]["recommendation"],
        "basis": rd["RD-004"]["recommendation"],
        "fidelity": rd["RD-005"]["recommendation"],
        "endpoints": rd["RD-006"]["recommendation"],
        "discovery": rd["RD-007"]["recommendation"],
        "nulls": rd["RD-008"]["recommendation"],
        "calibration": rd["RD-009"]["recommendation"],
        "interpretation_units": {
            "population_unit": "teacher_seed",
            "student_initialization": "conditional_realization_sensitivity",
            "hard_soft_pooling": False,
            "tasks_as_population_replicates": False,
        },
        "optional_tier3_attempt_policy": {
            "eligible_target_per_condition": 1,
            "fixed_attempt_slots_per_condition": 2,
            "stop_after_first_eligible": True,
            "replacement": "none",
            "derivation": "approved lowest-eligible singular roster and approved 20/30/40 attempt projection",
        },
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
    }
    protocol_path = ROOT / "followup/configs/stage13/frozen_scientific_protocol_v1.json"
    write(protocol_path, protocol)
    protocol_sha = file_sha256(protocol_path)

    profiles = {
        "schema_version": "stage13-production-profiles/v1",
        "approval_sha256": approval_sha,
        "scientific_protocol_sha256": protocol_sha,
        "training_profile": rd["RD-003"]["recommendation"],
        "fidelity_profile": rd["RD-005"]["recommendation"],
        "endpoint_profile": rd["RD-006"]["recommendation"],
        "discovery_profiles": rd["RD-007"]["recommendation"],
        "null_profiles": rd["RD-008"]["recommendation"],
        "calibration_profile": rd["RD-009"]["recommendation"],
        "storage_retry_export_profile": rd["RD-014"]["recommendation"],
        "terminal_states": ["sealed_success", "failed", "unavailable", "censored", "not_admitted"],
        "hidden_defaults_permitted": False,
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
    }
    profiles_path = ROOT / "followup/configs/stage13/production_profiles_v1.json"
    write(profiles_path, profiles)
    profiles_sha = file_sha256(profiles_path)

    analysis = {
        "schema_version": "stage13-analysis-report-plan/v1",
        "approval_sha256": approval_sha,
        "scientific_protocol_sha256": protocol_sha,
        "production_profiles_sha256": profiles_sha,
        "hierarchical_analysis": rd["RD-010"]["recommendation"],
        "fourier": rd["RD-011"]["recommendation"],
        "tiering": rd["RD-012"]["recommendation"],
        "orchestration_boundary": rd["RD-014"]["recommendation"],
        "outcome_categories": rd["RD-010"]["recommendation"]["outcome_categories"],
        "claim_rules": {
            "endpoint1": "procedure-relative primary; never global minimum",
            "packing": "procedure-relative lower bound and key secondary",
            "nulls": "calibration diagnostics only",
            "tractable": "small-instance procedure calibration only",
            "basis": "paired conditional sensitivity",
            "method": "within-method primary; cross-method descriptive",
            "fourier": "shared abstraction only under full six-condition success rule; uniqueness prohibited",
        },
        "report_surfaces": report_surfaces(),
        "every_terminal_dataset_maps_to_report": True,
        "manual_setting_selection": False,
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
    }
    analysis_path = ROOT / "followup/configs/stage13/analysis_report_plan_v1.json"
    write(analysis_path, analysis)
    analysis_sha = file_sha256(analysis_path)

    excluded = {
        "schema_version": "stage13-excluded-evidence-output-register/v1",
        "approval_sha256": approval_sha,
        "excluded_classes": [
            {
                "class": "development",
                "examples": ["Stage 8-10 diagnostics", "method-development outputs"],
                "reason": "technical development only",
            },
            {
                "class": "synthetic",
                "examples": ["Stage 12 synthetic fixtures", "Stage 13 complete-report fixture"],
                "reason": "topology/correctness only",
            },
            {
                "class": "red_team",
                "examples": ["red-team briefing and evidence dossier"],
                "reason": "design authority, never endpoint evidence",
            },
            {
                "class": "benchmark",
                "examples": ["Stage 9/10 timing", "Stage 13 constructed timing"],
                "reason": "feasibility only",
            },
            {
                "class": "registered_or_private",
                "examples": [
                    "registered checkpoints",
                    "dense-output caches",
                    "unpublished local results",
                ],
                "reason": "firewall prohibition",
            },
        ],
        "forbidden_uses": [
            "select threshold",
            "select basis",
            "select method",
            "select phase",
            "select pair",
            "choose favorable analysis",
            "make definitive inference",
        ],
        "registered_or_private_artifacts_accessed": False,
        "scientific_jobs_executed": False,
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
    }
    excluded_path = ROOT / "followup/manifests/stage13_excluded_evidence_v1.json"
    write(excluded_path, excluded)

    resource_path = ROOT / "followup/manifests/stage13_scope_resource_projection_v3.json"
    arrays = build_arrays(profiles_sha)
    array_spec = {
        "schema_version": "stage13-job-array-spec/v1",
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
        "approval_sha256": approval_sha,
        "scientific_profile_sha256": profiles_sha,
        "analysis_plan_sha256": analysis_sha,
        "resource_projection_sha256": file_sha256(resource_path),
        "seed_derivation": "stage4-domain-separated-seed/v1",
        "resource_classes": [
            {
                "id": "qualified-accelerator-or-cpu/v1",
                "stage14_binding_required": True,
                "memory": "measured per hardware class",
                "scratch": "measured active worker plus staging",
                "runtime": "minimum-of-three qualified throughput",
                "provider": None,
            },
            {
                "id": "qualified-cpu/v1",
                "stage14_binding_required": True,
                "memory": "measured per hardware class",
                "scratch": "measured active worker plus staging",
                "runtime": "minimum-of-three qualified throughput",
                "provider": None,
            },
        ],
        "retry_policy": {
            "maximum_attempts": 3,
            "maximum_retries": 2,
            "logical_identity_unchanged": True,
            "retryable": [
                "worker_error",
                "resource_exhaustion",
                "interruption",
                "stale_claim",
                "transfer_interruption",
            ],
            "nonretryable": ["scientific_failure", "validation_failure"],
        },
        "terminal_states": ["sealed_success", "failed", "unavailable", "censored", "not_admitted"],
        "human_gates": ["approval", "Stage 14 protected-core launch", "primary completeness/exit"],
        "arrays": arrays,
    }
    array_path = ROOT / "followup/manifests/stage13_job_array_spec_v1.json"
    write(array_path, array_spec)
    expanded_first = expand_job_arrays(array_spec)
    expanded_second = expand_job_arrays(load(array_path))
    if canonical_json_bytes(expanded_first) != canonical_json_bytes(expanded_second):
        raise ValueError("independent manifest expansions differ")
    seal = expansion_seal(expanded_first)
    seal_path = ROOT / "followup/manifests/stage13_expanded_manifest_seal_v1.json"
    write(seal_path, seal)

    fixture = {
        "schema_version": "stage13-synthetic-complete-fixture/v1",
        "scientific_data": False,
        "production_eligible": False,
        "coverage": {
            "tasks": [1, 2, 3, 4, 5, "mod7-calibration"],
            "task1_teacher_seeds": list(range(15)),
            "optional_teacher_seeds": list(range(5)),
            "phases": ["pre-grokking", "50%", "stable post-grokking"],
            "conditions": ["hard", "soft"],
            "architectures": [
                item["id"] for item in rd["RD-002"]["recommendation"]["architectures"]
            ],
            "bases": rd["RD-004"]["recommendation"]["assignment"]["bases_on_every_available_model"],
            "methods": [item["id"] for item in rd["RD-007"]["recommendation"]["methods"]],
            "endpoints": ["Endpoint 1", "procedure-relative packing lower bound"],
            "tiers": ["tier1", "tier2-protected", "tier3-task3", "tier3-task4", "tier3-task5"],
            "campaign_completion": [
                "tier1-intact-partial-tier2",
                "partial-tier3",
                "tier1-incomplete",
            ],
            "disagreements": ["architecture", "basis", "method"],
        },
        "terminal_cases": [
            {"case_id": "eligible-hard", "state": "eligible"},
            {"case_id": "eligible-soft", "state": "eligible"},
            {"case_id": "attempt-failed", "state": "failed_attempt"},
            {"case_id": "student-insufficient", "state": "insufficient_eligible"},
            {"case_id": "teacher-unavailable", "state": "teacher_unavailable"},
            {"case_id": "phase-unavailable", "state": "phase_unavailable"},
            {"case_id": "budget-censored", "state": "budget_exhausted"},
            {"case_id": "search-failed", "state": "search_failure"},
            {"case_id": "packing-valid-zero", "state": "packing_zero"},
            {"case_id": "packing-lower-bound", "state": "packing_lower_bound"},
            {"case_id": "packing-censored", "state": "packing_censored"},
            {"case_id": "null-complete", "state": "eligible"},
            {"case_id": "tractable-complete", "state": "eligible"},
            {"case_id": "nonfinite", "state": "nonfinite_rejected"},
            {"case_id": "corrupt", "state": "corrupted_rejected"},
            {"case_id": "duplicate", "state": "duplicate_rejected"},
            {"case_id": "stale", "state": "stale_rejected"},
            {"case_id": "conflict", "state": "conflicting_rejected"},
        ],
        "fourier_cases": [
            {
                "case_id": f"fourier-{outcome}",
                "outcome": outcome,
                "controls": list(rd["RD-011"]["recommendation"]["conditions"]),
            }
            for outcome in ("winning", "tying", "losing", "failing", "incomplete")
        ],
    }
    fixture_path = ROOT / "followup/fixtures/stage13/synthetic_complete_fixture_v1.json"
    write(fixture_path, fixture)
    report = generate_synthetic_report(fixture, analysis)
    report_path = ROOT / "followup/reports/stage13_synthetic_complete_report_v1.json"
    write(report_path, report)

    root = {
        "schema_version": "stage13-campaign-root-manifest/v1",
        "campaign_reference": "stage13-stage15-prospective-campaign/v1",
        "implementation_base": dossier["implementation_base"],
        "approval": {"path": str(approval_path.relative_to(ROOT)), "sha256": approval_sha},
        "approved_dossier": {"path": str(DOSSIER_PATH.relative_to(ROOT)), "sha256": DOSSIER_SHA256},
        "scientific_protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": protocol_sha,
        },
        "production_profiles": {
            "path": str(profiles_path.relative_to(ROOT)),
            "sha256": profiles_sha,
        },
        "analysis_report_plan": {
            "path": str(analysis_path.relative_to(ROOT)),
            "sha256": analysis_sha,
        },
        "job_array_spec": {
            "path": str(array_path.relative_to(ROOT)),
            "sha256": file_sha256(array_path),
        },
        "expanded_manifest_seal": {
            "path": str(seal_path.relative_to(ROOT)),
            "sha256": file_sha256(seal_path),
            "logical_job_count": seal["logical_job_count"],
            "canonical_members_sha256": seal["canonical_members_sha256"],
        },
        "excluded_evidence": {
            "path": str(excluded_path.relative_to(ROOT)),
            "sha256": file_sha256(excluded_path),
        },
        "resource_projection": {
            "path": str(resource_path.relative_to(ROOT)),
            "sha256": file_sha256(resource_path),
        },
        "synthetic_fixture": {
            "path": str(fixture_path.relative_to(ROOT)),
            "sha256": file_sha256(fixture_path),
        },
        "synthetic_report": {
            "path": str(report_path.relative_to(ROOT)),
            "sha256": file_sha256(report_path),
            "canonical_sha256": canonical_sha256(report),
        },
        "operation_counts": seal["scientific_operation_counts"],
        "operation_counts_by_scope": seal["scientific_operation_counts_by_scope"],
        "tier_job_counts": seal["counts_by_tier"],
        "scope_job_counts": seal["counts_by_scope"],
        "coverage_counts": {
            "protected_core": {
                "teacher_training": {"task1": 15, "task2": 5, "tractable_calibration": 1},
                "student_attempt_maximum": {"task1": 540, "task2": 80, "architecture_panel": 320},
                "discovery_ledgers_per_method": {
                    "task1_direct": 45,
                    "task1_students": 270,
                    "task2_direct": 10,
                    "task2_students": 40,
                    "architecture_panel": 160,
                    "basis_direct": 40,
                    "basis_students": 80,
                },
                "phases": {"task1": 3, "task2": 2, "architecture": 2, "basis": 2},
                "teacher_seeds": {
                    "task1": 15,
                    "task2": 5,
                    "architecture_anchor": 5,
                    "basis_anchor": 5,
                },
                "conditions": {"hard": "separate", "soft": "separate"},
                "methods": 2,
                "frontier_thresholds": 5,
                "packing_grid_cells": 12,
                "fourier_conditions": 6,
            },
            "each_optional_task_increment": {
                "teacher_training": 5,
                "student_attempt_maximum": 40,
                "discovery_ledgers_per_method": 30,
                "phases": 2,
                "teacher_seeds": 5,
                "conditions": 2,
                "methods": 2,
            },
        },
        "resource_gate": "Stage 14 must bind actual qualified resources; if no verified configuration completes the protected core within H_science, launch blocks pending prospective amendment",
        "optional_order": [3, 4, 5],
        "registered_or_private_artifacts_accessed": False,
        "scientific_jobs_executed": False,
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
        "stage14_started": False,
        "stage15_started": False,
    }
    root_path = ROOT / "followup/manifests/stage13_campaign_root_v1.json"
    write(root_path, root)
    print(f"approval_sha256={approval_sha}")
    print(f"scientific_protocol_sha256={protocol_sha}")
    print(f"production_profiles_sha256={profiles_sha}")
    print(f"analysis_plan_sha256={analysis_sha}")
    print(f"expanded_logical_jobs={seal['logical_job_count']}")
    print(f"expanded_manifest_sha256={seal['canonical_members_sha256']}")
    print(f"synthetic_report_sha256={file_sha256(report_path)}")
    print(f"campaign_root_sha256={file_sha256(root_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

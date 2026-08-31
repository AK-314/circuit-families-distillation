#!/usr/bin/env python3
"""Build the final amended Stage 13 approval-gate dossier and projections."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from circuit_families.stage12p1.tasks import (
    TASK_CONFIG_SCHEMA_VERSION,
    ModularPolynomialImplementation,
    build_task_record,
)

ROOT = Path(__file__).resolve().parents[1]
V2_DOSSIER = ROOT / "followup/decisions/stage13_decision_dossier_v2.json"
RESOURCE_OUTPUT = ROOT / "followup/manifests/stage13_scope_resource_projection_v3.json"
DOSSIER_OUTPUT = ROOT / "followup/decisions/stage13_decision_dossier_v3.json"
OPTIONAL_DIRECTORY = ROOT / "followup/manifests/stage13_optional_tasks"

SCENARIOS = ("lower", "central", "conservative")
UPDATES_PER_ATTEMPT = (5000, 20000, 40000)
TRAIN_SECONDS = (0.018544916, 0.035, 0.141523208)
HARD_SECONDS = (0.035, 0.14, 0.28)
EXACT_SECONDS = 0.088445375
CALIBRATION_SECONDS = (0.0005, 0.005, EXACT_SECONDS)
EFFICIENCY = (0.85, 0.70, 0.50)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def mapping_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ceil_power_two_gib(value_gib: float, *, minimum: int = 1) -> int:
    target = max(float(minimum), value_gib)
    return 2 ** math.ceil(math.log2(target))


def split_identity() -> dict[str, Any]:
    return {
        "kind": "stage13-shared-hash-rank-split",
        "version": "v1",
        "example_order": "lexicographic x then y over 0..58",
        "ranking_material": "SHA256(ASCII('stage13/reduced-task-split/v1\\0' + decimal(x) + '\\0' + decimal(y)))",
        "ranking_tie_break": "lexicographic x then y",
        "train_count": 1044,
        "test_count": 2437,
        "shared_across_optional_tasks_3_through_5": True,
    }


def task_config(
    *,
    task_id: str,
    terms: list[dict[str, Any]],
) -> dict[str, Any]:
    implementation = ModularPolynomialImplementation()
    return {
        "schema_version": TASK_CONFIG_SCHEMA_VERSION,
        "task_id": task_id,
        "implementation": implementation.name,
        "implementation_version": implementation.version,
        "modulus": 59,
        "input_domains": [list(range(59)), list(range(59))],
        "parameters": {"terms": terms},
        "split_identity": split_identity(),
        "architecture_compatibility": {
            "input_arity": 2,
            "output_class_count": 59,
            "model_family": "predecessor-matched/v1",
            "tokenization": "two residues then equals token",
            "classification": "prospective-stage13-optional-task",
        },
        "scientific_data": False,
        "production_eligible": False,
    }


TASK_SPECS = {
    3: {
        "task_id": "stage13-task3-symmetric-quadratic-m59/v1",
        "formula": "(x^2 + x*y + y^2) mod 59",
        "terms": [
            {"coefficient": 1, "exponents": [0, 2]},
            {"coefficient": 1, "exponents": [1, 1]},
            {"coefficient": 1, "exponents": [2, 0]},
        ],
        "justification": "retains the accepted homogeneous symmetric quadratic with both pure and mixed degree-2 terms",
    },
    4: {
        "task_id": "stage13-task4-asymmetric-separable-mixed-degree-m59/v1",
        "formula": "(x^3 + y^2) mod 59",
        "terms": [
            {"coefficient": 1, "exponents": [0, 2]},
            {"coefficient": 1, "exponents": [3, 0]},
        ],
        "justification": "an asymmetric separable polynomial with unequal degrees and no interaction term; unlike subtraction, it is not a linear relabelling of Task 1, and unlike Task 2/3 it is neither bilinear nor homogeneous quadratic",
    },
    5: {
        "task_id": "stage13-task5-coupled-homogeneous-cubic-m59/v1",
        "formula": "(x^2*y + x*y^2) mod 59 = x*y*(x+y) mod 59",
        "terms": [
            {"coefficient": 1, "exponents": [1, 2]},
            {"coefficient": 1, "exponents": [2, 1]},
        ],
        "justification": "a homogeneous degree-3 coupled interaction with no pure monomials; it differs from separable Task 4, degree-2 Task 3, and single-monomial multiplication",
    },
}


def build_optional_manifests() -> dict[int, dict[str, Any]]:
    bindings: dict[int, dict[str, Any]] = {}
    OPTIONAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for task_number, spec in TASK_SPECS.items():
        task_record = build_task_record(
            task_config(task_id=spec["task_id"], terms=spec["terms"])
        )
        predecessor = (
            "protected-core-complete-and-secure"
            if task_number == 3
            else f"task-{task_number - 1}-increment-terminal-and-sealed"
        )
        manifest = {
            "schema_version": "stage13-optional-task-increment/v1",
            "increment_id": f"stage13-optional-task-{task_number}-increment/v1",
            "optional_order": task_number - 2,
            "launch_required": False,
            "scientific_data": False,
            "production_eligible": False,
            "definitive_execution_started": False,
            "dependency": predecessor,
            "capacity_gate": "may start only when protected completion and the 12-hour audit reserve are secure and Stage 14 measured remaining capacity can finish this entire increment",
            "task_number": task_number,
            "formula": spec["formula"],
            "algebraic_justification": spec["justification"],
            "task_record": task_record,
            "panel": {
                "teacher_seeds": [0, 1, 2, 3, 4],
                "phases": ["pre-grokking", "stable post-grokking"],
                "model_roles": [
                    "direct teacher",
                    "lowest fixed eligible canonical hard student",
                    "lowest fixed eligible canonical soft student",
                ],
                "student_selection": "lowest predeclared initialization slot that is eligible; no reroll or replacement",
                "architecture": "predecessor-matched/v1 canonical only",
                "basis": "canonical heads-plus-neurons only",
                "discovery_methods": [
                    "greedy-deletion-centred-logit/v1",
                    "hard-concrete-gates-centred-logit/v1",
                ],
                "exact_allowance_per_model_method_including_intact": 256,
                "teacher_phase_failure": "retain unavailable; no replacement task, seed, or phase",
            },
            "interpretation": {
                "population_unit": "teacher_seed within task",
                "tasks_are_independent_population_replicates": False,
                "cross_task_summary": "descriptive task-indexed external-validity panel only",
                "failure": "failed or partial increment remains failed/partial and is never substituted",
            },
            "resource_projection_reference": {
                "path": "followup/manifests/stage13_scope_resource_projection_v3.json",
                "scope_key": f"optional_task_{task_number}_increment",
            },
        }
        path = OPTIONAL_DIRECTORY / f"task{task_number}_increment_v1.json"
        write_json(path, manifest)
        digest = file_sha256(path)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
        bindings[task_number] = {
            "path": str(path.relative_to(ROOT)),
            "file_sha256": digest,
            "task_identity_sha256": task_record["hashes"]["task_identity_sha256"],
            "task_config_sha256": task_record["hashes"]["task_config_sha256"],
            "dataset_sha256": task_record["hashes"]["dataset_sha256"],
            "sidecar_path": str(sidecar.relative_to(ROOT)),
        }
    return bindings


def scope_workload(scope: str, scenario_index: int) -> dict[str, Any]:
    optional = scope.startswith("optional_task_")
    attempts = (20, 30, 40)[scenario_index] if optional else (470, 660, 940)[scenario_index]
    teacher_jobs = 5 if optional else 21
    canonical_attempts = attempts if optional else (310, 420, 620)[scenario_index]
    architecture_attempts = 0 if optional else (160, 240, 320)[scenario_index]
    architecture_ratio = 1.2903
    updates = UPDATES_PER_ATTEMPT[scenario_index]
    teacher_updates = teacher_jobs * updates
    canonical_student_updates = canonical_attempts * updates
    architecture_student_updates = architecture_attempts * updates * architecture_ratio
    training_updates = teacher_updates + canonical_student_updates + architecture_student_updates
    teacher_gpu_h = teacher_updates * TRAIN_SECONDS[scenario_index] / 3600.0
    student_gpu_h = (canonical_student_updates + architecture_student_updates) * TRAIN_SECONDS[scenario_index] / 3600.0

    discovery_ledgers_per_method = 30 if optional else 645
    hard_restarts = discovery_ledgers_per_method * 4
    main_hard_steps = hard_restarts * 5000
    ordinary_hard_restarts = 0 if optional else 30 * 16
    ordinary_hard_steps = ordinary_hard_restarts * 5000
    hard_steps = main_hard_steps + ordinary_hard_steps
    hard_gpu_h = hard_steps * HARD_SECONDS[scenario_index] / 3600.0
    gpu_device_h = teacher_gpu_h + student_gpu_h + hard_gpu_h

    discovery_exact = 2 * discovery_ledgers_per_method * 256
    ordinary_exact = 0 if optional else 60 * 256
    local_exact = 0 if optional else 60 * 128
    calibration_exact = 0 if optional else 2**18
    exact_cpu_h = (
        (discovery_exact + ordinary_exact + local_exact) * EXACT_SECONDS
        + calibration_exact * CALIBRATION_SECONDS[scenario_index]
    ) / 3600.0
    greedy_cpu_h = discovery_ledgers_per_method * 0.13140775 / 3600.0
    combinatorial_cpu_h = 0 if optional else 600000 * (2e-6, 2e-5, 2e-4)[scenario_index] / 3600.0
    fourier_cpu_h = 0 if optional else (0.019, 0.179, 1.166)[scenario_index]
    report_export_cpu_h = (0.25, 1.0, 4.0)[scenario_index] if optional else (4.0, 16.0, 64.0)[scenario_index]
    standalone_cpu_h = exact_cpu_h + greedy_cpu_h + combinatorial_cpu_h + fourier_cpu_h + report_export_cpu_h
    serial_cpu_h = (0.5, 1.0, 3.0)[scenario_index] if optional else (2.0, 5.0, 12.0)[scenario_index]
    max_cpu_concurrency = 60 if optional else 256
    max_accelerator_concurrency = hard_restarts

    checkpoint_bytes = (8_499_412, 11_227_236, 22_454_472)[scenario_index]
    architecture_checkpoint_bytes = (
        8_499_412,
        5_771_588 + 2 * 2_727_824 * architecture_ratio,
        2 * (5_771_588 + 2 * 2_727_824 * architecture_ratio),
    )[scenario_index]
    training_raw = (teacher_jobs + canonical_attempts) * checkpoint_bytes + architecture_attempts * architecture_checkpoint_bytes
    hard_run_count = hard_restarts + ordinary_hard_restarts
    hard_raw = hard_run_count * (2_260_000, 4_520_000, 9_040_000)[scenario_index]
    total_exact = discovery_exact + ordinary_exact + local_exact + calibration_exact
    ledger_raw = total_exact * (512, 1024, 2048)[scenario_index]
    fourier_raw = 0 if optional else 92160 * (512, 1024, 2048)[scenario_index]
    reports_raw = ((0.10, 0.25, 0.50) if optional else (0.50, 2.0, 6.0))[scenario_index] * 2**30
    raw_retained = training_raw + hard_raw + ledger_raw + fourier_raw + reports_raw
    hard_compact_factor = (0.25, 0.35, 0.50)[scenario_index]
    compact_retained = training_raw + hard_raw * hard_compact_factor + ledger_raw + fourier_raw + reports_raw
    retry_staging_worst = raw_retained * 3 + compact_retained
    persistent_request = ceil_power_two_gib(compact_retained / 2**30 / 0.8)
    scratch_request = ceil_power_two_gib(retry_staging_worst / 2**30 / 0.8)
    gpu_scratch_gib = (0.25, 0.50, 1.0)[scenario_index]
    cpu_scratch_gib = (0.05, 0.10, 0.25)[scenario_index]

    ideal_max_wall = (
        teacher_gpu_h / max(1, teacher_jobs)
        + student_gpu_h / max(1, attempts)
        + hard_gpu_h / max(1, hard_restarts)
        + max(standalone_cpu_h / max_cpu_concurrency, serial_cpu_h)
    )
    adjusted_max_wall = (
        teacher_gpu_h / (max(1, teacher_jobs) * EFFICIENCY[scenario_index])
        + student_gpu_h / (max(1, attempts) * EFFICIENCY[scenario_index])
        + hard_gpu_h / (max(1, hard_restarts) * EFFICIENCY[scenario_index])
        + max(
            standalone_cpu_h / (max_cpu_concurrency * EFFICIENCY[scenario_index]),
            serial_cpu_h / EFFICIENCY[scenario_index],
        )
    )

    return {
        "scenario": SCENARIOS[scenario_index],
        "jobs": {
            "teacher_training": teacher_jobs,
            "student_attempts": attempts,
            "greedy_discovery_ledgers": discovery_ledgers_per_method,
            "hard_concrete_discovery_ledgers": discovery_ledgers_per_method,
            "hard_concrete_restart_runs": hard_restarts,
            "ordinary_restart_subjobs": 0 if optional else 960,
            "exact_evaluations": total_exact,
            "fourier_input_condition_trials": 0 if optional else 92160,
            "merge_export": 1,
        },
        "compute": {
            "training_canonical_equivalent_updates": round(training_updates),
            "hard_concrete_model_in_loop_steps": hard_steps,
            "gpu_device_hours": round(gpu_device_h, 3),
            "gpu_device_hour_breakdown": {
                "teacher_training": round(teacher_gpu_h, 3),
                "student_training": round(student_gpu_h, 3),
                "hard_concrete_main_and_ordinary": round(hard_gpu_h, 3),
            },
            "host_cpu_core_hours_reserved_for_gpu_jobs": f"{round(gpu_device_h, 3)} * h_support, where Stage 14 binds host cores reserved per active accelerator",
            "standalone_cpu_core_hours": round(standalone_cpu_h, 3),
            "standalone_cpu_breakdown": {
                "exact_evaluation_and_calibration": round(exact_cpu_h, 3),
                "greedy_ranking": round(greedy_cpu_h, 3),
                "combinatorial_null": round(combinatorial_cpu_h, 3),
                "fourier": round(fourier_cpu_h, 3),
                "reduction_report_merge_export": round(report_export_cpu_h, 3),
            },
            "serial_or_weakly_parallel_cpu_critical_path_hours": serial_cpu_h,
            "maximum_useful_standalone_cpu_concurrency": max_cpu_concurrency,
            "maximum_useful_accelerator_concurrency": max_accelerator_concurrency,
            "ideal_wall_hours_at_maximum_useful_concurrency": round(ideal_max_wall, 3),
            "efficiency_adjusted_wall_hours_at_maximum_useful_concurrency": round(adjusted_max_wall, 3),
            "ideal_wall_formula": "teacher_device_h/min(G,teacher_jobs) + student_device_h/min(G,student_attempts) + hard_concrete_device_h/min(G,hard_restart_runs) + max(standalone_CPU_core_h/min(C,max_useful_CPU), serial_CPU_critical_h)",
            "efficiency_adjusted_formula": "replace each denominator by denominator*Stage14_measured_efficiency and divide serial critical path by measured weak-path efficiency; validate against actual availability intervals rather than assuming uninterrupted access",
        },
        "storage": {
            "expected_retained_compact_output_gib": round(compact_retained / 2**30, 3),
            "peak_active_scratch_formula_gib": f"{gpu_scratch_gib}*G_active + {cpu_scratch_gib}*C_active + one measured merge staging set; G_active/C_active are Stage 14 bindings",
            "uncompressed_staging_retry_worst_case_gib": round(retry_staging_worst / 2**30, 3),
            "requested_persistent_safety_quota_gib": persistent_request,
            "requested_scratch_safety_quota_gib": scratch_request,
            "quota_rule": "requested quota is the next power of two at or above projected demand/0.80; Stage 14 must request at least this amount or recompute from measured artifact sizes",
        },
        "assumptions": {
            "training_updates_per_attempt": updates,
            "training_seconds_per_canonical_equivalent_update": TRAIN_SECONDS[scenario_index],
            "model_in_loop_hard_concrete_seconds_per_step": HARD_SECONDS[scenario_index],
            "planning_efficiency": EFFICIENCY[scenario_index],
            "actual_rates_and_efficiency_must_be_replaced_by_stage14_measurements": True,
        },
    }


def build_resource_projection(optional_bindings: dict[int, dict[str, Any]]) -> dict[str, Any]:
    scopes = {
        "protected_core": {
            "launch_required": True,
            "includes": [
                "Tier 1 Task 1 fixed seeds 0-14",
                "protected Tier 2 Task 2 modular multiplication",
                "paired architecture and basis panels",
                "exact 2^18 calibration",
                "Fourier aligned condition and all five controls",
                "both discovery methods, Endpoint 1, packing, frontier, and protected null calibration",
            ],
            "scenarios": [scope_workload("protected_core", index) for index in range(3)],
        }
    }
    for task_number in (3, 4, 5):
        scopes[f"optional_task_{task_number}_increment"] = {
            "launch_required": False,
            "manifest": optional_bindings[task_number],
            "order": task_number - 2,
            "scenarios": [
                scope_workload(f"optional_task_{task_number}", index)
                for index in range(3)
            ],
        }
    return {
        "schema_version": "stage13-scope-resource-projection/v3",
        "classification": "prospective non-scientific planning evidence",
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
        "no_fixed_provider_or_hardware_assumption": True,
        "scientific_window": {
            "stage14_binding": "H_total = total qualified permitted hours across verified availability intervals",
            "audit_reserve_hours": 12,
            "science_hours": "H_science = max(0, H_total - 12)",
            "uninterrupted_access_assumed": False,
        },
        "stage14_required_bindings": [
            "provider/scheduler and account only if actually available",
            "machine count and hardware class for every worker",
            "qualified CPU, CUDA, or MPS status per hardware class",
            "minimum-of-three post-warmup training, model-loop hard-concrete, and exact-evaluation throughput",
            "host CPU cores reserved per active accelerator",
            "per-machine RAM/VRAM and active scratch",
            "per-machine availability intervals and permitted hours",
            "queue latency, preemption behavior, and measured scheduling efficiency",
            "persistent/scratch quotas and merge/export throughput",
        ],
        "prohibited_assumptions": [
            "Symbolica availability",
            "Eton availability",
            "16 GPUs",
            "256 CPU cores",
            "MPS qualification",
            "uninterrupted access",
        ],
        "school_mac_parameterized_branch": {
            "machine_count": "M, bound by Stage 14",
            "hardware_classes": "class k records machine count M_k, CPU cores, RAM, optional MPS device, and disk quotas",
            "permitted_hours": "availability intervals I_{k,m}; capacity is integrated over intervals, never count*nominal wall time",
            "qualified_rates": "r_train_cpu[k], r_hc_cpu[k], r_exact_cpu[k], and optional r_train_mps[k]/r_hc_mps[k] only after exact and semantic MPS qualification",
            "capacity_equations": [
                "training_capacity = sum over machines and intervals of qualified_training_rate * usable_interval_seconds * measured_efficiency",
                "hard_concrete_capacity = analogous sum using model-in-loop rate",
                "exact_capacity = analogous sum using standalone CPU exact rate",
                "host CPU support reservations are subtracted before standalone CPU capacity is computed",
            ],
            "schedule_rule": "run the dependency-closed protected DAG against actual intervals; CPU-only is mandatory when MPS is absent or unqualified",
            "pass_rule": "protected core must finish within H_science with memory/storage quotas and audit reserve intact; otherwise launch blocks",
            "optional_rule": "after protected completion is secure, admit Task 3, then Task 4, then Task 5 only when measured remaining interval capacity can finish the entire next increment",
        },
        "launch_rule": "if no verified configuration completes the protected core within H_science, block launch pending prospective amendment",
        "scopes": scopes,
    }


def decision(record: dict[str, Any], decision_id: str) -> dict[str, Any]:
    return next(item for item in record["decisions"] if item["decision_id"] == decision_id)


def build_dossier(
    optional_bindings: dict[int, dict[str, Any]],
    resource_projection: dict[str, Any],
) -> dict[str, Any]:
    v2 = json.loads(V2_DOSSIER.read_text(encoding="utf-8"))
    dossier = deepcopy(v2)
    dossier["schema_version"] = "stage13-decision-dossier/v3"
    dossier["recommended_package_id"] = "stage13-package-a-protected-core-optional-five-task/v3"
    dossier["supersedes"] = {
        "path": "followup/decisions/stage13_decision_dossier_v2.json",
        "file_sha256": file_sha256(V2_DOSSIER),
        "approval_status": "never_approved",
    }
    amended_ids = {"RD-001", "RD-007", "RD-012", "RD-014"}
    dossier["accepted_in_principle_unchanged_from_v2"] = [
        {
            "decision_id": item["decision_id"],
            "canonical_sha256": mapping_sha256(item),
        }
        for item in v2["decisions"]
        if item["decision_id"] not in amended_ids
    ]
    dossier["final_amendment_scope"] = {
        "amended_decisions": sorted(amended_ids),
        "other_decisions_reopened": False,
        "package_c_selection_used": False,
    }
    dossier["resource_projection"] = {
        "path": str(RESOURCE_OUTPUT.relative_to(ROOT)),
        "file_sha256": file_sha256(RESOURCE_OUTPUT),
        "launch_scope": "Tier 1 plus protected Tier 2 only",
        "optional_scope": "Tasks 3, 4, and 5 are separate ordered increments and are excluded from launch feasibility",
    }
    dossier["optional_task_increment_manifests"] = [
        {"task_number": task_number, **optional_bindings[task_number]}
        for task_number in (3, 4, 5)
    ]
    dossier["approval_sentence_template"] = (
        "I approve stage13-package-a-protected-core-optional-five-task/v3 exactly as described in the Stage 13 v3 decision dossier with SHA-256 <DOSSIER_SHA256>; Parts C-H may proceed subject to the Stage 14 protected-core launch gate."
    )

    rd1 = decision(dossier, "RD-001")
    rd1["viable_options"] = [
        {
            "option": "A",
            "description": "Fixed Task 1 seeds 0-14 without replacement, protected Task 2, and ordered optional Tasks 3-5.",
            "tradeoff": "Prospective fixed population and breadth panel; optional breadth cannot rescue or replace protected failures.",
        },
        {
            "option": "invalid-C",
            "description": "Phase-complete first-15 selection from seeds 0-19.",
            "tradeoff": "Explicitly rejected; it changes the population selection rule.",
        },
    ]
    rd1["protected_tier_consequences"] = {
        "tier1": "Task 1 uses exactly seeds 0-14 and the accepted three phases, with no replacement.",
        "tier2": "Task 2 modular multiplication uses seeds 0-4, pre/stable phases, and remains protected launch scope.",
        "tier3": "Tasks 3-5 are optional ordered increments, each seeds 0-4, pre/stable, direct plus lowest eligible hard/soft, canonical architecture/basis, both methods.",
    }
    rd1["recommendation"]["teacher_seeds"] = list(range(15))
    rd1["recommendation"]["replacement_rule"] = "no replacement of task, teacher seed, phase, or failed training/student slot beyond its existing fixed cap"
    rd1["recommendation"]["package_c_phase_complete_selection"] = "prohibited"
    rd1["recommendation"]["task3"].update(
        {
            "task_id": TASK_SPECS[3]["task_id"],
            "manifest": optional_bindings[3],
            "optional_order": 1,
        }
    )
    rd1["recommendation"]["task4"] = {
        "task_id": TASK_SPECS[4]["task_id"],
        "family": "modular_polynomial",
        "modulus": 59,
        "domain": "full 59x59",
        "formula": TASK_SPECS[4]["formula"],
        "justification": TASK_SPECS[4]["justification"],
        "manifest": optional_bindings[4],
        "role": "optional Tier 3",
        "optional_order": 2,
    }
    rd1["recommendation"]["task5"] = {
        "task_id": TASK_SPECS[5]["task_id"],
        "family": "modular_polynomial",
        "modulus": 59,
        "domain": "full 59x59",
        "formula": TASK_SPECS[5]["formula"],
        "justification": TASK_SPECS[5]["justification"],
        "manifest": optional_bindings[5],
        "role": "optional Tier 3",
        "optional_order": 3,
    }
    rd1["recommendation"]["optional_task_common_panel"] = {
        "tasks": [3, 4, 5],
        "teacher_seeds": [0, 1, 2, 3, 4],
        "phases": ["pre-grokking", "stable post-grokking"],
        "model_roles": ["direct teacher", "lowest eligible hard student", "lowest eligible soft student"],
        "architecture": "canonical",
        "basis": "canonical",
        "discovery_methods": ["greedy-deletion-centred-logit/v1", "hard-concrete-gates-centred-logit/v1"],
        "replacement": "none",
        "population_interpretation": "tasks are not independent population replicates; report task-indexed descriptive external validity",
    }
    rd1["rationale"] = "The protected population remains the fixed Package A seed roster. Tasks 3-5 add algebraically predeclared breadth only after protected security and cannot affect the launch estimand or replace failures."

    rd7 = decision(dossier, "RD-007")
    rd7["recommendation"]["backend_determinism"] = "Stage 14 must qualify the actual CPU/CUDA/MPS backend. MPS has no presumed authority; a school-Mac MPS class becomes eligible only after exact and semantic qualification on that class, otherwise its qualified MPS throughput is zero and CPU is used."

    rd12 = decision(dossier, "RD-012")
    rd12["protected_tier_consequences"] = {
        "tier1": "All accepted Task 1 work remains protected.",
        "tier2": "Task 2, architecture/basis panels, exact calibration, frontier, and Fourier controls are part of the protected launch core.",
        "tier3": "Only Tasks 3, 4, and 5, in that order, are optional and excluded from launch feasibility.",
    }
    rd12["recommendation"]["protected_launch_scope"] = {
        "definition": "Tier 1 plus protected Tier 2",
        "includes": resource_projection["scopes"]["protected_core"]["includes"],
        "launch_feasibility_must_cover": True,
    }
    rd12["recommendation"]["tier3_order"] = [
        "Task 3 symmetric quadratic increment",
        "Task 4 asymmetric separable mixed-degree increment",
        "Task 5 coupled homogeneous cubic increment",
    ]
    rd12["recommendation"]["optional_increment_rule"] = "each task increment has its own sealed manifest/hash; after protected completion is secure, admit only the next increment when measured remaining capacity can finish it; a terminal failure is retained and never replaced"
    rd12["recommendation"]["shedding_order"] = ["Task 5 not admitted", "Task 4 not admitted", "Task 3 not admitted", "protected core is never shed"]
    rd12["recommendation"]["school_mac_order"] = ["protected core", "Task 3 if whole-increment capacity passes", "Task 4 if whole-increment capacity passes", "Task 5 if whole-increment capacity passes"]
    rd12["uncertainty"] = "Optional tasks are excluded from launch feasibility. Their task-specific convergence and throughput remain Stage 14 measurements; insufficient optional capacity yields non-admission, not protected-core failure."

    rd14 = decision(dossier, "RD-014")
    rd14["viable_options"][0]["description"] = "Bind actual Stage 14 machine inventory, hardware classes, availability intervals, qualified CPU/CUDA/MPS rates, quotas, and permitted hours; require protected-core completion and keep optional tasks ordered."
    rd14["recommendation"].pop("intended_envelope_branch_not_verified_grant", None)
    rd14["recommendation"].pop("stage14_throughput_branch_freeze", None)
    rd14["recommendation"]["launch_scope"] = "Tier 1 plus protected Tier 2 only"
    rd14["recommendation"]["actual_resource_binding"] = resource_projection["stage14_required_bindings"]
    rd14["recommendation"]["prohibited_assumptions"] = resource_projection["prohibited_assumptions"]
    rd14["recommendation"]["scientific_window"] = resource_projection["scientific_window"]
    rd14["recommendation"]["school_mac_branch"] = resource_projection["school_mac_parameterized_branch"]
    rd14["recommendation"]["launch_blocker"] = resource_projection["launch_rule"]
    rd14["recommendation"]["optional_admission"] = "Task 3, then Task 4, then Task 5; only after protected completion is secure and only when the entire next manifest increment fits measured remaining capacity"
    rd14["stage12_evidence"] = rd14["stage12_evidence"] + " The v3 accounting separates accelerator device-hours, host support reservation, standalone CPU work, weak CPU critical path, compact retention, active scratch, and retry/staging worst case."
    rd14["uncertainty"] = "No provider, institution, accelerator count, CPU count, MPS status, uninterrupted window, or quota is presumed. Stage 14 must bind all such facts and block if the protected core cannot finish."

    dossier["package_alternatives"] = [
        {
            "package_id": dossier["recommended_package_id"],
            "summary": "Package A fixed Task 1 seeds 0-14; required protected core is Tier 1 plus protected Tier 2; Tasks 3-5 are separate optional ordered increments and are excluded from launch feasibility.",
        }
    ]
    return dossier


def main() -> int:
    optional_bindings = build_optional_manifests()
    resource_projection = build_resource_projection(optional_bindings)
    write_json(RESOURCE_OUTPUT, resource_projection)
    dossier = build_dossier(optional_bindings, resource_projection)
    write_json(DOSSIER_OUTPUT, dossier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the prospective Stage 13 package resource projection.

The arithmetic is intentionally explicit and uses only technical benchmarks,
constructed timing evidence, and declared planning assumptions.  It does not
read registered checkpoints or scientific results.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCENARIOS = ("lower", "central", "conservative")

PACKAGES = {
    "A": {
        "package_id": "stage13-package-a-conservative-protected/v2",
        "teacher_jobs": 26,
        "student_attempts": [490, 690, 980],
        "canonical_student_attempts": [330, 450, 660],
        "architecture_student_attempts": [160, 240, 320],
        "architecture_average_parameter_ratio": 1.2903,
        "discovery_ledgers_per_method": 675,
        "exact_allowance": 256,
        "hard_restarts": 4,
        "hard_steps": 5000,
        "ordinary_profiles": 60,
        "ordinary_restarts_per_profile": 16,
        "ordinary_exact_allowance": 256,
        "local_exact_per_profile": 128,
        "combinatorial_draws_per_profile": 10000,
        "basis_ledgers_both_methods": 240,
        "architecture_discovery_ledgers_both_methods": 320,
        "fourier_comparison_sets": 60,
        "fourier_inputs_per_condition": 256,
    },
    "B": {
        "package_id": "stage13-package-b-reduced-protected/v2",
        "teacher_jobs": 26,
        "student_attempts": [260, 390, 520],
        "canonical_student_attempts": [220, 330, 440],
        "architecture_student_attempts": [40, 60, 80],
        "architecture_average_parameter_ratio": 1.4350,
        "discovery_ledgers_per_method": 415,
        "exact_allowance": 128,
        "hard_restarts": 2,
        "hard_steps": 2500,
        "ordinary_profiles": 60,
        "ordinary_restarts_per_profile": 8,
        "ordinary_exact_allowance": 128,
        "local_exact_per_profile": 64,
        "combinatorial_draws_per_profile": 2000,
        "basis_ledgers_both_methods": 180,
        "architecture_discovery_ledgers_both_methods": 80,
        "fourier_comparison_sets": 60,
        "fourier_inputs_per_condition": 256,
    },
    "C": {
        "package_id": "stage13-package-c-expanded-roster/v2",
        "teacher_jobs": 31,
        "student_attempts": [490, 690, 980],
        "canonical_student_attempts": [330, 450, 660],
        "architecture_student_attempts": [160, 240, 320],
        "architecture_average_parameter_ratio": 1.2903,
        "discovery_ledgers_per_method": 675,
        "exact_allowance": 256,
        "hard_restarts": 4,
        "hard_steps": 5000,
        "ordinary_profiles": 60,
        "ordinary_restarts_per_profile": 16,
        "ordinary_exact_allowance": 256,
        "local_exact_per_profile": 128,
        "combinatorial_draws_per_profile": 10000,
        "basis_ledgers_both_methods": 240,
        "architecture_discovery_ledgers_both_methods": 320,
        "fourier_comparison_sets": 60,
        "fourier_inputs_per_condition": 256,
    },
}

# Scenario assumptions. CUDA rates are prospectively qualified thresholds, not
# measurements. The conservative training rate is the measured CPU reference.
UPDATES_PER_ATTEMPT = [5000, 20000, 40000]
TRAIN_SECONDS_PER_EQUIVALENT_UPDATE = [0.018544916, 0.035, 0.141523208]
HARD_SECONDS_PER_MODEL_OBJECTIVE_STEP = [0.035, 0.14, 0.28]
GPU_EFFICIENCY = [0.85, 0.70, 0.50]
CPU_EFFICIENCY = [0.85, 0.70, 0.50]
EXACT_SECONDS = 0.088445375
CALIBRATION_SECONDS_PER_MASK = [0.0005, 0.005, EXACT_SECONDS]
FOURIER_SECONDS_PER_INPUT_CONDITION = [0.00005, 0.0005, 0.005]
FOURIER_ALIGNMENT_SECONDS_PER_SET = [1.0, 10.0, 60.0]


def _round(value: float, digits: int = 3) -> float:
    return round(value, digits)


def _scenario_projection(package: dict[str, object], index: int) -> dict[str, object]:
    teacher_jobs = int(package["teacher_jobs"])
    canonical_attempts = int(package["canonical_student_attempts"][index])
    architecture_attempts = int(package["architecture_student_attempts"][index])
    ratio = float(package["architecture_average_parameter_ratio"])
    updates = UPDATES_PER_ATTEMPT[index]

    # The tiny calibration teacher is conservatively charged as canonical.
    teacher_equivalent_updates = teacher_jobs * updates
    canonical_student_equivalent_updates = canonical_attempts * updates
    architecture_student_equivalent_updates = architecture_attempts * updates * ratio
    training_equivalent_updates = (
        teacher_equivalent_updates
        + canonical_student_equivalent_updates
        + architecture_student_equivalent_updates
    )
    teacher_training_gpu_hours = teacher_equivalent_updates * TRAIN_SECONDS_PER_EQUIVALENT_UPDATE[index] / 3600.0
    canonical_student_gpu_hours = canonical_student_equivalent_updates * TRAIN_SECONDS_PER_EQUIVALENT_UPDATE[index] / 3600.0
    architecture_student_gpu_hours = architecture_student_equivalent_updates * TRAIN_SECONDS_PER_EQUIVALENT_UPDATE[index] / 3600.0
    training_gpu_hours = teacher_training_gpu_hours + canonical_student_gpu_hours + architecture_student_gpu_hours

    discovery_jobs = int(package["discovery_ledgers_per_method"])
    main_hard_steps = (
        discovery_jobs
        * int(package["hard_restarts"])
        * int(package["hard_steps"])
    )
    ordinary_hard_profiles = int(package["ordinary_profiles"]) // 2
    ordinary_hard_steps = (
        ordinary_hard_profiles
        * int(package["ordinary_restarts_per_profile"])
        * int(package["hard_steps"])
    )
    total_hard_steps = main_hard_steps + ordinary_hard_steps
    main_hard_gpu_hours = main_hard_steps * HARD_SECONDS_PER_MODEL_OBJECTIVE_STEP[index] / 3600.0
    ordinary_hard_gpu_hours = ordinary_hard_steps * HARD_SECONDS_PER_MODEL_OBJECTIVE_STEP[index] / 3600.0
    hard_gpu_hours = main_hard_gpu_hours + ordinary_hard_gpu_hours

    discovery_exact = 2 * discovery_jobs * int(package["exact_allowance"])
    ordinary_exact = (
        int(package["ordinary_profiles"])
        * int(package["ordinary_exact_allowance"])
    )
    local_exact = (
        int(package["ordinary_profiles"])
        * int(package["local_exact_per_profile"])
    )
    calibration_exact = 2**18
    total_exact = discovery_exact + ordinary_exact + local_exact + calibration_exact
    discovery_exact_cpu_hours = discovery_exact * EXACT_SECONDS / 3600.0
    ordinary_exact_cpu_hours = ordinary_exact * EXACT_SECONDS / 3600.0
    local_exact_cpu_hours = local_exact * EXACT_SECONDS / 3600.0
    calibration_cpu_hours = calibration_exact * CALIBRATION_SECONDS_PER_MASK[index] / 3600.0
    exact_cpu_hours = discovery_exact_cpu_hours + ordinary_exact_cpu_hours + local_exact_cpu_hours + calibration_cpu_hours

    greedy_jobs = discovery_jobs
    greedy_ranking_cpu_hours = greedy_jobs * 0.13140775 / 3600.0
    combinatorial_draws = (
        int(package["ordinary_profiles"])
        * int(package["combinatorial_draws_per_profile"])
    )
    # Constructed packing work has no production timing; use transparent rates.
    null_cpu_hours = combinatorial_draws * [2e-6, 2e-5, 2e-4][index] / 3600.0

    comparison_sets = int(package["fourier_comparison_sets"])
    fourier_input_trials = (
        comparison_sets * 6 * int(package["fourier_inputs_per_condition"])
    )
    fourier_cpu_hours = (
        fourier_input_trials * FOURIER_SECONDS_PER_INPUT_CONDITION[index]
        + comparison_sets * FOURIER_ALIGNMENT_SECONDS_PER_SET[index]
    ) / 3600.0

    merge_export_cpu_hours = [4.0, 16.0, 64.0][index]
    other_cpu_hours = greedy_ranking_cpu_hours + null_cpu_hours + fourier_cpu_hours
    cpu_core_hours = exact_cpu_hours + other_cpu_hours + merge_export_cpu_hours
    gpu_hours = training_gpu_hours + hard_gpu_hours
    gpu_wall = gpu_hours / (16.0 * GPU_EFFICIENCY[index])
    cpu_wall = cpu_core_hours / (256.0 * CPU_EFFICIENCY[index])
    # Dependency/pipeline allowance beyond pure capacity lower bounds.
    orchestration_hours = [1.0, 4.0, 12.0][index]
    projected_science_wall = gpu_wall + cpu_wall + orchestration_hours

    # Persistent bytes: rolling checkpoint retention=2, final dense output,
    # compact search records, reports. Conservative includes a 2x staging copy.
    canonical_training_bytes = [8_499_412, 11_227_236, 22_454_472][index]
    architecture_training_bytes = (
        [8_499_412, 5_771_588 + 2 * 2_727_824 * ratio, 2 * (5_771_588 + 2 * 2_727_824 * ratio)][index]
    )
    training_storage = (
        (teacher_jobs + canonical_attempts) * canonical_training_bytes
        + architecture_attempts * architecture_training_bytes
    )
    hard_runs = (
        discovery_jobs * int(package["hard_restarts"])
        + ordinary_hard_profiles * int(package["ordinary_restarts_per_profile"])
    )
    hard_storage = hard_runs * [2_260_000, 4_520_000, 9_040_000][index]
    ledger_storage = total_exact * [512, 1024, 2048][index]
    fourier_storage = fourier_input_trials * [512, 1024, 2048][index]
    report_storage = [1e9, 3e9, 8e9][index]
    persistent_bytes = (
        training_storage + hard_storage + ledger_storage + fourier_storage + report_storage
    )
    scratch_bytes = persistent_bytes * [1.5, 2.0, 3.0][index]

    per_training_worker_gib = [0.861, 1.72, 3.44][index]
    per_cpu_worker_gib = [0.482, 0.75, 1.0][index]

    return {
        "scenario": SCENARIOS[index],
        "assumptions": {
            "updates_per_training_attempt": updates,
            "seconds_per_canonical_equivalent_training_update": TRAIN_SECONDS_PER_EQUIVALENT_UPDATE[index],
            "seconds_per_model_in_loop_hard_concrete_step": HARD_SECONDS_PER_MODEL_OBJECTIVE_STEP[index],
            "gpu_scheduling_efficiency": GPU_EFFICIENCY[index],
            "cpu_scheduling_efficiency": CPU_EFFICIENCY[index],
        },
        "compute": {
            "canonical_equivalent_training_updates": _round(training_equivalent_updates, 0),
            "training_gpu_hours": _round(training_gpu_hours),
            "hard_concrete_optimizer_steps": total_hard_steps,
            "hard_concrete_gpu_hours": _round(hard_gpu_hours),
            "total_gpu_hours": _round(gpu_hours),
            "exact_evaluation_cpu_core_hours": _round(exact_cpu_hours),
            "other_cpu_core_hours": _round(other_cpu_hours),
            "merge_export_cpu_core_hours": merge_export_cpu_hours,
            "total_cpu_core_hours": _round(cpu_core_hours),
            "capacity_lower_bound_gpu_wall_hours": _round(gpu_wall),
            "capacity_lower_bound_cpu_wall_hours": _round(cpu_wall),
            "orchestration_dependency_allowance_hours": orchestration_hours,
            "projected_science_wall_hours": _round(projected_science_wall),
            "audit_reserve_hours": 12,
            "projected_total_wall_hours": _round(projected_science_wall + 12),
            "fits_84_hour_science_window": projected_science_wall <= 84,
            "fits_96_hour_total_window": projected_science_wall + 12 <= 96,
            "category_breakdown_additive": {
                "teacher_training_gpu_hours": _round(teacher_training_gpu_hours),
                "canonical_student_training_gpu_hours": _round(canonical_student_gpu_hours),
                "architecture_student_training_gpu_hours": _round(architecture_student_gpu_hours),
                "greedy_discovery_ranking_cpu_core_hours": _round(greedy_ranking_cpu_hours),
                "hard_concrete_main_gpu_hours": _round(main_hard_gpu_hours),
                "ordinary_restart_hard_concrete_gpu_hours": _round(ordinary_hard_gpu_hours),
                "discovery_exact_evaluation_cpu_core_hours": _round(discovery_exact_cpu_hours),
                "ordinary_restart_exact_cpu_core_hours": _round(ordinary_exact_cpu_hours),
                "local_null_exact_cpu_core_hours": _round(local_exact_cpu_hours),
                "combinatorial_null_cpu_core_hours": _round(null_cpu_hours),
                "exact_2pow18_calibration_cpu_core_hours": _round(calibration_cpu_hours),
                "fourier_cpu_core_hours": _round(fourier_cpu_hours),
                "merge_and_export_cpu_core_hours": merge_export_cpu_hours,
            },
            "panel_subset_attribution_nonadditive": {
                "basis_discovery_ledgers_both_methods": int(package["basis_ledgers_both_methods"]),
                "basis_exact_evaluations": int(package["basis_ledgers_both_methods"]) * int(package["exact_allowance"]),
                "basis_hard_concrete_steps": int(package["basis_ledgers_both_methods"]) // 2 * int(package["hard_restarts"]) * int(package["hard_steps"]),
                "architecture_training_gpu_hours": _round(architecture_student_gpu_hours),
                "architecture_discovery_ledgers_both_methods": int(package["architecture_discovery_ledgers_both_methods"]),
                "architecture_exact_evaluations": int(package["architecture_discovery_ledgers_both_methods"]) * int(package["exact_allowance"]),
                "architecture_hard_concrete_steps": int(package["architecture_discovery_ledgers_both_methods"]) // 2 * int(package["hard_restarts"]) * int(package["hard_steps"]),
            },
        },
        "memory": {
            "per_training_or_hard_concrete_worker_gib": per_training_worker_gib,
            "sixteen_training_workers_aggregate_gib": _round(per_training_worker_gib * 16),
            "per_exact_or_reducer_cpu_worker_gib": per_cpu_worker_gib,
            "two_hundred_fifty_six_cpu_workers_aggregate_gib": _round(per_cpu_worker_gib * 256),
            "gpu_vram_requirement": "unbenchmarked; Stage 14 qualification threshold applies",
        },
        "storage": {
            "training_artifacts_gib": _round(training_storage / 2**30),
            "search_checkpoints_gib": _round(hard_storage / 2**30),
            "exact_ledgers_gib": _round(ledger_storage / 2**30),
            "fourier_records_gib": _round(fourier_storage / 2**30),
            "reports_and_export_metadata_gib": _round(report_storage / 2**30),
            "persistent_total_gib": _round(persistent_bytes / 2**30),
            "scratch_peak_gib": _round(scratch_bytes / 2**30),
            "fits_1_tib_persistent": persistent_bytes <= 2**40,
            "fits_4_tib_scratch": scratch_bytes <= 4 * 2**40,
        },
    }


def _job_counts(package: dict[str, object]) -> dict[str, object]:
    discovery = int(package["discovery_ledgers_per_method"])
    profiles = int(package["ordinary_profiles"])
    comparisons = int(package["fourier_comparison_sets"])
    exact_discovery = 2 * discovery * int(package["exact_allowance"])
    exact_ordinary = profiles * int(package["ordinary_exact_allowance"])
    exact_local = profiles * int(package["local_exact_per_profile"])
    return {
        "teacher_training_jobs": int(package["teacher_jobs"]),
        "student_attempt_jobs_by_scenario": dict(zip(SCENARIOS, package["student_attempts"], strict=True)),
        "architecture_student_attempt_jobs_by_scenario": dict(zip(SCENARIOS, package["architecture_student_attempts"], strict=True)),
        "greedy_discovery_ledger_jobs": discovery,
        "hard_concrete_discovery_ledger_jobs": discovery,
        "hard_concrete_native_restart_runs": discovery * int(package["hard_restarts"]),
        "ordinary_restart_profile_jobs": profiles,
        "ordinary_restart_subjobs": profiles * int(package["ordinary_restarts_per_profile"]),
        "discovery_exact_evaluations_including_intact": exact_discovery,
        "ordinary_restart_exact_evaluations_including_intact": exact_ordinary,
        "local_null_exact_evaluations": exact_local,
        "combinatorial_null_draws": profiles * int(package["combinatorial_draws_per_profile"]),
        "basis_panel_discovery_ledgers_both_methods_subset": int(package["basis_ledgers_both_methods"]),
        "architecture_panel_discovery_ledgers_both_methods_subset": int(package["architecture_discovery_ledgers_both_methods"]),
        "fourier_comparison_sets": comparisons,
        "fourier_condition_jobs": comparisons * 6,
        "fourier_input_condition_trials": comparisons * 6 * int(package["fourier_inputs_per_condition"]),
        "exact_calibration_enumeration_jobs": 1,
        "exact_calibration_masks": 2**18,
        "all_exact_evaluations": exact_discovery + exact_ordinary + exact_local + 2**18,
        "merge_jobs": 1,
        "export_jobs": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = {
        "schema_version": "stage13-package-resource-projection/v2",
        "classification": "prospective non-scientific planning evidence",
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
        "envelope": {
            "total_hours": 96,
            "science_execution_hours": 84,
            "final_audit_reserve_hours": 12,
            "cpu_cores": 256,
            "cuda_gpus": 16,
            "scratch_tib": 4,
            "persistent_tib": 1,
            "grant_verified": False,
        },
        "benchmark_bindings": {
            "stage9_training": "followup/benchmarks/stage9_training_backend_benchmark_m5max_v1.json",
            "stage10_exact": "followup/benchmarks/stage10_discovery_compute_benchmark_m5max_v1.json",
            "stage13_constructed_search": "followup/benchmarks/stage13_search_profile_benchmark_v1.json",
        },
        "packages": {},
        "unbenchmarked_quantities": [
            "CUDA teacher and student training update throughput and VRAM",
            "model-in-the-loop hard-concrete objective step throughput and VRAM",
            "concurrent filesystem and scheduler efficiency at 16 GPU / 256 CPU scale",
            "alternate-architecture throughput beyond parameter-count scaling",
            "Fourier activation capture, alignment, and intervention throughput",
            "tiny mod-7 exact-enumeration throughput",
            "production merge compression and verified export throughput",
            "actual provider hardware, quotas, paths, and queue latency",
        ],
    }
    for key, package in PACKAGES.items():
        record["packages"][key] = {
            "package_id": package["package_id"],
            "job_counts": _job_counts(package),
            "scenarios": [_scenario_projection(package, i) for i in range(3)],
        }
    payload = json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

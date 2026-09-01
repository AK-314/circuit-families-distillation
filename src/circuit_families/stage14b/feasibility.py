"""Dependency- and interval-aware protected-core feasibility binding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .records import Stage14BError, canonical_sha256, with_boundary


def solve_feasibility(
    projection: Mapping[str, Any],
    qualified_pools: Sequence[Mapping[str, Any]],
    *,
    total_hours: float,
    final_window_hours: float = 12.0,
) -> dict[str, Any]:
    if total_hours < 0 or final_window_hours != 12.0:
        raise Stage14BError("Stage 14-B requires the frozen 12-hour final window")
    real_pools = [
        pool
        for pool in qualified_pools
        if pool.get("production_qualified") is True and pool.get("production_pool") is True
    ]
    science_hours = max(0.0, total_hours - final_window_hours)
    if not real_pools:
        record = with_boundary(
            {
                "schema_version": "stage14b-feasibility/v1",
                "status": "WAITING_FOR_AUTHORIZED_RESOURCE_FACTS",
                "protected_core_launch": "PROTECTED_CORE_LAUNCH_BLOCKED",
                "protected_core_feasible": None,
                "total_hours": total_hours,
                "science_hours": science_hours,
                "final_window_hours": final_window_hours,
                "qualified_pool_count": 0,
                "missing_facts": [
                    "Symbolica grant and scheduler/account/queue",
                    "verified hardware counts/classes and permitted 96-hour interval",
                    "qualified per-class rates and host-support cores",
                    "verified RAM/VRAM and per-job memory",
                    "verified scratch/persistent quotas and permitted roots",
                    "container runtime and immutable image digest",
                    "transfer destination and measured merge/export path",
                    "Eton approvals, machine classes, and allowed intervals",
                ],
                "optional_tasks_assessed_after_protected": False,
            }
        )
        record["feasibility_sha256"] = canonical_sha256(record)
        return record
    device_hours = sum(float(pool["available_accelerator_device_hours"]) for pool in real_pools)
    cpu_hours = sum(float(pool["available_standalone_cpu_core_hours"]) for pool in real_pools)
    host_support_available = sum(
        float(pool["available_host_support_cpu_core_hours"]) for pool in real_pools
    )
    memory_ok = all(float(pool["peak_memory_fraction"]) <= 0.8 for pool in real_pools)
    storage_ok = all(float(pool["peak_storage_fraction"]) <= 0.8 for pool in real_pools)
    protected_scenarios = projection["scopes"]["protected_core"]["scenarios"]
    try:
        protected = next(item for item in protected_scenarios if item["scenario"] == "conservative")
    except StopIteration as exc:
        raise Stage14BError("frozen conservative protected scenario is missing") from exc
    compute = protected["compute"]
    accelerator_required = float(compute["gpu_device_hours"])
    cpu_required = float(compute["standalone_cpu_core_hours"])
    critical_path = float(compute["serial_or_weakly_parallel_cpu_critical_path_hours"])
    host_support_required = sum(
        float(pool["allocated_accelerator_device_hours"])
        * float(pool["host_support_cores_per_active_accelerator"])
        for pool in real_pools
    )
    actual_accelerator_concurrency = sum(
        int(pool["maximum_available_accelerator_concurrency"]) for pool in real_pools
    )
    actual_cpu_concurrency = sum(
        int(pool["maximum_available_standalone_cpu_concurrency"]) for pool in real_pools
    )
    interval_ok = all(pool.get("dependency_interval_schedule_pass") is True for pool in real_pools)
    pass_gate = (
        device_hours >= accelerator_required
        and cpu_hours >= cpu_required
        and host_support_available >= host_support_required
        and critical_path <= science_hours
        and memory_ok
        and storage_ok
        and interval_ok
    )
    record = with_boundary(
        {
            "schema_version": "stage14b-feasibility/v1",
            "status": "PASS" if pass_gate else "FAILED",
            "protected_core_launch": "READY_FOR_ALEX6_REVIEW"
            if pass_gate
            else "PROTECTED_CORE_LAUNCH_BLOCKED",
            "protected_core_feasible": pass_gate,
            "total_hours": total_hours,
            "science_hours": science_hours,
            "final_window_hours": final_window_hours,
            "qualified_pool_count": len(real_pools),
            "accelerator_device_hours": {
                "required": accelerator_required,
                "available": device_hours,
            },
            "standalone_cpu_core_hours": {"required": cpu_required, "available": cpu_hours},
            "host_support_cpu_core_hours": {
                "required": host_support_required,
                "available": host_support_available,
            },
            "serial_or_weak_cpu_critical_path_hours": critical_path,
            "maximum_useful_concurrency": {
                "accelerators": compute["maximum_useful_accelerator_concurrency"],
                "standalone_cpu": compute["maximum_useful_standalone_cpu_concurrency"],
            },
            "actually_available_concurrency": {
                "accelerators": actual_accelerator_concurrency,
                "standalone_cpu": actual_cpu_concurrency,
            },
            "ideal_wall_hours": compute["ideal_wall_hours_at_maximum_useful_concurrency"],
            "efficiency_adjusted_wall_hours": compute[
                "efficiency_adjusted_wall_hours_at_maximum_useful_concurrency"
            ],
            "storage_gib": {
                "retained_compact": protected["storage"]["expected_retained_compact_output_gib"],
                "staging_retry_worst_case": protected["storage"][
                    "uncompressed_staging_retry_worst_case_gib"
                ],
                "requested_persistent_quota": protected["storage"][
                    "requested_persistent_safety_quota_gib"
                ],
                "requested_scratch_quota": protected["storage"][
                    "requested_scratch_safety_quota_gib"
                ],
            },
            "memory_headroom_pass": memory_ok,
            "storage_headroom_pass": storage_ok,
            "dependency_and_interval_pass": interval_ok,
            "optional_tasks_assessed_after_protected": pass_gate,
        }
    )
    record["feasibility_sha256"] = canonical_sha256(record)
    return record

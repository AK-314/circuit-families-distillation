"""Outcome-neutral monitoring, human gates, alerts, and final-window rules."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from circuit_families.stage12p3.contracts import validate_operational_status

from .records import Stage14BError, canonical_sha256, with_boundary

FINAL_WINDOW_SECONDS = 12 * 60 * 60


def model_storage(
    *,
    active_scratch_bytes: int,
    rolling_recovery_bytes: int,
    staging_retry_worst_case_bytes: int,
    retained_compact_bytes: int,
    destination_transfer_bytes: int,
    scratch_quota_bytes: int,
    persistent_quota_bytes: int,
) -> dict[str, Any]:
    values = (
        active_scratch_bytes,
        rolling_recovery_bytes,
        staging_retry_worst_case_bytes,
        retained_compact_bytes,
        destination_transfer_bytes,
        scratch_quota_bytes,
        persistent_quota_bytes,
    )
    if any(isinstance(value, bool) or value < 0 for value in values):
        raise Stage14BError("storage coordinates must be non-negative integers")
    scratch_peak = active_scratch_bytes + rolling_recovery_bytes + staging_retry_worst_case_bytes
    persistent_peak = retained_compact_bytes + destination_transfer_bytes
    return with_boundary(
        {
            "schema_version": "stage14b-storage-model/v1",
            "active_scratch_bytes": active_scratch_bytes,
            "rolling_recovery_bytes": rolling_recovery_bytes,
            "staging_retry_worst_case_bytes": staging_retry_worst_case_bytes,
            "retained_compact_bytes": retained_compact_bytes,
            "destination_transfer_bytes": destination_transfer_bytes,
            "scratch_quota_bytes": scratch_quota_bytes,
            "persistent_quota_bytes": persistent_quota_bytes,
            "scratch_peak_bytes": scratch_peak,
            "persistent_peak_bytes": persistent_peak,
            "scratch_headroom_pass": scratch_quota_bytes > 0
            and scratch_peak <= int(scratch_quota_bytes * 0.8),
            "persistent_headroom_pass": persistent_quota_bytes > 0
            and persistent_peak <= int(persistent_quota_bytes * 0.8),
            "git_or_lfs_destination": False,
        }
    )


def final_window_start(total_start: int, total_hours: float) -> int:
    if total_start < 0 or total_hours < 0:
        raise Stage14BError("campaign time coordinates must be non-negative")
    return total_start + max(0, int(total_hours * 3600) - FINAL_WINDOW_SECONDS)


def monitor_campaign(
    job_records: Sequence[Mapping[str, Any]],
    *,
    now: int,
    total_start: int,
    total_hours: float,
    storage: Mapping[str, int],
    gates: Mapping[str, str],
) -> dict[str, Any]:
    states = Counter(str(item["state"]) for item in job_records)
    families: dict[str, Counter[str]] = {}
    retries = 0
    alerts = []
    for item in job_records:
        family = str(item["family"])
        families.setdefault(family, Counter())[str(item["state"])] += 1
        retries += max(0, int(item.get("attempts", 1)) - 1)
        heartbeat = item.get("heartbeat_at")
        if heartbeat is not None and int(heartbeat) > now:
            alerts.append(
                {
                    "category": "clock_anomaly",
                    "logical_job_id": item["logical_job_id"],
                    "action": "pause_and_check_clock",
                }
            )
        if item["state"] in {"claimed", "running"} and (
            heartbeat is None or now - int(heartbeat) > int(item.get("lease_seconds", 300))
        ):
            alerts.append(
                {
                    "category": "stale_heartbeat",
                    "logical_job_id": item["logical_job_id"],
                    "action": "reconcile",
                }
            )
        if item.get("duplicate_claim"):
            alerts.append(
                {
                    "category": "duplicate_claim",
                    "logical_job_id": item["logical_job_id"],
                    "action": "audit_claim",
                }
            )
        if item.get("partial_output"):
            alerts.append(
                {
                    "category": "partial_object",
                    "logical_job_id": item["logical_job_id"],
                    "action": "verify_or_retry",
                }
            )
        queued_at = item.get("queued_at")
        if item.get("scheduler_state") == "pending" and queued_at is not None:
            if now - int(queued_at) > int(item.get("queue_stall_seconds", 3600)):
                alerts.append(
                    {
                        "category": "queue_stall",
                        "logical_job_id": item["logical_job_id"],
                        "action": "inspect_scheduler_without_reprioritizing_science",
                    }
                )
        if int(item.get("preemptions", 0)) > 0:
            alerts.append(
                {
                    "category": "preemption",
                    "logical_job_id": item["logical_job_id"],
                    "action": "resume_with_frozen_retry_policy",
                }
            )
        if item.get("output_conflict"):
            alerts.append(
                {
                    "category": "conflicting_output",
                    "logical_job_id": item["logical_job_id"],
                    "action": "stop_and_audit",
                }
            )
        if item.get("transfer_failure"):
            alerts.append(
                {
                    "category": "transfer_failure",
                    "logical_job_id": item["logical_job_id"],
                    "action": "resume_verified_prefix",
                }
            )
    used = int(storage.get("used_bytes", 0))
    quota = int(storage.get("quota_bytes", 0))
    if quota <= 0:
        alerts.append(
            {"category": "quota_unknown", "logical_job_id": None, "action": "bind_verified_quota"}
        )
    elif used > quota:
        alerts.append({"category": "quota_breach", "logical_job_id": None, "action": "pause"})
    elif used > int(quota * 0.8):
        alerts.append(
            {"category": "storage_warning", "logical_job_id": None, "action": "compact_or_pause"}
        )
    boundary = final_window_start(total_start, total_hours)
    final_active = now >= boundary
    if final_active and any(
        item.get("optional") and item["state"] == "planned" for item in job_records
    ):
        alerts.append(
            {
                "category": "final_window_optional_work",
                "logical_job_id": None,
                "action": "stop_optional_dispatch",
            }
        )
    if final_active and any(
        not item.get("optional")
        and item["state"] not in {"sealed_success", "terminal_failure", "unavailable", "censored"}
        for item in job_records
    ):
        alerts.append(
            {
                "category": "final_window_risk",
                "logical_job_id": None,
                "action": "stop_optional_and_close_protected_state",
            }
        )
    record = with_boundary(
        {
            "schema_version": "stage14b-operational-status/v1",
            "counts": dict(sorted(states.items())),
            "counts_by_family": {
                key: dict(sorted(value.items())) for key, value in sorted(families.items())
            },
            "failures": states.get("terminal_failure", 0),
            "retries": retries,
            "resources": {"active_jobs": states.get("running", 0)},
            "storage": dict(storage),
            "hashes": {
                "job_inventory_sha256": canonical_sha256([dict(item) for item in job_records])
            },
            "gates": dict(gates),
            "scheduler_running_is_logical_success": False,
            "sealed_output_required": True,
            "final_window": {
                "reserved_seconds": FINAL_WINDOW_SECONDS,
                "starts_at": boundary,
                "active": final_active,
                "new_optional_work_permitted": not final_active,
            },
            "alerts": alerts,
        }
    )
    validate_operational_status(record)
    return record


def gate_15_1(
    *,
    exact_sha: bool,
    freeze_verified: bool,
    environment_verified: bool,
    inputs_verified: bool,
    resources_qualified: bool,
    tiny_pipeline_equal: bool,
    feasibility_pass: bool,
) -> dict[str, Any]:
    checks = {
        "exact_sha": exact_sha,
        "freeze_verified": freeze_verified,
        "environment_verified": environment_verified,
        "inputs_verified": inputs_verified,
        "resources_qualified": resources_qualified,
        "tiny_pipeline_equal": tiny_pipeline_equal,
        "protected_core_feasible": feasibility_pass,
        "production_authorized": False,
    }
    return with_boundary(
        {
            "schema_version": "stage14b-gate-15.1/v1",
            "checks": checks,
            "gate_status": "BLOCKED_UNAUTHORIZED"
            if all(value for key, value in checks.items() if key != "production_authorized")
            else "BLOCKED_NOT_READY",
            "production_release": False,
        }
    )


def gate_15_2(
    protected_jobs: Sequence[Mapping[str, Any]], *, projected_secure: bool
) -> dict[str, Any]:
    counts = Counter(item["state"] for item in protected_jobs)
    complete = all(
        item["state"] in {"sealed_success", "terminal_failure", "unavailable", "censored"}
        for item in protected_jobs
    )
    return with_boundary(
        {
            "schema_version": "stage14b-gate-15.2/v1",
            "planned": len(protected_jobs),
            "terminal": sum(
                counts[state]
                for state in ("sealed_success", "terminal_failure", "unavailable", "censored")
            ),
            "state_counts": dict(sorted(counts.items())),
            "protected_completion_secure": projected_secure,
            "optional_admission_permitted": complete and projected_secure,
            "optional_order": [3, 4, 5],
        }
    )


def gate_15_3(
    *,
    final_window_active: bool,
    protected_closed: bool,
    recomputation_equal: bool,
    export_verified: bool,
    source_preserved: bool,
) -> dict[str, Any]:
    checks = {
        "final_window_active": final_window_active,
        "new_optional_work_stopped": final_window_active,
        "protected_terminal_state_closed": protected_closed,
        "independent_recomputation_equal": recomputation_equal,
        "destination_reread_verified": export_verified,
        "source_preserved": source_preserved,
    }
    return with_boundary(
        {
            "schema_version": "stage14b-gate-15.3/v1",
            "checks": checks,
            "gate_status": "PASS" if all(checks.values()) else "BLOCKED",
            "custody_approval_required_before_deletion": True,
        }
    )

from __future__ import annotations

import pytest

from circuit_families.stage14b.monitoring import (
    FINAL_WINDOW_SECONDS,
    final_window_start,
    gate_15_3,
    model_storage,
)
from circuit_families.stage14b.records import Stage14BError
from circuit_families.stage14b.scheduler import (
    SlurmClassAdapter,
    SlurmConfig,
    VerifiedLaunchAuthorization,
    WorkerCapabilities,
    deterministic_placement_plan,
    verify_launch_authorization,
)


def _adapter() -> SlurmClassAdapter:
    return SlurmClassAdapter(
        SlurmConfig(
            "slurm/test/v1",
            "submit-injected",
            "status-injected",
            "accounting-injected",
            "cancel-injected",
            "account-injected",
            "partition-injected",
            "worker-injected",
        )
    )


def test_slurm_plan_keeps_identity_and_production_is_unreachable() -> None:
    adapter = _adapter()
    plan = adapter.dry_run(["job-b", "job-a"], concurrency_cap=2)
    assert [item["logical_job_id"] for item in plan["entries"]] == ["job-b", "job-a"]
    assert plan["provider_job_id"] is None
    with pytest.raises(Stage14BError, match="authorization absent"):
        adapter.submit(plan, capability=None)
    with pytest.raises(Stage14BError, match="cannot be constructed"):
        VerifiedLaunchAuthorization("0" * 64, "0" * 40, object())


def test_exact_authorization_is_single_use_and_edited_json_fails() -> None:
    stage14_sha = "1" * 40
    bindings = {"environment_sha256": "2" * 64, "input_bundle_sha256": "3" * 64}
    authorization = {
        "schema_version": "alex6-stage15-launch-authorization/v1",
        "authority": "Alex 6",
        "approved": True,
        "stage14_sha": stage14_sha,
        "bindings": bindings,
        "authorization_nonce": "prospective-human-record",
        "scientific_data": False,
        "production_eligible": True,
        "definitive_execution_started": False,
    }
    capability = verify_launch_authorization(
        authorization,
        current_sha=stage14_sha,
        expected_bindings=bindings,
        operator_confirmation=f"LAUNCH_STAGE15_AT_{stage14_sha}",
    )
    adapter = _adapter()
    plan = adapter.dry_run(["job-a"], concurrency_cap=1)
    with pytest.raises(Stage14BError, match="no authorized scheduler executor"):
        adapter.submit(plan, capability=capability)
    with pytest.raises(Stage14BError, match="replayed"):
        adapter.submit(plan, capability=capability)
    edited = dict(authorization)
    edited["authority"] = "Austin 6"
    with pytest.raises(Stage14BError, match="authority"):
        verify_launch_authorization(
            edited,
            current_sha=stage14_sha,
            expected_bindings=bindings,
            operator_confirmation=f"LAUNCH_STAGE15_AT_{stage14_sha}",
        )


def test_capability_matching_honors_per_host_memory_and_interval() -> None:
    jobs = [
        {
            "logical_job_id": "job-a",
            "priority": 10,
            "requirements": {
                "resource_class": "qualified-cpu/v1",
                "cpu_cores": 2,
                "memory_bytes": 1024,
                "accelerators": 0,
                "scratch_bytes": 100,
                "wall_seconds": 20,
                "availability_interval": [10, 20],
            },
        },
        {
            "logical_job_id": "job-b",
            "priority": 5,
            "requirements": {
                "resource_class": "qualified-cpu/v1",
                "cpu_cores": 1,
                "memory_bytes": 4096,
                "accelerators": 0,
                "scratch_bytes": 100,
                "wall_seconds": 20,
                "availability_interval": [10, 20],
            },
        },
    ]
    worker = WorkerCapabilities(
        "pool-a",
        ("qualified-cpu/v1",),
        2,
        2048,
        0,
        1000,
        60,
        ((0, 30),),
    )
    plan = deterministic_placement_plan(jobs, [worker])
    assert plan["placements"][0]["status"] == "PLACED"
    assert plan["placements"][1]["status"] == "UNPLACED"
    assert not plan["logical_identity_changed"]


def test_final_window_is_exactly_twelve_hours() -> None:
    assert FINAL_WINDOW_SECONDS == 43_200
    assert final_window_start(100, 96) == 100 + 84 * 3600
    gate = gate_15_3(
        final_window_active=True,
        protected_closed=True,
        recomputation_equal=True,
        export_verified=True,
        source_preserved=True,
    )
    assert gate["gate_status"] == "PASS"
    assert gate["custody_approval_required_before_deletion"]
    storage = model_storage(
        active_scratch_bytes=10,
        rolling_recovery_bytes=10,
        staging_retry_worst_case_bytes=20,
        retained_compact_bytes=10,
        destination_transfer_bytes=10,
        scratch_quota_bytes=100,
        persistent_quota_bytes=100,
    )
    assert storage["scratch_headroom_pass"]
    assert storage["persistent_headroom_pass"]
    assert storage["git_or_lfs_destination"] is False

from __future__ import annotations

import json
from pathlib import Path

from circuit_families.stage14b.feasibility import solve_feasibility
from circuit_families.stage14b.rehearsal import compare_rehearsals, run_rehearsal
from circuit_families.stage14b.resources import inventory_resource_pool, qualify_backend


def test_uninterrupted_and_resumed_rehearsals_match(tmp_path: Path) -> None:
    first = run_rehearsal(tmp_path / "first", interrupted=False, hash_seed=1)
    second = run_rehearsal(tmp_path / "second", interrupted=True, hash_seed=271828)
    comparison = compare_rehearsals(first, second)
    assert comparison["canonical_content_equal"]
    assert first["job_count"] == 48
    assert first["forced_failure_count"] == 20
    assert second["transfer_interruption_observed"]
    assert first["unauthorized_production_launch_rejected"]
    assert first["production_release_enabled"] is False


def test_local_cpu_is_technical_only(tmp_path: Path) -> None:
    inventory = inventory_resource_pool(
        provider="local-test",
        site="pytest",
        pool_reference="pytest-host/v1",
        permitted_roots=(tmp_path,),
        production_pool=False,
    )
    qualification = qualify_backend(
        inventory,
        backend="cpu",
        output_root=tmp_path / "qualification",
        absolute_tolerance=1e-6,
        relative_tolerance=1e-5,
        memory_ceiling_bytes=4 * 1024**3,
        disk_ceiling_bytes=4096,
        time_ceiling_seconds=60,
    )
    assert qualification["technical_suite"] == "PASS"
    assert qualification["qualification_status"] == "TECHNICAL_ONLY_PASS"
    assert qualification["production_qualified"] is False
    assert len(qualification["repeats"]) == 3


def test_missing_real_resources_blocks_without_salvage() -> None:
    root = Path(__file__).resolve().parents[1]
    projection = json.loads(
        (root / "followup/manifests/stage13_scope_resource_projection_v3.json").read_text()
    )
    result = solve_feasibility(projection, (), total_hours=96)
    assert result["status"] == "WAITING_FOR_AUTHORIZED_RESOURCE_FACTS"
    assert result["protected_core_launch"] == "PROTECTED_CORE_LAUNCH_BLOCKED"
    assert result["protected_core_feasible"] is None
    assert result["science_hours"] == 84


def test_verified_capacity_uses_conservative_protected_equation() -> None:
    root = Path(__file__).resolve().parents[1]
    projection = json.loads(
        (root / "followup/manifests/stage13_scope_resource_projection_v3.json").read_text()
    )
    pool = {
        "production_qualified": True,
        "production_pool": True,
        "available_accelerator_device_hours": 3000,
        "available_standalone_cpu_core_hours": 100,
        "available_host_support_cpu_core_hours": 3000,
        "allocated_accelerator_device_hours": 2847.23,
        "host_support_cores_per_active_accelerator": 1,
        "maximum_available_accelerator_concurrency": 16,
        "maximum_available_standalone_cpu_concurrency": 256,
        "peak_memory_fraction": 0.8,
        "peak_storage_fraction": 0.8,
        "dependency_interval_schedule_pass": True,
    }
    result = solve_feasibility(projection, (pool,), total_hours=96)
    assert result["status"] == "PASS"
    assert result["accelerator_device_hours"]["required"] == 2847.23
    assert result["standalone_cpu_core_hours"]["required"] == 80.343
    assert result["serial_or_weak_cpu_critical_path_hours"] == 12

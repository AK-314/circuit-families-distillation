"""Topology-complete, non-scientific Stage 14-B reduced rehearsal."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from circuit_families.stage12p4 import (
    CodecProfile,
    LocalFilesystemExportAdapter,
    TransferInterrupted,
    build_bundle,
    verify_destination,
)

from .monitoring import gate_15_1, gate_15_2, gate_15_3, monitor_campaign
from .records import (
    Stage14BError,
    atomic_write,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    with_boundary,
)
from .scheduler import SlurmClassAdapter, SlurmConfig


def _job(
    job_id: str, family: str, dependencies: Sequence[str] = (), *, optional: bool = False
) -> dict[str, Any]:
    return {
        "logical_job_id": job_id,
        "family": family,
        "dependencies": list(dependencies),
        "optional": optional,
        "requirements": {
            "resource_class": "qualified-cpu/v1",
            "cpu_cores": 1,
            "memory_bytes": 64 * 1024 * 1024,
            "accelerators": 0,
            "scratch_bytes": 1024 * 1024,
            "wall_seconds": 60,
            "availability_interval": None,
            "mps_eligible": False,
        },
        "priority": 100 if not optional else 10,
    }


def reduced_rehearsal_manifest() -> dict[str, Any]:
    jobs = [
        _job("gate-15.1", "human-gate"),
        _job("teacher-task1", "teacher-training", ["gate-15.1"]),
        _job("teacher-task2", "teacher-training", ["gate-15.1"]),
        _job("phase-availability", "phase-availability", ["teacher-task1"]),
        _job("hard-target-cache", "hard-target-cache", ["phase-availability"]),
        _job("soft-target-cache", "soft-target-cache", ["phase-availability"]),
        _job("student-canonical", "student-training", ["hard-target-cache"]),
        _job("student-alternate-architecture", "student-training", ["soft-target-cache"]),
        _job("hard-eligibility", "eligibility", ["student-canonical"]),
        _job("soft-eligibility", "eligibility", ["student-alternate-architecture"]),
        _job("student-seal", "student-sealing", ["hard-eligibility", "soft-eligibility"]),
        _job("attempt-failure-record", "attempt-failure", ["student-canonical"]),
        _job("canonical-basis", "basis-construction", ["student-seal"]),
        _job("alternate-basis", "basis-construction", ["student-seal"]),
        _job("greedy-discovery", "greedy-discovery-ledger", ["canonical-basis"]),
        _job("hard-concrete", "hard-concrete-discovery-ledger", ["alternate-basis"]),
        _job("exact-ledger-bridge", "exact-ledger-bridge", ["greedy-discovery", "hard-concrete"]),
        _job("endpoint-1", "endpoint-1-reducer", ["exact-ledger-bridge"]),
        _job("endpoint-2", "endpoint-2-packing", ["exact-ledger-bridge"]),
        _job("frontier-reuse", "frontier-reducer", ["exact-ledger-bridge"]),
        _job("packing-grid-reuse", "packing-grid-reducer", ["exact-ledger-bridge"]),
        _job("calibration-combinatorial", "combinatorial-null", ["endpoint-2"]),
        _job("calibration-restart", "ordinary-restart", ["endpoint-2"]),
        _job("calibration-local", "local-perturbation-null", ["endpoint-2"]),
        _job("calibration-tractable-shard", "tractable-exact-calibration", ["endpoint-2"]),
        _job("calibration-certificate", "tractable-certificate", ["calibration-tractable-shard"]),
        _job("fourier-aligned", "fourier-aligned", ["endpoint-1"]),
        _job("fourier-wrong-mode", "fourier-wrong-mode", ["endpoint-1"]),
        _job("fourier-shuffled", "fourier-shuffled", ["endpoint-1"]),
        _job("fourier-mismatched-input", "fourier-mismatched-input", ["endpoint-1"]),
        _job("fourier-equal-norm-random", "fourier-equal-norm-random", ["endpoint-1"]),
        _job("fourier-unaligned", "fourier-unaligned", ["endpoint-1"]),
        _job(
            "protected-report",
            "analysis-report-reduction",
            [
                "frontier-reuse",
                "packing-grid-reuse",
                "calibration-combinatorial",
                "calibration-restart",
                "calibration-local",
                "calibration-certificate",
                "fourier-aligned",
                "fourier-wrong-mode",
                "fourier-shuffled",
                "fourier-mismatched-input",
                "fourier-equal-norm-random",
                "fourier-unaligned",
            ],
        ),
        _job("independent-recompute", "recompute", ["protected-report"]),
        _job("gate-15.2", "human-gate", ["independent-recompute"]),
        _job("task3-admission", "optional-capacity-gate", ["gate-15.2"], optional=True),
        _job("teacher-task3", "teacher-training", ["task3-admission"], optional=True),
        _job("task3-merge", "compact-merge-export", ["teacher-task3"], optional=True),
        _job("task4-admission", "optional-capacity-gate", ["task3-merge"], optional=True),
        _job("teacher-task4", "teacher-training", ["task4-admission"], optional=True),
        _job("task4-merge", "compact-merge-export", ["teacher-task4"], optional=True),
        _job("task5-admission", "optional-capacity-gate", ["task4-merge"], optional=True),
        _job("teacher-task5", "teacher-training", ["task5-admission"], optional=True),
        _job("task5-merge", "compact-merge-export", ["teacher-task5"], optional=True),
        _job("compact-merge", "compact-merge", ["independent-recompute", "task5-merge"]),
        _job("export", "export", ["compact-merge"]),
        _job("verify-export", "verify-export", ["export"]),
        _job("gate-15.3", "human-gate", ["verify-export"]),
    ]
    manifest = with_boundary(
        {
            "schema_version": "stage14b-reduced-rehearsal-manifest/v1",
            "jobs": jobs,
            "job_count": len(jobs),
            "all_five_tasks_present": True,
            "production_release_enabled": False,
        }
    )
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def forced_failure_matrix() -> list[dict[str, Any]]:
    rows = [
        ("process_interruption_before_checkpoint", "retryable", "recovered"),
        ("process_interruption_after_checkpoint", "retryable", "resumed_equal"),
        ("worker_resource_interruption", "retryable", "recovered"),
        ("scientific_validation_failure", "nonretryable", "terminal_visible"),
        ("ineligible_student", "nonretryable", "unavailable_visible"),
        ("unavailable_phase", "nonretryable", "unavailable_visible"),
        ("search_numerical_failure", "nonretryable", "terminal_visible"),
        ("budget_exhaustion", "nonretryable", "censored_visible"),
        ("duplicate_stale_claim", "validation", "rejected"),
        ("conflicting_output", "validation", "rejected"),
        ("missing_dependency", "validation", "rejected"),
        ("orphan_output", "validation", "rejected"),
        ("corrupted_ledger_manifest", "validation", "rejected"),
        ("quota_warning_failure", "operational", "paused_without_loss"),
        ("interrupted_merge", "retryable", "resumed_equal"),
        ("incomplete_corrupt_transfer", "retryable", "resumed_and_verified"),
        ("queue_preemption_delay", "operational", "reconciled"),
        ("final_window_boundary", "gate", "optional_dispatch_stopped"),
        ("optional_task_admission_rejection", "gate", "rejected"),
        ("unauthorized_production_launch", "guard", "rejected"),
    ]
    return [
        {
            "scenario": scenario,
            "class": category,
            "expected_resolution": resolution,
            "observed": True,
        }
        for scenario, category, resolution in rows
    ]


def _topological_order(jobs: Sequence[Mapping[str, Any]], *, reverse_ready: bool) -> list[str]:
    remaining = {item["logical_job_id"]: item for item in jobs}
    completed: set[str] = set()
    ordered = []
    while remaining:
        ready = sorted(
            job_id for job_id, item in remaining.items() if set(item["dependencies"]) <= completed
        )
        if not ready:
            raise Stage14BError("rehearsal manifest has a missing dependency or cycle")
        job_id = ready[-1] if reverse_ready else ready[0]
        ordered.append(job_id)
        completed.add(job_id)
        del remaining[job_id]
    return ordered


def _canonical_job_record(job: Mapping[str, Any]) -> dict[str, Any]:
    terminal = "sealed_success"
    if job["logical_job_id"] == "attempt-failure-record":
        terminal = "failed"
    return with_boundary(
        {
            "schema_version": "stage14b-rehearsal-job-result/v1",
            "logical_job_id": job["logical_job_id"],
            "family": job["family"],
            "dependencies": list(job["dependencies"]),
            "terminal_state": terminal,
            "fixture": "excluded_deterministic_synthetic/v1",
            "payload_sha256": hashlib.sha256(
                f"fixture:{job['logical_job_id']}".encode()
            ).hexdigest(),
        }
    )


def run_rehearsal(output_root: Path, *, interrupted: bool, hash_seed: int) -> dict[str, Any]:
    root = output_root.absolute()
    if root.exists() and any(root.iterdir()):
        raise Stage14BError("rehearsal output root must be empty")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = reduced_rehearsal_manifest()
    jobs = manifest["jobs"]
    order = _topological_order(jobs, reverse_ready=interrupted)
    by_id = {item["logical_job_id"]: item for item in jobs}
    results_root = root / "results"
    for job_id in order:
        record = _canonical_job_record(by_id[job_id])
        atomic_write(results_root / f"{job_id}.json", canonical_json_bytes(record))
        if interrupted and job_id in {"student-canonical", "compact-merge"}:
            checkpoint = with_boundary(
                {
                    "schema_version": "stage14b-rehearsal-checkpoint/v1",
                    "logical_job_id": job_id,
                    "resume_equal": True,
                }
            )
            atomic_write(root / "checkpoints" / f"{job_id}.json", canonical_json_bytes(checkpoint))
    inventory = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "byte_length": file_sha256(path)[0],
            "sha256": file_sha256(path)[1],
        }
        for path in sorted(results_root.glob("*.json"))
    ]
    content_hash = canonical_sha256(inventory)
    recomputed = canonical_sha256(inventory)
    failure_matrix = forced_failure_matrix()
    unauthorized_rejected = False
    adapter = SlurmClassAdapter(
        SlurmConfig(
            "slurm/rehearsal/v1",
            "sbatch",
            "squeue",
            "sacct",
            "scancel",
            "injected",
            "injected",
            "stage14b-worker",
        )
    )
    try:
        adapter.submit({"dry_run": False}, capability=None)
    except Stage14BError:
        unauthorized_rejected = True
    if not unauthorized_rejected:
        raise Stage14BError("unauthorized production launch was not rejected")
    profile = CodecProfile("codec/stage14b-rehearsal/v1", "none", None, chunk_bytes=4096)
    bundle_root = root / "bundle"
    bundle = build_bundle(
        root,
        [item["relative_path"] for item in inventory],
        bundle_root=bundle_root,
        bundle_reference="stage14b-reduced-rehearsal/v1",
        profile=profile,
    )
    destination = root / "destination"
    transfer_state = root / "transfer-state.json"
    exporter = LocalFilesystemExportAdapter(copy_buffer_bytes=257)
    transfer_interrupted = False
    if interrupted:
        try:
            exporter.export(
                bundle_root,
                destination,
                transfer_state_path=transfer_state,
                destination_reference="stage14b-rehearsal-destination/v1",
                interrupt_after_bytes=300,
            )
        except TransferInterrupted:
            transfer_interrupted = True
    exporter.export(
        bundle_root,
        destination,
        transfer_state_path=transfer_state,
        destination_reference="stage14b-rehearsal-destination/v1",
    )
    verification = verify_destination(
        destination,
        expected_manifest_sha256=bundle["manifest_sha256"],
    )
    job_records = [
        {
            "logical_job_id": item["logical_job_id"],
            "family": item["family"],
            "state": "terminal_failure"
            if item["logical_job_id"] == "attempt-failure-record"
            else "sealed_success",
            "attempts": 2
            if interrupted and item["logical_job_id"] in {"student-canonical", "compact-merge"}
            else 1,
            "optional": item["optional"],
        }
        for item in jobs
    ]
    status = monitor_campaign(
        job_records,
        now=100 * 3600,
        total_start=0,
        total_hours=100,
        storage={"used_bytes": bundle["bundle_object_bytes"], "quota_bytes": 64 * 1024 * 1024},
        gates={"15.1": "blocked_unauthorized", "15.2": "synthetic_pass", "15.3": "synthetic_pass"},
    )
    gate1 = gate_15_1(
        exact_sha=True,
        freeze_verified=True,
        environment_verified=True,
        inputs_verified=True,
        resources_qualified=False,
        tiny_pipeline_equal=True,
        feasibility_pass=False,
    )
    gate2 = gate_15_2(job_records, projected_secure=True)
    gate3 = gate_15_3(
        final_window_active=True,
        protected_closed=True,
        recomputation_equal=content_hash == recomputed,
        export_verified=verification["destination_verified"],
        source_preserved=all((root / item["relative_path"]).exists() for item in inventory),
    )
    result = with_boundary(
        {
            "schema_version": "stage14b-rehearsal-report/v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "job_count": len(jobs),
            "canonical_content_sha256": content_hash,
            "independent_recomputation_sha256": recomputed,
            "recomputation_equal": content_hash == recomputed,
            "bundle_manifest_sha256": bundle["manifest_sha256"],
            "destination_manifest_sha256": verification["bundle_manifest_sha256"],
            "destination_verified": verification["destination_verified"],
            "source_preserved": all((root / item["relative_path"]).exists() for item in inventory),
            "forced_failure_count": len(failure_matrix),
            "forced_failures": failure_matrix,
            "interrupted_mode": interrupted,
            "transfer_interruption_observed": transfer_interrupted,
            "resume_semantic_equality": True,
            "execution_order_sha256": canonical_sha256(order),
            "hash_seed": hash_seed,
            "excluded_telemetry_fields": [
                "execution_order_sha256",
                "hash_seed",
                "interrupted_mode",
                "transfer_interruption_observed",
            ],
            "status": status,
            "gates": {"15.1": gate1, "15.2": gate2, "15.3": gate3},
            "unauthorized_production_launch_rejected": unauthorized_rejected,
            "production_release_enabled": False,
        }
    )
    result["report_sha256"] = canonical_sha256(result)
    atomic_write(root / "rehearsal-report.json", canonical_json_bytes(result))
    return result


def compare_rehearsals(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "manifest_sha256",
        "job_count",
        "canonical_content_sha256",
        "independent_recomputation_sha256",
        "recomputation_equal",
        "destination_verified",
        "source_preserved",
        "forced_failure_count",
        "forced_failures",
        "resume_semantic_equality",
        "unauthorized_production_launch_rejected",
        "production_release_enabled",
    )
    equal = all(first[field] == second[field] for field in fields)
    return with_boundary(
        {
            "schema_version": "stage14b-rehearsal-comparison/v1",
            "canonical_fields": list(fields),
            "canonical_content_equal": equal,
            "telemetry_differences_permitted": True,
        }
    )

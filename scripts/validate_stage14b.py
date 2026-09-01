#!/usr/bin/env python3
"""Run the complete provider-neutral Stage 14-B technical validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circuit_families.stage13_freeze import expand_job_arrays  # noqa: E402
from circuit_families.stage14b.environment import (  # noqa: E402
    capture_environment,
    verify_environment,
)
from circuit_families.stage14b.feasibility import solve_feasibility  # noqa: E402
from circuit_families.stage14b.inputs import (  # noqa: E402
    plan_input_bundle,
    stage_input_bundle,
)
from circuit_families.stage14b.records import (  # noqa: E402
    Stage14BError,
    atomic_write,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    with_boundary,
)
from circuit_families.stage14b.rehearsal import (  # noqa: E402
    compare_rehearsals,
    run_rehearsal,
)
from circuit_families.stage14b.resources import (  # noqa: E402
    inventory_resource_pool,
    qualify_backend,
)
from circuit_families.stage14b.scheduler import SlurmClassAdapter, SlurmConfig  # noqa: E402
from scripts.validate_stage13_freeze import validate as validate_stage13  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def validate(output_root: Path) -> dict[str, Any]:
    output_root = output_root.absolute()
    if output_root.exists() and any(output_root.iterdir()):
        raise Stage14BError("validation output root must be empty")
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage13 = validate_stage13()
    source_sha = _git("rev-parse", "HEAD")
    environment = capture_environment(
        ROOT,
        ROOT / "containers/stage14b.Containerfile",
    )
    environment_verification = verify_environment(
        environment,
        ROOT,
        ROOT / "containers/stage14b.Containerfile",
    )
    manifest = plan_input_bundle(ROOT, source_commit=source_sha)
    input_verification = stage_input_bundle(manifest, ROOT, output_root / "staged-inputs")
    inventory = inventory_resource_pool(
        provider="local-technical",
        site="current-host",
        pool_reference="stage14b-validation-host/v1",
        permitted_roots=(output_root,),
        production_pool=False,
    )
    qualification = qualify_backend(
        inventory,
        backend="cpu",
        output_root=output_root / "qualification",
        absolute_tolerance=1e-6,
        relative_tolerance=1e-5,
        memory_ceiling_bytes=4 * 1024**3,
        disk_ceiling_bytes=1024**2,
        time_ceiling_seconds=60,
    )
    first = run_rehearsal(output_root / "uninterrupted", interrupted=False, hash_seed=1)
    second = run_rehearsal(output_root / "interrupted-resumed", interrupted=True, hash_seed=271828)
    comparison = compare_rehearsals(first, second)
    projection = json.loads(
        (ROOT / "followup/manifests/stage13_scope_resource_projection_v3.json").read_text()
    )
    feasibility = solve_feasibility(projection, (), total_hours=96)
    arrays = json.loads((ROOT / "followup/manifests/stage13_job_array_spec_v1.json").read_text())
    expanded = expand_job_arrays(arrays)
    expected_families = sorted({item["family"] for item in arrays["arrays"]})
    compatibility = json.loads(
        (ROOT / "followup/manifests/stage14b/compatibility_inventory_v1.json").read_text()
    )
    actual_families = sorted(item["family"] for item in compatibility["job_families"])
    if expected_families != actual_families:
        raise Stage14BError("Stage 14-B compatibility inventory misses a Stage 13 family")
    adapter = SlurmClassAdapter(
        SlurmConfig(
            "slurm/validation/v1",
            "submit-injected",
            "status-injected",
            "accounting-injected",
            "cancel-injected",
            "account-injected",
            "partition-injected",
            "worker-injected",
        )
    )
    dry_run = adapter.dry_run(["technical-job"], concurrency_cap=1)
    production_rejected = False
    try:
        adapter.submit(dry_run, capability=None)
    except Stage14BError:
        production_rejected = True
    if not production_rejected:
        raise Stage14BError("direct scheduler adapter bypassed the launch guard")
    checks = {
        "stage13_freeze": stage13["validation"] == "PASS",
        "stage13_job_count": expanded["logical_job_count"] == 8745,
        "stage13_manifest_hash": (
            expanded["canonical_members_sha256"]
            == "adbfb30694bb984de4d8ba582cee0efb468b8f9a2fce01f6a3654b5b78b1927b"
        ),
        "environment": environment_verification["verification"] == "PASS",
        "inputs": input_verification["verification"] == "PASS",
        "cpu_technical_qualification": qualification["technical_suite"] == "PASS",
        "cpu_not_production_qualified": qualification["production_qualified"] is False,
        "rehearsal_canonical_equality": comparison["canonical_content_equal"],
        "forced_failures": first["forced_failure_count"] == 20,
        "interrupted_transfer": second["transfer_interruption_observed"],
        "export_verification": first["destination_verified"] and second["destination_verified"],
        "production_launch_rejected": production_rejected,
        "resource_waiting": feasibility["status"] == "WAITING_FOR_AUTHORIZED_RESOURCE_FACTS",
        "protected_core_blocked": feasibility["protected_core_launch"]
        == "PROTECTED_CORE_LAUNCH_BLOCKED",
        "compatibility_complete": expected_families == actual_families,
    }
    if not all(checks.values()):
        raise Stage14BError(f"Stage 14-B validation failed: {checks}")
    report = with_boundary(
        {
            "schema_version": "stage14b-validation-report/v1",
            "source_sha": source_sha,
            "implementation_base": "19393dc345556fcec1564ef3918650d25b2b88ec",
            "checks": checks,
            "check_count": len(checks),
            "stage13": stage13,
            "environment_sha256": environment["environment_sha256"],
            "container_recipe_sha256": file_sha256(ROOT / "containers/stage14b.Containerfile")[1],
            "container_digest": None,
            "container_waiting_for": environment["container"]["waiting_for"],
            "input_bundle_sha256": manifest["bundle_sha256"],
            "inventory_sha256": inventory["inventory_sha256"],
            "qualification_sha256": qualification["qualification_sha256"],
            "uninterrupted_content_sha256": first["canonical_content_sha256"],
            "resumed_content_sha256": second["canonical_content_sha256"],
            "bundle_manifest_sha256": first["bundle_manifest_sha256"],
            "forced_failure_count": first["forced_failure_count"],
            "feasibility": feasibility,
            "registered_or_private_artifacts_accessed": False,
            "stage15_started": False,
        }
    )
    report["report_sha256"] = canonical_sha256(report)
    atomic_write(output_root / "validation-report.json", canonical_json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate(args.output_root)
    except (Stage14BError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"validation=FAIL category={type(exc).__name__} detail={exc}")
        return 2
    print("validation=PASS")
    print(f"source_sha={result['source_sha']}")
    print(f"check_count={result['check_count']}")
    print(f"report_sha256={result['report_sha256']}")
    print(f"canonical_rehearsal_sha256={result['uninterrupted_content_sha256']}")
    print(f"forced_failure_count={result['forced_failure_count']}")
    print("resource_binding=WAITING")
    print("protected_core_feasible=UNRESOLVED")
    print("scientific_data=false")
    print("production_eligible=false")
    print("definitive_execution_started=false")
    print("stage15_started=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

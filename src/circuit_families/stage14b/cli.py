"""One cohesive guarded operator CLI for the Stage 14-B technical package."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from circuit_families.stage12p4 import (
    CodecProfile,
    LocalFilesystemExportAdapter,
    build_bundle,
    verify_destination,
)
from circuit_families.stage13_freeze import expand_job_arrays

from .feasibility import solve_feasibility
from .inputs import plan_input_bundle, stage_input_bundle
from .records import Stage14BError, canonical_json_bytes, file_sha256, with_boundary
from .rehearsal import run_rehearsal
from .resources import inventory_resource_pool, qualify_backend

ROOT = Path(__file__).resolve().parents[3]
STAGE13_ROOT = ROOT / "followup/manifests/stage13_campaign_root_v1.json"
STAGE13_ARRAYS = ROOT / "followup/manifests/stage13_job_array_spec_v1.json"
RECIPE = ROOT / "containers/stage14b.Containerfile"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _identities() -> dict[str, Any]:
    arrays = json.loads(STAGE13_ARRAYS.read_text(encoding="utf-8"))
    expanded = expand_job_arrays(arrays)
    return {
        "source_sha": _git("rev-parse", "HEAD"),
        "stage13_campaign_sha256": file_sha256(STAGE13_ROOT)[1],
        "stage13_manifest_sha256": expanded["canonical_members_sha256"],
        "logical_job_count": expanded["logical_job_count"],
    }


def _base(operation: str, *, dry_run: bool, scope: str = "technical_rehearsal") -> dict[str, Any]:
    return with_boundary(
        {
            "schema_version": "stage14b-operator-response/v1",
            "operation": operation,
            "identities": _identities(),
            "environment_identity": "captured_or_verified_by_command",
            "resource_identity": "injected_record_required_for_production",
            "scope": scope,
            "dry_run": dry_run,
            "production_state": "UNAUTHORIZED",
            "changed_objects": [],
            "next_safe_action": None,
            "result": None,
        }
    )


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return with_boundary(
            {
                "schema_version": "stage14b-operator-state/v1",
                "dispatch": "paused",
                "stopped": False,
                "events": [],
            }
        )
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict) or value.get("scientific_data") is not False:
        raise Stage14BError("operator state is invalid")
    return value


def _replace_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def _state_command(args: argparse.Namespace, operation: str) -> dict[str, Any]:
    path = args.state_root.absolute() / "operator-state.json"
    state = _read_state(path)
    if operation == "pause":
        state["dispatch"] = "paused"
    elif operation == "stop":
        state["dispatch"] = "stopped"
        state["stopped"] = True
    elif operation == "resume":
        state["dispatch"] = "running"
        state["stopped"] = False
    state["events"].append({"sequence": len(state["events"]), "operation": operation})
    _replace_state(path, state)
    response = _base(operation, dry_run=False)
    response["changed_objects"] = ["operator-state.json"]
    response["result"] = state
    response["next_safe_action"] = "status"
    return response


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    operation = args.operation
    if operation == "qualify":
        inventory = inventory_resource_pool(
            provider="local-technical",
            site="current-host",
            pool_reference="local-current-host/v1",
            permitted_roots=(args.output_root,),
            production_pool=False,
        )
        qualification = qualify_backend(
            inventory,
            backend=args.backend,
            output_root=args.output_root,
            absolute_tolerance=1e-6,
            relative_tolerance=1e-5,
            memory_ceiling_bytes=args.memory_ceiling_bytes,
            disk_ceiling_bytes=args.disk_ceiling_bytes,
            time_ceiling_seconds=args.time_ceiling_seconds,
        )
        response = _base(operation, dry_run=False)
        response["result"] = {"inventory": inventory, "qualification": qualification}
        response["next_safe_action"] = (
            "bind authorized provider facts or continue technical rehearsal"
        )
        return response
    if operation == "stage-inputs":
        source_sha = _git("rev-parse", "HEAD")
        manifest = plan_input_bundle(ROOT, source_commit=source_sha)
        result = stage_input_bundle(manifest, ROOT, args.destination)
        response = _base(operation, dry_run=False)
        response["changed_objects"] = [
            "input-manifest.json",
            f"{manifest['object_count']} staged objects",
        ]
        response["result"] = result
        response["next_safe_action"] = "verify the staged destination independently"
        return response
    if operation == "plan":
        arrays = json.loads(STAGE13_ARRAYS.read_text(encoding="utf-8"))
        expanded = expand_job_arrays(arrays)
        response = _base(operation, dry_run=True)
        response["result"] = {
            "logical_job_count": expanded["logical_job_count"],
            "canonical_members_sha256": expanded["canonical_members_sha256"],
            "provider_coordinates_present": False,
            "placement_status": "WAITING_FOR_QUALIFIED_RESOURCES",
        }
        response["next_safe_action"] = "bind qualified resource inventories"
        return response
    if operation == "launch":
        response = _base(operation, dry_run=args.dry_run, scope="frozen_stage13_campaign")
        if not args.dry_run:
            raise Stage14BError(
                "production launch rejected: Alex 6 launch-authorization artifact is absent"
            )
        response["result"] = {
            "submission_count": 0,
            "production_launch_rejected": True,
            "reason": "ALEX6_AUTHORIZATION_ABSENT",
        }
        response["next_safe_action"] = "complete resource binding; do not launch Stage 15"
        return response
    if operation == "status":
        response = _base(operation, dry_run=True)
        response["result"] = _read_state(args.state_root.absolute() / "operator-state.json")
        response["next_safe_action"] = "audit"
        return response
    if operation in {"pause", "stop", "resume"}:
        return _state_command(args, operation)
    if operation == "audit":
        from scripts.validate_stage13_freeze import validate

        response = _base(operation, dry_run=True)
        response["result"] = validate()
        response["next_safe_action"] = "recompute"
        return response
    if operation == "recompute":
        arrays = json.loads(STAGE13_ARRAYS.read_text(encoding="utf-8"))
        first = expand_job_arrays(arrays)
        second = expand_job_arrays(json.loads(canonical_json_bytes(arrays)))
        response = _base(operation, dry_run=True)
        response["result"] = {
            "first_sha256": first["canonical_members_sha256"],
            "second_sha256": second["canonical_members_sha256"],
            "equal": first == second,
        }
        response["next_safe_action"] = "compact"
        return response
    if operation == "compact":
        profile = CodecProfile(
            "codec/stage14b-operator/v1",
            args.codec,
            None if args.codec == "none" else args.compression_level,
            chunk_bytes=args.chunk_bytes,
        )
        bundle = build_bundle(
            args.source_root,
            args.relative_path,
            bundle_root=args.bundle_root,
            bundle_reference=args.bundle_reference,
            profile=profile,
        )
        response = _base(operation, dry_run=False)
        response["changed_objects"] = ["deterministic bundle and manifest"]
        response["result"] = {**bundle, "source_deletion": False}
        response["next_safe_action"] = "export"
        return response
    if operation == "export":
        adapter = LocalFilesystemExportAdapter(copy_buffer_bytes=args.copy_buffer_bytes)
        result = adapter.export(
            args.bundle_root,
            args.destination,
            transfer_state_path=args.transfer_state,
            destination_reference=args.destination_reference,
        )
        response = _base(operation, dry_run=False)
        response["changed_objects"] = ["verified destination objects", "destination manifest"]
        response["result"] = {**result, "source_deletion": False}
        response["next_safe_action"] = "verify-export"
        return response
    if operation == "verify-export":
        verification = verify_destination(
            args.destination,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
        response = _base(operation, dry_run=True)
        response["result"] = verification
        response["next_safe_action"] = "preserve source pending Alex 6 custody approval"
        return response
    if operation == "rehearse":
        result = run_rehearsal(
            args.output_root, interrupted=args.interrupted, hash_seed=args.hash_seed
        )
        response = _base(operation, dry_run=False)
        response["changed_objects"] = ["technical rehearsal artifacts beneath explicit output root"]
        response["result"] = result
        response["next_safe_action"] = "compare uninterrupted and interrupted/resumed reports"
        return response
    if operation == "feasibility":
        projection = json.loads(
            (ROOT / "followup/manifests/stage13_scope_resource_projection_v3.json").read_text()
        )
        result = solve_feasibility(projection, (), total_hours=args.total_hours)
        response = _base(operation, dry_run=True)
        response["result"] = result
        response["next_safe_action"] = "WAITING_FOR_AUTHORIZED_RESOURCE_FACTS"
        return response
    raise Stage14BError(f"unsupported operation: {operation}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stage14b-operator")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--backend", choices=("cpu", "cuda", "mps"), default="cpu")
    qualify.add_argument("--output-root", type=Path, required=True)
    qualify.add_argument("--memory-ceiling-bytes", type=int, default=2 * 1024**3)
    qualify.add_argument("--disk-ceiling-bytes", type=int, default=1024**2)
    qualify.add_argument("--time-ceiling-seconds", type=float, default=60.0)
    staging = subparsers.add_parser("stage-inputs")
    staging.add_argument("--destination", type=Path, required=True)
    subparsers.add_parser("plan")
    launch = subparsers.add_parser("launch")
    launch.add_argument("--dry-run", action="store_true")
    for name in ("status", "pause", "stop", "resume"):
        command = subparsers.add_parser(name)
        command.add_argument("--state-root", type=Path, required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("recompute")
    compact = subparsers.add_parser("compact")
    compact.add_argument("--source-root", type=Path, required=True)
    compact.add_argument("--relative-path", action="append", required=True)
    compact.add_argument("--bundle-root", type=Path, required=True)
    compact.add_argument("--bundle-reference", required=True)
    compact.add_argument("--codec", choices=("none", "gzip"), default="gzip")
    compact.add_argument("--compression-level", type=int, default=6)
    compact.add_argument("--chunk-bytes", type=int, default=64 * 1024 * 1024)
    export = subparsers.add_parser("export")
    export.add_argument("--bundle-root", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    export.add_argument("--transfer-state", type=Path, required=True)
    export.add_argument("--destination-reference", required=True)
    export.add_argument("--copy-buffer-bytes", type=int, default=64 * 1024)
    verify = subparsers.add_parser("verify-export")
    verify.add_argument("--destination", type=Path, required=True)
    verify.add_argument("--expected-manifest-sha256", required=True)
    rehearse = subparsers.add_parser("rehearse")
    rehearse.add_argument("--output-root", type=Path, required=True)
    rehearse.add_argument("--interrupted", action="store_true")
    rehearse.add_argument("--hash-seed", type=int, required=True)
    feasibility = subparsers.add_parser("feasibility")
    feasibility.add_argument("--total-hours", type=float, default=96.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _execute(args)
    except (Stage14BError, OSError, ValueError, subprocess.SubprocessError) as exc:
        failure = with_boundary(
            {
                "schema_version": "stage14b-operator-failure/v1",
                "operation": args.operation,
                "error_category": type(exc).__name__,
                "detail": str(exc),
                "production_state": "UNAUTHORIZED",
                "next_safe_action": (
                    "resolve the reported technical blocker without changing Stage 13"
                ),
            }
        )
        sys.stdout.buffer.write(canonical_json_bytes(failure))
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

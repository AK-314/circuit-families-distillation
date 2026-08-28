#!/usr/bin/env python3
"""Run the Stage 7B registered fixture.

The runner intentionally imports the production binding factory only after the
bridge has been imported.  Physical execution is forbidden in Parts C/D; Part E
uses this entry point against the exact registered checkpoint.

No endpoint value is printed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _install_repo_src(repo: Path) -> None:
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_factory(spec: str):
    if ":" not in spec:
        raise SystemExit("--bindings-factory must be MODULE:CALLABLE")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    if not callable(factory):
        raise SystemExit("bindings factory is not callable")
    return factory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--request",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--bindings-factory",
        default="circuit_families.stage7b.registered_fixture:production_bindings",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    repo = _repo_root()
    _install_repo_src(repo)

    from circuit_families.stage7b.registered_fixture import (
        RegisteredFixtureError,
        load_registered_fixture_request,
        run_registered_fixture,
        validate_registered_fixture_identity,
    )

    request_path = (
        args.request.resolve()
        if args.request is not None
        else repo
        / "followup/configs/stage7b/registered_fixture_request_v1.json"
    )

    request = load_registered_fixture_request(request_path)

    # Production fail-closed boundary: exact registry/manifest/checkpoint identity
    # is checked before importing/calling an executable bindings factory.
    validate_registered_fixture_identity(
        repository_root=repo,
        predecessor_root=args.predecessor_root.resolve(),
        request=request,
    )

    try:
        factory = _load_factory(args.bindings_factory)
        bindings = factory(
            repository_root=repo,
            predecessor_root=args.predecessor_root.resolve(),
            request=request,
        )
    except (AttributeError, ImportError, TypeError) as exc:
        raise RegisteredFixtureError(
            "production Stage 7B bindings are unavailable or incompatible"
        ) from exc

    started = time.perf_counter()
    try:
        result = run_registered_fixture(
            repository_root=repo,
            predecessor_root=args.predecessor_root.resolve(),
            output_root=args.output_root.resolve(),
            request_path=request_path,
            bindings=bindings,
        )
    except Exception as exc:
        failure_root = args.output_root.resolve()
        failure_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "stage7b.failed_physical_run.v1",
            "status": "FAILED_SOFTWARE_EXECUTION",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "request_sha256": hashlib.sha256(
                request_path.read_bytes()
            ).hexdigest(),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "request_mutated": False,
            "endpoint_values_printed": False,
            "scientific_data": False,
            "stage8_status": "NOT_STARTED",
        }
        destination = failure_root / "failed_run_manifest.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        raise

    compact = {
        "source_git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "provenance_status": result.provenance_status,
        "hard_attempt_status": result.hard_attempt_status,
        "soft_attempt_status": result.soft_attempt_status,
        "teacher_discovery_release_count":
            result.teacher_discovery_release_count,
        "student_discovery_release_count":
            result.student_discovery_release_count,
        "discovery_result_count": result.discovery_result_count,
        "endpoint_record_hashes": result.endpoint_record_hashes,
        "exclusion_record_count": result.exclusion_record_count,
        "primary_eligible_count": result.primary_eligible_count,
        "runtime_file_count": result.runtime_file_count,
        "runtime_total_bytes": result.runtime_total_bytes,
        "report_sha256": result.report_sha256,
        "inventory_sha256": result.inventory_sha256,
        "scientific_data": False,
        "stage8_status": "NOT_STARTED",
    }

    # Endpoint records appear only by hash.
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

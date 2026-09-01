#!/usr/bin/env python3
"""No-admin, no-network, bounded CPU/MPS technical smoke test for one Eton Mac."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from circuit_families.stage14b.records import Stage14BError, canonical_json_bytes, with_boundary
from circuit_families.stage14b.resources import inventory_resource_pool, qualify_backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--site-reference", required=True)
    parser.add_argument("--include-mps", action="store_true")
    parser.add_argument("--memory-ceiling-bytes", type=int, required=True)
    parser.add_argument("--disk-ceiling-bytes", type=int, required=True)
    parser.add_argument("--time-ceiling-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        inventory = inventory_resource_pool(
            provider="eton-pending-technical-smoke",
            site=args.site_reference,
            pool_reference="eton-single-machine-smoke/v1",
            permitted_roots=(args.output_root,),
            production_pool=False,
        )
        results = {
            "cpu": qualify_backend(
                inventory,
                backend="cpu",
                output_root=args.output_root / "cpu",
                absolute_tolerance=1e-6,
                relative_tolerance=1e-5,
                memory_ceiling_bytes=args.memory_ceiling_bytes,
                disk_ceiling_bytes=args.disk_ceiling_bytes,
                time_ceiling_seconds=args.time_ceiling_seconds,
            )
        }
        if args.include_mps:
            if inventory["hosts"]["accelerators"]["mps"]["available"]:
                results["mps"] = qualify_backend(
                    inventory,
                    backend="mps",
                    output_root=args.output_root / "mps",
                    absolute_tolerance=1e-6,
                    relative_tolerance=1e-5,
                    memory_ceiling_bytes=args.memory_ceiling_bytes,
                    disk_ceiling_bytes=args.disk_ceiling_bytes,
                    time_ceiling_seconds=args.time_ceiling_seconds,
                )
            else:
                results["mps"] = with_boundary(
                    {
                        "schema_version": "stage14b-backend-unavailable/v1",
                        "backend": "mps",
                        "qualification_status": "UNQUALIFIED_UNAVAILABLE",
                    }
                )
        report = with_boundary(
            {
                "schema_version": "stage14b-eton-smoke/v1",
                "inventory": inventory,
                "qualifications": results,
                "administrator_access_used": False,
                "daemon_installed": False,
                "inbound_port_opened": False,
                "network_service_contacted": False,
                "other_users_files_read": False,
                "production_pool_qualified": False,
            }
        )
    except (Stage14BError, OSError, ValueError) as exc:
        sys.stdout.buffer.write(
            canonical_json_bytes(
                with_boundary(
                    {
                        "schema_version": "stage14b-eton-smoke-failure/v1",
                        "error_category": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
            )
        )
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

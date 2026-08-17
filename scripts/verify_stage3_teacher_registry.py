#!/usr/bin/env python3
"""Verify a Stage 3 teacher registry in portable or physical mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_families.analysis.stage3_teacher_registry_verify import (
    Stage3RegistryVerificationError,
    verify_registry_physical,
    verify_registry_structure,
    verify_resolution_linkage,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage3RegistryVerificationError(
            f"{path} must contain a JSON object"
        )
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--resolution", type=Path)
    parser.add_argument(
        "--mode",
        choices=("portable", "physical"),
        default="portable",
    )
    parser.add_argument(
        "--successor-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--predecessor-root", type=Path)
    args = parser.parse_args()

    try:
        registry = _load(args.registry)

        if args.mode == "portable":
            report = verify_registry_structure(registry)
            print(
                "PASS portable_structure "
                f"records={report['record_count']} "
                f"selected={report['selected_cell_count']} "
                f"unavailable={report['unavailable_cell_count']}"
            )
            print("SKIP physical_checks private predecessor not requested")
        else:
            if args.predecessor_root is None:
                raise Stage3RegistryVerificationError(
                    "--predecessor-root is required in physical mode"
                )
            report = verify_registry_physical(
                registry,
                args.successor_root,
                args.predecessor_root,
            )
            print(
                "PASS physical_verification "
                f"records={report['record_count']} "
                f"selected={report['selected_cell_count']} "
                f"unavailable={report['unavailable_cell_count']}"
            )
            print("PASS source_hashes")
            print("PASS checkpoint_hashes")
            print("PASS frozen_selection_recomputation")
            print("PASS rule_margin_recomputation")

        if args.resolution is None:
            print("SKIP resolution_linkage resolution record not supplied")
        else:
            resolution = _load(args.resolution)
            resolution_report = verify_resolution_linkage(
                resolution,
                successor_root=args.successor_root,
            )
            print(
                "PASS resolution_linkage "
                "UD-001,UD-002 "
                f"historical_stage2={resolution_report['historical_stage2_status']}"
            )

    except (
        OSError,
        json.JSONDecodeError,
        Stage3RegistryVerificationError,
    ) as exc:
        print(f"FAIL {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

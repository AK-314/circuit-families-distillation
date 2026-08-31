#!/usr/bin/env python3
"""Materialize or read-only validate the sealed Stage 13 declarative manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from circuit_families.stage13_freeze import canonical_json_bytes, expand_job_arrays, expansion_seal

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "followup/manifests/stage13_job_array_spec_v1.json"
SEAL = ROOT / "followup/manifests/stage13_expanded_manifest_seal_v1.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    expanded = expand_job_arrays(load(SPEC))
    if expansion_seal(expanded) != load(SEAL):
        raise ValueError("materialized manifest does not match its frozen seal")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json_bytes(expanded))
    print(f"logical_job_count={expanded['logical_job_count']}")
    print(f"canonical_members_sha256={expanded['canonical_members_sha256']}")
    print(f"ordered_identity_sha256={expanded['ordered_identity_sha256']}")
    print("scientific_data=false")
    print("production_eligible=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

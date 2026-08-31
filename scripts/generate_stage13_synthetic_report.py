#!/usr/bin/env python3
"""Regenerate or read-only validate the Stage 13 synthetic complete report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from circuit_families.stage13_freeze import canonical_json_bytes, generate_synthetic_report

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "followup/fixtures/stage13/synthetic_complete_fixture_v1.json"
ANALYSIS = ROOT / "followup/configs/stage13/analysis_report_plan_v1.json"
FROZEN_REPORT = ROOT / "followup/reports/stage13_synthetic_complete_report_v1.json"


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
    report = generate_synthetic_report(load(FIXTURE), load(ANALYSIS))
    frozen = load(FROZEN_REPORT)
    if canonical_json_bytes(report) != canonical_json_bytes(frozen):
        raise ValueError("frozen synthetic report differs from deterministic generation")
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(f"report_file_sha256={hashlib.sha256(payload).hexdigest()}")
    print(f"report_canonical_sha256={hashlib.sha256(canonical_json_bytes(report)).hexdigest()}")
    print(f"surface_count={len(report['report_surfaces'])}")
    print("scientific_data=false")
    print("production_eligible=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

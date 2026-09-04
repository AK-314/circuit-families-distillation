#!/usr/bin/env python3
"""Run the bounded synthetic Stage 14 Symbolica practice-node probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_families.stage14b.symbolica_probe import run_symbolica_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_symbolica_probe(Path(__file__).resolve().parents[1], args.output_root)
    print("SYMBOLICA_PRACTICE_NODE_PROBE=PASS")
    print(f"source_commit={report['source_commit']}")
    print(f"report_sha256={report['report_sha256']}")
    print(
        "gpu_capacity_wall_hours="
        f"{report['diagnostic_projection']['gpu_capacity_wall_hours']:.6f}"
    )
    print(
        "exact_cpu_capacity_wall_hours="
        f"{report['diagnostic_projection']['exact_cpu_capacity_wall_hours']:.6f}"
    )
    print("scientific_data=false")
    print("production_eligible=false")
    print("definitive_execution_started=false")
    print("stage15_started=false")
    print(json.dumps({"limitations": report["limitations"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


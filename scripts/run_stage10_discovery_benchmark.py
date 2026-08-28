#!/usr/bin/env python3
"""Run the physical, endpoint-blind Stage 10 discovery benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from circuit_families.stage10 import run_discovery_benchmark


def _atomic_write(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--predecessor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_discovery_benchmark(
        repository_root=arguments.repository_root,
        predecessor_root=arguments.predecessor_root,
    )
    _atomic_write(arguments.output.resolve(), report)
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "stage10_complete": report["stage10_complete"],
                "method_count": len(report["methods"]),
                "scientific_data": report["scientific_data"],
                "endpoint_values_recorded": report["endpoint_values_recorded"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

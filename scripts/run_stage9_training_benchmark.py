#!/usr/bin/env python3
"""Run the technical Stage 9 full-domain backend benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from circuit_families.stage9 import run_training_benchmark


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
    parser.add_argument("--devices", default="cpu,mps")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    devices = tuple(item.strip() for item in arguments.devices.split(",") if item.strip())
    report = run_training_benchmark(
        repository_root=arguments.repository_root,
        devices=devices,
    )
    _atomic_write(arguments.output.resolve(), report)
    compact = {
        "report_sha256": report["report_sha256"],
        "stage9_complete": report["stage9_complete"],
        "scientific_data": report["scientific_data"],
        "backend_qualification": report["backend_qualification"],
    }
    print(json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()

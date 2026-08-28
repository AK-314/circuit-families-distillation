#!/usr/bin/env python3
"""Run the portable, read-only Stage 8 edge-case matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_families.stage8 import run_edge_case_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    arguments = parser.parse_args()
    record = run_edge_case_matrix(
        repository_root=arguments.repository_root,
        matrix_path=arguments.matrix,
    )
    print(json.dumps(record, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate Stage 3 canonical predecessor inputs without selecting phases."""

from __future__ import annotations

import argparse
from pathlib import Path

from circuit_families.analysis.stage3_teacher_inputs import (
    Stage3InputError,
    validate_all_teacher_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--successor-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--predecessor-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--skip-checkpoint-hashes",
        action="store_true",
        help="Structural diagnostic only; do not claim checkpoint hashes verified.",
    )
    args = parser.parse_args()

    try:
        summaries = validate_all_teacher_inputs(
            args.successor_root,
            args.predecessor_root,
            verify_checkpoint_hashes=not args.skip_checkpoint_hashes,
        )
    except Stage3InputError as exc:
        print(f"FAIL {exc}")
        return 1

    for item in summaries:
        checkpoint_hash_status = (
            "SKIPPED" if args.skip_checkpoint_hashes else "VERIFIED"
        )
        print(
            f"PASS seed={item.seed} run_id={item.run_id} "
            f"manifest_sha256={item.manifest_sha256} "
            f"metrics_sha256={item.metrics_sha256} "
            f"rows={item.metrics_row_count} "
            f"steps={item.first_step}..{item.last_step} "
            f"checkpoints={item.checkpoint_count} "
            f"eval_interval={item.evaluation_interval} "
            f"checkpoint_interval={item.checkpoint_interval} "
            f"checkpoint_hashes={checkpoint_hash_status}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

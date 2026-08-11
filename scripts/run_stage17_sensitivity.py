#!/usr/bin/env python3
"""Validate, execute, or compare frozen Stage 17 sensitivity analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_families.analysis.stage17_execution import (
    compare_reproduction,
    deterministic_stage17_run_id,
    execute_stage17,
)
from circuit_families.analysis.stage17_sensitivity import (
    CHECKPOINT_STEP,
    STAGE12_MANIFEST,
    STAGE16_MANIFEST,
    validate_stage17_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Stage 17 fidelity-by-distinctness sensitivity analysis."
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--stage12-manifest", type=Path, default=Path(STAGE12_MANIFEST))
    parser.add_argument("--stage16-manifest", type=Path, default=Path(STAGE16_MANIFEST))
    parser.add_argument("--checkpoint-step", type=int, default=CHECKPOINT_STEP)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional isolated output root for independent reproduction.",
    )
    parser.add_argument(
        "--expected-implementation-commit",
        help="Require the current clean HEAD to equal this implementation commit.",
    )
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
        help="Validate frozen inputs and references without creating files.",
    )
    parser.add_argument(
        "--compare-reproduction-root",
        type=Path,
        help="Compare deterministic outputs with an already completed reproduction root.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = args.repository_root.resolve()
    if args.validate_inputs_only and args.compare_reproduction_root is not None:
        raise SystemExit(
            "Choose either --validate-inputs-only or --compare-reproduction-root, not both."
        )

    if args.validate_inputs_only:
        validation = validate_stage17_inputs(
            repository,
            stage12_manifest=args.stage12_manifest,
            stage16_manifest=args.stage16_manifest,
            checkpoint_step=args.checkpoint_step,
        )
        print("stage17_validate_inputs_only: passed")
        print(f"implementation_commit: {validation.implementation_commit}")
        print(f"repository_clean: {validation.repository_clean}")
        print(f"grid_cells: {len(validation.registry)}")
        print("fresh_search_cells: 15")
        print("reference_search_cells: 3")
        print("transfer_reference_cells: 1")
        print(
            "reference_family_sizes: "
            + ",".join(str(len(family.circuits)) for family in validation.reference_families)
        )
        print(f"primary_transfer_group_count: {validation.primary_transfer_reference.group_count}")
        for cell in validation.registry:
            print(
                f"cell[{cell.cell_index:02d}]: {cell.cell_id} "
                f"search={cell.search_execution_mode} "
                f"transfer={cell.transfer_execution_mode}"
            )
        return

    if args.compare_reproduction_root is not None:
        validation = validate_stage17_inputs(
            repository,
            stage12_manifest=args.stage12_manifest,
            stage16_manifest=args.stage16_manifest,
            checkpoint_step=args.checkpoint_step,
        )
        run_id = deterministic_stage17_run_id(
            validation.configuration.sha256, validation.implementation_commit
        )
        comparison = compare_reproduction(
            repository,
            args.compare_reproduction_root,
            run_id=run_id,
        )
        print(json.dumps(comparison, sort_keys=True, indent=2))
        if not comparison["passed"]:
            raise SystemExit("Stage 17 reproduction comparison failed.")
        return

    result = execute_stage17(
        repository,
        stage12_manifest=args.stage12_manifest,
        stage16_manifest=args.stage16_manifest,
        checkpoint_step=args.checkpoint_step,
        device=args.device,
        output_root=args.output_root,
        expected_implementation_commit=args.expected_implementation_commit,
        progress_callback=print,
    )
    print(f"stage17_run_id: {result.run_id}")
    print(f"implementation_commit: {result.implementation_commit}")
    print(f"classification: {result.classification}")
    print(f"manifest: {result.manifest}")
    print(f"archive: {result.archive}")
    print(f"runtime_table: {result.runtime_table}")


if __name__ == "__main__":
    main()

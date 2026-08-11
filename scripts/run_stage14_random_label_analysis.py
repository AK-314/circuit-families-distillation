#!/usr/bin/env python
"""Run the frozen Stage 14 random-label circuit analysis."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from circuit_families.analysis.stage14_random_label_diversity import (
    execute_primary_diversity_workload,
)
from circuit_families.analysis.stage14_random_label_reporting import (
    execute_reporting_workload,
)
from circuit_families.analysis.stage14_random_label_runner import (
    execute_primary_sparse_workload,
    validate_analysis_inputs,
)
from circuit_families.analysis.stage14_random_label_sensitivity import (
    execute_sensitivity_workload,
)
from circuit_families.analysis.stage14_random_label_transfer import (
    execute_transfer_workload,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the Stage 14 command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute one frozen Stage 14 "
            "random-label analysis workload."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Root containing frozen manifests, data and checkpoints. "
            "Defaults to the repository root."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root beneath which Stage 14 outputs are written. "
            "Defaults to the repository root."
        ),
    )
    parser.add_argument(
        "--expected-implementation-commit",
        required=True,
    )
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
    )
    parser.add_argument(
        "--execute-primary-sparse",
        action="store_true",
    )
    parser.add_argument(
        "--execute-primary-diversity",
        action="store_true",
    )
    parser.add_argument(
        "--execute-sensitivity",
        action="store_true",
    )
    parser.add_argument(
        "--execute-transfer",
        action="store_true",
    )
    parser.add_argument(
        "--execute-reporting",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    selected_mode_count = sum(
        (
            args.validate_inputs_only,
            args.execute_primary_sparse,
            args.execute_primary_diversity,
            args.execute_sensitivity,
            args.execute_transfer,
            args.execute_reporting,
        )
    )

    if selected_mode_count != 1:
        parser.error(
            "Choose exactly one of --validate-inputs-only, "
            "--execute-primary-sparse, "
            "--execute-primary-diversity, "
            "--execute-sensitivity, --execute-transfer, "
            "or --execute-reporting."
        )

    common = {
        "repository_root": args.repository_root,
        "expected_implementation_commit": (
            args.expected_implementation_commit
        ),
        "output_root": args.output_root,
    }

    if args.validate_inputs_only:
        report = validate_analysis_inputs(
            **common,
            input_root=args.input_root,
            require_clean_repository=True,
            require_outputs_absent=True,
            verify_checkpoint_hashes=True,
        )
        print(
            json.dumps(
                report.to_record(),
                sort_keys=True,
                indent=2,
            )
        )
        print()
        print("stage14_validate_inputs_only: passed")
        print("scientific_search_started: false")
        print("scientific_outputs_created: false")
        print("stage15_started: false")
        return 0

    if args.execute_primary_sparse:
        result = execute_primary_sparse_workload(
            **common,
            input_root=args.input_root,
            device=args.device,
            progress_callback=print,
        )
        label = "primary_sparse_cell_count"
        count = len(result.cells)
        outputs = (
            ("sparse_search_table", result.sparse_search_table),
            ("runtime_table", result.runtime_table),
        )
        completion = "stage14_primary_sparse_execution"
    elif args.execute_primary_diversity:
        result = execute_primary_diversity_workload(
            **common,
            input_root=args.input_root,
            device=args.device,
            progress_callback=print,
        )
        label = "primary_diversity_cell_count"
        count = len(result.cells)
        outputs = (
            ("family_summary_table", result.family_summary_table),
            ("circuits_table", result.circuits_table),
            (
                "pairwise_overlap_table",
                result.pairwise_overlap_table,
            ),
            ("restarts_table", result.restarts_table),
            ("runtime_table", result.runtime_table),
        )
        completion = "stage14_primary_diversity_execution"
    elif args.execute_sensitivity:
        result = execute_sensitivity_workload(
            **common,
            input_root=args.input_root,
            device=args.device,
            progress_callback=print,
        )
        label = "sensitivity_fresh_cell_count"
        count = len(result.executed_cells)
        outputs = (
            ("fidelity_table", result.fidelity_table),
            (
                "distinctness_table",
                result.distinctness_table,
            ),
            ("runtime_table", result.runtime_table),
        )
        completion = "stage14_sensitivity_execution"
    elif args.execute_transfer:
        result = execute_transfer_workload(
            **common,
            input_root=args.input_root,
            device=args.device,
            progress_callback=print,
        )
        label = "transfer_cell_count"
        count = (
            result.global_cell_count
            + result.subset_discovery_cell_count
            + result.grouping_cell_count
        )
        outputs = (
            ("transfer_table", result.transfer_table),
            ("runtime_table", result.runtime_table),
        )
        completion = "stage14_transfer_execution"
    else:
        result = execute_reporting_workload(
            **common,
            progress_callback=print,
        )
        label = "reporting_output_count"
        count = 4
        outputs = (
            ("frontier_table", result.frontier_table),
            ("analysis_note", result.analysis_note),
            ("manifest", result.manifest),
            ("archive", result.archive),
        )
        completion = "stage14_reporting_execution"

    print()
    print(
        f"analysis_run_id: {result.analysis_run_id}"
    )
    print(
        "implementation_commit: "
        f"{result.implementation_commit}"
    )
    print(f"{label}: {count}")

    for name, file_name in outputs:
        print(f"{name}: {file_name}")

    print(f"{completion}: complete")
    print("stage15_started: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate or execute Stage 16 genuine-task functional transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_families.analysis.stage14_random_label_runner import (
    current_git_commit,
)
from circuit_families.analysis.stage16_transfer import (
    compare_reproduction,
    execute_stage16,
    validate_stage16_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic Stage 16 genuine-task functional transfer."
    )
    parser.add_argument(
        "--stage12-manifest",
        type=Path,
        default=Path(
            "manifests/stage12_diversity_"
            "stage12-diversity-s1-020ebf1b5814.json"
        ),
    )
    parser.add_argument("--checkpoint-step", type=int, default=9050)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Write outputs to an isolated root for reproduction.",
    )
    parser.add_argument(
        "--expected-implementation-commit",
        help="Require this exact clean implementation commit (defaults to HEAD).",
    )
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
        help="Validate every frozen input and create no files.",
    )
    parser.add_argument(
        "--compare-reference-root",
        type=Path,
        help=(
            "After isolated reproduction, compare deterministic outputs "
            "byte-for-byte with this completed Stage 16 root."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository_root.resolve()
    expected_commit = args.expected_implementation_commit or current_git_commit(
        repository
    )
    output_root = args.output_root.resolve() if args.output_root else repository

    if args.validate_inputs_only:
        if args.compare_reference_root is not None:
            raise ValueError("Validation-only mode cannot compare reproduction outputs.")
        report = validate_stage16_inputs(
            repository_root=repository,
            expected_implementation_commit=expected_commit,
            stage12_manifest=args.stage12_manifest,
            checkpoint_step=args.checkpoint_step,
            device=args.device,
            output_root=output_root,
            require_outputs_absent=False,
        )
        print("Stage 16 input validation: passed")
        print(f"implementation_commit: {report.implementation_commit}")
        print(f"stage16_run_id: {report.configuration.run_id}")
        print(f"checkpoint_step: {report.context.checkpoint_step}")
        print(f"checkpoint_sha256: {report.context.checkpoint_sha256}")
        print(f"source_family_size: {len(report.circuits)}")
        for circuit in report.circuits:
            print(
                f"source_circuit: {circuit.circuit_id} "
                f"retained={circuit.mask.retained_component_count} "
                f"mask_sha256={circuit.mask_sha256}"
            )
        for subset_name in report.subset_indices:
            print(
                f"subset: {subset_name} "
                f"count={report.subset_indices[subset_name].size} "
                f"membership_sha256={report.subset_hashes[subset_name]}"
            )
        print("files_created: 0")
        return

    result = execute_stage16(
        repository_root=repository,
        expected_implementation_commit=expected_commit,
        stage12_manifest=args.stage12_manifest,
        checkpoint_step=args.checkpoint_step,
        device=args.device,
        output_root=output_root,
        progress_callback=lambda message: print(message, flush=True),
    )
    print("Stage 16 execution: complete")
    print(f"stage16_run_id: {result.run_id}")
    print(f"implementation_commit: {result.implementation_commit}")
    print(f"manifest: {result.manifest}")
    print(f"archive: {result.archive}")

    if args.compare_reference_root is not None:
        comparison = compare_reproduction(
            reference_root=args.compare_reference_root,
            reproduction_root=output_root,
        )
        print(json.dumps(comparison, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

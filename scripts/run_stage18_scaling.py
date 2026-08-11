"""Validate or execute the frozen Stage 18 main-seed scaling workload."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from circuit_families.analysis.stage18_scaling import (
    deterministic_stage18_run_id,
    load_stage18_configuration,
    validate_stage18_inputs,
)


def _validate_implementation_ancestor(
    repository: Path,
    implementation_commit: str,
) -> None:
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"),
        cwd=repository,
        check=True,
    )
    completed = subprocess.run(
        ("git", "diff", "--name-only", f"{implementation_commit}..HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    allowed_prefixes = (
        "checkpoints/stage18-main-training-",
        "manifests/training_stage18-main-training-",
        "manifests/stage18_checkpoints_seed_",
        "manifests/stage18_training.json",
        "results/raw/stage18-main-training-",
        "results/tables/stage18_checkpoint_registry.csv",
        "results/tables/stage18_main_seed_registry.csv",
        "results/tables/stage18_training_runs.csv",
    )
    disallowed = [
        path
        for path in completed.stdout.splitlines()
        if path and not path.startswith(allowed_prefixes)
    ]
    if disallowed:
        raise ValueError(
            "Files changed after the Stage 18 implementation commit: " + ", ".join(disallowed)
        )


def _validate_finalization_commit(
    repository: Path,
    implementation_commit: str,
    finalization_commit: str,
) -> None:
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", implementation_commit, finalization_commit),
        cwd=repository,
        check=True,
    )
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != finalization_commit:
        raise ValueError("Current commit does not match the expected finalization commit.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Stage 18 multi-seed scaling.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("manifests/post_stage17_checkpoint_grid_and_concurrency_freeze.json"),
    )
    parser.add_argument(
        "--stage17-manifest",
        type=Path,
        default=Path("manifests/stage17_sensitivity_stage17-sensitivity-s1-7801e7938531.json"),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--expected-implementation-commit")
    parser.add_argument("--expected-finalization-commit")
    parser.add_argument("--finalize-existing-run", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--compare-reproduction-root", type=Path)
    parser.add_argument("--reproduction-mode", action="store_true")
    args = parser.parse_args()
    if args.workers != 12:
        parser.error("Stage 18 production requires exactly 12 workers.")
    if args.threads_per_worker != 1:
        parser.error("Stage 18 requires exactly one thread per worker.")
    if args.expected_finalization_commit and not args.expected_implementation_commit:
        parser.error("A finalization commit requires an implementation commit.")
    if args.finalize_existing_run and not args.expected_finalization_commit:
        parser.error("Resume-only finalization requires an expected finalization commit.")
    if args.finalize_existing_run and args.validate_inputs_only:
        parser.error("Resume-only finalization cannot be combined with validation-only mode.")
    if args.finalize_existing_run and args.compare_reproduction_root is not None:
        parser.error("Resume-only finalization cannot be combined with reproduction comparison.")
    return args


def main() -> None:
    args = parse_args()
    if args.compare_reproduction_root is not None:
        from circuit_families.analysis.stage18_execution import compare_reproduction

        repository = args.repository_root.resolve()
        configuration = load_stage18_configuration(repository)
        implementation_commit = args.expected_implementation_commit
        if implementation_commit is None:
            raise ValueError(
                "Existing-output comparison requires --expected-implementation-commit."
            )
        run_id = deterministic_stage18_run_id(configuration.sha256, implementation_commit)
        comparison = compare_reproduction(
            repository,
            args.compare_reproduction_root.resolve(),
            run_id=run_id,
            progress=True,
        )
        print(json.dumps(comparison, sort_keys=True, indent=2))
        if not comparison["passed"]:
            raise SystemExit(1)
        return
    validation = validate_stage18_inputs(
        args.repository_root,
        reproduction_mode=args.reproduction_mode,
    )
    implementation_commit = args.expected_implementation_commit or validation.implementation_commit
    finalization_commit = args.expected_finalization_commit
    run_id = deterministic_stage18_run_id(validation.configuration.sha256, implementation_commit)
    if args.validate_inputs_only:
        if finalization_commit is None:
            _validate_implementation_ancestor(args.repository_root.resolve(), implementation_commit)
        else:
            _validate_finalization_commit(
                args.repository_root.resolve(), implementation_commit, finalization_commit
            )
        print("stage18_validate_inputs_only: passed")
        print(f"implementation_commit: {implementation_commit}")
        if finalization_commit is not None:
            print(f"finalization_commit: {finalization_commit}")
        print(f"stage18_run_id: {run_id}")
        print(f"registered_cells: {len(validation.cells)}")
        print("fresh_cells: 612")
        print("reference_cells: 18")
        print(f"worker_shards: {len(validation.shards)}")
        return
    if finalization_commit is None:
        _validate_implementation_ancestor(args.repository_root.resolve(), implementation_commit)
    else:
        _validate_finalization_commit(
            args.repository_root.resolve(), implementation_commit, finalization_commit
        )
    from circuit_families.analysis.stage18_execution import execute_stage18

    manifest = execute_stage18(
        args.repository_root.resolve(),
        run_id=run_id,
        implementation_commit=implementation_commit,
        configuration_sha256=validation.configuration.sha256,
        finalization_commit=finalization_commit,
        finalize_existing=args.finalize_existing_run,
    )
    print(f"stage18_run_id: {run_id}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()

"""Validate or run deterministic Stage 19 matched comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuit_families.analysis.stage18_scaling import stable_json, write_csv
from circuit_families.analysis.stage19_matched_comparisons import (
    CHECKPOINT_STEPS,
    FIDELITY_TOLERANCE,
    SPARSITY_TOLERANCE,
    build_stage19_tables,
    read_csv_rows,
)
from circuit_families.training import file_sha256

STAGE18_RUN_ID = "stage18-scaling-24a9adb84176"
STAGE18_MANIFEST = Path(f"manifests/stage18_scaling_{STAGE18_RUN_ID}.json")
CONCURRENCY_AMENDMENT = Path("manifests/stage18_reproduction_downstream_concurrency_amendment.json")
SOURCE_TABLES = (
    Path("results/tables/stage18_circuits.csv"),
    Path("results/tables/stage18_family_summary.csv"),
    Path("results/tables/stage18_pairwise_overlap.csv"),
)
PROVENANCE_PATHS = (
    Path("results/tables/stage18_cell_registry.csv"),
    Path("results/tables/stage18_search_failures.csv"),
    Path("results/archives/stage18-scaling-24a9adb84176/index.json"),
)
SEED_REGISTRY = Path("results/tables/stage18_main_seed_registry.csv")
OUTPUT_TABLES = {
    "input_registry": Path("results/tables/stage19_input_registry.csv"),
    "matched_fidelity_pairs": Path("results/tables/stage19_matched_fidelity_pairs.csv"),
    "matched_fidelity_summary": Path("results/tables/stage19_matched_fidelity_summary.csv"),
    "matched_sparsity_pairs": Path("results/tables/stage19_matched_sparsity_pairs.csv"),
    "matched_sparsity_summary": Path("results/tables/stage19_matched_sparsity_summary.csv"),
    "unmatched_circuits": Path("results/tables/stage19_unmatched_circuits.csv"),
    "pareto_frontiers": Path("results/tables/stage19_pareto_frontiers.csv"),
    "empty_cells": Path("results/tables/stage19_empty_cells.csv"),
    "excluded_comparisons": Path("results/tables/stage19_excluded_comparisons.csv"),
    "comparison_sources": Path("results/tables/stage19_comparison_sources.csv"),
}
NOTE_PATH = Path("results/notes/stage19_matched_comparisons.md")
RUNTIME_PATH = Path("results/tables/stage19_runtime.csv")


def _object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object.")
    return value


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_tracked_repository(repository: Path) -> str:
    if _git(repository, "branch", "--show-current") != "main":
        raise ValueError("Stage 19 requires branch main.")
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        completed = subprocess.run(("git", *arguments), cwd=repository, check=False)
        if completed.returncode != 0:
            raise ValueError("Stage 19 requires a clean tracked repository.")
    return _git(repository, "rev-parse", "HEAD")


def _reproduction_passed(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    comparison = _object(path, "Stage 18 reproduction comparison")
    if comparison.get("stage18_run_id") != STAGE18_RUN_ID:
        raise ValueError("Stage 18 reproduction comparison run ID mismatch.")
    return comparison.get("passed") is True


def validate_inputs(
    repository: Path,
    reproduction_comparison: Path | None,
) -> tuple[str, str, bool, dict[str, str]]:
    commit = _validate_tracked_repository(repository)
    manifest_path = repository / STAGE18_MANIFEST
    manifest = _object(manifest_path, "Stage 18 manifest")
    if manifest.get("schema_version") != 2 or manifest.get("experiment_stage") != 18:
        raise ValueError("Stage 19 requires the complete Stage 18 schema-v2 manifest.")
    if manifest.get("stage18_run_id") != STAGE18_RUN_ID:
        raise ValueError("Stage 18 source run ID mismatch.")
    if manifest.get("stage19_started") is not False:
        raise ValueError("Stage 18 manifest must record that Stage 19 had not started.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Stage 18 manifest outputs must be a hash mapping.")
    hashes = {str(STAGE18_MANIFEST): file_sha256(manifest_path)}
    for relative in SOURCE_TABLES:
        expected = outputs.get(str(relative))
        actual = file_sha256(repository / relative)
        if actual != expected:
            raise ValueError(f"Stage 18 source hash mismatch: {relative}")
        hashes[str(relative)] = actual
    for relative in PROVENANCE_PATHS:
        expected = outputs.get(str(relative))
        actual = file_sha256(repository / relative)
        if actual != expected:
            raise ValueError(f"Stage 18 provenance hash mismatch: {relative}")
        hashes[str(relative)] = actual
    _git(repository, "ls-files", "--error-unmatch", str(SEED_REGISTRY))
    hashes[str(SEED_REGISTRY)] = file_sha256(repository / SEED_REGISTRY)
    amendment = _object(repository / CONCURRENCY_AMENDMENT, "concurrency amendment")
    lifecycle = amendment.get("lifecycle", {})
    if lifecycle.get("stage19_to_stage22_may_begin_during_reproduction") is not True:
        raise ValueError("The concurrency amendment does not permit Stage 19 to begin.")
    if (
        lifecycle.get("downstream_results_finalization_requires_successful_stage18_reproduction")
        is not True
    ):
        raise ValueError("The Stage 19 reproduction finalization gate is missing.")
    hashes[str(CONCURRENCY_AMENDMENT)] = file_sha256(repository / CONCURRENCY_AMENDMENT)
    comparison = (
        reproduction_comparison
        if reproduction_comparison is None or reproduction_comparison.is_absolute()
        else repository / reproduction_comparison
    )
    reproduced = _reproduction_passed(comparison)
    return commit, hashes[str(STAGE18_MANIFEST)], reproduced, hashes


def deterministic_run_id(stage18_manifest_sha256: str, implementation_commit: str) -> str:
    identity = (
        f"stage19|{stage18_manifest_sha256}|{implementation_commit}|"
        f"fidelity={FIDELITY_TOLERANCE}|sparsity={SPARSITY_TOLERANCE}|"
        f"steps={','.join(str(step) for step in CHECKPOINT_STEPS)}"
    )
    return f"stage19-matched-{hashlib.sha256(identity.encode('ascii')).hexdigest()[:12]}"


def _ensure_outputs_absent(repository: Path, manifest_path: Path) -> None:
    paths = [repository / path for path in OUTPUT_TABLES.values()]
    paths.extend((repository / NOTE_PATH, repository / RUNTIME_PATH, manifest_path))
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Stage 19 outputs already exist: " + ", ".join(map(str, existing)))


def _write_note(
    path: Path,
    *,
    run_id: str,
    status: str,
    counts: dict[str, int],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "# Stage 19 matched comparisons",
                "",
                f"- Run ID: `{run_id}`",
                f"- Lifecycle status: `{status}`",
                f"- Matched-fidelity comparisons: {counts['matched_fidelity_comparisons']}",
                f"- Matched-sparsity comparisons: {counts['matched_sparsity_comparisons']}",
                f"- Explicit empty cells: {counts['empty_cells']}",
                f"- Pareto-frontier rows: {counts['pareto_frontier_rows']}",
                f"- Complete Stage 18 input cells: {counts['input_cells']}",
                f"- Nonempty / empty / singleton cells: {counts['nonempty_cells']} / "
                f"{counts['empty_cells']} / {counts['singleton_cells']}.",
                "- Fidelity matching tolerance: 0.01 exact global fidelity.",
                "- Sparsity matching tolerance: five retained components.",
                "- Matching maximises valid cardinality and then minimises total "
                "absolute difference.",
                "- Matching is symmetric joint one-to-one selection without replacement. "
                "Exact observed circuit values are used; interpolation and rounded-display "
                "matching are prohibited. Remaining ties are broken by deterministic circuit "
                "identity order.",
                "- Empty families are reported directly, never imputed, and excluded "
                "only where the metric is undefined.",
                "- Circuits and search restarts are not treated as independent experimental units.",
                "- Pareto axes are exact circuit fidelity (higher is better) and retained "
                "component count (fewer is better); duplicate masks are canonicalised "
                "deterministically and no smoothing is used.",
                "",
                "These outputs were generated from the definitive Stage 18 run. Independent "
                "Stage 18 reproduction comparison was pending at the time of generation. "
                "The results are therefore provisionally validated and must be revalidated "
                "after reproduction completes.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 19 matched comparisons.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--allow-pending-reproduction", action="store_true")
    parser.add_argument("--stage18-reproduction-comparison", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository_root.resolve()
    started = time.monotonic()
    commit, source_manifest_hash, reproduced, source_hashes = validate_inputs(
        repository, args.stage18_reproduction_comparison
    )
    run_id = deterministic_run_id(source_manifest_hash, commit)
    if args.validate_inputs_only:
        print("stage19_validate_inputs_only: passed")
        print(f"implementation_commit: {commit}")
        print(f"stage19_run_id: {run_id}")
        print(f"stage18_reproduction_passed: {str(reproduced).lower()}")
        print("checkpoint_comparisons_per_seed: 21")
        print("total_checkpoint_comparisons: 105")
        return
    if not reproduced and not args.allow_pending_reproduction:
        raise ValueError(
            "Stage 18 reproduction remains pending; use --allow-pending-reproduction "
            "only for advisor-authorised provisional Stage 19 outputs."
        )
    manifest_path = repository / f"manifests/stage19_matched_{run_id}.json"
    _ensure_outputs_absent(repository, manifest_path)
    tables = build_stage19_tables(
        stage18_run_id=STAGE18_RUN_ID,
        circuit_rows=read_csv_rows(repository / SOURCE_TABLES[0]),
        family_rows=read_csv_rows(repository / SOURCE_TABLES[1]),
        overlap_rows=read_csv_rows(repository / SOURCE_TABLES[2]),
    )
    outputs = []
    for name, relative in OUTPUT_TABLES.items():
        outputs.append(write_csv(repository / relative, getattr(tables, name)))
    status = "finalized" if reproduced else "provisionally_validated"
    counts = {
        "input_cells": len(tables.input_registry),
        "nonempty_cells": sum(not row["empty_family"] for row in tables.input_registry),
        "singleton_cells": sum(row["singleton_family"] for row in tables.input_registry),
        "right_censored_cells": sum(row["right_censored"] for row in tables.input_registry),
        "failed_search_cells": sum(
            row["search_status"] == "sparsity_failure" for row in tables.input_registry
        ),
        "unavailable_control_cells": sum(
            row["unavailable_control"] for row in tables.input_registry
        ),
        "matched_fidelity_comparisons": len(tables.matched_fidelity_summary),
        "matched_fidelity_pairs": len(tables.matched_fidelity_pairs),
        "matched_sparsity_comparisons": len(tables.matched_sparsity_summary),
        "matched_sparsity_pairs": len(tables.matched_sparsity_pairs),
        "unmatched_circuit_records": len(tables.unmatched_circuits),
        "pareto_frontier_rows": len(tables.pareto_frontiers),
        "empty_cells": len(tables.empty_cells),
        "excluded_comparisons": len(tables.excluded_comparisons),
        "comparison_source_rows": len(tables.comparison_sources),
    }
    outputs.append(
        write_csv(
            repository / RUNTIME_PATH,
            (
                {
                    "stage19_run_id": run_id,
                    "operation": "saved_table_matching_and_reporting",
                    "elapsed_seconds": f"{time.monotonic() - started:.6f}",
                    "retraining": False,
                    "circuit_search": False,
                },
            ),
        )
    )
    outputs.append(_write_note(repository / NOTE_PATH, run_id=run_id, status=status, counts=counts))
    stage18_manifest = _object(repository / STAGE18_MANIFEST, "Stage 18 manifest")
    stable_json(
        manifest_path,
        {
            "schema_version": 1,
            "experiment_stage": 19,
            "stage19_run_id": run_id,
            "creation_timestamp_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "stage19_finalized": reproduced,
            "stage18_run_id": STAGE18_RUN_ID,
            "stage18_reproduction_passed": reproduced,
            "stage18_reproduction_status": "passed" if reproduced else "pending",
            "implementation_commit": commit,
            "upstream_stage18": {
                "run_id": STAGE18_RUN_ID,
                "implementation_commit": stage18_manifest["implementation_commit"],
                "scientific_output_commit": _git(
                    repository, "log", "-1", "--format=%H", "--", str(STAGE18_MANIFEST)
                ),
                "manifest_path": str(STAGE18_MANIFEST),
                "manifest_sha256": source_manifest_hash,
                "consumed_table_hashes": {
                    str(path): source_hashes[str(path)] for path in SOURCE_TABLES
                },
                "archive_index_path": str(PROVENANCE_PATHS[-1]),
                "archive_index_sha256": source_hashes[str(PROVENANCE_PATHS[-1])],
                "main_seed_registry_path": str(SEED_REGISTRY),
                "main_seed_registry_sha256": source_hashes[str(SEED_REGISTRY)],
                "model_seeds": [0, 1, 2, 3, 4],
                "checkpoint_steps": list(CHECKPOINT_STEPS),
                "cell_count": 630,
                "primary_cell": {
                    "fidelity_numerator": 99,
                    "fidelity_denominator": 100,
                    "jaccard_cutoff_numerator": 1,
                    "jaccard_cutoff_denominator": 2,
                },
                "reproduction_status": "passed" if reproduced else "pending",
            },
            "source_artifacts": source_hashes,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "comparison_design": {
                "within_seed": True,
                "all_unordered_checkpoint_pairs": True,
                "checkpoint_comparisons_per_seed": 21,
                "independent_unit": "trained_model_seed",
                "matched_fidelity_tolerance": str(FIDELITY_TOLERANCE),
                "matched_sparsity_tolerance_components": int(SPARSITY_TOLERANCE),
                "one_to_one_without_replacement": True,
                "matching_direction": "symmetric_joint_pair_selection",
                "interpolation": False,
                "matching_values": "exact_observed_circuit_values",
                "deterministic_tie_break": "circuit_identity_lexicographic",
                "objective_order": [
                    "maximum_valid_cardinality",
                    "minimum_total_absolute_difference",
                ],
                "empty_family_imputation": False,
            },
            "counts": counts,
            "outputs": {str(path.relative_to(repository)): file_sha256(path) for path in outputs},
        },
    )
    print(f"stage19_run_id: {run_id}")
    print(f"status: {status}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

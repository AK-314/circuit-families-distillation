"""Validate or run provisional Stage 20 seed-level paired inference."""

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
from circuit_families.analysis.stage19_matched_comparisons import read_csv_rows
from circuit_families.analysis.stage20_paired_inference import (
    METRICS,
    build_checkpoint_metrics,
    build_paired_deltas,
    build_seed_level_summaries,
    build_trajectory_source,
    generate_seed_trajectory_figures,
)
from circuit_families.training import file_sha256

STAGE18_RUN_ID = "stage18-scaling-24a9adb84176"
STAGE19_RUN_ID = "stage19-matched-02b89bd79ed8"
STAGE19_MANIFEST = Path(f"manifests/stage19_matched_{STAGE19_RUN_ID}.json")
SOURCE_PATHS = (
    Path("results/tables/stage18_family_summary.csv"),
    Path("results/tables/stage18_circuit_size_summary.csv"),
    Path("results/tables/stage18_transfer_profiles.csv"),
    Path("manifests/stage18_training.json"),
    Path("results/tables/stage19_matched_fidelity_summary.csv"),
    Path("results/tables/stage19_matched_sparsity_summary.csv"),
)
OUTPUT_PATHS = {
    "checkpoint_metrics": Path("results/tables/stage20_seed_checkpoint_metrics.csv"),
    "comparison_registry": Path("results/tables/stage20_comparison_registry.csv"),
    "paired_deltas": Path("results/tables/stage20_paired_deltas.csv"),
    "seed_level_summaries": Path("results/tables/stage20_seed_level_summaries.csv"),
    "trajectory_source": Path("results/tables/stage20_trajectory_source.csv"),
}
NOTE_PATH = Path("results/notes/stage20_paired_seed_inference.md")
CAPTION_PATH = Path("figures/stage20_trajectory_captions.txt")
RUNTIME_PATH = Path("results/tables/stage20_runtime.csv")
TRAJECTORY_PATHS = (
    Path("figures/stage20_seed_checkpoint_trajectories.png"),
    Path("figures/stage20_seed_checkpoint_trajectories.pdf"),
    Path("figures/stage20_matched_adjacent_trajectories.png"),
    Path("figures/stage20_matched_adjacent_trajectories.pdf"),
)


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


def validate_inputs(
    repository: Path,
) -> tuple[str, str, bool, dict[str, str], dict[str, Any]]:
    if _git(repository, "branch", "--show-current") != "main":
        raise ValueError("Stage 20 requires branch main.")
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        if subprocess.run(("git", *arguments), cwd=repository, check=False).returncode:
            raise ValueError("Stage 20 requires a clean tracked repository.")
    commit = _git(repository, "rev-parse", "HEAD")
    stage19_path = repository / STAGE19_MANIFEST
    stage19 = _object(stage19_path, "Stage 19 manifest")
    if stage19.get("stage19_run_id") != STAGE19_RUN_ID:
        raise ValueError("Stage 19 source run ID mismatch.")
    if stage19.get("status") != "provisionally_validated":
        raise ValueError("Stage 19 must be provisionally validated before Stage 20.")
    if stage19.get("stage18_reproduction_status") not in ("pending", "pass"):
        raise ValueError("Stage 19 has an invalid Stage 18 reproduction status.")
    stage19_outputs = stage19.get("outputs")
    if not isinstance(stage19_outputs, dict):
        raise ValueError("Stage 19 output hashes are missing.")
    stage18_manifest = _object(
        repository / f"manifests/stage18_scaling_{STAGE18_RUN_ID}.json",
        "Stage 18 manifest",
    )
    stage18_outputs = stage18_manifest.get("outputs")
    if not isinstance(stage18_outputs, dict):
        raise ValueError("Stage 18 output hashes are missing.")
    hashes = {str(STAGE19_MANIFEST): file_sha256(stage19_path)}
    for relative in SOURCE_PATHS:
        actual = file_sha256(repository / relative)
        expected = (
            stage19_outputs.get(str(relative))
            if str(relative).startswith("results/tables/stage19_")
            else stage18_outputs.get(str(relative))
        )
        if relative == Path("manifests/stage18_training.json"):
            _git(repository, "ls-files", "--error-unmatch", str(relative))
            expected = actual
        if actual != expected:
            raise ValueError(f"Stage 20 source hash mismatch: {relative}")
        hashes[str(relative)] = actual
    reproduced = stage19.get("stage18_reproduction_status") == "pass"
    return commit, hashes[str(STAGE19_MANIFEST)], reproduced, hashes, stage19


def deterministic_run_id(stage19_manifest_sha256: str, implementation_commit: str) -> str:
    identity = f"stage20|{stage19_manifest_sha256}|{implementation_commit}|seed-level-paired-v1"
    return f"stage20-paired-{hashlib.sha256(identity.encode('ascii')).hexdigest()[:12]}"


def _write_note(path: Path, *, run_id: str, status: str, counts: dict[str, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "# Stage 20 paired seed-level inference",
                "",
                f"- Run ID: `{run_id}`",
                f"- Lifecycle status: `{status}`",
                "- These outputs were generated from the definitive Stage 18 run. Independent "
                "Stage 18 reproduction comparison was pending at generation, so the results are "
                "provisionally validated and must be revalidated after reproduction completes.",
                "- Independent unit: independently trained model seed.",
                "- The exact seven-step grid is shared across all seeds.",
                "- Seed-specific phase labels are retained and phase misalignment is explicit.",
                "- Undefined empty-family metrics remain undefined; family size remains zero.",
                "- The prespecified exact two-sided sign test tests equiprobable positive and "
                "negative nonzero seed-level changes; zero differences are reported but excluded.",
                "- No multiple-comparison adjustment is applied: probabilities are descriptive "
                "small-sample summaries, not decision thresholds.",
                "- Every trajectory line represents one independently trained model seed; "
                "circuits within a seed are not treated as replications.",
                "- Gaps in trajectories are undefined empty-family-dependent quantities, "
                "not imputed observations.",
                f"- Paired delta rows: {counts['paired_delta_rows']}",
                f"- Seed-level summary rows: {counts['seed_level_summary_rows']}",
                f"- Trajectory figures: {counts['trajectory_figures']}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _write_caption(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n\n".join(
            (
                "Stage 20 trajectory figures (provisional pending independent Stage 18 "
                "reproduction). The independently trained model seed is the replication unit "
                "(n=5). No circuits, restarts, checkpoints, or transfer subsets are counted as "
                "independent replicates.",
                "Raw seed trajectories. Every coloured line is one seed across the complete "
                "seven-checkpoint grid (steps 200, 3400, 7450, 8150, 8500, 8650, and 9050). "
                "Panels show recovered structural-family size, transfer-distinct-group count, "
                "median pairwise Jaccard overlap, median retained circuit size, and mean transfer "
                "fidelity. Family size zero is an observed empty-family result; gaps in dependent "
                "metrics are undefined empty-family outcomes and are neither zero-filled nor "
                "interpolated. Right-censoring flags are retained in the source table; none occur "
                "in this Stage 18 run.",
                "Adjacent matched-comparison trajectories. Values are right-minus-left changes "
                "for adjacent fixed checkpoints. Matched-fidelity structural diversity uses "
                "one-to-one observed circuit pairs within seed at absolute fidelity tolerance "
                "0.01. Matched-sparsity fidelity uses one-to-one observed pairs within seed at "
                "retained-component tolerance 5. The first checkpoint has no preceding pair and "
                "is therefore shown as a gap. Other gaps mean the prespecified match was "
                "undefined; no interpolation or imputation is used.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 20 paired inference.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--allow-pending-reproduction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository_root.resolve()
    commit, stage19_hash, reproduced, source_hashes, stage19 = validate_inputs(repository)
    run_id = deterministic_run_id(stage19_hash, commit)
    if args.validate_inputs_only:
        print("stage20_validate_inputs_only: passed")
        print(f"implementation_commit: {commit}")
        print(f"stage20_run_id: {run_id}")
        print(f"stage18_reproduction_passed: {str(reproduced).lower()}")
        print("seed_count: 5")
        print("grid_comparisons_per_seed: 17")
        print(f"metrics_per_comparison: {len(METRICS)}")
        return
    if not reproduced and not args.allow_pending_reproduction:
        raise ValueError(
            "Stage 18 reproduction remains pending; provisional Stage 20 execution requires "
            "--allow-pending-reproduction."
        )
    manifest_path = repository / f"manifests/stage20_paired_{run_id}.json"
    candidates = [repository / path for path in OUTPUT_PATHS.values()]
    candidates.extend(repository / path for path in TRAJECTORY_PATHS)
    candidates.extend(
        (
            repository / NOTE_PATH,
            repository / CAPTION_PATH,
            repository / RUNTIME_PATH,
            manifest_path,
        )
    )
    existing = [path for path in candidates if path.exists()]
    if existing:
        raise FileExistsError("Stage 20 outputs already exist: " + ", ".join(map(str, existing)))

    started = time.monotonic()
    training = _object(repository / SOURCE_PATHS[3], "Stage 18 training manifest")
    checkpoint_metrics = build_checkpoint_metrics(
        family_rows=read_csv_rows(repository / SOURCE_PATHS[0]),
        circuit_size_rows=read_csv_rows(repository / SOURCE_PATHS[1]),
        transfer_profile_rows=read_csv_rows(repository / SOURCE_PATHS[2]),
        training_runs=training["runs"],
    )
    paired = build_paired_deltas(
        checkpoint_metrics=checkpoint_metrics,
        matched_fidelity_rows=read_csv_rows(repository / SOURCE_PATHS[4]),
        matched_sparsity_rows=read_csv_rows(repository / SOURCE_PATHS[5]),
    )
    comparison_registry = tuple(
        {
            key: row[key]
            for key in (
                "comparison_type",
                "comparison_label",
                "model_seed",
                "left_checkpoint_step",
                "right_checkpoint_step",
                "left_phase_label",
                "right_phase_label",
                "phase_alignment_status",
            )
        }
        for row in paired
        if row["metric"] == METRICS[0]
    )
    summaries = build_seed_level_summaries(paired)
    trajectory_source = build_trajectory_source(checkpoint_metrics, paired)
    table_values = {
        "checkpoint_metrics": checkpoint_metrics,
        "comparison_registry": comparison_registry,
        "paired_deltas": paired,
        "seed_level_summaries": summaries,
        "trajectory_source": trajectory_source,
    }
    outputs = [
        write_csv(repository / relative, table_values[name])
        for name, relative in OUTPUT_PATHS.items()
    ]
    outputs.extend(
        generate_seed_trajectory_figures(
            repository,
            trajectory_rows=trajectory_source,
        )
    )
    status = "finalized" if reproduced else "provisionally_validated"
    counts = {
        "seed_checkpoint_rows": len(checkpoint_metrics),
        "comparison_registry_rows": len(comparison_registry),
        "paired_delta_rows": len(paired),
        "defined_paired_delta_rows": sum(row["metric_status"] == "defined" for row in paired),
        "undefined_paired_delta_rows": sum(row["metric_status"] == "undefined" for row in paired),
        "seed_level_summary_rows": len(summaries),
        "trajectory_source_rows": len(trajectory_source),
        "phase_misaligned_comparison_rows": sum(
            row["phase_alignment_status"] == "phase_misaligned" for row in comparison_registry
        ),
        "trajectory_figures": 2,
        "trajectory_figure_files": 4,
    }
    outputs.append(_write_note(repository / NOTE_PATH, run_id=run_id, status=status, counts=counts))
    outputs.append(_write_caption(repository / CAPTION_PATH))
    elapsed_seconds = time.monotonic() - started
    outputs.append(
        write_csv(
            repository / RUNTIME_PATH,
            (
                {
                    "stage20_run_id": run_id,
                    "wall_clock_seconds": f"{elapsed_seconds:.6f}",
                    "operation": "saved_table_only_paired_analysis_and_plotting",
                    "scientific_training_or_search_performed": False,
                },
            ),
        )
    )
    source_commits = {
        str(relative): _git(repository, "log", "-1", "--format=%H", "--", str(relative))
        for relative in (STAGE19_MANIFEST, *SOURCE_PATHS)
    }
    stable_json(
        manifest_path,
        {
            "schema_version": 1,
            "experiment_stage": 20,
            "stage20_run_id": run_id,
            "creation_timestamp_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "stage20_finalized": reproduced,
            "stage18_reproduction_status": "pass" if reproduced else "pending",
            "stage18_run_id": STAGE18_RUN_ID,
            "stage19_run_id": STAGE19_RUN_ID,
            "stage18_reproduction_passed": reproduced,
            "implementation_commit": commit,
            "source_artifacts": source_hashes,
            "source_commits": source_commits,
            "upstream_stage19": {
                "manifest_path": str(STAGE19_MANIFEST),
                "manifest_sha256": stage19_hash,
                "run_id": STAGE19_RUN_ID,
                "status": stage19["status"],
                "implementation_commit": stage19["implementation_commit"],
            },
            "upstream_stage18": stage19["upstream_stage18"],
            "independent_unit": "trained_model_seed",
            "metrics": list(METRICS),
            "statistical_plan": {
                "exact_test": "two_sided_exact_sign_test_nonzero_differences",
                "null_hypothesis": "positive_and_negative_directions_are_equiprobable",
                "zero_difference_handling": "reported_but_excluded_from_sign_test",
                "multiple_comparison_adjustment": "none_prespecified_descriptive_summaries",
                "inference_scope": "descriptive_small_sample_seed_level",
            },
            "counts": counts,
            "outputs": {str(path.relative_to(repository)): file_sha256(path) for path in outputs},
        },
    )
    print(f"stage20_run_id: {run_id}")
    print(f"status: {status}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

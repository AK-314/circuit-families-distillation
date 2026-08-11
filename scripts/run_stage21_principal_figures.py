"""Validate or generate provisional Stage 21 principal figures."""

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
from circuit_families.analysis.stage21_figures import generate_principal_figures
from circuit_families.training import file_sha256

STAGE18_RUN_ID = "stage18-scaling-24a9adb84176"
STAGE18_MANIFEST = Path(f"manifests/stage18_scaling_{STAGE18_RUN_ID}.json")
STAGE20_RUN_ID = "stage20-paired-a70dc23368a6"
STAGE20_MANIFEST = Path(f"manifests/stage20_paired_{STAGE20_RUN_ID}.json")
EXTRA_SOURCES = (
    Path("results/tables/stage18_family_summary.csv"),
    Path("results/tables/stage18_pairwise_overlap.csv"),
    Path("results/tables/stage18_transfer_profiles.csv"),
    Path("manifests/stage18_training.json"),
    Path("results/tables/seed_0_stage14_random_label_family_summary.csv"),
)
SOURCE_REGISTRY = Path("results/tables/stage21_figure_source_registry.csv")
RUNTIME = Path("results/tables/stage21_runtime.csv")
FIGURE_STEMS = (
    "stage21_figure1_training_trajectories",
    "stage21_figure2_family_dynamics",
    "stage21_figure3_structural_sensitivity",
    "stage21_figure4_transfer_dynamics",
    "stage21_figure5_controls",
)
SOURCE_TABLES = tuple(
    Path(f"results/tables/stage21_figure{index}_{suffix}_source.csv")
    for index, suffix in enumerate(
        ("training_curves", "family_dynamics", "structural", "transfer", "controls"),
        start=1,
    )
)
CAPTION = Path("figures/stage21_principal_figures_caption.txt")


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


def _tracked_commit(repository: Path, relative: Path) -> str:
    _git(repository, "ls-files", "--error-unmatch", str(relative))
    commit = _git(repository, "log", "-1", "--format=%H", "--", str(relative))
    if not commit:
        raise ValueError(f"No source commit found for Stage 21 input: {relative}")
    return commit


def _validate_allowed_untracked(repository: Path) -> tuple[str, ...]:
    status = _git(repository, "status", "--porcelain", "--untracked-files=all")
    untracked = tuple(
        line[3:]
        for line in status.splitlines()
        if line.startswith("?? ")
    )
    archive_prefix = f"results/archives/{STAGE18_RUN_ID}/"
    unexpected = tuple(
        path
        for path in untracked
        if path != "stage17_inspection.md" and not path.startswith(archive_prefix)
    )
    if unexpected:
        raise ValueError("Unexpected untracked files before Stage 21: " + ", ".join(unexpected))
    return untracked


def validate_inputs(
    repository: Path,
) -> tuple[str, str, bool, dict[str, str], dict[str, str], dict[str, Any], tuple[str, ...]]:
    if _git(repository, "branch", "--show-current") != "main":
        raise ValueError("Stage 21 requires branch main.")
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        if subprocess.run(("git", *arguments), cwd=repository, check=False).returncode:
            raise ValueError("Stage 21 requires a clean tracked repository.")
    commit = _git(repository, "rev-parse", "HEAD")
    manifest_path = repository / STAGE20_MANIFEST
    manifest = _object(manifest_path, "Stage 20 manifest")
    if manifest.get("stage20_run_id") != STAGE20_RUN_ID:
        raise ValueError("Stage 20 source run ID mismatch.")
    if manifest.get("status") != "provisionally_validated":
        raise ValueError("Stage 20 must be provisionally validated before Stage 21.")
    if manifest.get("stage18_reproduction_status") not in ("pending", "pass"):
        raise ValueError("Stage 20 has an invalid Stage 18 reproduction status.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Stage 20 output hashes are missing.")
    hashes = {str(STAGE20_MANIFEST): file_sha256(manifest_path)}
    for relative, expected in outputs.items():
        actual = file_sha256(repository / relative)
        if actual != expected:
            raise ValueError(f"Stage 20 output hash mismatch: {relative}")
        hashes[relative] = actual
    stage18 = _object(repository / STAGE18_MANIFEST, "Stage 18 manifest")
    if stage18.get("stage18_run_id") != STAGE18_RUN_ID:
        raise ValueError("Stage 18 source run ID mismatch.")
    stage18_outputs = stage18.get("outputs")
    if not isinstance(stage18_outputs, dict):
        raise ValueError("Stage 18 output hashes are missing.")
    hashes[str(STAGE18_MANIFEST)] = file_sha256(repository / STAGE18_MANIFEST)
    for relative in EXTRA_SOURCES[:3]:
        actual = file_sha256(repository / relative)
        if actual != stage18_outputs.get(str(relative)):
            raise ValueError(f"Stage 18 source hash mismatch: {relative}")
        hashes[str(relative)] = actual

    training = _object(repository / EXTRA_SOURCES[3], "Stage 18 training registry")
    hashes[str(EXTRA_SOURCES[3])] = file_sha256(repository / EXTRA_SOURCES[3])
    for run in training.get("runs", ()):
        training_manifest_path = Path(str(run["manifest_path"]))
        training_manifest = _object(
            repository / training_manifest_path,
            f"training manifest for seed {run['model_seed']}",
        )
        actual_manifest_hash = file_sha256(repository / training_manifest_path)
        if actual_manifest_hash != run["manifest_sha256"]:
            raise ValueError(f"Training manifest hash mismatch: {training_manifest_path}")
        hashes[str(training_manifest_path)] = actual_manifest_hash
        metrics_path = Path(str(run["metrics_path"]))
        actual_metrics_hash = file_sha256(repository / metrics_path)
        expected_metrics_hash = training_manifest["hashes"]["metrics_jsonl_sha256"]
        if actual_metrics_hash != expected_metrics_hash:
            raise ValueError(f"Training metrics hash mismatch: {metrics_path}")
        hashes[str(metrics_path)] = actual_metrics_hash

    random_control = EXTRA_SOURCES[4]
    if not (repository / random_control).is_file():
        raise FileNotFoundError(f"Missing Stage 21 source: {random_control}")
    hashes[str(random_control)] = file_sha256(repository / random_control)
    source_commits = {
        relative: _tracked_commit(repository, Path(relative)) for relative in hashes
    }
    allowed_untracked = _validate_allowed_untracked(repository)
    return (
        commit,
        hashes[str(STAGE20_MANIFEST)],
        manifest.get("stage18_reproduction_status") == "pass",
        hashes,
        source_commits,
        manifest,
        allowed_untracked,
    )


def deterministic_run_id(stage20_manifest_sha256: str, implementation_commit: str) -> str:
    identity = f"stage21|{stage20_manifest_sha256}|{implementation_commit}|principal-figures-v1"
    return f"stage21-figures-{hashlib.sha256(identity.encode('ascii')).hexdigest()[:12]}"


def _source_registry_rows(repository: Path) -> tuple[dict[str, object], ...]:
    specifications = (
        (
            1,
            "A: training accuracy; B: test accuracy",
            SOURCE_TABLES[0],
            "model_seed, training_step, train_accuracy, test_accuracy, analysed_checkpoint",
            "all five genuine-task seeds and all logged steps",
            "model_seed ascending; training_step ascending",
            "none; raw logged values",
        ),
        (
            2,
            "A: primary family trajectories",
            SOURCE_TABLES[1],
            "model_seed, checkpoint_step, family_size, right_censored",
            "fidelity=0.990; displayed_jaccard_cutoff=0.50",
            "model_seed ascending; frozen checkpoint order",
            "none; one line per seed",
        ),
        (
            2,
            "B: fidelity sensitivity",
            SOURCE_TABLES[1],
            "model_seed, checkpoint_step, displayed_fidelity, family_size",
            "displayed_jaccard_cutoff=0.50; all six frozen fidelity thresholds",
            "fidelity ascending; frozen checkpoint order; model_seed ascending",
            "unweighted arithmetic mean across five seeds",
        ),
        (
            3,
            "A-E: final structural-overlap matrices",
            SOURCE_TABLES[2],
            "model_seed, circuit_i, circuit_j, jaccard_overlap, family_size",
            "record_type=overlap_matrix; checkpoint_step=9050; fidelity=0.990; cutoff=0.50",
            "model_seed ascending; numeric C1..Cn circuit order",
            "none; symmetric counterpart displayed deterministically",
        ),
        (
            3,
            "F: distinctness sensitivity",
            SOURCE_TABLES[2],
            "model_seed, checkpoint_step, displayed_jaccard_cutoff, family_size",
            "record_type=distinctness_sensitivity; fidelity=0.990; cutoffs=0.25,0.50,0.75",
            "cutoff ascending; frozen checkpoint order; model_seed ascending",
            "unweighted arithmetic mean across five seeds",
        ),
        (
            4,
            "A-E: final functional-transfer matrices",
            SOURCE_TABLES[3],
            "model_seed, circuit_id, test_subset, transfer_fidelity",
            "record_type=functional_transfer_matrix; checkpoint_step=9050; primary family",
            "model_seed ascending; numeric C1..Cn circuit order; Q1,Q2,Q3,Q4",
            "none; each matrix cell is an observed circuit-subset fidelity",
        ),
        (
            4,
            "F: transfer-distinct-group trajectories",
            SOURCE_TABLES[3],
            "model_seed, checkpoint_step, transfer_distinct_group_count",
            "record_type=transfer_group_trajectory; primary family",
            "model_seed ascending; frozen checkpoint order",
            "none; one line per seed and null retained for empty families",
        ),
        (
            5,
            "single panel: genuine task and available controls",
            SOURCE_TABLES[4],
            "condition, model_seed, checkpoint_step, family_size, availability",
            "primary family; seven fixed checkpoints",
            "condition; model_seed ascending; frozen checkpoint order",
            "individual genuine seeds plus their unweighted mean; random-label seed 0 raw",
        ),
    )
    rows = []
    for figure_number, panel, source, columns, filters, sort_order, aggregation in specifications:
        stem = FIGURE_STEMS[figure_number - 1]
        for suffix in ("png", "pdf"):
            output = Path(f"figures/{stem}.{suffix}")
            rows.append(
                {
                    "figure_number": figure_number,
                    "panel": panel,
                    "source_table": str(source),
                    "source_columns": columns,
                    "filters": filters,
                    "sort_order": sort_order,
                    "aggregation_rule": aggregation,
                    "output_file": str(output),
                    "output_sha256": file_sha256(repository / output),
                }
            )
    return tuple(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 21 principal figures.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--allow-pending-reproduction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository_root.resolve()
    (
        commit,
        stage20_hash,
        reproduced,
        source_hashes,
        source_commits,
        stage20,
        allowed_untracked,
    ) = validate_inputs(repository)
    run_id = deterministic_run_id(stage20_hash, commit)
    if args.validate_inputs_only:
        print("stage21_validate_inputs_only: passed")
        print(f"implementation_commit: {commit}")
        print(f"stage21_run_id: {run_id}")
        print(f"stage18_reproduction_passed: {str(reproduced).lower()}")
        print("principal_figure_count: 5")
        return
    if not reproduced and not args.allow_pending_reproduction:
        raise ValueError(
            "Stage 18 reproduction remains pending; provisional Stage 21 execution requires "
            "--allow-pending-reproduction."
        )
    manifest_path = repository / f"manifests/stage21_figures_{run_id}.json"
    candidates = [
        repository / Path(f"figures/{stem}.{suffix}")
        for stem in FIGURE_STEMS
        for suffix in ("png", "pdf")
    ]
    candidates.extend(repository / path for path in SOURCE_TABLES)
    candidates.extend(
        (repository / CAPTION, repository / SOURCE_REGISTRY, repository / RUNTIME, manifest_path)
    )
    existing = [path for path in candidates if path.exists()]
    if existing:
        raise FileExistsError("Stage 21 outputs already exist: " + ", ".join(map(str, existing)))
    started = time.monotonic()
    outputs = generate_principal_figures(repository)
    outputs = tuple(outputs) + (
        write_csv(repository / SOURCE_REGISTRY, _source_registry_rows(repository)),
    )
    elapsed_seconds = time.monotonic() - started
    outputs = tuple(outputs) + (
        write_csv(
            repository / RUNTIME,
            (
                {
                    "stage21_run_id": run_id,
                    "wall_clock_seconds": f"{elapsed_seconds:.6f}",
                    "operation": "committed_source_table_only_figure_generation",
                    "training_performed": False,
                    "circuit_search_performed": False,
                    "network_evaluation_performed": False,
                },
            ),
        ),
    )
    status = "finalized" if reproduced else "provisionally_validated"
    figure_files = [path for path in outputs if path.suffix in (".png", ".pdf")]
    expected_source_tables = tuple(repository / path for path in SOURCE_TABLES)
    source_tables = [path for path in outputs if path in expected_source_tables]
    stable_json(
        manifest_path,
        {
            "schema_version": 1,
            "experiment_stage": 21,
            "stage21_run_id": run_id,
            "creation_timestamp_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "stage21_finalized": reproduced,
            "provisional_notice": (
                "These outputs were generated from the definitive Stage 18 run. Independent "
                "Stage 18 reproduction comparison was pending at generation; the results are "
                "provisionally validated and must be revalidated after reproduction completes."
            ),
            "stage18_reproduction_status": "pass" if reproduced else "pending",
            "stage18_reproduction_passed": reproduced,
            "stage18_run_id": STAGE18_RUN_ID,
            "stage19_run_id": stage20["upstream_stage19"]["run_id"],
            "stage20_run_id": STAGE20_RUN_ID,
            "implementation_commit": commit,
            "source_artifacts": source_hashes,
            "source_commits": source_commits,
            "upstream_stage20": {
                "manifest_path": str(STAGE20_MANIFEST),
                "manifest_sha256": stage20_hash,
                "run_id": STAGE20_RUN_ID,
                "status": stage20["status"],
                "implementation_commit": stage20["implementation_commit"],
            },
            "upstream_stage19": stage20["upstream_stage19"],
            "upstream_stage18": stage20["upstream_stage18"],
            "repository_hygiene": {
                "allowed_untracked_at_generation": list(allowed_untracked),
                "allowed_untracked_policy": (
                    "user-owned stage17_inspection.md and Stage 18 archive shards covered by "
                    "the committed deterministic archive index only"
                ),
                "private_temporary_sources_used": False,
            },
            "generation": {
                "input_mode": "committed_hash_checked_saved_artifacts_only",
                "training_performed": False,
                "circuit_search_performed": False,
                "network_evaluation_performed": False,
                "interpolation": False,
            },
            "counts": {
                "principal_figures": 5,
                "figure_files": len(figure_files),
                "figure_source_tables": len(source_tables),
                "source_registry_rows": len(_source_registry_rows(repository)),
            },
            "outputs": {str(path.relative_to(repository)): file_sha256(path) for path in outputs},
        },
    )
    print(f"stage21_run_id: {run_id}")
    print(f"status: {status}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

"""Record Stage 18 reproduction and revalidate Stages 19--21 without recomputation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuit_families.analysis.downstream_finalization import (
    read_last_json_object,
    stage20_training_projection,
    validate_stage18_comparison,
    verify_manifest_outputs,
)
from circuit_families.analysis.stage18_scaling import stable_json
from circuit_families.training import file_sha256

STAGE18_RUN_ID = "stage18-scaling-24a9adb84176"
STAGE18_IMPLEMENTATION_COMMIT = "2edaa427d7da142191a484b69949d72406fe51ac"
STAGE18_MANIFEST = Path(f"manifests/stage18_scaling_{STAGE18_RUN_ID}.json")
STAGE18_TRAINING = Path("manifests/stage18_training.json")
COMPARISON_MANIFEST = Path(f"manifests/stage18_reproduction_comparison_{STAGE18_RUN_ID}.json")
STAGE18_NOTE = Path("results/notes/stage18_independent_reproduction.md")
DOWNSTREAM_NOTE = Path("results/notes/stages19_21_reproduction_revalidation.md")

STAGES = {
    19: {
        "run_id": "stage19-matched-02b89bd79ed8",
        "manifest": Path("manifests/stage19_matched_stage19-matched-02b89bd79ed8.json"),
        "finalization": Path("manifests/stage19_finalization_stage19-matched-02b89bd79ed8.json"),
        "finalized_field": "stage19_finalized",
        "thresholds": {
            "input_cells": 630,
            "empty_cells": 330,
            "matched_fidelity_comparisons": 105,
            "matched_sparsity_comparisons": 105,
            "pareto_frontier_rows": 178,
        },
    },
    20: {
        "run_id": "stage20-paired-a70dc23368a6",
        "manifest": Path("manifests/stage20_paired_stage20-paired-a70dc23368a6.json"),
        "finalization": Path("manifests/stage20_finalization_stage20-paired-a70dc23368a6.json"),
        "finalized_field": "stage20_finalized",
        "thresholds": {
            "seed_checkpoint_rows": 35,
            "paired_delta_rows": 595,
            "seed_level_summary_rows": 119,
            "trajectory_figures": 2,
            "trajectory_figure_files": 4,
        },
    },
    21: {
        "run_id": "stage21-figures-e50a93433996",
        "manifest": Path("manifests/stage21_figures_stage21-figures-e50a93433996.json"),
        "finalization": Path("manifests/stage21_finalization_stage21-figures-e50a93433996.json"),
        "finalized_field": "stage21_finalized",
        "thresholds": {
            "principal_figures": 5,
            "figure_files": 10,
            "figure_source_tables": 5,
            "source_registry_rows": 16,
        },
    },
}


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_main(repository: Path) -> str:
    if _git(repository, "branch", "--show-current") != "main":
        raise ValueError("Finalization requires branch main.")
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        if subprocess.run(("git", *arguments), cwd=repository, check=False).returncode:
            raise ValueError("Finalization requires a clean tracked repository.")
    return _git(repository, "rev-parse", "HEAD")


def _semantic_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finalization_id(
    stage: int, run_id: str, manifest_sha: str, comparison_sha: str, commit: str
) -> str:
    value = f"stage{stage}|{run_id}|{manifest_sha}|{comparison_sha}|{commit}|revalidation-v1"
    return f"stage{stage}-finalization-{hashlib.sha256(value.encode('ascii')).hexdigest()[:12]}"


def _verify_stage(
    repository: Path, stage: int, specification: dict[str, Any]
) -> tuple[dict[str, Any], str, dict[str, str]]:
    path = repository / specification["manifest"]
    manifest = _object(path)
    if manifest.get(f"stage{stage}_run_id") != specification["run_id"]:
        raise ValueError(f"Stage {stage} run ID mismatch.")
    if manifest.get("status") != "provisionally_validated":
        raise ValueError(f"Stage {stage} source manifest must be provisional.")
    if manifest.get(specification["finalized_field"]) is not False:
        raise ValueError(f"Stage {stage} source manifest must not already be finalized.")
    if manifest.get("stage18_reproduction_status") != "pending":
        raise ValueError(f"Stage {stage} source manifest must record pending reproduction.")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError(f"Stage {stage} counts are missing.")
    for key, expected in specification["thresholds"].items():
        if counts.get(key) != expected:
            raise ValueError(
                f"Stage {stage} threshold {key} is {counts.get(key)!r}; expected {expected}."
            )
    return manifest, file_sha256(path), verify_manifest_outputs(repository, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize Stages 19--21 after independent Stage 18 reproduction."
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--reproduction-root", type=Path, required=True)
    parser.add_argument("--comparison-log", type=Path, required=True)
    parser.add_argument(
        "--comparison-primary-commit",
        default="c81ae4e77a7dd21f6e1c538495f56a89d86f1761",
    )
    parser.add_argument(
        "--comparison-reproduction-commit",
        default="3e8652582e207bd1b552d01521bdb9179bc31b98",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository_root.resolve()
    reproduction = args.reproduction_root.resolve()
    comparison_log = args.comparison_log.resolve()
    implementation_commit = _require_clean_main(repository)
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", args.comparison_primary_commit, "HEAD"),
        cwd=repository,
        check=False,
    ).returncode:
        raise ValueError("The primary comparison commit is not an ancestor of current HEAD.")
    if _git(reproduction, "rev-parse", "HEAD") != args.comparison_reproduction_commit:
        raise ValueError("Reproduction repository commit changed after comparison.")

    comparison = read_last_json_object(comparison_log)
    validate_stage18_comparison(comparison, run_id=STAGE18_RUN_ID)
    primary_manifest_path = repository / STAGE18_MANIFEST
    reproduction_manifest_path = reproduction / STAGE18_MANIFEST
    primary_manifest = _object(primary_manifest_path)
    reproduction_manifest = _object(reproduction_manifest_path)
    for name, manifest in (
        ("primary", primary_manifest),
        ("reproduction", reproduction_manifest),
    ):
        if manifest.get("stage18_run_id") != STAGE18_RUN_ID:
            raise ValueError(f"{name} Stage 18 manifest run ID mismatch.")
        if manifest.get("implementation_commit") != STAGE18_IMPLEMENTATION_COMMIT:
            raise ValueError(f"{name} Stage 18 implementation commit mismatch.")

    primary_training = _object(repository / STAGE18_TRAINING)
    reproduction_training = _object(reproduction / STAGE18_TRAINING)
    primary_projection = stage20_training_projection(primary_training)
    reproduction_projection = stage20_training_projection(reproduction_training)
    if primary_projection != reproduction_projection:
        raise ValueError("The Stage 20-consumed training fields differ in the reproduction.")
    training_projection_sha = _semantic_sha(primary_projection)

    stage_records: dict[int, tuple[dict[str, Any], str, dict[str, str]]] = {}
    for stage, specification in STAGES.items():
        stage_records[stage] = _verify_stage(repository, stage, specification)
    stage19_hash = stage_records[19][1]
    if stage_records[20][0]["upstream_stage19"]["manifest_sha256"] != stage19_hash:
        raise ValueError("Stage 20 does not point to the verified Stage 19 manifest.")
    stage20_hash = stage_records[20][1]
    if stage_records[21][0]["upstream_stage20"]["manifest_sha256"] != stage20_hash:
        raise ValueError("Stage 21 does not point to the verified Stage 20 manifest.")

    output_paths = [repository / COMPARISON_MANIFEST, repository / STAGE18_NOTE]
    output_paths.extend(repository / spec["finalization"] for spec in STAGES.values())
    output_paths.append(repository / DOWNSTREAM_NOTE)
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Finalization outputs already exist: " + ", ".join(map(str, existing))
        )

    created = datetime.now(UTC).isoformat()
    comparison_record = {
        "schema_version": 1,
        "experiment_stage": 18,
        "record_type": "independent_reproduction_comparison",
        "creation_timestamp_utc": created,
        "stage18_run_id": STAGE18_RUN_ID,
        "status": "independently_reproduced",
        "stage18_reproduction_status": "passed",
        "passed": True,
        "implementation_commit": STAGE18_IMPLEMENTATION_COMMIT,
        "comparison_primary_commit": args.comparison_primary_commit,
        "comparison_reproduction_commit": args.comparison_reproduction_commit,
        "primary": {
            "manifest_path": str(STAGE18_MANIFEST),
            "manifest_sha256": file_sha256(primary_manifest_path),
            "scientific_output_commit": _git(
                repository, "log", "-1", "--format=%H", "--", str(STAGE18_MANIFEST)
            ),
        },
        "reproduction": {
            "repository": str(reproduction),
            "manifest_path": str(STAGE18_MANIFEST),
            "manifest_sha256": file_sha256(reproduction_manifest_path),
            "finalization_commit": "1e55dbfd09066cbdc116ae92db69746a89349ebd",
        },
        "comparison_log": {
            "path": str(comparison_log),
            "sha256": file_sha256(comparison_log),
        },
        "stage20_training_input_semantics": {
            "fields": [
                "model_seed",
                "first_ten_percent_test_step",
                "stable_post_step",
            ],
            "seed_count": 5,
            "primary_and_reproduction_semantic_sha256": training_projection_sha,
            "equal": True,
        },
        **comparison,
    }
    stable_json(repository / COMPARISON_MANIFEST, comparison_record)
    comparison_sha = file_sha256(repository / COMPARISON_MANIFEST)

    finalization_hashes: dict[int, str] = {}
    for stage, specification in STAGES.items():
        manifest, manifest_sha, verified_outputs = stage_records[stage]
        finalization = {
            "schema_version": 1,
            "experiment_stage": stage,
            "record_type": "post_reproduction_revalidation",
            f"stage{stage}_run_id": specification["run_id"],
            f"stage{stage}_finalization_id": _finalization_id(
                stage,
                specification["run_id"],
                manifest_sha,
                comparison_sha,
                implementation_commit,
            ),
            "creation_timestamp_utc": created,
            "status": "finalized",
            specification["finalized_field"]: True,
            "passed": True,
            "implementation_commit": implementation_commit,
            "stage18_run_id": STAGE18_RUN_ID,
            "stage18_reproduction_status": "passed",
            "stage18_reproduction_comparison": {
                "path": str(COMPARISON_MANIFEST),
                "sha256": comparison_sha,
            },
            "provisional_source": {
                "run_id": specification["run_id"],
                "manifest_path": str(specification["manifest"]),
                "manifest_sha256": manifest_sha,
                "status_at_generation": manifest["status"],
                "note_retained_as_historical_generation_record": True,
            },
            "revalidation": {
                "method": (
                    "verify all declared output SHA-256 hashes after exact Stage 18 reproduction"
                ),
                "scientific_recomputation_required": False,
                "reason": (
                    "The independent reproduction matched the definitive Stage 18 scientific "
                    "outputs, and every saved downstream artifact remains byte-identical to its "
                    "audited provisional manifest."
                ),
                "verified_output_count": len(verified_outputs),
                "verified_outputs": verified_outputs,
            },
            "acceptance_thresholds": {
                key: {
                    "expected": expected,
                    "observed": manifest["counts"][key],
                    "passed": manifest["counts"][key] == expected,
                }
                for key, expected in specification["thresholds"].items()
            },
        }
        if stage > 19:
            upstream = stage - 1
            finalization["upstream_finalization"] = {
                "stage": upstream,
                "path": str(STAGES[upstream]["finalization"]),
                "sha256": finalization_hashes[upstream],
                "status": "finalized",
            }
        path = repository / specification["finalization"]
        stable_json(path, finalization)
        finalization_hashes[stage] = file_sha256(path)

    stage18_note = repository / STAGE18_NOTE
    stage18_note.parent.mkdir(parents=True, exist_ok=True)
    stage18_note.write_text(
        "# Stage 18 independent reproduction\n\n"
        "Status: `independently_reproduced`.\n\n"
        "The independent reproduction passed the finalized bidirectional comparison for "
        f"`{STAGE18_RUN_ID}`. All 818,386 scientific paths matched, all 35 normalized "
        "archive inventories matched, and the deterministic mismatch count was zero. The "
        "comparison additionally normalized 1,224 metadata-wrapper files under the recorded "
        "policy. Stage 20's consumed training-phase fields were compared separately and were "
        "identical for all five seeds.\n\n"
        f"Authoritative comparison record: `{COMPARISON_MANIFEST}`.\n",
        encoding="utf-8",
    )
    downstream_note = repository / DOWNSTREAM_NOTE
    downstream_note.write_text(
        "# Stages 19--21 post-reproduction revalidation\n\n"
        "Status: `finalized`.\n\n"
        "Stages 19, 20, and 21 were revalidated after the independent Stage 18 comparison "
        "passed. Recalculation was unnecessary: the reproduced Stage 18 scientific inputs "
        "match the definitive run, the exact Stage 20 training-phase inputs also match, and "
        "every declared downstream output still matches its audited SHA-256 hash. The original "
        "provisional notes are retained as honest generation-time records; the finalization "
        "manifests below are the current lifecycle authority.\n\n"
        "- Stage 19: finalized; 630 input cells, 330 empty cells, and 105 comparisons under "
        "each matching design verified.\n"
        "- Stage 20: finalized; 35 seed-checkpoint rows, 595 paired deltas, 119 seed summaries, "
        "and both trajectory figures verified.\n"
        "- Stage 21: finalized; five principal figures, ten figure files, five source tables, "
        "and the 16-row source registry verified.\n\n"
        + "\n".join(
            f"- Stage {stage} finalization: `{specification['finalization']}`"
            for stage, specification in STAGES.items()
        )
        + "\n",
        encoding="utf-8",
    )

    print("stage18_reproduction_status: passed")
    print("stage19_status: finalized")
    print("stage20_status: finalized")
    print("stage21_status: finalized")
    print(f"comparison_manifest: {repository / COMPARISON_MANIFEST}")


if __name__ == "__main__":
    main()

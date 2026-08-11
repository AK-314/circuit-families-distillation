"""Verify an independent reproduction of Stage 14."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from circuit_families.analysis.random_label_control import (
    MAIN_MODEL_REFERENCE_CHECKPOINTS,
)
from circuit_families.training.random_label import (
    CHECKPOINT_INTERVAL,
    FINAL_STEP,
    file_sha256,
)

TRAINING_MANIFEST_GLOB = "manifests/training_stage14-random-label-training-s0-*.json"
CHECKPOINT_SELECTION_MANIFEST = Path("manifests/stage14_random_label_checkpoints.json")
REPRODUCTION_MANIFEST = Path("manifests/stage14_random_label_reproduction.json")
TRAINING_METRICS_TABLE = Path("results/tables/seed_0_stage14_random_label_training_metrics.csv")
CHECKPOINT_TABLE = Path("results/tables/seed_0_stage14_random_label_checkpoints.csv")
MASKING_TABLE = Path("results/tables/seed_0_stage14_random_label_masking_validation.csv")

MASK_FILENAMES = (
    "all_retained.json",
    "all_ablated.json",
    "head_H0_ablated.json",
    "neuron_N0_ablated.json",
    "saved_arbitrary_reloaded.json",
)


def parse_args() -> argparse.Namespace:
    """Parse independent-reproduction arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare two complete Stage 14 executions and record "
            "the independent-reproduction result."
        )
    )
    parser.add_argument(
        "--primary-output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reproduction-output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--expected-implementation-commit",
        required=True,
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=REPRODUCTION_MANIFEST,
    )
    return parser.parse_args()


def load_json(file_path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    payload = json.loads(file_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {file_path}.")

    return payload


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialise JSON-safe content canonically."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_payload_sha256(payload: Any) -> str:
    """Hash canonical JSON-safe content."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def resolve_path(root: Path, file_path: Path) -> Path:
    """Resolve one output-root-relative path."""

    return file_path.resolve() if file_path.is_absolute() else (root / file_path).resolve()


def relative_path(file_path: Path, root: Path) -> str:
    """Return one output-root-relative path."""

    return str(file_path.resolve().relative_to(root.resolve()))


def require_clean_implementation(
    repository: Path,
    *,
    expected_commit: str,
) -> str:
    """Require the frozen implementation commit and clean repository."""

    head = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    if head != expected_commit:
        raise RuntimeError(
            f"Implementation commit mismatch: expected {expected_commit}, found {head}."
        )

    status = subprocess.run(
        [
            "/usr/bin/git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    if status:
        raise RuntimeError("Independent reproduction requires a clean repository.")

    return head


def require_distinct_output_roots(
    primary_root: Path,
    reproduction_root: Path,
) -> None:
    """Reject aliases between supposedly independent runs."""

    primary = primary_root.resolve()
    reproduction = reproduction_root.resolve()

    if primary == reproduction:
        raise ValueError("Primary and reproduction output roots must differ.")

    try:
        primary.relative_to(reproduction)
    except ValueError:
        pass
    else:
        raise ValueError("Primary output root may not be inside reproduction root.")

    try:
        reproduction.relative_to(primary)
    except ValueError:
        pass
    else:
        raise ValueError("Reproduction output root may not be inside primary root.")


def single_training_manifest(output_root: Path) -> Path:
    """Locate exactly one Stage 14 training manifest."""

    matches = sorted(output_root.glob(TRAINING_MANIFEST_GLOB))

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Stage 14 training manifest in {output_root}; found {matches}."
        )

    return matches[0].resolve()


def normalise_training_manifest(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Remove only the creation timestamp."""

    normalised = copy.deepcopy(manifest)

    if "timestamp_utc" not in normalised:
        raise ValueError("Training manifest is missing timestamp_utc.")

    del normalised["timestamp_utc"]
    return normalised


def normalise_selection_manifest(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Remove the hash induced only by the training timestamp."""

    normalised = copy.deepcopy(manifest)
    source_training = normalised.get("source_training_manifest")

    if not isinstance(source_training, dict):
        raise ValueError("Checkpoint-selection manifest lacks its source training-manifest record.")

    if "sha256" not in source_training:
        raise ValueError("Checkpoint-selection source training hash is missing.")

    del source_training["sha256"]
    return normalised


def checkpoint_hash_vectors(
    training_manifest: dict[str, Any],
) -> dict[str, list[Any]]:
    """Extract the complete checkpoint identity trajectory."""

    checkpoints = training_manifest.get("checkpoints")

    if not isinstance(checkpoints, list):
        raise ValueError("Training checkpoint records are missing.")

    expected_steps = list(range(0, FINAL_STEP + 1, CHECKPOINT_INTERVAL))
    observed_steps = [int(record["training_step"]) for record in checkpoints]

    if observed_steps != expected_steps:
        raise ValueError("Training checkpoint steps differ from the frozen schedule.")

    if len(checkpoints) != 182:
        raise ValueError("Training manifest must contain 182 checkpoints.")

    if not all(record.get("reload_verified") is True for record in checkpoints):
        raise ValueError("Every training checkpoint must be reload-verified.")

    return {
        "training_steps": observed_steps,
        "file_sha256": [record["file_sha256"] for record in checkpoints],
        "model_state_sha256": [record["model_state_sha256"] for record in checkpoints],
        "optimizer_state_sha256": [record["optimizer_state_sha256"] for record in checkpoints],
    }


def require_equal(
    primary: Any,
    reproduction: Any,
    description: str,
) -> None:
    """Require exact equality."""

    if primary != reproduction:
        raise ValueError(f"Independent reproduction mismatch for {description}.")


def compare_file_bytes(
    *,
    primary_root: Path,
    reproduction_root: Path,
    relative_file: Path,
) -> dict[str, Any]:
    """Require one artifact to be byte-identical."""

    primary = resolve_path(primary_root, relative_file)
    reproduction = resolve_path(
        reproduction_root,
        relative_file,
    )

    if not primary.is_file():
        raise FileNotFoundError(primary)

    if not reproduction.is_file():
        raise FileNotFoundError(reproduction)

    primary_sha256 = file_sha256(primary)
    reproduction_sha256 = file_sha256(reproduction)

    require_equal(
        primary_sha256,
        reproduction_sha256,
        str(relative_file),
    )

    return {
        "path": str(relative_file),
        "primary_sha256": primary_sha256,
        "reproduction_sha256": reproduction_sha256,
        "byte_identical": True,
    }


def selected_checkpoint_steps(
    selection_manifest: dict[str, Any],
) -> list[int]:
    """Return and validate the exact seven matched steps."""

    rows = selection_manifest.get("matched_checkpoints")

    if not isinstance(rows, list):
        raise ValueError("Matched checkpoint records are missing.")

    observed = [int(row["requested_step"]) for row in rows]
    expected = [step for _, step in MAIN_MODEL_REFERENCE_CHECKPOINTS]

    if observed != expected:
        raise ValueError("Matched checkpoint step grid differs from the frozen grid.")

    if any(
        int(row["selected_random_label_step"]) != int(row["requested_step"])
        or int(row["absolute_step_mismatch"]) != 0
        for row in rows
    ):
        raise ValueError("Checkpoint matching is not exact in training-step space.")

    return observed


def masking_directory(
    output_root: Path,
    stage14_run_id: str,
) -> Path:
    """Return the deterministic Stage 14 mask directory."""

    return (output_root / "results/raw" / f"{stage14_run_id}-masking" / "masks").resolve()


def write_json(
    file_path: Path,
    payload: dict[str, Any],
) -> Path:
    """Write deterministic JSON without a timestamp."""

    if file_path.exists():
        raise FileExistsError(f"Reproduction manifest already exists: {file_path}")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return file_path


def main() -> None:
    """Verify and record the independent reproduction."""

    args = parse_args()
    repository = args.repository_root.resolve()
    primary_root = args.primary_output_root.resolve()
    reproduction_root = args.reproduction_output_root.resolve()

    require_distinct_output_roots(
        primary_root,
        reproduction_root,
    )
    implementation_commit = require_clean_implementation(
        repository,
        expected_commit=args.expected_implementation_commit,
    )

    primary_training_path = single_training_manifest(primary_root)
    reproduction_training_path = single_training_manifest(reproduction_root)

    primary_training = load_json(primary_training_path)
    reproduction_training = load_json(reproduction_training_path)

    require_equal(
        primary_training["run_id"],
        reproduction_training["run_id"],
        "training run ID",
    )
    require_equal(
        primary_training["git_commit"],
        implementation_commit,
        "primary training implementation commit",
    )
    require_equal(
        reproduction_training["git_commit"],
        implementation_commit,
        "reproduction implementation commit",
    )
    require_equal(
        primary_training["configs"]["combined_sha256"],
        reproduction_training["configs"]["combined_sha256"],
        "combined configuration SHA-256",
    )
    require_equal(
        primary_training["hashes"]["metrics_jsonl_sha256"],
        reproduction_training["hashes"]["metrics_jsonl_sha256"],
        "raw metrics SHA-256",
    )
    require_equal(
        primary_training["final_metrics"],
        reproduction_training["final_metrics"],
        "final metrics",
    )

    primary_checkpoint_vectors = checkpoint_hash_vectors(primary_training)
    reproduction_checkpoint_vectors = checkpoint_hash_vectors(reproduction_training)

    require_equal(
        primary_checkpoint_vectors,
        reproduction_checkpoint_vectors,
        "complete 182-checkpoint identity trajectory",
    )

    primary_training_normalised = normalise_training_manifest(primary_training)
    reproduction_training_normalised = normalise_training_manifest(reproduction_training)

    require_equal(
        primary_training_normalised,
        reproduction_training_normalised,
        "normalised training manifests",
    )

    primary_selection_path = resolve_path(
        primary_root,
        CHECKPOINT_SELECTION_MANIFEST,
    )
    reproduction_selection_path = resolve_path(
        reproduction_root,
        CHECKPOINT_SELECTION_MANIFEST,
    )

    primary_selection = load_json(primary_selection_path)
    reproduction_selection = load_json(reproduction_selection_path)

    require_equal(
        primary_selection["stage14_run_id"],
        primary_training["run_id"],
        "primary selected run ID",
    )
    require_equal(
        reproduction_selection["stage14_run_id"],
        reproduction_training["run_id"],
        "reproduction selected run ID",
    )
    require_equal(
        selected_checkpoint_steps(primary_selection),
        selected_checkpoint_steps(reproduction_selection),
        "seven exact matched checkpoint steps",
    )
    require_equal(
        primary_selection["classification"],
        reproduction_selection["classification"],
        "control classification",
    )
    require_equal(
        normalise_selection_manifest(primary_selection),
        normalise_selection_manifest(reproduction_selection),
        "normalised checkpoint-selection manifests",
    )

    deterministic_files = (
        TRAINING_METRICS_TABLE,
        CHECKPOINT_TABLE,
        MASKING_TABLE,
    )

    artifact_comparisons = [
        compare_file_bytes(
            primary_root=primary_root,
            reproduction_root=reproduction_root,
            relative_file=relative_file,
        )
        for relative_file in deterministic_files
    ]

    run_id = str(primary_training["run_id"])
    primary_mask_directory = masking_directory(
        primary_root,
        run_id,
    )
    reproduction_mask_directory = masking_directory(
        reproduction_root,
        run_id,
    )

    mask_comparisons = []

    for filename in MASK_FILENAMES:
        primary_mask = primary_mask_directory / filename
        reproduction_mask = reproduction_mask_directory / filename

        if not primary_mask.is_file():
            raise FileNotFoundError(primary_mask)

        if not reproduction_mask.is_file():
            raise FileNotFoundError(reproduction_mask)

        primary_sha256 = file_sha256(primary_mask)
        reproduction_sha256 = file_sha256(reproduction_mask)

        require_equal(
            primary_sha256,
            reproduction_sha256,
            f"mask artifact {filename}",
        )

        mask_comparisons.append(
            {
                "path": (f"results/raw/{run_id}-masking/masks/{filename}"),
                "primary_sha256": primary_sha256,
                "reproduction_sha256": reproduction_sha256,
                "byte_identical": True,
            }
        )

    output_manifest = resolve_path(
        primary_root,
        args.output_manifest,
    )

    record = {
        "schema_version": 1,
        "stage": 14,
        "experiment_type": ("random_label_independent_reproduction"),
        "implementation_commit": implementation_commit,
        "stage14_run_id": run_id,
        "primary_output": {
            "training_manifest": {
                "path": relative_path(
                    primary_training_path,
                    primary_root,
                ),
                "sha256": file_sha256(primary_training_path),
                "normalised_sha256": (canonical_payload_sha256(primary_training_normalised)),
            },
            "checkpoint_selection_manifest": {
                "path": str(CHECKPOINT_SELECTION_MANIFEST),
                "sha256": file_sha256(primary_selection_path),
                "normalised_sha256": (
                    canonical_payload_sha256(normalise_selection_manifest(primary_selection))
                ),
            },
        },
        "reproduction_output": {
            "training_manifest": {
                "path": relative_path(
                    reproduction_training_path,
                    reproduction_root,
                ),
                "sha256": file_sha256(reproduction_training_path),
                "normalised_sha256": (canonical_payload_sha256(reproduction_training_normalised)),
            },
            "checkpoint_selection_manifest": {
                "path": str(CHECKPOINT_SELECTION_MANIFEST),
                "sha256": file_sha256(reproduction_selection_path),
                "normalised_sha256": (
                    canonical_payload_sha256(normalise_selection_manifest(reproduction_selection))
                ),
            },
        },
        "training_reproduction": {
            "run_id_identical": True,
            "combined_config_sha256_identical": True,
            "raw_metrics_sha256_identical": True,
            "final_metrics_identical": True,
            "normalised_training_manifest_identical": True,
            "checkpoint_count": 182,
            "checkpoint_steps_identical": True,
            "checkpoint_file_sha256_identical": True,
            "model_state_sha256_identical": True,
            "optimizer_state_sha256_identical": True,
            "all_checkpoints_reload_verified_in_both_runs": True,
        },
        "checkpoint_selection_reproduction": {
            "matched_checkpoint_count": 7,
            "matched_steps": selected_checkpoint_steps(primary_selection),
            "absolute_step_mismatch": 0,
            "classification_identical": True,
            "normalised_manifest_identical": True,
        },
        "byte_identical_artifacts": artifact_comparisons,
        "byte_identical_masks": mask_comparisons,
        "scientific_scope": {
            "random_label_training_reproduced": True,
            "exact_checkpoint_matching_reproduced": True,
            "masking_machinery_validation_reproduced": True,
            "random_label_sparse_search_started": False,
            "diversity_search_started": False,
            "stage15_started": False,
        },
        "reproduction_status": "passed",
    }

    write_json(output_manifest, record)

    print("===== STAGE 14 INDEPENDENT REPRODUCTION =====")
    print(f"implementation_commit: {implementation_commit}")
    print(f"stage14_run_id: {run_id}")
    print("training_runs_compared: 2")
    print("training_metric_records_per_run: 182")
    print("checkpoint_records_per_run: 182")
    print("all_checkpoint_file_hashes_identical: passed")
    print("all_model_state_hashes_identical: passed")
    print("all_optimizer_state_hashes_identical: passed")
    print("raw_metrics_bytes_identical: passed")
    print("normalised_training_manifests_identical: passed")
    print("matched_checkpoint_count: 7")
    print("matched_checkpoint_steps_identical: passed")
    print("classification_identical: passed")
    print("training_metrics_table_bytes_identical: passed")
    print("checkpoint_table_bytes_identical: passed")
    print("masking_table_bytes_identical: passed")
    print("saved_mask_bytes_identical: passed")
    print(f"reproduction_manifest: {output_manifest}")
    print(f"reproduction_manifest_sha256: {file_sha256(output_manifest)}")
    print("random_label_sparse_search_started: false")
    print("diversity_search_started: false")
    print("stage15_started: false")
    print("stage14_independent_reproduction: passed")


if __name__ == "__main__":
    main()

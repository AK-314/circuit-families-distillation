#!/usr/bin/env python3
"""Run frozen Stage 11 matched-size random-mask calibration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuit_families.analysis.fidelity_calibration import (
    CANDIDATE_THRESHOLDS,
    RANDOM_MASK_EVALUATION_COLUMNS,
    RANDOM_MASKS_PER_THRESHOLD,
    RUNTIME_COLUMNS,
    SOURCE_TRAINING_RUN_ID,
    STABLE_POST_CHECKPOINT_STEP,
    THRESHOLD_CALIBRATION_COLUMNS,
    derive_random_seed,
    file_sha256,
    load_calibration_source_records,
    minimum_agreement_count,
    random_mask_evaluation_record,
    sample_matched_size_masks,
    sampled_mask_to_component_mask,
    select_primary_threshold,
    summarise_threshold_evaluations,
    threshold_decimal,
    write_csv_records,
    write_deterministic_tar_gz,
)
from circuit_families.interpretability.fidelity import (
    compute_full_model_reference,
    evaluate_component_mask,
    load_checkpoint_evaluation_context,
)
from circuit_families.manifests import package_versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic Stage 11 matched-size random-mask calibration."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage8-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage9-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage9-table",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage9-archive",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage10-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument("--expected-implementation-commit")
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
    )
    return parser.parse_args()


def git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_clean_repository(repository: Path) -> str:
    status = git_output(repository, "status", "--short")
    if status:
        raise RuntimeError(
            "Stage 11 scientific outputs require a clean implementation "
            "commit. Current status:\n" + status
        )
    return git_output(repository, "rev-parse", "HEAD")


def relative_path(repository: Path, file_path: Path) -> str:
    return str(file_path.resolve().relative_to(repository))


def stable_json_write(file_path: Path, value: Any) -> Path:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return file_path


def build_output_paths(
    repository: Path,
    run_id: str,
) -> dict[str, Path]:
    return {
        "raw_directory": repository / "results" / "raw" / run_id,
        "evaluation_table": (
            repository
            / "results"
            / "tables"
            / "seed_1_stage11_random_mask_evaluations.csv"
        ),
        "calibration_table": (
            repository
            / "results"
            / "tables"
            / "seed_1_stage11_threshold_calibration.csv"
        ),
        "runtime_table": (
            repository
            / "results"
            / "tables"
            / "seed_1_stage11_random_mask_runtime.csv"
        ),
        "record": (
            repository
            / "results"
            / "notes"
            / "seed_1_stage11_threshold_calibration.md"
        ),
        "archive": (
            repository / "results" / "archives" / f"{run_id}.tar.gz"
        ),
        "manifest": (
            repository / "manifests" / f"stage11_calibration_{run_id}.json"
        ),
    }


def validate_absent_outputs(outputs: dict[str, Path]) -> None:
    existing = [
        file_path
        for file_path in outputs.values()
        if file_path.exists()
    ]
    if existing:
        rendered = "\n".join(str(item) for item in existing)
        raise FileExistsError(
            "Stage 11 refuses to overwrite existing outputs:\n" + rendered
        )


def main() -> None:
    args = parse_args()
    repository = args.repository_root.resolve()

    stage8_manifest_file = repository / args.stage8_manifest
    if not stage8_manifest_file.is_file():
        raise FileNotFoundError(
            f"Stage 8 manifest does not exist: {stage8_manifest_file}"
        )

    sources = load_calibration_source_records(
        stage9_manifest_path=repository / args.stage9_manifest,
        stage9_table_path=repository / args.stage9_table,
        stage9_archive_path=repository / args.stage9_archive,
        stage10_manifest_path=repository / args.stage10_manifest,
    )

    if args.validate_inputs_only:
        print(f"stage9_run_id: {sources.stage9_run_id}")
        print(f"stage10_run_id: {sources.stage10_run_id}")
        print(f"candidate_count: {len(sources.circuits)}")
        print(f"random_masks_per_threshold: {RANDOM_MASKS_PER_THRESHOLD}")
        print(
            "planned_random_mask_evaluations: "
            f"{len(sources.circuits) * RANDOM_MASKS_PER_THRESHOLD}"
        )
        print(
            "retained_components: "
            + ", ".join(
                str(record.retained_components)
                for record in sources.circuits
            )
        )

        for rank, circuit in enumerate(sources.circuits, start=1):
            seed = derive_random_seed(circuit.threshold)
            print(
                "candidate: "
                f"rank={rank} "
                f"threshold={seed.threshold_decimal} "
                f"retained_components={circuit.retained_components} "
                f"stage9_exact_evaluations={circuit.exact_evaluations} "
                f"minimum_agreement_count="
                f"{minimum_agreement_count(circuit.threshold)}"
            )
            print(f"seed_material: {seed.seed_material}")
            print(f"seed_digest: {seed.seed_digest}")
            print(f"seed_uint64: {seed.seed_uint64}")
            print(f"bit_generator: {seed.bit_generator}")
            print(f"numpy_version: {seed.numpy_version}")

        print("input_validation: passed")
        return

    implementation_commit = require_clean_repository(repository)
    if (
        args.expected_implementation_commit
        and implementation_commit != args.expected_implementation_commit
    ):
        raise RuntimeError(
            "implementation commit mismatch: expected "
            f"{args.expected_implementation_commit}, found "
            f"{implementation_commit}"
        )

    outputs = build_output_paths(repository, args.run_id)
    validate_absent_outputs(outputs)

    context = load_checkpoint_evaluation_context(
        repository_root=repository,
        run_id=SOURCE_TRAINING_RUN_ID,
        checkpoint_manifest_path=args.checkpoint_manifest,
        checkpoint_step=STABLE_POST_CHECKPOINT_STEP,
        device_override=args.device,
    )
    full_reference = compute_full_model_reference(
        context.model,
        context.inputs,
        context.targets,
        batch_size=args.evaluation_batch_size,
    )

    from circuit_families.training import canonical_state_hash

    model_state_before = canonical_state_hash(context.model.state_dict())
    hook_counts_before = {
        "blocks.0.attn.hook_z": len(
            context.model.blocks[0].attn.hook_z.fwd_hooks
        ),
        "blocks.0.mlp.hook_post": len(
            context.model.blocks[0].mlp.hook_post.fwd_hooks
        ),
    }

    raw_directory = outputs["raw_directory"]
    evaluation_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    candidates = []

    fourier_by_threshold = {
        record.threshold: record for record in sources.fourier_records
    }

    try:
        for descending_rank, circuit in enumerate(
            sources.circuits,
            start=1,
        ):
            started = time.perf_counter()
            sampled_masks = sample_matched_size_masks(
                circuit.threshold,
                retained_count=circuit.retained_components,
            )
            threshold_rows: list[dict[str, object]] = []
            threshold_directory = (
                raw_directory
                / f"threshold_{threshold_decimal(circuit.threshold)}"
            )

            for sampled in sampled_masks:
                mask = sampled_mask_to_component_mask(sampled)
                mask_file = (
                    threshold_directory
                    / "masks"
                    / f"mask_{sampled.mask_index:03d}.json"
                )
                metrics = evaluate_component_mask(
                    context.model,
                    context.inputs,
                    context.targets,
                    mask,
                    batch_size=args.evaluation_batch_size,
                    full_model_reference=full_reference,
                )
                row = random_mask_evaluation_record(
                    stage11_run_id=args.run_id,
                    circuit=circuit,
                    sampled=sampled,
                    metrics=metrics,
                )

                raw_mask_record = {
                    "schema_version": 1,
                    "stage11_run_id": args.run_id,
                    "source_training_run_id": SOURCE_TRAINING_RUN_ID,
                    "checkpoint_step": STABLE_POST_CHECKPOINT_STEP,
                    "fidelity_threshold": float(circuit.threshold),
                    "mask_index": sampled.mask_index,
                    "sampling": {
                        "seed_material": row["seed_material"],
                        "seed_digest": row["seed_digest"],
                        "seed_uint64": row["seed_uint64"],
                        "bit_generator": row["bit_generator"],
                        "numpy_version": row["numpy_version"],
                        "definition": row["sampling_definition"],
                    },
                    "mask": mask.to_record(),
                    "metrics": metrics.to_record(),
                    "passes_candidate_threshold": row[
                        "passes_candidate_threshold"
                    ],
                }
                stable_json_write(mask_file, raw_mask_record)
                threshold_rows.append(row)
                evaluation_rows.append(row)

            candidate, calibration_row = summarise_threshold_evaluations(
                stage11_run_id=args.run_id,
                descending_evaluation_rank=descending_rank,
                circuit=circuit,
                fourier_record=fourier_by_threshold[circuit.threshold],
                evaluation_rows=threshold_rows,
                masks=sampled_masks,
            )
            candidates.append(candidate)
            calibration_rows.append(calibration_row)

            elapsed = time.perf_counter() - started
            runtime_rows.append(
                {
                    "stage11_run_id": args.run_id,
                    "fidelity_threshold": float(circuit.threshold),
                    "mask_count": RANDOM_MASKS_PER_THRESHOLD,
                    "elapsed_seconds": elapsed,
                    "seconds_per_mask": elapsed / RANDOM_MASKS_PER_THRESHOLD,
                    "included_in_deterministic_scientific_hashes": False,
                }
            )

        selected, qualification_results = select_primary_threshold(candidates)

        model_state_after = canonical_state_hash(context.model.state_dict())
        hook_counts_after = {
            "blocks.0.attn.hook_z": len(
                context.model.blocks[0].attn.hook_z.fwd_hooks
            ),
            "blocks.0.mlp.hook_post": len(
                context.model.blocks[0].mlp.hook_post.fwd_hooks
            ),
        }
        if model_state_after != model_state_before:
            raise RuntimeError(
                "model state changed during Stage 11 calibration"
            )
        if hook_counts_after != hook_counts_before:
            raise RuntimeError(
                "hook counts changed during Stage 11 calibration"
            )

        for row, qualification in zip(
            calibration_rows,
            qualification_results,
            strict=True,
        ):
            row["selected_primary_threshold"] = (
                selected is not None
                and float(qualification.threshold) == float(selected)
            )

        write_csv_records(
            outputs["evaluation_table"],
            fieldnames=RANDOM_MASK_EVALUATION_COLUMNS,
            rows=evaluation_rows,
        )
        write_csv_records(
            outputs["calibration_table"],
            fieldnames=THRESHOLD_CALIBRATION_COLUMNS,
            rows=calibration_rows,
        )
        write_csv_records(
            outputs["runtime_table"],
            fieldnames=RUNTIME_COLUMNS,
            rows=runtime_rows,
        )

        selected_text = (
            threshold_decimal(selected)
            if selected is not None
            else "none"
        )
        note_lines = [
            "# Stage 11 threshold calibration",
            "",
            f"- Stage 11 run ID: `{args.run_id}`",
            f"- Source training run: `{SOURCE_TRAINING_RUN_ID}`",
            f"- Stable-post checkpoint: `{STABLE_POST_CHECKPOINT_STEP}`",
            f"- Random masks evaluated: `{len(evaluation_rows)}`",
            "- Random masks per threshold: `100`",
            "- Sampling: uniform subset without replacement over all 516 "
            "searchable components; no head/neuron stratification",
            "- Primary fidelity: exact prediction agreement with the full model",
            "- Random-control pass rule: exact integer comparison to each "
            "candidate threshold",
            "- Qualification: retained components <=258; random passes <=5; "
            "Stage 10 compatible or explained; Stage 9 exact evaluations "
            "<=10000",
            f"- Selected primary threshold: `{selected_text}`",
            "",
            "No pre-grokking, transition, diversity-family, control-task, "
            "across-seed, Stage 12, or hypothesis-effect outcomes entered "
            "the selection.",
            "",
        ]
        outputs["record"].parent.mkdir(parents=True, exist_ok=True)
        outputs["record"].write_text(
            "\n".join(note_lines),
            encoding="utf-8",
        )

        write_deterministic_tar_gz(
            source_directory=raw_directory,
            archive_path=outputs["archive"],
        )

        deterministic_outputs = {
            name: {
                "path": relative_path(repository, outputs[name]),
                "sha256": file_sha256(outputs[name]),
            }
            for name in (
                "evaluation_table",
                "calibration_table",
                "record",
                "archive",
            )
        }
        runtime_output = {
            "path": relative_path(repository, outputs["runtime_table"]),
            "sha256": file_sha256(outputs["runtime_table"]),
            "included_in_deterministic_scientific_hashes": False,
        }

        manifest = {
            "schema_version": 1,
            "experiment_type": "stage11_fidelity_threshold_calibration",
            "stage11_run_id": args.run_id,
            "creation_timestamp_utc": datetime.now(UTC).isoformat(),
            "stage11_implementation_git_commit": implementation_commit,
            "source_training_run_id": SOURCE_TRAINING_RUN_ID,
            "checkpoint": {
                "training_step": context.checkpoint_step,
                "phase": context.checkpoint_phase,
                "path": relative_path(repository, context.checkpoint_path),
                "checkpoint_sha256": context.checkpoint_sha256,
                "model_state_sha256": context.model_state_sha256,
            },
            "source_manifests": {
                "checkpoint_selection": {
                    "path": str(args.checkpoint_manifest),
                    "sha256": file_sha256(
                        repository / args.checkpoint_manifest
                    ),
                },
                "stage8_masking": {
                    "path": str(args.stage8_manifest),
                    "sha256": file_sha256(
                        repository / args.stage8_manifest
                    ),
                },
                "stage9_sparse_search": {
                    "manifest_path": str(args.stage9_manifest),
                    "manifest_sha256": file_sha256(
                        repository / args.stage9_manifest
                    ),
                    "table_path": str(args.stage9_table),
                    "table_sha256": file_sha256(
                        repository / args.stage9_table
                    ),
                    "archive_path": str(args.stage9_archive),
                    "archive_sha256": file_sha256(
                        repository / args.stage9_archive
                    ),
                    "source_masks": [
                        {
                            "fidelity_threshold": float(circuit.threshold),
                            "archive_member": circuit.final_mask_member,
                            "sha256": circuit.final_mask_sha256,
                            "mask_id": circuit.mask.mask_id,
                        }
                        for circuit in sources.circuits
                    ],
                },
                "stage10_fourier": {
                    "path": str(args.stage10_manifest),
                    "sha256": file_sha256(repository / args.stage10_manifest),
                },
            },
            "configuration": {
                "candidate_thresholds": [
                    float(value) for value in CANDIDATE_THRESHOLDS
                ],
                "candidate_execution_order": [
                    float(value) for value in CANDIDATE_THRESHOLDS
                ],
                "random_masks_per_threshold": RANDOM_MASKS_PER_THRESHOLD,
                "component_universe_size": 516,
                "sampling": (
                    "uniform subset without replacement over all searchable "
                    "components with no head/neuron stratification"
                ),
                "bit_generator": "PCG64",
                "seed_derivation": (
                    "SHA-256 of the frozen threshold-specific seed material; "
                    "first 16 hexadecimal characters interpreted as uint64"
                ),
                "evaluation_batch_size": args.evaluation_batch_size,
                "device": str(context.device),
                "evaluated_example_count": len(context.targets),
                "example_ordering": context.example_ordering,
                "percentile_method": "linear",
            },
            "selection": {
                "mechanical_rule": (
                    "select the highest threshold satisfying all four frozen "
                    "qualification criteria"
                ),
                "selected_primary_threshold": (
                    None if selected is None else float(selected)
                ),
                "qualification_results": [
                    {
                        **asdict(result),
                        "threshold": float(result.threshold),
                    }
                    for result in qualification_results
                ],
                "prohibited_evidence_excluded": True,
            },
            "integrity": {
                "clean_implementation_commit_verified": True,
                "scientific_outputs_generated_from_exact_commit": (
                    implementation_commit
                ),
                "model_state_sha256_before": model_state_before,
                "model_state_sha256_after": model_state_after,
                "model_state_unchanged": (
                    model_state_before == model_state_after
                ),
                "hook_counts_before": hook_counts_before,
                "hook_counts_after": hook_counts_after,
                "hook_counts_unchanged": (
                    hook_counts_before == hook_counts_after
                ),
                "stage12_started": False,
                "runtime_excluded_from_deterministic_scientific_hashes": True,
            },
            "outputs": {
                **deterministic_outputs,
                "runtime_table": runtime_output,
            },
            "software": {
                "python": sys.version,
                "packages": package_versions(),
            },
        }
        stable_json_write(outputs["manifest"], manifest)

    except Exception:
        if raw_directory.exists():
            shutil.rmtree(raw_directory)
        raise

    print(f"stage11_run_id: {args.run_id}")
    print(f"implementation_git_commit: {implementation_commit}")
    print(f"random_mask_evaluations: {len(evaluation_rows)}")
    print(
        "selected_primary_threshold: "
        + (
            threshold_decimal(selected)
            if selected is not None
            else "none"
        )
    )
    print(f"manifest: {relative_path(repository, outputs['manifest'])}")


if __name__ == "__main__":
    main()

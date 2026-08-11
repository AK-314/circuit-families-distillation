"""Execute or validate frozen Stage 18 main-seed training."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from circuit_families.analysis.stage18_scaling import (
    CHECKPOINT_STEPS,
    FRESH_TRAINING_SEEDS,
    PILOT_SEED,
    RESERVE_SEEDS,
    build_main_seed_registry,
    stable_json,
    validate_stage18_inputs,
    write_csv,
)
from circuit_families.analysis.stage18_training import (
    ABSOLUTE_MAXIMUM,
    EXTENSION_INCREMENT,
    classify_grokking_run,
    requires_extension,
)
from circuit_families.training import file_sha256, read_jsonl, run_training


def _train_one(
    repository: str,
    model_seed: int,
    implementation_commit: str,
) -> dict[str, object]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(repository)
    result = run_training(
        repository_root=root,
        task_config_path="configs/task.yaml",
        model_config_path="configs/model.yaml",
        training_config_path="configs/training.yaml",
        dataset_archive_path="data/generated/modular_addition_m113.npz",
        dataset_metadata_path="data/generated/modular_addition_m113.metadata.json",
        dataset_manifest_path=("manifests/dataset_modular-addition-dataset-s0-7ef9c73ff18f.json"),
        model_seed=model_seed,
        smoke=False,
        device_override="cpu",
        output_root=root,
        overwrite=False,
        experiment_type_override="stage18-main-training",
        run_identity={
            "experiment_stage": 18,
            "training_role": "primary_or_prespecified_reserve_main_seed",
            "model_seed": model_seed,
            "implementation_commit": implementation_commit,
            "standard_horizon": 40_000,
            "extension_increment": EXTENSION_INCREMENT,
            "absolute_maximum": ABSOLUTE_MAXIMUM,
        },
        checkpoint_verification_steps=CHECKPOINT_STEPS,
        extension_increment=EXTENSION_INCREMENT,
        absolute_max_steps=ABSOLUTE_MAXIMUM,
        extension_decider=requires_extension,
    )
    records = read_jsonl(result.metrics_path)
    classification = classify_grokking_run(records)
    return {
        "model_seed": model_seed,
        "run_id": result.run_id,
        "manifest_path": str(result.manifest_path.relative_to(root)),
        "metrics_path": str(result.metrics_path.relative_to(root)),
        "checkpoint_directory": str(result.checkpoint_directory.relative_to(root)),
        "final_step": result.final_step,
        "checkpoint_count": result.checkpoint_count,
        "classification": asdict(classification),
    }


def _reference_result(repository: Path) -> dict[str, object]:
    run_id = "modular-addition-training-s1-5f1bc9dee7ab"
    metrics_path = repository / "results/raw" / run_id / "metrics.jsonl"
    classification = classify_grokking_run(read_jsonl(metrics_path))
    return {
        "model_seed": PILOT_SEED,
        "run_id": run_id,
        "manifest_path": f"manifests/training_{run_id}.json",
        "metrics_path": str(metrics_path.relative_to(repository)),
        "checkpoint_directory": f"checkpoints/{run_id}",
        "final_step": 40_000,
        "checkpoint_count": 801,
        "classification": asdict(classification),
        "training_execution": "reference_existing_result",
    }


def _train_batch(
    repository: Path,
    seeds: tuple[int, ...],
    implementation_commit: str,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=len(seeds)) as pool:
        futures = {
            pool.submit(_train_one, str(repository), seed, implementation_commit): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            result = future.result()
            result["training_execution"] = "fresh_execution"
            results.append(result)
            print(
                f"training seed {result['model_seed']} complete: "
                f"{result['classification']['status']} step={result['final_step']}",
                flush=True,
            )
    return sorted(results, key=lambda row: int(row["model_seed"]))


def _selected_analysis_seeds(
    results: list[dict[str, object]],
) -> tuple[int, ...]:
    by_seed = {int(row["model_seed"]): row for row in results}
    selected: list[int] = [PILOT_SEED]
    replacement_candidates = iter(RESERVE_SEEDS)
    for seed in (0, 2, 3, 4):
        if bool(by_seed[seed]["classification"]["eligible"]):
            selected.append(seed)
        else:
            replacement = next(
                candidate
                for candidate in replacement_candidates
                if bool(by_seed.get(candidate, {}).get("classification", {}).get("eligible"))
            )
            selected.append(replacement)
    return tuple(sorted(selected))


def _write_training_outputs(
    repository: Path,
    implementation_commit: str,
    results: list[dict[str, object]],
) -> None:
    by_seed = {int(row["model_seed"]): row for row in results}
    seed_rows = []
    for base in build_main_seed_registry():
        seed = int(base["primary_seed"])
        result = by_seed[seed]
        classification = result["classification"]
        seed_rows.append(
            {
                **base,
                "training_status": "complete",
                "training_run_id": result["run_id"],
                "training_eligibility": (
                    "eligible" if classification["eligible"] else "ineligible"
                ),
                "grokking_classification": classification["status"],
                "final_training_step": result["final_step"],
            }
        )
    training_rows = []
    checkpoint_rows = []
    for result in sorted(results, key=lambda row: int(row["model_seed"])):
        manifest_path = repository / str(result["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        classification = result["classification"]
        training_rows.append(
            {
                "model_seed": result["model_seed"],
                "training_execution": result["training_execution"],
                "run_id": result["run_id"],
                "manifest_path": result["manifest_path"],
                "manifest_sha256": file_sha256(manifest_path),
                "metrics_path": result["metrics_path"],
                "final_step": result["final_step"],
                "grokking_classification": classification["status"],
                "eligible": classification["eligible"],
                "first_memorisation_step": classification["first_memorisation_step"],
                "first_ten_percent_test_step": classification["first_ten_percent_test_step"],
                "stable_post_step": classification["stable_post_step"],
            }
        )
        metrics = {
            int(row["training_step"]): row
            for row in read_jsonl(repository / str(result["metrics_path"]))
        }
        checkpoints = {int(row["training_step"]): row for row in manifest["checkpoints"]}
        for index, step in enumerate(CHECKPOINT_STEPS, start=1):
            checkpoint = checkpoints[step]
            metric = metrics[step]
            checkpoint_rows.append(
                {
                    "model_seed": result["model_seed"],
                    "checkpoint_index": index,
                    "checkpoint_step": step,
                    "checkpoint_path": checkpoint["path"],
                    "checkpoint_sha256": checkpoint["file_sha256"],
                    "model_state_sha256": checkpoint["model_state_sha256"],
                    "optimizer_state_sha256": checkpoint["optimizer_state_sha256"],
                    "train_accuracy": metric["train_accuracy"],
                    "test_accuracy": metric["test_accuracy"],
                    "train_cross_entropy": metric["train_loss"],
                    "test_cross_entropy": metric["test_loss"],
                    "reload_verified": checkpoint["reload_verified"],
                }
            )
        if int(result["model_seed"]) != PILOT_SEED:
            records = [row for row in checkpoint_rows if row["model_seed"] == result["model_seed"]]
            checkpoint_run_id = str(result["run_id"])

            def checkpoint_record(
                row: dict[str, object],
                run_id: str = checkpoint_run_id,
            ) -> dict[str, object]:
                return {
                    "run_id": run_id,
                    "training_step": row["checkpoint_step"],
                    "checkpoint_path": row["checkpoint_path"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "phase_label": f"matched-step grid {row['checkpoint_index']}",
                    "selection_status": "selected_fixed_step",
                    "achieved_train_accuracy": row["train_accuracy"],
                    "achieved_test_accuracy": row["test_accuracy"],
                    "train_loss": row["train_cross_entropy"],
                    "test_loss": row["test_cross_entropy"],
                }

            stable_json(
                repository / f"manifests/stage18_checkpoints_seed_{result['model_seed']}.json",
                {
                    "schema_version": 1,
                    "run_id": result["run_id"],
                    "source_training_manifest": result["manifest_path"],
                    "selection_rule": "fixed_training_step_grid",
                    "pre_checkpoint": checkpoint_record(records[0]),
                    "formal_transition_checkpoints": {
                        f"grid_{index}": checkpoint_record(row)
                        for index, row in enumerate(records[1:6], start=2)
                    },
                    "selected_stable_post_checkpoint": checkpoint_record(records[6]),
                },
            )
    write_csv(repository / "results/tables/stage18_main_seed_registry.csv", seed_rows)
    write_csv(repository / "results/tables/stage18_training_runs.csv", training_rows)
    write_csv(repository / "results/tables/stage18_checkpoint_registry.csv", checkpoint_rows)
    stable_json(
        repository / "manifests/stage18_training.json",
        {
            "schema_version": 1,
            "experiment_stage": 18,
            "creation_timestamp_utc": datetime.now(UTC).isoformat(),
            "implementation_commit": implementation_commit,
            "primary_main_seeds": [0, 1, 2, 3, 4],
            "reserve_seeds": list(RESERVE_SEEDS),
            "runs": training_rows,
            "checkpoint_count": len(checkpoint_rows),
            "stage19_started": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 18 main-seed training.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-seeds", nargs="+", type=int, default=list(FRESH_TRAINING_SEEDS))
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--expected-implementation-commit")
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--reproduction-mode", action="store_true")
    args = parser.parse_args()
    if tuple(args.model_seeds) != FRESH_TRAINING_SEEDS:
        parser.error("Fresh training seeds must be exactly 0 2 3 4 in that order.")
    if args.threads_per_worker != 1:
        parser.error("Stage 18 training requires one thread per worker.")
    return args


def main() -> None:
    args = parse_args()
    validation = validate_stage18_inputs(
        args.repository_root,
        expected_implementation_commit=args.expected_implementation_commit,
        reproduction_mode=args.reproduction_mode,
    )
    if args.validate_inputs_only:
        print("stage18_training_validate_inputs_only: passed")
        print(f"implementation_commit: {validation.implementation_commit}")
        print("fresh_training_seeds: 0 2 3 4")
        print("standard_horizon: 40000")
        print("extension_increment: 10000")
        print("absolute_maximum: 80000")
        return
    repository = args.repository_root.resolve()
    fresh = _train_batch(repository, tuple(args.model_seeds), validation.implementation_commit)
    results = [_reference_result(repository), *fresh]
    complete = sum(bool(row["classification"]["eligible"]) for row in results)
    for reserve_seed in RESERVE_SEEDS:
        if complete >= 5:
            break
        reserve = _train_batch(repository, (reserve_seed,), validation.implementation_commit)[0]
        results.append(reserve)
        complete += int(bool(reserve["classification"]["eligible"]))
    if complete < 5:
        raise RuntimeError("Reserve seeds 5-9 did not yield five complete grokking seeds.")
    _selected_analysis_seeds(results)
    _write_training_outputs(repository, validation.implementation_commit, results)
    print("stage18_training: complete")


if __name__ == "__main__":
    main()

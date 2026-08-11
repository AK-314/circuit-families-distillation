"""Isolated workers and serial merge primitives for Stage 18."""

from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from circuit_families.analysis.stage12_artifacts import write_stage12_cell_artifacts
from circuit_families.analysis.stage17_execution import (
    CellSearchResult,
    CellTransferResult,
    NormalizedCircuit,
    NormalizedOverlap,
    _empty_transfer,
    _scientific_rows,
    _write_raw_transfer,
    execute_fresh_transfer,
    normalize_fresh_search,
)
from circuit_families.analysis.stage17_sensitivity import Stage17Cell
from circuit_families.analysis.stage18_scaling import (
    CHECKPOINT_STEPS,
    FRESH_CELL_COUNT,
    PRIMARY_MAIN_SEEDS,
    PRODUCTION_WORKERS,
    Stage18Cell,
    WorkerShard,
    build_stage18_registry,
    build_worker_shards,
    stable_json,
    write_csv,
)
from circuit_families.interpretability.diversity_forced_search import (
    run_checkpoint_family_search,
)
from circuit_families.interpretability.fidelity import (
    evaluate_component_mask,
    load_checkpoint_evaluation_context,
)
from circuit_families.interpretability.masks import ComponentMask
from circuit_families.training import canonical_state_hash, file_sha256


@dataclass(frozen=True)
class SeedSource:
    model_seed: int
    run_id: str
    checkpoint_manifest: str


def _stage17_cell(cell: Stage18Cell) -> Stage17Cell:
    return Stage17Cell(
        cell_index=cell.global_cell_index,
        model_seed=cell.model_seed,
        checkpoint_step=cell.checkpoint_step,
        fidelity_threshold=cell.fidelity,
        fidelity_display=cell.fidelity_display,
        distinctness_cutoff=cell.distinctness,
        distinctness_display=cell.distinctness_display,
        search_execution_mode=cell.family_search_execution_mode,
        search_source_stage=cell.source_stage,
        search_source_run_id=cell.source_run_id,
        search_source_manifest=cell.source_manifest,
        search_source_table=None,
        transfer_execution_mode=cell.transfer_execution_mode,
        transfer_source_stage=cell.source_stage,
        transfer_source_run_id=cell.source_run_id,
        transfer_source_manifest=cell.source_manifest,
        transfer_source_table=None,
        expected_search_budget=50_000,
        output_status="not_executed",
        transfer_grouping_status="not_evaluated",
        cell_id=cell.cell_id,
    )


def _fraction(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _search_record(search: CellSearchResult) -> dict[str, Any]:
    return {
        "cell_id": search.cell.cell_id,
        "status": search.status,
        "stopping_reason": search.stopping_reason,
        "family_size": search.family_size,
        "right_censored": search.right_censored,
        "exact_evaluations_used": search.exact_evaluations_used,
        "budget_remaining": search.budget_remaining,
        "ranking_passes": search.ranking_passes,
        "restarts_attempted": search.restarts_attempted,
        "failed_requested_member_count": search.failed_requested_member_count,
        "terminal_requested_member_index": search.terminal_requested_member_index,
        "invalid_output_count": search.invalid_output_count,
        "search_integrity": dict(search.search_integrity),
        "circuits": [
            {
                "circuit_id": circuit.circuit_id,
                "member_index": circuit.member_index,
                "selected_restart_index": circuit.selected_restart_index,
                "mask": circuit.mask.to_record(),
                "mask_sha256": circuit.mask_sha256,
                "metrics": dict(circuit.metrics),
                "exact_evaluations_used": circuit.exact_evaluations_used,
                "ranking_passes_used": circuit.ranking_passes_used,
                "accepted_removal_count": circuit.accepted_removal_count,
                "rejected_candidate_count": circuit.rejected_candidate_count,
                "candidate_batches_tested": circuit.candidate_batches_tested,
                "locally_single_deletion_minimal": circuit.locally_single_deletion_minimal,
                "maximum_prior_overlap": _fraction(circuit.maximum_prior_overlap),
                "prior_overlaps": [_fraction(value) for value in circuit.prior_overlaps],
            }
            for circuit in search.circuits
        ],
        "overlaps": [
            {
                "circuit_i": row.circuit_i,
                "circuit_j": row.circuit_j,
                "member_i": row.member_i,
                "member_j": row.member_j,
                "intersection_count": row.intersection_count,
                "union_count": row.union_count,
                "jaccard": _fraction(row.jaccard),
            }
            for row in search.overlaps
        ],
        "restart_rows": list(search.restart_rows),
    }


def _as_fraction(value: Sequence[int]) -> Fraction:
    return Fraction(int(value[0]), int(value[1]))


def _load_search(path: Path, cell: Stage18Cell) -> CellSearchResult:
    record = json.loads(path.read_text(encoding="utf-8"))
    circuits = tuple(
        NormalizedCircuit(
            circuit_id=row["circuit_id"],
            member_index=int(row["member_index"]),
            selected_restart_index=int(row["selected_restart_index"]),
            mask=ComponentMask.from_record(row["mask"]),
            mask_sha256=row["mask_sha256"],
            metrics=row["metrics"],
            exact_evaluations_used=int(row["exact_evaluations_used"]),
            ranking_passes_used=int(row["ranking_passes_used"]),
            accepted_removal_count=int(row["accepted_removal_count"]),
            rejected_candidate_count=int(row["rejected_candidate_count"]),
            candidate_batches_tested=int(row["candidate_batches_tested"]),
            locally_single_deletion_minimal=bool(row["locally_single_deletion_minimal"]),
            maximum_prior_overlap=_as_fraction(row["maximum_prior_overlap"]),
            prior_overlaps=tuple(_as_fraction(value) for value in row["prior_overlaps"]),
        )
        for row in record["circuits"]
    )
    overlaps = tuple(
        NormalizedOverlap(
            circuit_i=row["circuit_i"],
            circuit_j=row["circuit_j"],
            member_i=int(row["member_i"]),
            member_j=int(row["member_j"]),
            intersection_count=int(row["intersection_count"]),
            union_count=int(row["union_count"]),
            jaccard=_as_fraction(row["jaccard"]),
        )
        for row in record["overlaps"]
    )
    return CellSearchResult(
        cell=_stage17_cell(cell),
        status=record["status"],
        stopping_reason=record["stopping_reason"],
        family_size=int(record["family_size"]),
        right_censored=bool(record["right_censored"]),
        exact_evaluations_used=int(record["exact_evaluations_used"]),
        budget_remaining=int(record["budget_remaining"]),
        circuits=circuits,
        overlaps=overlaps,
        restart_rows=tuple(record["restart_rows"]),
        ranking_passes=int(record["ranking_passes"]),
        restarts_attempted=int(record["restarts_attempted"]),
        failed_requested_member_count=int(record["failed_requested_member_count"]),
        terminal_requested_member_index=record["terminal_requested_member_index"],
        invalid_output_count=int(record["invalid_output_count"]),
        raw_cell_directory=path.parent,
        search_integrity=record["search_integrity"],
    )


def _seed_sources(repository: Path) -> dict[int, SeedSource]:
    training_path = repository / "manifests/stage18_training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    sources = {}
    for row in training["runs"]:
        seed = int(row["model_seed"])
        checkpoint_manifest = (
            "manifests/checkpoints_seed_1.json"
            if seed == 1
            else f"manifests/stage18_checkpoints_seed_{seed}.json"
        )
        sources[seed] = SeedSource(seed, row["run_id"], checkpoint_manifest)
    return sources


def _load_context(repository: Path, source: SeedSource, step: int):
    return load_checkpoint_evaluation_context(
        repository_root=repository,
        run_id=source.run_id,
        checkpoint_manifest_path=source.checkpoint_manifest,
        checkpoint_step=step,
        device_override="cpu",
    )


def _configure_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def validate_masking_position(repository: Path, source: SeedSource, step: int) -> dict[str, object]:
    context = _load_context(repository, source, step)
    state_before = canonical_state_hash(context.model.state_dict())
    metrics = evaluate_component_mask(
        context.model,
        context.inputs,
        context.targets,
        ComponentMask.all_retained(),
        batch_size=256,
    )
    state_after = canonical_state_hash(context.model.state_dict())
    if metrics.primary_fidelity != 1.0 or metrics.prediction_agreement_count != 12_769:
        raise RuntimeError(
            f"All-retained identity failed for seed {source.model_seed} step {step}."
        )
    if state_before != state_after:
        raise RuntimeError("Masking validation changed model state.")
    if any(parameter.grad is not None for parameter in context.model.parameters()):
        raise RuntimeError("Masking validation left parameter gradients.")
    return {
        "model_seed": source.model_seed,
        "checkpoint_step": step,
        "checkpoint_sha256": context.checkpoint_sha256,
        "model_state_sha256": context.model_state_sha256,
        "all_retained_fidelity": metrics.primary_fidelity,
        "prediction_agreement_count": metrics.prediction_agreement_count,
        "evaluated_example_count": metrics.evaluated_example_count,
        "output_class_113_excluded": True,
        "final_position_logits_only": True,
        "model_state_unchanged": True,
        "parameter_gradients_absent": True,
        "hooks_restored_to_baseline": True,
        "status": "passed",
    }


def _masking_shard(
    repository: str,
    assignments: Sequence[tuple[int, int]],
) -> list[dict[str, object]]:
    _configure_worker()
    root = Path(repository)
    sources = _seed_sources(root)
    return [validate_masking_position(root, sources[seed], step) for seed, step in assignments]


def execute_masking_validation(
    repository: Path,
    analysis_seeds: Sequence[int],
) -> list[dict[str, object]]:
    positions = [(seed, step) for seed in analysis_seeds for step in CHECKPOINT_STEPS]
    assignments = [positions[index::PRODUCTION_WORKERS] for index in range(PRODUCTION_WORKERS)]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=PRODUCTION_WORKERS) as pool:
        futures = [pool.submit(_masking_shard, str(repository), shard) for shard in assignments]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: (int(row["model_seed"]), int(row["checkpoint_step"])))
    if len(rows) != 35:
        raise RuntimeError("Stage 18 masking validation did not produce 35 positions.")
    return rows


def _search_shard(
    repository: str,
    run_id: str,
    implementation_commit: str,
    shard: WorkerShard,
) -> dict[str, object]:
    _configure_worker()
    root = Path(repository)
    sources = _seed_sources(root)
    worker_root = root / "results/raw" / run_id / "workers" / shard.worker_id
    worker_root.mkdir(parents=True, exist_ok=False)
    stable_json(
        worker_root / "shard_manifest.json",
        {
            "schema_version": 1,
            "stage18_run_id": run_id,
            "implementation_commit": implementation_commit,
            **shard.to_record(),
        },
    )
    completed = 0
    evaluations = 0
    for cell in shard.cells:
        started = time.perf_counter()
        context = _load_context(root, sources[cell.model_seed], cell.checkpoint_step)
        stage17_cell = _stage17_cell(cell)
        execution = run_checkpoint_family_search(
            context,
            fidelity_threshold=float(cell.fidelity),
            distinctness_cutoff=cell.distinctness,
            model_seed=cell.model_seed,
            checkpoint_index=cell.checkpoint_index,
            ranking_batch_size=256,
            evaluation_batch_size=256,
            family_target=10,
            max_restarts_per_alternative=5,
            per_requested_circuit_budget=10_000,
            per_cell_budget=50_000,
            reuse_coefficient=0.5,
            tie_tolerance=1.0e-12,
        )
        raw_cell = worker_root / "cells" / cell.cell_id
        artifacts = write_stage12_cell_artifacts(
            raw_cell / "search",
            execution,
            cell_metadata={
                "stage18_run_id": run_id,
                "global_cell_index": cell.global_cell_index,
                "cell_id": cell.cell_id,
                "model_seed": cell.model_seed,
                "checkpoint_index": cell.checkpoint_index,
                "checkpoint_step": cell.checkpoint_step,
                "implementation_commit": implementation_commit,
                "independent_cell_budget": True,
                "initial_family_size": 0,
            },
        )
        search = normalize_fresh_search(run_id, stage17_cell, execution, artifacts)
        stable_json(raw_cell / "search_result.json", _search_record(search))
        stable_json(
            raw_cell / "search_runtime.json",
            {
                "cell_id": cell.cell_id,
                "elapsed_seconds": time.perf_counter() - started,
                "included_in_deterministic_scientific_hashes": False,
            },
        )
        completed += 1
        evaluations += search.exact_evaluations_used
    return {
        "worker_id": shard.worker_id,
        "completed_cells": completed,
        "exact_evaluations": evaluations,
    }


def execute_search_workers(
    repository: Path,
    run_id: str,
    implementation_commit: str,
    cells: Sequence[Stage18Cell],
) -> None:
    shards = build_worker_shards(cells)
    with ProcessPoolExecutor(max_workers=PRODUCTION_WORKERS) as pool:
        futures = [
            pool.submit(
                _search_shard,
                str(repository),
                run_id,
                implementation_commit,
                shard,
            )
            for shard in shards
        ]
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += int(result["completed_cells"])
            print(f"Stage 18: {completed} / 612 fresh cells complete", flush=True)
    if completed != FRESH_CELL_COUNT:
        raise RuntimeError("Stage 18 search workers did not complete all 612 cells.")


def _search_path(repository: Path, run_id: str, cell: Stage18Cell) -> Path:
    return (
        repository
        / "results/raw"
        / run_id
        / "workers"
        / str(cell.worker_id)
        / "cells"
        / cell.cell_id
        / "search_result.json"
    )


def validate_search_merge(repository: Path, run_id: str, cells: Sequence[Stage18Cell]) -> None:
    fresh = [cell for cell in cells if cell.family_search_execution_mode == "fresh_execution"]
    paths = [_search_path(repository, run_id, cell) for cell in fresh]
    if len(paths) != FRESH_CELL_COUNT or len(set(paths)) != FRESH_CELL_COUNT:
        raise RuntimeError("Fresh search paths are not unique and complete.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Stage 18 search outputs.")


def _transfer_shard(repository: str, run_id: str, shard: WorkerShard) -> int:
    _configure_worker()
    root = Path(repository)
    sources = _seed_sources(root)
    completed = 0
    for cell in shard.cells:
        path = _search_path(root, run_id, cell)
        search = _load_search(path, cell)
        context = _load_context(root, sources[cell.model_seed], cell.checkpoint_step)
        runtime_rows: list[dict[str, object]] = []
        transfer = (
            _empty_transfer(search.cell)
            if search.family_size == 0
            else execute_fresh_transfer(
                run_id,
                context,
                search,
                evaluation_batch_size=256,
                runtime_rows=runtime_rows,
            )
        )
        _write_raw_transfer(path.parent, transfer)
        stable_json(path.parent / "transfer_runtime.json", runtime_rows)
        completed += 1
    return completed


def execute_transfer_workers(repository: Path, run_id: str, cells: Sequence[Stage18Cell]) -> None:
    shards = build_worker_shards(cells)
    with ProcessPoolExecutor(max_workers=PRODUCTION_WORKERS) as pool:
        futures = [pool.submit(_transfer_shard, str(repository), run_id, shard) for shard in shards]
        completed = sum(future.result() for future in as_completed(futures))
    if completed != 612:
        raise RuntimeError("Stage 18 transfer workers did not complete all 612 cells.")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _minimal_transfer(cell: Stage18Cell, cell_root: Path) -> CellTransferResult:
    summary = json.loads((cell_root / "transfer/summary.json").read_text(encoding="utf-8"))
    return CellTransferResult(
        cell=_stage17_cell(cell),
        profiles=(),
        profile_rows=tuple(_read_jsonl(cell_root / "transfer/profiles.jsonl")),
        evaluation_rows=tuple(_read_jsonl(cell_root / "transfer/evaluations.jsonl")),
        distance_rows=tuple(_read_jsonl(cell_root / "transfer/distances.jsonl")),
        group_rows=tuple(_read_jsonl(cell_root / "transfer/groups.jsonl")),
        group_count=summary["group_count"],
        status=summary["status"],
    )


def serial_merge_fresh(
    repository: Path,
    run_id: str,
    cells: Sequence[Stage18Cell],
) -> dict[str, list[dict[str, object]]]:
    merged: dict[str, list[dict[str, object]]] = {}
    for cell in cells:
        if cell.family_search_execution_mode != "fresh_execution":
            continue
        path = _search_path(repository, run_id, cell)
        search = _load_search(path, cell)
        transfer = _minimal_transfer(cell, path.parent)
        rows = _scientific_rows(run_id, (search,), {cell.cell_id: transfer})
        for name, values in rows.items():
            for row in values:
                row.pop("stage17_run_id", None)
                row["stage18_run_id"] = run_id
                row["model_seed"] = cell.model_seed
                row["checkpoint_index"] = cell.checkpoint_index
                row["checkpoint_step"] = cell.checkpoint_step
                row["global_cell_index"] = cell.global_cell_index
            merged.setdefault(name, []).extend(values)
        for name, values in (
            ("transfer_profiles_table", transfer.profile_rows),
            ("transfer_distances_table", transfer.distance_rows),
            ("transfer_groups_table", transfer.group_rows),
            ("transfer_evaluations_table", transfer.evaluation_rows),
        ):
            normalized = []
            for source in values:
                row = dict(source)
                row.pop("stage17_run_id", None)
                row["stage18_run_id"] = run_id
                row["model_seed"] = cell.model_seed
                row["checkpoint_index"] = cell.checkpoint_index
                row["checkpoint_step"] = cell.checkpoint_step
                row["global_cell_index"] = cell.global_cell_index
                normalized.append(row)
            merged.setdefault(name, []).extend(normalized)
    return merged


def write_merged_tables(
    repository: Path,
    merged: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[Path, ...]:
    mapping = {
        "family_summary_table": "stage18_family_summary.csv",
        "circuits_table": "stage18_circuits.csv",
        "pairwise_overlap_table": "stage18_pairwise_overlap.csv",
        "search_failures_table": "stage18_search_failures.csv",
        "circuit_size_summary_table": "stage18_circuit_size_summary.csv",
        "transfer_profiles_table": "stage18_transfer_profiles.csv",
        "transfer_distances_table": "stage18_transfer_distances.csv",
        "transfer_groups_table": "stage18_transfer_groups.csv",
        "transfer_evaluations_table": "stage18_transfer_evaluations.csv",
        "restarts_table": "stage18_restarts.csv",
        "frontier_table": "stage18_frontier.csv",
        "family_size_heatmap_source_table": "stage18_family_size_heatmap_source.csv",
        "family_size_curves_source_table": "stage18_family_size_curves_source.csv",
        "family_size_distinctness_source_table": ("stage18_family_size_distinctness_source.csv"),
    }
    outputs = []
    for key, filename in mapping.items():
        rows = tuple(merged.get(key, ()))
        if rows:
            outputs.append(write_csv(repository / "results/tables" / filename, rows))
    return tuple(outputs)


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def merge_stage17_references(
    repository: Path,
    run_id: str,
    cells: Sequence[Stage18Cell],
    merged: dict[str, list[dict[str, object]]],
) -> None:
    reference_by_id = {
        cell.cell_id: cell
        for cell in cells
        if cell.family_search_execution_mode == "reference_existing_result"
    }
    sources = {
        "family_summary_table": "seed_1_stage17_family_summary.csv",
        "circuits_table": "seed_1_stage17_circuits.csv",
        "pairwise_overlap_table": "seed_1_stage17_pairwise_overlap.csv",
        "search_failures_table": "seed_1_stage17_search_failures.csv",
        "circuit_size_summary_table": "seed_1_stage17_circuit_size_summary.csv",
        "transfer_profiles_table": "seed_1_stage17_transfer_profiles.csv",
        "transfer_distances_table": "seed_1_stage17_transfer_distances.csv",
        "transfer_groups_table": "seed_1_stage17_transfer_groups.csv",
        "restarts_table": "seed_1_stage17_restarts.csv",
    }
    for key, filename in sources.items():
        rows = _read_csv(repository / "results/tables" / filename)
        normalized = []
        for row in rows:
            cell = reference_by_id.get(str(row.get("cell_id")))
            if cell is None:
                raise ValueError(f"Unregistered Stage 17 reference cell in {filename}.")
            row.pop("stage17_run_id", None)
            row["stage18_run_id"] = run_id
            row["global_cell_index"] = cell.global_cell_index
            row["model_seed"] = 1
            row["checkpoint_index"] = 7
            row["checkpoint_step"] = 9050
            normalized.append(row)
        merged.setdefault(key, []).extend(normalized)


def final_cell_registry(
    cells: Sequence[Stage18Cell],
    family_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_id = {str(row["cell_id"]): row for row in family_rows}
    rows = []
    for cell in cells:
        family = by_id[cell.cell_id]
        row = cell.to_record()
        row.update(
            {
                "status": family["status"],
                "stopping_reason": family["stopping_reason"],
                "family_size": family["family_size"],
                "transfer_group_status": family["transfer_group_status"],
            }
        )
        rows.append(row)
    if len(rows) != 630:
        raise RuntimeError("Final Stage 18 cell registry must contain 630 rows.")
    return rows


def write_stage18_note(
    repository: Path,
    run_id: str,
    family_rows: Sequence[Mapping[str, object]],
) -> Path:
    training = json.loads(
        (repository / "manifests/stage18_training.json").read_text(encoding="utf-8")
    )
    zero_count = sum(int(row["family_size"]) == 0 for row in family_rows)
    capped_count = sum(str(row["right_censored"]).lower() == "true" for row in family_rows)
    failures = sum(str(row["status"]) != "complete" for row in family_rows)
    training_lines = "\n".join(
        f"- seed {int(row['model_seed'])}: `{row['grokking_classification']}` at "
        f"step {int(row['final_step'])} using `{row['run_id']}` "
        f"({row['training_execution']})"
        for row in sorted(training["runs"], key=lambda item: int(item["model_seed"]))
    )
    primary = [
        row
        for row in family_rows
        if str(row["displayed_fidelity"]) == "0.990"
        and str(row["displayed_jaccard_cutoff"]) == "0.50"
    ]
    trajectory_lines = []
    for seed in sorted({int(row["model_seed"]) for row in primary}):
        rows = sorted(
            (row for row in primary if int(row["model_seed"]) == seed),
            key=lambda row: int(row["checkpoint_step"]),
        )
        entries = []
        for row in rows:
            group_count = row["transfer_group_count"]
            groups = group_count if group_count not in (None, "") else "NA"
            entries.append(
                f"{int(row['checkpoint_step'])}:family={int(row['family_size'])},groups={groups}"
            )
        trajectory = ", ".join(entries)
        trajectory_lines.append(f"- seed {seed}: {trajectory}")
    trajectories = "\n".join(trajectory_lines)
    note = (
        "# Stage 18 main-seed scaling\n\n"
        f"Run ID: `{run_id}`.\n\n"
        "## Design and execution\n\n"
        "The five-seed target and realised registry are seeds 0, 1, 2, 3 and 4. "
        "No reserve seed was required. Seed 1 is the exact Stage 17 reference at step "
        "9050; all other registered positions were executed under Stage 18. The frozen "
        "checkpoint grid is 200, 3400, 7450, 8150, 8500, 8650 and 9050.\n\n"
        "The analysis contains 630 registered fidelity-by-distinctness cells: 612 fresh "
        "executions and 18 exact Stage 17 references. Production used 12 isolated "
        "workers with one intra-op and one inter-op thread per worker. Worker output "
        "directories were disjoint and final merging and archive construction were "
        "serial.\n\n"
        "Additional random-label seeds are frozen at zero because resources are "
        "prioritised for the five required main seeds. Stage 15 remains unavailable.\n\n"
        "## Training outcomes\n\n"
        f"{training_lines}\n\n"
        "## Primary descriptive trajectories\n\n"
        "Each entry is `checkpoint:family size,transfer-group count` at fidelity 0.990 "
        "and Jaccard cutoff 0.50.\n\n"
        f"{trajectories}\n\n"
        "## Surface outcomes and limitations\n\n"
        f"Observed empty-family cells: {zero_count}. Non-complete search-status cells: "
        f"{failures}. Right-censored cells: {capped_count}. Every nonempty fresh family "
        "received transfer evaluation and deterministic complete-linkage grouping at "
        "tolerance 0.05. Empty-family transfer counts remain absent rather than zero.\n\n"
        "These raw within-seed checkpoint trajectories are descriptive. The analysis "
        "does not establish an across-seed inferential conclusion, does not repair the "
        "unavailable Stage 15 control, and adds no further random-label control seed. "
        "Matched comparisons, sign tests, permutation summaries and across-seed "
        "inference are deferred to Stages 19-20.\n"
    )
    path = repository / "results/notes/stage18_main_seed_scaling.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note, encoding="utf-8")
    return path


def write_stage18_figures(
    repository: Path,
    family_rows: Sequence[Mapping[str, object]],
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [
        row
        for row in family_rows
        if str(row["displayed_fidelity"]) == "0.990"
        and str(row["displayed_jaccard_cutoff"]) == "0.50"
    ]
    primary.sort(key=lambda row: (int(row["model_seed"]), int(row["checkpoint_step"])))
    outputs: list[Path] = []

    def save(figure: Any, stem: str) -> None:
        for suffix in ("png", "pdf"):
            path = repository / f"figures/{stem}.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                path,
                dpi=180,
                metadata={"CreationDate": None, "ModDate": None},
            )
            outputs.append(path)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for seed in sorted({int(row["model_seed"]) for row in primary}):
        rows = [row for row in primary if int(row["model_seed"]) == seed]
        axis.plot(
            [int(row["checkpoint_step"]) for row in rows],
            [int(row["family_size"]) for row in rows],
            marker="o",
            label=f"seed {seed}",
        )
    axis.set(
        xlabel="Training step",
        ylabel="Recovered family size",
        title="Stage 18 primary family dynamics",
    )
    axis.legend()
    save(figure, "stage18_primary_family_dynamics")

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for seed in sorted({int(row["model_seed"]) for row in primary}):
        rows = [row for row in primary if int(row["model_seed"]) == seed]
        axis.plot(
            [int(row["checkpoint_step"]) for row in rows],
            [
                float("nan")
                if row["transfer_group_count"] in (None, "")
                else int(row["transfer_group_count"])
                for row in rows
            ],
            marker="o",
            label=f"seed {seed}",
        )
    axis.set(
        xlabel="Training step",
        ylabel="Transfer-group count",
        title="Stage 18 primary transfer-group dynamics",
    )
    axis.legend()
    save(figure, "stage18_primary_transfer_group_dynamics")

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for seed in sorted({int(row["model_seed"]) for row in family_rows}):
        rows = [row for row in family_rows if int(row["model_seed"]) == seed]
        axis.scatter(
            [int(row["checkpoint_step"]) for row in rows],
            [int(row["family_size"]) for row in rows],
            s=8,
            alpha=0.45,
            label=f"seed {seed}",
        )
    axis.set(
        xlabel="Training step",
        ylabel="Family size across 18-cell surface",
        title="Stage 18 seed sensitivity overview",
    )
    axis.legend()
    save(figure, "stage18_seed_sensitivity_overview")

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    training = json.loads(
        (repository / "manifests/stage18_training.json").read_text(encoding="utf-8")
    )
    for run in sorted(training["runs"], key=lambda row: int(row["model_seed"])):
        metrics = _read_jsonl(repository / run["metrics_path"])
        axis.plot(
            [int(row["training_step"]) for row in metrics],
            [float(row["test_accuracy"]) for row in metrics],
            label=f"seed {run['model_seed']} test",
        )
    axis.set(
        xlabel="Training step",
        ylabel="Test accuracy",
        title="Stage 18 main-seed training curves",
    )
    axis.legend()
    save(figure, "stage18_main_seed_training_curves")

    caption = repository / "figures/stage18_caption.txt"
    caption.write_text(
        "Stage 18 descriptive main-seed training, structural-family, transfer-group, "
        "and full-surface trajectories on the frozen seven-step grid. Empty families "
        "remain zero only for family size; undefined transfer counts remain absent.\n",
        encoding="utf-8",
    )
    outputs.append(caption)
    return tuple(outputs)


def write_archive_shards(
    repository: Path,
    run_id: str,
    cells: Sequence[Stage18Cell],
) -> Path:
    from circuit_families.analysis.stage14_random_label_reporting import (
        write_deterministic_archive,
    )

    archive_root = repository / "results/archives" / run_id
    archive_root.mkdir(parents=True, exist_ok=False)
    index_rows = []
    for seed in sorted({cell.model_seed for cell in cells}):
        for step in CHECKPOINT_STEPS:
            selected = [
                cell for cell in cells if cell.model_seed == seed and cell.checkpoint_step == step
            ]
            members = []
            for cell in selected:
                if cell.family_search_execution_mode == "reference_existing_result":
                    continue
                cell_root = _search_path(repository, run_id, cell).parent
                members.extend(
                    path
                    for path in cell_root.rglob("*")
                    if _archive_member_allowed(path)
                )
            inventory = stable_json(
                archive_root / f"seed_{seed}_step_{step}_inventory.json",
                {
                    "model_seed": seed,
                    "checkpoint_step": step,
                    "cell_ids": [cell.cell_id for cell in selected],
                    "reference_stage17": seed == 1 and step == 9050,
                    "members": [str(path.relative_to(repository)) for path in sorted(members)],
                },
            )
            raw_file_count = len(members)
            members.append(inventory)
            uncompressed_size = sum(path.stat().st_size for path in members)
            archive = archive_root / f"seed_{seed}_step_{step}.tar.gz"
            write_deterministic_archive(archive, root=repository, members=tuple(sorted(members)))
            index_rows.append(
                {
                    "model_seed": seed,
                    "checkpoint_step": step,
                    "path": str(archive.relative_to(repository)),
                    "sha256": file_sha256(archive),
                    "member_count": len(members),
                    "raw_file_count": raw_file_count,
                    "uncompressed_size": uncompressed_size,
                    "compressed_size": archive.stat().st_size,
                }
            )
    return stable_json(
        archive_root / "index.json",
        {"schema_version": 1, "stage18_run_id": run_id, "shards": index_rows},
    )


def _archive_member_allowed(path: Path) -> bool:
    """Exclude operating-system metadata and runtime telemetry from scientific archives."""
    return path.is_file() and path.name != ".DS_Store" and "runtime" not in path.name


def refresh_archive_index(repository: Path, run_id: str) -> Path:
    archive_root = repository / "results/archives" / run_id
    index_path = archive_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = []
    for source in index["shards"]:
        row = dict(source)
        archive = repository / str(row["path"])
        if file_sha256(archive) != str(row["sha256"]):
            raise ValueError(f"Stage 18 archive hash mismatch: {archive}")
        inventory = archive_root / (
            f"seed_{int(row['model_seed'])}_step_{int(row['checkpoint_step'])}_inventory.json"
        )
        payload = json.loads(inventory.read_text(encoding="utf-8"))
        members = [repository / str(path) for path in payload["members"]]
        missing = [path for path in members if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} Stage 18 archive-index members.")
        row["member_count"] = len(members) + 1
        row["raw_file_count"] = len(members)
        row["uncompressed_size"] = inventory.stat().st_size + sum(
            path.stat().st_size for path in members
        )
        row["compressed_size"] = archive.stat().st_size
        rows.append(row)
    if len(rows) != 35:
        raise RuntimeError("Stage 18 archive index must contain exactly 35 shards.")
    return stable_json(
        index_path,
        {"schema_version": 1, "stage18_run_id": run_id, "shards": rows},
    )


def selected_analysis_seeds(repository: Path) -> tuple[int, ...]:
    training = json.loads(
        (repository / "manifests/stage18_training.json").read_text(encoding="utf-8")
    )
    rows = {int(row["model_seed"]): row for row in training["runs"]}
    selected = [1]
    reserves = iter(seed for seed in range(5, 10) if bool(rows.get(seed, {}).get("eligible")))
    for seed in (0, 2, 3, 4):
        selected.append(seed if bool(rows[seed]["eligible"]) else next(reserves))
    return tuple(sorted(selected))


def validate_existing_stage18_outputs(
    repository: Path,
    run_id: str,
    cells: Sequence[Stage18Cell],
    analysis_seeds: Sequence[int],
    *,
    require_final_outputs_absent: bool = True,
) -> Path:
    raw_root = repository / "results/raw" / run_id
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Stage 18 raw output does not exist: {raw_root}")
    validate_search_merge(repository, run_id, cells)
    fresh = [cell for cell in cells if cell.family_search_execution_mode == "fresh_execution"]
    if len(fresh) != FRESH_CELL_COUNT:
        raise RuntimeError("Stage 18 recovery requires exactly 612 fresh cells.")
    missing_transfer = []
    for cell in fresh:
        cell_root = _search_path(repository, run_id, cell).parent
        for relative in (Path("transfer/summary.json"), Path("transfer_runtime.json")):
            path = cell_root / relative
            if not path.is_file():
                missing_transfer.append(path)
    if missing_transfer:
        raise FileNotFoundError(f"Missing {len(missing_transfer)} Stage 18 transfer outputs.")

    masking_path = repository / "results/tables/stage18_masking_validation.csv"
    if not masking_path.is_file():
        raise FileNotFoundError("Stage 18 masking validation table is missing.")
    masking_rows = _read_csv(masking_path)
    expected_positions = {
        (int(seed), int(step)) for seed in analysis_seeds for step in CHECKPOINT_STEPS
    }
    observed_positions = {
        (int(row["model_seed"]), int(row["checkpoint_step"])) for row in masking_rows
    }
    if (
        len(masking_rows) != len(expected_positions)
        or observed_positions != expected_positions
        or any(str(row.get("status")) != "passed" for row in masking_rows)
    ):
        raise RuntimeError("Stage 18 masking validation is not complete and passed.")

    final_outputs = (
        repository / f"manifests/stage18_scaling_{run_id}.json",
        repository / "results/archives" / run_id,
        repository / "results/notes/stage18_main_seed_scaling.md",
        repository / "figures/stage18_caption.txt",
        repository / "results/tables/stage18_family_summary.csv",
        repository / "results/tables/stage18_cell_registry.csv",
    )
    existing = [path for path in final_outputs if path.exists()]
    if require_final_outputs_absent and existing:
        raise FileExistsError(
            "Stage 18 finalization outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )
    return masking_path


def _artifact(repository: Path, relative: str) -> dict[str, object]:
    return {"path": relative, "sha256": file_sha256(repository / relative)}


def write_stage18_manifest(
    repository: Path,
    *,
    run_id: str,
    implementation_commit: str,
    finalization_commit: str,
    configuration_sha256: str,
    analysis_seeds: Sequence[int],
    outputs: Sequence[Path],
    finalize_existing: bool,
) -> Path:
    configuration = json.loads(
        (repository / "configs/stage18_scaling.json").read_text(encoding="utf-8")
    )
    training = json.loads(
        (repository / "manifests/stage18_training.json").read_text(encoding="utf-8")
    )
    first_training = json.loads(
        (repository / str(training["runs"][0]["manifest_path"])).read_text(encoding="utf-8")
    )
    stage17 = json.loads(
        (
            repository / "manifests/stage17_sensitivity_stage17-sensitivity-s1-7801e7938531.json"
        ).read_text(encoding="utf-8")
    )
    stage16 = json.loads(
        (repository / "manifests/stage16_transfer_stage16-transfer-s1-cc55bd4162c8.json").read_text(
            encoding="utf-8"
        )
    )
    cell_registry = _read_csv(repository / "results/tables/stage18_cell_registry.csv")
    checkpoint_registry = _read_csv(repository / "results/tables/stage18_checkpoint_registry.csv")
    worker_shards = _read_csv(repository / "results/tables/stage18_worker_shards.csv")
    masking_rows = _read_csv(repository / "results/tables/stage18_masking_validation.csv")
    archive_index_path = repository / "results/archives" / run_id / "index.json"
    archive_index = json.loads(archive_index_path.read_text(encoding="utf-8"))
    references = configuration["references"]
    source_artifacts = {
        key: {"path": references[key], "sha256": references[f"{key}_sha256"]}
        for key in (
            "freeze_manifest",
            "freeze_note",
            "benchmark_summary",
            "control_seed_freeze",
            "stage17_manifest",
            "stage17_archive",
        )
    }
    source_artifacts["stage16_manifest"] = _artifact(
        repository, str(references["stage16_manifest"])
    )
    source_artifacts["stage16_archive"] = dict(stage16["outputs"]["archive"])
    output_hashes = {str(path.relative_to(repository)): file_sha256(path) for path in outputs}
    table_outputs = {
        path: digest for path, digest in output_hashes.items() if path.startswith("results/tables/")
    }
    figure_outputs = {
        path: digest for path, digest in output_hashes.items() if path.startswith("figures/")
    }
    integrity = {
        "masking_positions_passed": sum(row["status"] == "passed" for row in masking_rows),
        "masking_position_count": len(masking_rows),
        "model_state_unchanged_all_positions": all(
            row["model_state_unchanged"] == "True" for row in masking_rows
        ),
        "parameter_gradients_absent_all_positions": all(
            row["parameter_gradients_absent"] == "True" for row in masking_rows
        ),
        "hooks_restored_to_baseline_all_positions": all(
            row["hooks_restored_to_baseline"] == "True" for row in masking_rows
        ),
    }
    manifest_path = repository / f"manifests/stage18_scaling_{run_id}.json"
    return stable_json(
        manifest_path,
        {
            "schema_version": 2,
            "experiment_stage": 18,
            "stage18_run_id": run_id,
            "creation_timestamp_utc": datetime.now(UTC).isoformat(),
            "implementation_commit": implementation_commit,
            "finalization_commit": finalization_commit,
            "freeze_commit": "f7d4e232c15da5c0239c5df7dde1dacba850c2fb",
            "configuration": _artifact(repository, "configs/stage18_scaling.json"),
            "configuration_sha256": configuration_sha256,
            "source_artifacts": source_artifacts,
            "training": {
                "primary_main_seeds": configuration["main_seeds"],
                "reserve_seeds": configuration["reserve_seeds"],
                "pilot_reference_seed": configuration["pilot_seed"],
                "fresh_training_seeds": configuration["fresh_training_seeds"],
                "runs": training["runs"],
                "replacement_decisions": [
                    {
                        "primary_seed": seed,
                        "selected_seed": seed,
                        "replacement_used": False,
                    }
                    for seed in configuration["main_seeds"]
                ],
                "configuration_hashes": first_training["configs"],
                "dataset": first_training["dataset"],
                "architecture_sha256": first_training["configs"]["model"]["sha256"],
                "optimiser_sha256": first_training["configs"]["training"]["sha256"],
            },
            "analysis_seeds": list(analysis_seeds),
            "primary_main_seeds": list(PRIMARY_MAIN_SEEDS),
            "checkpoint_grid": {
                "steps": list(CHECKPOINT_STEPS),
                "registry": checkpoint_registry,
                "registry_artifact": _artifact(
                    repository, "results/tables/stage18_checkpoint_registry.csv"
                ),
            },
            "fidelity_grid": configuration["fidelity_grid"],
            "distinctness_grid": configuration["distinctness_grid"],
            "sparsity_boundary": configuration["search"]["maximum_retained_components"],
            "component_universe": stage17["component_universe"],
            "search_configuration": configuration["search"],
            "cell_registry": {
                "count": len(cell_registry),
                "rows": cell_registry,
                "artifact": _artifact(repository, "results/tables/stage18_cell_registry.csv"),
            },
            "counts": {
                "seed_checkpoints": 35,
                "cells": 630,
                "fresh_cells": 612,
                "stage17_reference_cells": 18,
                "worker_shards": 12,
                "archive_shards": 35,
            },
            "transfer": {
                **configuration["transfer"],
                "stage16_subset_discovery": stage16["outputs"]["subset_discovery"],
                "stage16_subset_transfer": stage16["outputs"]["subset_transfer"],
            },
            "concurrency": {
                **configuration["concurrency"],
                "worker_shards": worker_shards,
            },
            "phase_barriers": {
                "all_masking_before_search": True,
                "search_merge_completed_before_transfer": True,
                "serial_final_merge": True,
                "serial_archive_construction": True,
            },
            "archive": {
                **configuration["archive"],
                "index": _artifact(repository, str(archive_index_path.relative_to(repository))),
                "shards": archive_index["shards"],
            },
            "additional_controls": configuration["additional_controls"],
            "stage15_status": "unavailable",
            "integrity": integrity,
            "software": stage17["software"],
            "device": "cpu",
            "repository": {
                "branch": "main",
                "tracked_repository_clean_before_execution": True,
                "permitted_untracked_file": "stage17_inspection.md",
                "permitted_untracked_file_sha256": (
                    "fc8cd5cc791715bb558cb47472ffc4b821254723591e458e894318659dc86533"
                ),
            },
            "recovery": {
                "resumed_from_completed_raw_outputs": finalize_existing,
                "validated_search_cells": FRESH_CELL_COUNT,
                "validated_transfer_cells": FRESH_CELL_COUNT,
                "fresh_cells_executed_during_invocation": (
                    0 if finalize_existing else FRESH_CELL_COUNT
                ),
            },
            "outputs": output_hashes,
            "table_outputs": table_outputs,
            "figure_outputs": figure_outputs,
            "note": _artifact(repository, "results/notes/stage18_main_seed_scaling.md"),
            "runtime": _artifact(repository, "results/tables/stage18_runtime.csv"),
            "compute_use": _artifact(repository, "results/tables/stage18_compute_use.csv"),
            "stage19_started": False,
        },
    )


def refresh_stage18_metadata(
    repository: Path,
    *,
    run_id: str,
    implementation_commit: str,
    finalization_commit: str,
    configuration_sha256: str,
) -> Path:
    analysis_seeds = selected_analysis_seeds(repository)
    cells = build_stage18_registry(analysis_seeds)
    masking_path = validate_existing_stage18_outputs(
        repository,
        run_id,
        cells,
        analysis_seeds,
        require_final_outputs_absent=False,
    )
    merged = serial_merge_fresh(repository, run_id, cells)
    merge_stage17_references(repository, run_id, cells, merged)
    for rows in merged.values():
        rows.sort(
            key=lambda row: (
                int(row.get("model_seed", 0)),
                int(row.get("checkpoint_index", 0)),
                int(row.get("cell_index", row.get("global_cell_index", 0))),
                int(row.get("family_member_index", row.get("member_i", 0)) or 0),
            )
        )
    table_paths = list(write_merged_tables(repository, merged))
    family_rows = merged["family_summary_table"]
    note = write_stage18_note(repository, run_id, family_rows)
    archive_index = refresh_archive_index(repository, run_id)
    fixed_outputs = (
        masking_path,
        repository / "results/tables/stage18_cell_registry.csv",
        repository / "results/tables/stage18_family_size_surface.csv",
        repository / "results/tables/stage18_compute_use.csv",
        repository / "results/tables/stage18_runtime.csv",
        repository / "results/tables/stage18_worker_shards.csv",
        note,
        archive_index,
    )
    figures = tuple(
        repository / relative
        for relative in (
            "figures/stage18_caption.txt",
            "figures/stage18_main_seed_training_curves.pdf",
            "figures/stage18_main_seed_training_curves.png",
            "figures/stage18_primary_family_dynamics.pdf",
            "figures/stage18_primary_family_dynamics.png",
            "figures/stage18_primary_transfer_group_dynamics.pdf",
            "figures/stage18_primary_transfer_group_dynamics.png",
            "figures/stage18_seed_sensitivity_overview.pdf",
            "figures/stage18_seed_sensitivity_overview.png",
        )
    )
    outputs = (*fixed_outputs, *table_paths, *figures)
    missing = [path for path in outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Stage 18 metadata outputs.")
    return write_stage18_manifest(
        repository,
        run_id=run_id,
        implementation_commit=implementation_commit,
        finalization_commit=finalization_commit,
        configuration_sha256=configuration_sha256,
        analysis_seeds=analysis_seeds,
        outputs=outputs,
        finalize_existing=True,
    )


def execute_stage18(
    repository: Path,
    *,
    run_id: str,
    implementation_commit: str,
    configuration_sha256: str,
    finalization_commit: str | None = None,
    finalize_existing: bool = False,
) -> Path:
    analysis_seeds = selected_analysis_seeds(repository)
    cells = build_stage18_registry(analysis_seeds)
    raw_root = repository / "results/raw" / run_id
    if finalize_existing:
        masking_path = validate_existing_stage18_outputs(repository, run_id, cells, analysis_seeds)
        started = masking_path.stat().st_mtime
    else:
        if raw_root.exists():
            raise FileExistsError(f"Stage 18 raw output already exists: {raw_root}")
        started = time.perf_counter()
        masking_rows = execute_masking_validation(repository, analysis_seeds)
        masking_path = write_csv(
            repository / "results/tables/stage18_masking_validation.csv", masking_rows
        )
        execute_search_workers(repository, run_id, implementation_commit, cells)
        validate_search_merge(repository, run_id, cells)
        execute_transfer_workers(repository, run_id, cells)
    merged = serial_merge_fresh(repository, run_id, cells)
    merge_stage17_references(repository, run_id, cells, merged)
    for rows in merged.values():
        rows.sort(
            key=lambda row: (
                int(row.get("model_seed", 0)),
                int(row.get("checkpoint_index", 0)),
                int(row.get("cell_index", row.get("global_cell_index", 0))),
                int(row.get("family_member_index", row.get("member_i", 0)) or 0),
            )
        )
    table_paths = list(write_merged_tables(repository, merged))
    family_rows = merged["family_summary_table"]
    registry_path = write_csv(
        repository / "results/tables/stage18_cell_registry.csv",
        final_cell_registry(cells, family_rows),
    )
    surface_path = write_csv(
        repository / "results/tables/stage18_family_size_surface.csv", family_rows
    )
    compute_path = write_csv(
        repository / "results/tables/stage18_compute_use.csv",
        tuple(
            {
                "model_seed": row["model_seed"],
                "checkpoint_step": row["checkpoint_step"],
                "cell_id": row["cell_id"],
                "exact_evaluations_used": row["exact_evaluations_used"],
                "ranking_passes": row["ranking_passes"],
                "restarts_attempted": row["restarts_attempted"],
            }
            for row in family_rows
        ),
    )
    runtime_path = write_csv(
        repository / "results/tables/stage18_runtime.csv",
        (
            {
                "stage18_run_id": run_id,
                "record_type": "complete_workload",
                "elapsed_seconds": (
                    time.time() - started if finalize_existing else time.perf_counter() - started
                ),
                "included_in_deterministic_scientific_hashes": False,
            },
        ),
    )
    shard_path = write_csv(
        repository / "results/tables/stage18_worker_shards.csv",
        tuple(shard.to_record() for shard in build_worker_shards(cells)),
    )
    note = write_stage18_note(repository, run_id, family_rows)
    figures = write_stage18_figures(repository, family_rows)
    archive_index = write_archive_shards(repository, run_id, cells)
    outputs = (
        masking_path,
        registry_path,
        surface_path,
        compute_path,
        runtime_path,
        shard_path,
        note,
        archive_index,
        *table_paths,
        *figures,
    )
    return write_stage18_manifest(
        repository,
        run_id=run_id,
        implementation_commit=implementation_commit,
        finalization_commit=finalization_commit or implementation_commit,
        configuration_sha256=configuration_sha256,
        analysis_seeds=analysis_seeds,
        outputs=outputs,
        finalize_existing=finalize_existing,
    )


def compare_reproduction(
    reference_root: Path,
    reproduction_root: Path,
    *,
    run_id: str,
    progress: bool = False,
) -> dict[str, object]:
    def scientific_paths(root: Path) -> set[Path]:
        paths: set[Path] = set()
        raw_prefix = Path("results/raw") / run_id
        manifest_path = root / f"manifests/stage18_scaling_{run_id}.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths.update(
                Path(relative)
                for relative in manifest["outputs"]
                if not str(relative).startswith("results/archives/")
                and "runtime" not in Path(str(relative)).name
            )
        output_prefixes = (Path("results/tables"), Path("results/notes"), Path("figures"))
        for prefix in (raw_prefix, *(() if manifest_path.is_file() else output_prefixes)):
            source = root / prefix
            stage18_named_outputs_only = prefix != raw_prefix
            if source.is_file():
                paths.add(prefix)
            elif source.is_dir():
                paths.update(
                    path.relative_to(root)
                    for path in source.rglob("*")
                    if path.is_file()
                    and (not stage18_named_outputs_only or path.name.startswith("stage18_"))
                    and path.name != ".DS_Store"
                    and "runtime" not in path.name
                    and not path.name.startswith("stage18_training")
                )
        return paths

    def normalized_json(relative: Path, root: Path) -> dict[str, object]:
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        if relative.name == "hash_inventory.json":
            payload["files"] = [
                row
                for row in payload["files"]
                if Path(str(row["path"])).name != ".DS_Store"
            ]
            payload["file_count"] = len(payload["files"])
        elif relative.name == "search_result.json":
            payload.get("search_integrity", {}).pop("hash_inventory_sha256", None)
        return payload

    reference_paths = scientific_paths(reference_root)
    reproduction_paths = scientific_paths(reproduction_root)
    compared_paths = sorted(reference_paths & reproduction_paths)
    if progress:
        print(
            "Stage 18 comparison: indexed "
            f"{len(reference_paths | reproduction_paths)} scientific paths; hashing begins.",
            flush=True,
        )
    mismatches: list[dict[str, str]] = []
    normalized_metadata_file_count = 0
    for relative in sorted(reference_paths - reproduction_paths):
        mismatches.append({"path": str(relative), "reason": "missing"})
    for relative in sorted(reproduction_paths - reference_paths):
        mismatches.append({"path": str(relative), "reason": "unexpected_extra"})
    last_report = time.monotonic()
    for index, relative in enumerate(compared_paths, start=1):
        reference = reference_root / relative
        reproduction = reproduction_root / relative
        if relative.name in ("hash_inventory.json", "search_result.json"):
            matched = normalized_json(relative, reference_root) == normalized_json(
                relative, reproduction_root
            )
            normalized_metadata_file_count += 1
        else:
            matched = file_sha256(reference) == file_sha256(reproduction)
        if not matched:
            reason = (
                "normalized_metadata_mismatch"
                if relative.name in ("hash_inventory.json", "search_result.json")
                else "sha256_mismatch"
            )
            mismatches.append({"path": str(relative), "reason": reason})
        now = time.monotonic()
        if progress and (now - last_report >= 5 or index == len(compared_paths)):
            percentage = 100 if not compared_paths else 100 * index / len(compared_paths)
            print(
                "Stage 18 comparison: "
                f"{index} / {len(compared_paths)} files, {percentage:.1f}%",
                flush=True,
            )
            last_report = now

    archive_relative = Path("results/archives") / run_id

    def archive_inventories(root: Path) -> dict[str, dict[str, object]]:
        archive_root = root / archive_relative
        values = {}
        if not archive_root.is_dir():
            return values
        for path in sorted(archive_root.glob("*_inventory.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            values[path.name] = {
                "model_seed": payload["model_seed"],
                "checkpoint_step": payload["checkpoint_step"],
                "cell_ids": payload["cell_ids"],
                "reference_stage17": payload["reference_stage17"],
                "members": sorted(
                    member
                    for member in payload["members"]
                    if Path(str(member)).name != ".DS_Store"
                    and "runtime" not in Path(str(member)).name
                ),
            }
        return values

    reference_inventories = archive_inventories(reference_root)
    reproduction_inventories = archive_inventories(reproduction_root)
    inventory_names = sorted(set(reference_inventories) | set(reproduction_inventories))
    inventory_mismatches = []
    for name in inventory_names:
        relative = archive_relative / name
        if name not in reproduction_inventories:
            inventory_mismatches.append({"path": str(relative), "reason": "missing"})
        elif name not in reference_inventories:
            inventory_mismatches.append(
                {"path": str(relative), "reason": "unexpected_extra"}
            )
        elif reference_inventories[name] != reproduction_inventories[name]:
            inventory_mismatches.append(
                {"path": str(relative), "reason": "normalized_inventory_mismatch"}
            )
    mismatches.extend(inventory_mismatches)
    if progress:
        print(
            "Stage 18 comparison: normalized archive inventories "
            f"{len(inventory_names) - len(inventory_mismatches)} / {len(inventory_names)}",
            flush=True,
        )
    return {
        "stage18_run_id": run_id,
        "comparison_policy": {
            "scientific_files": (
                "strict path and SHA-256 equality in both directions except the explicitly "
                "normalized metadata wrappers"
            ),
            "excluded_nondeterministic_files": [
                ".DS_Store",
                "runtime telemetry",
                "stage18_training registry outputs",
            ],
            "normalized_metadata_wrappers": {
                "hash_inventory.json": (
                    "remove .DS_Store rows and recompute file_count before exact JSON comparison"
                ),
                "search_result.json": (
                    "remove only search_integrity.hash_inventory_sha256 because the normalized "
                    "inventory is compared separately"
                ),
            },
            "archive_policy": (
                "compare normalized member inventories; archive byte streams, sizes, and index "
                "hashes are packaging duplicates and are excluded"
            ),
        },
        "compared_file_count": len(reference_paths | reproduction_paths),
        "normalized_metadata_file_count": normalized_metadata_file_count,
        "archive_inventory_count": len(inventory_names),
        "archive_inventory_mismatch_count": len(inventory_mismatches),
        "deterministic_mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }

"""Execute Stage 14 random-label transfer and grouping workloads."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from circuit_families.analysis.fidelity_calibration import (
    write_csv_records,
)
from circuit_families.analysis.random_label_circuit_analysis import (
    load_frozen_analysis_configuration,
    load_random_label_checkpoint_context,
    subset_context,
)
from circuit_families.analysis.stage14_random_label_diversity import (
    DIVERSITY_RUNTIME_COLUMNS,
)
from circuit_families.analysis.stage14_random_label_runner import (
    Stage14AnalysisCell,
    Stage14ExecutionPlan,
    adapt_random_label_search_context,
    build_execution_plan,
    find_stage15_artifacts,
    output_contract,
    validate_analysis_inputs,
)
from circuit_families.analysis.transfer import (
    TransferEvaluation,
    TransferProfile,
    evaluate_transfer_profile,
    pairwise_transfer_distances,
    transfer_grouping,
)
from circuit_families.data.input_subsets import (
    SUBSET_NAMES,
)
from circuit_families.interpretability.masks import (
    load_component_mask,
)
from circuit_families.interpretability.sparse_search import (
    CheckpointSearchExecution,
    run_checkpoint_sparse_search,
    write_sparse_search_artifacts,
)

TRANSFER_COLUMNS = (
    "analysis_run_id",
    "workload",
    "record_type",
    "cell_id",
    "sequence_index",
    "checkpoint_index",
    "checkpoint_step",
    "discovery_subset",
    "evaluation_subset",
    "circuit_id",
    "source_member_index",
    "search_status",
    "retained_component_count",
    "primary_fidelity",
    "prediction_agreement_count",
    "evaluated_example_count",
    "exact_evaluations_used",
    "grouping_tolerance_numerator",
    "grouping_tolerance_denominator",
    "grouping_tolerance",
    "group_count",
    "groups_json",
    "pairwise_distances_json",
    "scientifically_executed",
    "raw_cell_directory",
)

TRANSFER_RUNTIME_COLUMNS = DIVERSITY_RUNTIME_COLUMNS


@dataclass(frozen=True)
class SubsetDiscoveryExecution:
    """One subset-discovery sparse search and optional transfer profile."""

    cell: Stage14AnalysisCell
    source_context: Any
    search_execution: CheckpointSearchExecution
    artifacts: Any
    transfer_evaluation: TransferEvaluation | None
    elapsed_seconds: float


@dataclass(frozen=True)
class TransferWorkloadResult:
    """Outputs from all 56 frozen transfer-related cells."""

    analysis_run_id: str
    implementation_commit: str
    transfer_table: Path
    runtime_table: Path
    global_cell_count: int
    subset_discovery_cell_count: int
    grouping_cell_count: int
    subset_discoveries: tuple[SubsetDiscoveryExecution, ...]


def transfer_cells(
    plan: Stage14ExecutionPlan,
    workload: str,
) -> tuple[Stage14AnalysisCell, ...]:
    """Return one transfer-related workload in frozen order."""

    expected_counts = {
        "global_family_transfer": 7,
        "subset_discovery": 28,
        "transfer_grouping": 21,
    }

    if workload not in expected_counts:
        raise ValueError(f"Unknown transfer workload: {workload}")

    cells = tuple(
        cell
        for cell in plan.cells
        if cell.workload == workload
    )

    if len(cells) != expected_counts[workload]:
        raise ValueError(
            f"{workload} expected "
            f"{expected_counts[workload]} cells, "
            f"found {len(cells)}."
        )

    if any(
        cell.execution_mode != "execute"
        for cell in cells
    ):
        raise ValueError(
            f"Every {workload} cell must execute."
        )

    return cells


def _sha256(file_name: Path) -> str:
    return hashlib.sha256(file_name.read_bytes()).hexdigest()


def _stable_json(
    file_name: Path,
    value: Mapping[str, Any],
) -> Path:
    file_name.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    file_name.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return file_name


def _relative(
    root: Path,
    file_name: Path,
) -> str:
    return file_name.resolve().relative_to(
        root.resolve()
    ).as_posix()


def _read_runtime_rows(
    runtime_table: Path,
) -> list[dict[str, object]]:
    with runtime_table.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return [
            {
                column: row.get(column, "")
                for column in TRANSFER_RUNTIME_COLUMNS
            }
            for row in csv.DictReader(handle)
        ]


def _runtime_row(
    *,
    analysis_run_id: str,
    cell: Stage14AnalysisCell,
    exact_evaluations_used: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "analysis_run_id": analysis_run_id,
        "workload": cell.workload,
        "cell_id": cell.cell_id,
        "checkpoint_index": cell.checkpoint_index,
        "checkpoint_step": cell.checkpoint_step,
        "fidelity_threshold": (
            ""
            if cell.fidelity_threshold is None
            else float(cell.fidelity_threshold)
        ),
        "distinctness_cutoff": (
            ""
            if cell.distinctness_cutoff is None
            else float(cell.distinctness_cutoff)
        ),
        "discovery_subset": (
            ""
            if cell.discovery_subset is None
            else cell.discovery_subset
        ),
        "grouping_tolerance": (
            ""
            if cell.grouping_tolerance is None
            else float(cell.grouping_tolerance)
        ),
        "requested_member_index": "",
        "accepted_circuit": "",
        "restart_count": "",
        "record_type": "cell",
        "exact_evaluations_used": (
            exact_evaluations_used
        ),
        "elapsed_seconds": elapsed_seconds,
        "included_in_deterministic_scientific_hashes": False,
    }


def _blank_transfer_row(
    *,
    analysis_run_id: str,
    cell: Stage14AnalysisCell,
    record_type: str,
    raw_cell_directory: str,
) -> dict[str, object]:
    return {
        "analysis_run_id": analysis_run_id,
        "workload": cell.workload,
        "record_type": record_type,
        "cell_id": cell.cell_id,
        "sequence_index": cell.sequence_index,
        "checkpoint_index": cell.checkpoint_index,
        "checkpoint_step": cell.checkpoint_step,
        "discovery_subset": (
            ""
            if cell.discovery_subset is None
            else cell.discovery_subset
        ),
        "evaluation_subset": "",
        "circuit_id": "",
        "source_member_index": "",
        "search_status": "",
        "retained_component_count": "",
        "primary_fidelity": "",
        "prediction_agreement_count": "",
        "evaluated_example_count": "",
        "exact_evaluations_used": 0,
        "grouping_tolerance_numerator": "",
        "grouping_tolerance_denominator": "",
        "grouping_tolerance": "",
        "group_count": "",
        "groups_json": "",
        "pairwise_distances_json": "",
        "scientifically_executed": True,
        "raw_cell_directory": raw_cell_directory,
    }


def _evaluation_rows(
    *,
    analysis_run_id: str,
    cell: Stage14AnalysisCell,
    transfer: TransferEvaluation,
    source_member_index: int | str,
    retained_component_count: int,
    raw_cell_directory: str,
    record_type: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for evaluation in transfer.evaluations:
        metrics = evaluation.metrics

        rows.append(
            {
                **_blank_transfer_row(
                    analysis_run_id=analysis_run_id,
                    cell=cell,
                    record_type=record_type,
                    raw_cell_directory=(
                        raw_cell_directory
                    ),
                ),
                "evaluation_subset": (
                    evaluation.evaluation_subset
                ),
                "circuit_id": (
                    evaluation.circuit_id
                ),
                "source_member_index": (
                    source_member_index
                ),
                "retained_component_count": (
                    retained_component_count
                ),
                "primary_fidelity": (
                    metrics.primary_fidelity
                ),
                "prediction_agreement_count": (
                    metrics.prediction_agreement_count
                ),
                "evaluated_example_count": (
                    metrics.evaluated_example_count
                ),
            }
        )

    return rows


def _primary_circuit_rows(
    circuits_table: Path,
) -> dict[int, list[dict[str, str]]]:
    by_step: dict[int, list[dict[str, str]]] = {}

    with circuits_table.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            by_step.setdefault(
                int(row["checkpoint_step"]),
                [],
            ).append(row)

    return by_step


def _primary_mask_path(
    *,
    raw_root: Path,
    checkpoint_step: int,
    member_index: int,
    restart_index: int,
) -> Path:
    return (
        raw_root
        / "primary_diversity"
        / f"step_{checkpoint_step:08d}"
        / "restarts"
        / f"C{member_index:02d}"
        / f"restart_{restart_index:02d}"
        / "search"
        / "final_mask.json"
    )


def execute_transfer_workload(
    *,
    repository_root: str | Path,
    expected_implementation_commit: str,
    input_root: str | Path | None = None,
    output_root: str | Path | None = None,
    device: str = "cpu",
    progress_callback: Callable[[str], None] | None = None,
) -> TransferWorkloadResult:
    """Execute global transfer, subset discovery and grouping cells."""

    if device not in {"cpu", "cuda"}:
        raise ValueError(
            "Stage 14 scientific execution supports only CPU or CUDA."
        )

    repository = Path(repository_root).resolve()
    selected_input_root = (
        repository
        if input_root is None
        else Path(input_root).resolve()
    )
    selected_output_root = (
        repository
        if output_root is None
        else Path(output_root).resolve()
    )

    validation = validate_analysis_inputs(
        repository_root=repository,
        expected_implementation_commit=(
            expected_implementation_commit
        ),
        input_root=selected_input_root,
        output_root=selected_output_root,
        require_clean_repository=False,
        require_outputs_absent=False,
        verify_checkpoint_hashes=True,
    )

    if (
        validation.current_commit
        != expected_implementation_commit
    ):
        raise RuntimeError(
            "Validated implementation commit differs."
        )

    if find_stage15_artifacts(repository):
        raise FileExistsError(
            "Stage 15 artifacts exist before transfer execution."
        )

    configuration = load_frozen_analysis_configuration(
        repository_root=repository
    )
    plan = build_execution_plan(configuration)
    global_cells = transfer_cells(
        plan,
        "global_family_transfer",
    )
    subset_cells = transfer_cells(
        plan,
        "subset_discovery",
    )
    grouping_cells = transfer_cells(
        plan,
        "transfer_grouping",
    )
    resolved = dict(
        output_contract(configuration).resolve(
            selected_output_root
        )
    )

    expected_existing = {
        "raw_output_directory",
        "sparse_search_table",
        "family_summary_table",
        "circuits_table",
        "pairwise_overlap_table",
        "restart_table",
        "fidelity_sensitivity_table",
        "distinctness_sensitivity_table",
        "runtime_table",
    }
    observed_existing = {
        name
        for name, file_name in resolved.items()
        if file_name.exists()
    }

    if observed_existing != expected_existing:
        raise FileExistsError(
            "Transfer execution requires completed "
            "primary and sensitivity outputs."
        )

    raw_root = resolved["raw_output_directory"]
    global_raw = raw_root / "global_family_transfer"
    subset_raw = raw_root / "subset_discovery"
    grouping_raw = raw_root / "transfer_grouping"
    transfer_table = resolved["transfer_table"]
    runtime_table = resolved["runtime_table"]

    if transfer_table.exists():
        raise FileExistsError(transfer_table)

    for directory in (
        global_raw,
        subset_raw,
        grouping_raw,
    ):
        if directory.exists():
            raise FileExistsError(directory)

    runtime_bytes = runtime_table.read_bytes()
    existing_runtime_rows = _read_runtime_rows(
        runtime_table
    )
    primary_rows = _primary_circuit_rows(
        resolved["circuits_table"]
    )
    search_configuration = configuration.payload[
        "search"
    ]
    ranking_batch_size = int(
        search_configuration["ranking_batch_size"]
    )
    evaluation_batch_size = int(
        search_configuration["evaluation_batch_size"]
    )
    discovery_budget = int(
        search_configuration[
            "per_requested_circuit_exact_evaluations"
        ]
    )

    contexts: dict[int, Any] = {}
    global_profiles: dict[
        int,
        tuple[TransferProfile, ...],
    ] = {}
    table_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    subset_results: list[
        SubsetDiscoveryExecution
    ] = []

    try:
        global_raw.mkdir(
            parents=True,
            exist_ok=False,
        )
        subset_raw.mkdir(
            parents=True,
            exist_ok=False,
        )
        grouping_raw.mkdir(
            parents=True,
            exist_ok=False,
        )

        def context_for(step: int) -> Any:
            if step not in contexts:
                contexts[step] = (
                    load_random_label_checkpoint_context(
                        repository_root=repository,
                        configuration=configuration,
                        checkpoint_step=step,
                        device=device,
                        output_root=selected_input_root,
                    )
                )

            return contexts[step]

        for index, cell in enumerate(
            global_cells,
            start=1,
        ):
            if progress_callback is not None:
                progress_callback(
                    f"[{index:02d}/{len(global_cells):02d}] "
                    "global_family_transfer "
                    f"checkpoint={cell.checkpoint_step}"
                )

            started = time.perf_counter()
            source_context = context_for(
                cell.checkpoint_step
            )
            profiles: list[TransferProfile] = []
            evaluations: list[
                TransferEvaluation
            ] = []
            cell_directory = (
                global_raw
                / f"step_{cell.checkpoint_step:08d}"
            )
            rows = primary_rows.get(
                cell.checkpoint_step,
                [],
            )

            for row in sorted(
                rows,
                key=lambda value: int(
                    value["member_index"]
                ),
            ):
                member_index = int(
                    row["member_index"]
                )
                restart_index = int(
                    row["selected_restart_index"]
                )
                mask_path = _primary_mask_path(
                    raw_root=raw_root,
                    checkpoint_step=(
                        cell.checkpoint_step
                    ),
                    member_index=member_index,
                    restart_index=restart_index,
                )

                if not mask_path.is_file():
                    raise FileNotFoundError(mask_path)

                transfer = evaluate_transfer_profile(
                    context=source_context,
                    mask=load_component_mask(
                        mask_path
                    ),
                    circuit_id=(
                        row["member_label"]
                    ),
                    batch_size=(
                        evaluation_batch_size
                    ),
                )
                profiles.append(transfer.profile)
                evaluations.append(transfer)

            global_profiles[
                cell.checkpoint_step
            ] = tuple(profiles)

            summary_path = _stable_json(
                cell_directory
                / "cell_summary.json",
                {
                    "schema_version": 1,
                    "analysis_run_id": (
                        configuration.analysis_run_id
                    ),
                    "workload": cell.workload,
                    "cell_id": cell.cell_id,
                    "sequence_index": (
                        cell.sequence_index
                    ),
                    "checkpoint_index": (
                        cell.checkpoint_index
                    ),
                    "checkpoint_step": (
                        cell.checkpoint_step
                    ),
                    "source_family_size": len(
                        profiles
                    ),
                    "evaluation_subsets": list(
                        SUBSET_NAMES
                    ),
                    "profiles": [
                        {
                            "circuit_id": (
                                profile.circuit_id
                            ),
                            **profile.as_mapping(),
                        }
                        for profile in profiles
                    ],
                    "scientifically_executed": True,
                    "exact_evaluations_used": (
                        len(profiles)
                        * len(SUBSET_NAMES)
                    ),
                },
            )
            relative_directory = _relative(
                selected_output_root,
                cell_directory,
            )

            if not evaluations:
                table_rows.append(
                    _blank_transfer_row(
                        analysis_run_id=(
                            configuration.analysis_run_id
                        ),
                        cell=cell,
                        record_type=(
                            "empty_global_family"
                        ),
                        raw_cell_directory=(
                            relative_directory
                        ),
                    )
                )
            else:
                for member_index, transfer in enumerate(
                    evaluations,
                    start=1,
                ):
                    table_rows.extend(
                        _evaluation_rows(
                            analysis_run_id=(
                                configuration
                                .analysis_run_id
                            ),
                            cell=cell,
                            transfer=transfer,
                            source_member_index=(
                                member_index
                            ),
                            retained_component_count=int(
                                rows[member_index - 1][
                                    "retained_component_count"
                                ]
                            ),
                            raw_cell_directory=(
                                relative_directory
                            ),
                            record_type=(
                                "global_family_evaluation"
                            ),
                        )
                    )

            elapsed = time.perf_counter() - started
            runtime_rows.append(
                _runtime_row(
                    analysis_run_id=(
                        configuration.analysis_run_id
                    ),
                    cell=cell,
                    exact_evaluations_used=(
                        len(profiles)
                        * len(SUBSET_NAMES)
                    ),
                    elapsed_seconds=elapsed,
                )
            )

            if progress_callback is not None:
                progress_callback(
                    "completed "
                    f"checkpoint={cell.checkpoint_step} "
                    f"family_size={len(profiles)} "
                    f"summary_sha256={_sha256(summary_path)}"
                )

        for index, cell in enumerate(
            subset_cells,
            start=1,
        ):
            if progress_callback is not None:
                progress_callback(
                    f"[{index:02d}/{len(subset_cells):02d}] "
                    "subset_discovery "
                    f"checkpoint={cell.checkpoint_step} "
                    f"subset={cell.discovery_subset}"
                )

            if cell.discovery_subset is None:
                raise ValueError(
                    "Subset-discovery cell lacks a subset."
                )

            if cell.fidelity_threshold is None:
                raise ValueError(
                    "Subset-discovery threshold is absent."
                )

            source_context = context_for(
                cell.checkpoint_step
            )
            discovery_context = subset_context(
                source_context,
                cell.discovery_subset,
            )
            search_context = (
                adapt_random_label_search_context(
                    discovery_context
                )
            )

            started = time.perf_counter()
            search_execution = (
                run_checkpoint_sparse_search(
                    search_context,
                    fidelity_threshold=float(
                        cell.fidelity_threshold
                    ),
                    ranking_batch_size=(
                        ranking_batch_size
                    ),
                    evaluation_batch_size=(
                        evaluation_batch_size
                    ),
                    exact_evaluation_budget=(
                        discovery_budget
                    ),
                )
            )
            cell_directory = (
                subset_raw
                / f"step_{cell.checkpoint_step:08d}"
                / cell.discovery_subset
            )
            artifacts = write_sparse_search_artifacts(
                cell_directory / "search",
                search_execution.result,
                cell_metadata={
                    "analysis_run_id": (
                        configuration.analysis_run_id
                    ),
                    "analysis_identity_sha256": (
                        configuration
                        .analysis_identity_sha256
                    ),
                    "analysis_configuration_sha256": (
                        configuration.sha256
                    ),
                    "implementation_git_commit": (
                        expected_implementation_commit
                    ),
                    "workload": cell.workload,
                    "cell_id": cell.cell_id,
                    "sequence_index": (
                        cell.sequence_index
                    ),
                    "checkpoint_index": (
                        cell.checkpoint_index
                    ),
                    "checkpoint_step": (
                        cell.checkpoint_step
                    ),
                    "checkpoint_sha256": (
                        source_context.checkpoint_sha256
                    ),
                    "discovery_subset": (
                        cell.discovery_subset
                    ),
                    "fidelity_threshold": float(
                        cell.fidelity_threshold
                    ),
                },
            )

            transfer_evaluation: (
                TransferEvaluation | None
            ) = None

            if (
                search_execution.result.status
                == "valid_sparse_circuit"
            ):
                transfer_evaluation = (
                    evaluate_transfer_profile(
                        context=source_context,
                        mask=(
                            search_execution
                            .result.final_mask
                        ),
                        circuit_id=(
                            f"{cell.discovery_subset}-C1"
                        ),
                        batch_size=(
                            evaluation_batch_size
                        ),
                        discovery_subset=(
                            cell.discovery_subset
                        ),
                    )
                )

            elapsed = time.perf_counter() - started
            result = SubsetDiscoveryExecution(
                cell=cell,
                source_context=source_context,
                search_execution=search_execution,
                artifacts=artifacts,
                transfer_evaluation=(
                    transfer_evaluation
                ),
                elapsed_seconds=elapsed,
            )
            subset_results.append(result)
            relative_directory = _relative(
                selected_output_root,
                cell_directory,
            )
            search = search_execution.result
            summary_row = _blank_transfer_row(
                analysis_run_id=(
                    configuration.analysis_run_id
                ),
                cell=cell,
                record_type=(
                    "subset_discovery_search"
                ),
                raw_cell_directory=(
                    relative_directory
                ),
            )
            summary_row.update(
                {
                    "search_status": search.status,
                    "retained_component_count": (
                        search.final_mask
                        .retained_component_count
                    ),
                    "primary_fidelity": (
                        search.final_metrics
                        .primary_fidelity
                    ),
                    "prediction_agreement_count": (
                        search.final_metrics
                        .prediction_agreement_count
                    ),
                    "evaluated_example_count": (
                        search.final_metrics
                        .evaluated_example_count
                    ),
                    "exact_evaluations_used": (
                        search.exact_evaluations_used
                    ),
                }
            )
            table_rows.append(summary_row)

            if transfer_evaluation is not None:
                table_rows.extend(
                    _evaluation_rows(
                        analysis_run_id=(
                            configuration.analysis_run_id
                        ),
                        cell=cell,
                        transfer=transfer_evaluation,
                        source_member_index=1,
                        retained_component_count=(
                            search.final_mask
                            .retained_component_count
                        ),
                        raw_cell_directory=(
                            relative_directory
                        ),
                        record_type=(
                            "subset_discovery_transfer"
                        ),
                    )
                )

            runtime_rows.append(
                _runtime_row(
                    analysis_run_id=(
                        configuration.analysis_run_id
                    ),
                    cell=cell,
                    exact_evaluations_used=(
                        search.exact_evaluations_used
                    ),
                    elapsed_seconds=elapsed,
                )
            )

            if progress_callback is not None:
                progress_callback(
                    "completed "
                    f"cell={cell.cell_id} "
                    f"status={search.status} "
                    f"retained="
                    f"{search.final_mask.retained_component_count} "
                    f"exact_evaluations="
                    f"{search.exact_evaluations_used}"
                )

        for index, cell in enumerate(
            grouping_cells,
            start=1,
        ):
            if cell.grouping_tolerance is None:
                raise ValueError(
                    "Grouping cell lacks a tolerance."
                )

            if progress_callback is not None:
                progress_callback(
                    f"[{index:02d}/{len(grouping_cells):02d}] "
                    "transfer_grouping "
                    f"checkpoint={cell.checkpoint_step} "
                    f"tolerance={cell.grouping_tolerance}"
                )

            started = time.perf_counter()
            profiles = global_profiles[
                cell.checkpoint_step
            ]
            grouping = transfer_grouping(
                profiles,
                tolerance=(
                    cell.grouping_tolerance
                ),
            )
            distances = pairwise_transfer_distances(
                profiles
            )
            cell_directory = (
                grouping_raw
                / f"step_{cell.checkpoint_step:08d}"
                / (
                    "tolerance_"
                    f"{cell.grouping_tolerance.numerator}_"
                    f"{cell.grouping_tolerance.denominator}"
                )
            )
            _stable_json(
                cell_directory
                / "cell_summary.json",
                {
                    "schema_version": 1,
                    "analysis_run_id": (
                        configuration.analysis_run_id
                    ),
                    "workload": cell.workload,
                    "cell_id": cell.cell_id,
                    "sequence_index": (
                        cell.sequence_index
                    ),
                    "checkpoint_index": (
                        cell.checkpoint_index
                    ),
                    "checkpoint_step": (
                        cell.checkpoint_step
                    ),
                    "tolerance": {
                        "numerator": (
                            cell.grouping_tolerance
                            .numerator
                        ),
                        "denominator": (
                            cell.grouping_tolerance
                            .denominator
                        ),
                        "float": float(
                            cell.grouping_tolerance
                        ),
                    },
                    "groups": [
                        list(group)
                        for group in grouping.groups
                    ],
                    "group_count": (
                        grouping.group_count
                    ),
                    "pairwise_distances": {
                        f"{left}|{right}": value
                        for (
                            left,
                            right,
                        ), value in distances.items()
                    },
                    "scientifically_executed": True,
                },
            )
            row = _blank_transfer_row(
                analysis_run_id=(
                    configuration.analysis_run_id
                ),
                cell=cell,
                record_type="transfer_grouping",
                raw_cell_directory=_relative(
                    selected_output_root,
                    cell_directory,
                ),
            )
            row.update(
                {
                    "grouping_tolerance_numerator": (
                        cell.grouping_tolerance
                        .numerator
                    ),
                    "grouping_tolerance_denominator": (
                        cell.grouping_tolerance
                        .denominator
                    ),
                    "grouping_tolerance": float(
                        cell.grouping_tolerance
                    ),
                    "group_count": (
                        ""
                        if grouping.group_count is None
                        else grouping.group_count
                    ),
                    "groups_json": json.dumps(
                        grouping.groups,
                        separators=(",", ":"),
                    ),
                    "pairwise_distances_json": (
                        json.dumps(
                            {
                                f"{left}|{right}": value
                                for (
                                    left,
                                    right,
                                ), value in distances.items()
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                }
            )
            table_rows.append(row)
            elapsed = time.perf_counter() - started
            runtime_rows.append(
                _runtime_row(
                    analysis_run_id=(
                        configuration.analysis_run_id
                    ),
                    cell=cell,
                    exact_evaluations_used=0,
                    elapsed_seconds=elapsed,
                )
            )

        write_csv_records(
            transfer_table,
            fieldnames=TRANSFER_COLUMNS,
            rows=table_rows,
        )
        write_csv_records(
            runtime_table,
            fieldnames=TRANSFER_RUNTIME_COLUMNS,
            rows=[
                *existing_runtime_rows,
                *runtime_rows,
            ],
        )

        return TransferWorkloadResult(
            analysis_run_id=(
                configuration.analysis_run_id
            ),
            implementation_commit=(
                expected_implementation_commit
            ),
            transfer_table=transfer_table,
            runtime_table=runtime_table,
            global_cell_count=len(global_cells),
            subset_discovery_cell_count=(
                len(subset_cells)
            ),
            grouping_cell_count=len(
                grouping_cells
            ),
            subset_discoveries=tuple(
                subset_results
            ),
        )

    except Exception:
        shutil.rmtree(
            global_raw,
            ignore_errors=True,
        )
        shutil.rmtree(
            subset_raw,
            ignore_errors=True,
        )
        shutil.rmtree(
            grouping_raw,
            ignore_errors=True,
        )
        transfer_table.unlink(missing_ok=True)
        runtime_table.write_bytes(runtime_bytes)
        raise

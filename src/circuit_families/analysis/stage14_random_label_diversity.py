"""Execute the frozen Stage 14 primary diversity workload."""

from __future__ import annotations

import csv
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from circuit_families.analysis.fidelity_calibration import (
    write_csv_records,
)
from circuit_families.analysis.random_label_circuit_analysis import (
    load_frozen_analysis_configuration,
    load_random_label_checkpoint_context,
)
from circuit_families.analysis.stage12_artifacts import (
    Stage12CellArtifacts,
    write_stage12_cell_artifacts,
)
from circuit_families.analysis.stage12_reporting import (
    CIRCUIT_COLUMNS,
    FAMILY_SUMMARY_COLUMNS,
    PAIRWISE_OVERLAP_COLUMNS,
    RESTART_COLUMNS,
    Stage12ReportCell,
    circuit_rows,
    family_summary_rows,
    pairwise_overlap_rows,
    restart_rows,
)
from circuit_families.analysis.stage14_random_label_runner import (
    RUNTIME_COLUMNS,
    Stage14AnalysisCell,
    Stage14ExecutionPlan,
    adapt_random_label_search_context,
    build_execution_plan,
    find_stage15_artifacts,
    output_contract,
    validate_analysis_inputs,
)
from circuit_families.interpretability.diversity_forced_search import (
    NUMERICALLY_INDISTINGUISHABLE_TOLERANCE,
    CheckpointFamilySearchExecution,
    run_checkpoint_family_search,
)

DIVERSITY_RUNTIME_COLUMNS = (
    *RUNTIME_COLUMNS[:-4],
    "requested_member_index",
    "accepted_circuit",
    "restart_count",
    *RUNTIME_COLUMNS[-4:],
)


@dataclass(frozen=True)
class PrimaryDiversityCellExecution:
    """One completed primary family search and its artifacts."""

    cell: Stage14AnalysisCell
    source_context: Any
    execution: CheckpointFamilySearchExecution
    artifacts: Stage12CellArtifacts
    member_elapsed_seconds: Mapping[int, float]
    cell_elapsed_seconds: float


@dataclass(frozen=True)
class PrimaryDiversityWorkloadResult:
    """Outputs created by the seven primary family searches."""

    analysis_run_id: str
    implementation_commit: str
    raw_output_directory: Path
    family_summary_table: Path
    circuits_table: Path
    pairwise_overlap_table: Path
    restarts_table: Path
    runtime_table: Path
    cells: tuple[PrimaryDiversityCellExecution, ...]


def _search_checkpoint_index(
    cell: Stage14AnalysisCell,
) -> int:
    """Map the frozen zero-based plan index to the positive search index."""

    return cell.checkpoint_index + 1


def primary_diversity_cells(
    plan: Stage14ExecutionPlan,
) -> tuple[Stage14AnalysisCell, ...]:
    """Return the seven primary family cells in frozen order."""

    cells = tuple(
        cell
        for cell in plan.cells
        if cell.workload == "primary_diversity"
    )

    if len(cells) != 7:
        raise ValueError(
            f"Expected seven primary diversity cells, found {len(cells)}."
        )

    expected_steps = (
        200,
        3_400,
        7_450,
        8_150,
        8_500,
        8_650,
        9_050,
    )
    observed_steps = tuple(
        cell.checkpoint_step
        for cell in cells
    )

    if observed_steps != expected_steps:
        raise ValueError(
            "Primary diversity checkpoint order differs from the freeze."
        )

    for cell in cells:
        if cell.execution_mode != "execute":
            raise ValueError(
                "Every primary diversity cell must require execution."
            )

        if cell.fidelity_threshold != Fraction(99, 100):
            raise ValueError(
                "Primary diversity fidelity threshold differs from 99/100."
            )

        if cell.distinctness_cutoff != Fraction(1, 2):
            raise ValueError(
                "Primary diversity cutoff differs from 1/2."
            )

    return cells


def _runtime_callbacks(
) -> tuple[
    Callable[[int], None],
    Callable[[int], None],
    dict[int, float],
]:
    started: dict[int, float] = {}
    elapsed: dict[int, float] = {}

    def member_started(member_index: int) -> None:
        if member_index in started:
            raise RuntimeError(
                "Primary diversity member timer started twice."
            )

        started[member_index] = time.perf_counter()

    def member_finished(member_index: int) -> None:
        if member_index not in started:
            raise RuntimeError(
                "Primary diversity member timer finished before starting."
            )

        if member_index in elapsed:
            raise RuntimeError(
                "Primary diversity member timer finished twice."
            )

        elapsed[member_index] = (
            time.perf_counter()
            - started[member_index]
        )

    return member_started, member_finished, elapsed


def _grouped_restart_outcomes(
    execution: CheckpointFamilySearchExecution,
) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = {}

    for outcome in execution.result.restart_outcomes:
        grouped.setdefault(
            outcome.requested_member_index,
            [],
        ).append(outcome)

    return grouped


def primary_diversity_runtime_rows(
    *,
    analysis_run_id: str,
    result: PrimaryDiversityCellExecution,
) -> list[dict[str, object]]:
    """Return cell and requested-member runtime telemetry."""

    cell = result.cell
    execution = result.execution
    family = execution.result
    grouped = _grouped_restart_outcomes(execution)
    member_elapsed = dict(
        result.member_elapsed_seconds
    )

    if set(grouped) != set(member_elapsed):
        raise RuntimeError(
            "Primary diversity member runtime coverage does not match "
            "restart outcomes."
        )

    if cell.fidelity_threshold is None:
        raise ValueError(
            "Primary diversity cell lacks a fidelity threshold."
        )

    if cell.distinctness_cutoff is None:
        raise ValueError(
            "Primary diversity cell lacks a distinctness cutoff."
        )

    accepted = {
        member.member_index
        for member in family.members
    }

    rows: list[dict[str, object]] = [
        {
            "analysis_run_id": analysis_run_id,
            "workload": cell.workload,
            "cell_id": cell.cell_id,
            "checkpoint_index": cell.checkpoint_index,
            "checkpoint_step": cell.checkpoint_step,
            "fidelity_threshold": float(
                cell.fidelity_threshold
            ),
            "distinctness_cutoff": float(
                cell.distinctness_cutoff
            ),
            "discovery_subset": "",
            "grouping_tolerance": "",
            "requested_member_index": "",
            "accepted_circuit": "",
            "restart_count": len(
                family.restart_outcomes
            ),
            "record_type": "cell",
            "exact_evaluations_used": (
                family.exact_evaluations_used
            ),
            "elapsed_seconds": (
                result.cell_elapsed_seconds
            ),
            "included_in_deterministic_scientific_hashes": False,
        }
    ]

    for member_index in sorted(grouped):
        outcomes = grouped[member_index]

        rows.append(
            {
                "analysis_run_id": analysis_run_id,
                "workload": cell.workload,
                "cell_id": cell.cell_id,
                "checkpoint_index": (
                    cell.checkpoint_index
                ),
                "checkpoint_step": (
                    cell.checkpoint_step
                ),
                "fidelity_threshold": float(
                    cell.fidelity_threshold
                ),
                "distinctness_cutoff": float(
                    cell.distinctness_cutoff
                ),
                "discovery_subset": "",
                "grouping_tolerance": "",
                "requested_member_index": (
                    member_index
                ),
                "accepted_circuit": (
                    member_index in accepted
                ),
                "restart_count": len(outcomes),
                "record_type": "requested_member",
                "exact_evaluations_used": sum(
                    outcome.execution.result
                    .exact_evaluations_used
                    for outcome in outcomes
                ),
                "elapsed_seconds": (
                    member_elapsed[member_index]
                ),
                "included_in_deterministic_scientific_hashes": False,
            }
        )

    return rows


def _read_existing_runtime_rows(
    runtime_table: Path,
) -> list[dict[str, object]]:
    with runtime_table.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 7:
        raise ValueError(
            "Existing primary sparse runtime table must contain seven rows."
        )

    if any(
        row["workload"] != "primary_sparse"
        for row in rows
    ):
        raise ValueError(
            "Existing runtime table contains non-primary-sparse rows."
        )

    return [
        {
            column: row.get(column, "")
            for column in DIVERSITY_RUNTIME_COLUMNS
        }
        for row in rows
    ]


def _tie_tolerance(
    search_configuration: Mapping[str, Any],
) -> float:
    for key in (
        "tie_tolerance",
        "numerically_indistinguishable_tolerance",
    ):
        if key in search_configuration:
            return float(search_configuration[key])

    return NUMERICALLY_INDISTINGUISHABLE_TOLERANCE


def _expected_existing_outputs(
    *,
    configuration: Any,
    output_root: Path,
) -> dict[str, Path]:
    resolved = dict(
        output_contract(configuration).resolve(
            output_root
        )
    )

    existing = {
        name: path
        for name, path in resolved.items()
        if path.exists()
    }
    expected_names = {
        "raw_output_directory",
        "sparse_search_table",
        "runtime_table",
    }

    if set(existing) != expected_names:
        raise FileExistsError(
            "Primary diversity requires the exact completed "
            "primary-sparse output state. "
            f"Observed names: {sorted(existing)}."
        )

    return resolved


def execute_primary_diversity_workload(
    *,
    repository_root: str | Path,
    expected_implementation_commit: str,
    input_root: str | Path | None = None,
    output_root: str | Path | None = None,
    device: str = "cpu",
    progress_callback: Callable[[str], None] | None = None,
) -> PrimaryDiversityWorkloadResult:
    """Run and serialize the seven frozen primary family searches."""

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
    configuration = load_frozen_analysis_configuration(
        repository_root=repository
    )
    plan = build_execution_plan(configuration)
    cells = primary_diversity_cells(plan)
    resolved = _expected_existing_outputs(
        configuration=configuration,
        output_root=selected_output_root,
    )

    stage15_artifacts = find_stage15_artifacts(
        repository
    )

    if stage15_artifacts:
        raise FileExistsError(
            "Stage 15 artifacts exist before primary diversity execution."
        )

    if validation.current_commit != expected_implementation_commit:
        raise RuntimeError(
            "Validated implementation commit differs from expectation."
        )

    raw_output_directory = resolved[
        "raw_output_directory"
    ]
    diversity_raw_directory = (
        raw_output_directory
        / "primary_diversity"
    )
    family_summary_table = resolved[
        "family_summary_table"
    ]
    circuits_table = resolved["circuits_table"]
    pairwise_overlap_table = resolved[
        "pairwise_overlap_table"
    ]
    restarts_table = resolved["restart_table"]
    runtime_table = resolved["runtime_table"]

    generated_tables = (
        family_summary_table,
        circuits_table,
        pairwise_overlap_table,
        restarts_table,
    )

    for file_name in generated_tables:
        if file_name.exists():
            raise FileExistsError(
                f"Primary diversity output already exists: {file_name}"
            )

    if diversity_raw_directory.exists():
        raise FileExistsError(
            "Primary diversity raw directory already exists."
        )

    existing_runtime_bytes = (
        runtime_table.read_bytes()
    )
    existing_runtime_rows = (
        _read_existing_runtime_rows(
            runtime_table
        )
    )

    search_configuration = configuration.payload[
        "search"
    ]
    source_configuration = configuration.payload[
        "source"
    ]
    model_seed = int(
        source_configuration["model_seed"]
    )
    tie_tolerance = _tie_tolerance(
        search_configuration
    )

    completed: list[
        PrimaryDiversityCellExecution
    ] = []

    try:
        diversity_raw_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        for execution_index, cell in enumerate(
            cells,
            start=1,
        ):
            if progress_callback is not None:
                progress_callback(
                    f"[{execution_index:02d}/{len(cells):02d}] "
                    "primary_diversity "
                    f"checkpoint={cell.checkpoint_step}"
                )

            source_context = (
                load_random_label_checkpoint_context(
                    repository_root=repository,
                    configuration=configuration,
                    checkpoint_step=(
                        cell.checkpoint_step
                    ),
                    device=device,
                    output_root=selected_input_root,
                )
            )
            search_context = (
                adapt_random_label_search_context(
                    source_context
                )
            )

            if cell.fidelity_threshold is None:
                raise ValueError(
                    "Primary diversity cell lacks a threshold."
                )

            if cell.distinctness_cutoff is None:
                raise ValueError(
                    "Primary diversity cell lacks a cutoff."
                )

            (
                member_started,
                member_finished,
                member_elapsed,
            ) = _runtime_callbacks()

            started = time.perf_counter()
            execution = run_checkpoint_family_search(
                search_context,
                fidelity_threshold=float(
                    cell.fidelity_threshold
                ),
                distinctness_cutoff=(
                    cell.distinctness_cutoff
                ),
                model_seed=model_seed,
                checkpoint_index=(
                    _search_checkpoint_index(cell)
                ),
                ranking_batch_size=int(
                    search_configuration[
                        "ranking_batch_size"
                    ]
                ),
                evaluation_batch_size=int(
                    search_configuration[
                        "evaluation_batch_size"
                    ]
                ),
                family_target=int(
                    search_configuration[
                        "family_target"
                    ]
                ),
                max_restarts_per_alternative=int(
                    search_configuration[
                        "maximum_restarts_per_"
                        "requested_alternative"
                    ]
                ),
                per_requested_circuit_budget=int(
                    search_configuration[
                        "per_requested_circuit_"
                        "exact_evaluations"
                    ]
                ),
                per_cell_budget=int(
                    search_configuration[
                        "per_cell_exact_evaluations"
                    ]
                ),
                reuse_coefficient=float(
                    search_configuration[
                        "reuse_coefficient"
                    ]
                ),
                tie_tolerance=tie_tolerance,
                member_started_callback=member_started,
                member_finished_callback=member_finished,
            )
            cell_elapsed_seconds = (
                time.perf_counter()
                - started
            )

            raw_cell_directory = (
                diversity_raw_directory
                / f"step_{cell.checkpoint_step:08d}"
            )
            artifacts = write_stage12_cell_artifacts(
                raw_cell_directory,
                execution,
                cell_metadata={
                    "analysis_run_id": (
                        configuration.analysis_run_id
                    ),
                    "analysis_identity_sha256": (
                        configuration
                        .analysis_identity_sha256
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
                        source_context
                        .checkpoint_sha256
                    ),
                    "fidelity_threshold": float(
                        cell.fidelity_threshold
                    ),
                    "distinctness_cutoff": float(
                        cell.distinctness_cutoff
                    ),
                    "analysis_configuration_sha256": (
                        configuration.sha256
                    ),
                },
            )

            completed_cell = (
                PrimaryDiversityCellExecution(
                    cell=cell,
                    source_context=source_context,
                    execution=execution,
                    artifacts=artifacts,
                    member_elapsed_seconds=dict(
                        member_elapsed
                    ),
                    cell_elapsed_seconds=(
                        cell_elapsed_seconds
                    ),
                )
            )
            completed.append(completed_cell)

            if progress_callback is not None:
                progress_callback(
                    "completed "
                    f"checkpoint={cell.checkpoint_step} "
                    f"status={execution.result.status} "
                    f"family_size={execution.result.family_size} "
                    "exact_evaluations="
                    f"{execution.result.exact_evaluations_used}"
                )

        report_cells = tuple(
            Stage12ReportCell(
                cell_id=result.cell.cell_id,
                checkpoint_step=(
                    result.cell.checkpoint_step
                ),
                distinctness_cutoff=(
                    result.cell.distinctness_cutoff
                ),
                execution=result.execution,
                raw_cell_directory=(
                    result.artifacts
                    .output_directory
                    .resolve()
                    .relative_to(
                        selected_output_root
                    )
                    .as_posix()
                ),
            )
            for result in completed
            if result.cell.distinctness_cutoff
            is not None
        )

        if len(report_cells) != 7:
            raise RuntimeError(
                "Primary diversity report-cell count differs from seven."
            )

        family_rows: list[
            dict[str, Any]
        ] = []
        selected_circuit_rows: list[
            dict[str, Any]
        ] = []
        overlap_rows: list[
            dict[str, Any]
        ] = []
        all_restart_rows: list[
            dict[str, Any]
        ] = []

        for report_cell in report_cells:
            execution_order = (
                report_cell.distinctness_cutoff,
            )
            single_cell = (report_cell,)

            family_rows.extend(
                family_summary_rows(
                    stage12_run_id=(
                        configuration.analysis_run_id
                    ),
                    cells=single_cell,
                    execution_order=execution_order,
                )
            )
            selected_circuit_rows.extend(
                circuit_rows(
                    stage12_run_id=(
                        configuration.analysis_run_id
                    ),
                    cells=single_cell,
                    execution_order=execution_order,
                )
            )
            overlap_rows.extend(
                pairwise_overlap_rows(
                    stage12_run_id=(
                        configuration.analysis_run_id
                    ),
                    cells=single_cell,
                    execution_order=execution_order,
                )
            )
            all_restart_rows.extend(
                restart_rows(
                    stage12_run_id=(
                        configuration.analysis_run_id
                    ),
                    cells=single_cell,
                    execution_order=execution_order,
                )
            )

        write_csv_records(
            family_summary_table,
            fieldnames=FAMILY_SUMMARY_COLUMNS,
            rows=family_rows,
        )
        write_csv_records(
            circuits_table,
            fieldnames=CIRCUIT_COLUMNS,
            rows=selected_circuit_rows,
        )
        write_csv_records(
            pairwise_overlap_table,
            fieldnames=PAIRWISE_OVERLAP_COLUMNS,
            rows=overlap_rows,
        )
        write_csv_records(
            restarts_table,
            fieldnames=RESTART_COLUMNS,
            rows=all_restart_rows,
        )

        diversity_runtime_rows = [
            row
            for result in completed
            for row in primary_diversity_runtime_rows(
                analysis_run_id=(
                    configuration.analysis_run_id
                ),
                result=result,
            )
        ]
        write_csv_records(
            runtime_table,
            fieldnames=DIVERSITY_RUNTIME_COLUMNS,
            rows=[
                *existing_runtime_rows,
                *diversity_runtime_rows,
            ],
        )

        return PrimaryDiversityWorkloadResult(
            analysis_run_id=(
                configuration.analysis_run_id
            ),
            implementation_commit=(
                expected_implementation_commit
            ),
            raw_output_directory=(
                diversity_raw_directory
            ),
            family_summary_table=(
                family_summary_table
            ),
            circuits_table=circuits_table,
            pairwise_overlap_table=(
                pairwise_overlap_table
            ),
            restarts_table=restarts_table,
            runtime_table=runtime_table,
            cells=tuple(completed),
        )

    except Exception:
        shutil.rmtree(
            diversity_raw_directory,
            ignore_errors=True,
        )

        for file_name in generated_tables:
            file_name.unlink(missing_ok=True)

        runtime_table.write_bytes(
            existing_runtime_bytes
        )
        raise

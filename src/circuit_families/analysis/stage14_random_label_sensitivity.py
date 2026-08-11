"""Execute the frozen Stage 14 sensitivity workloads."""

from __future__ import annotations

import csv
import hashlib
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
from circuit_families.analysis.stage14_random_label_diversity import (
    DIVERSITY_RUNTIME_COLUMNS,
    PrimaryDiversityCellExecution,
    _runtime_callbacks,
    _search_checkpoint_index,
    _tie_tolerance,
    primary_diversity_runtime_rows,
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
from circuit_families.interpretability.diversity_forced_search import (
    CheckpointFamilySearchExecution,
    run_checkpoint_family_search,
)

SensitivityWorkload = Literal[
    "fidelity_sensitivity",
    "distinctness_sensitivity",
]

SENSITIVITY_COLUMNS = (
    "analysis_run_id",
    "workload",
    "cell_id",
    "sequence_index",
    "checkpoint_index",
    "checkpoint_step",
    "execution_mode",
    "dependency_cell_id",
    "fidelity_threshold_numerator",
    "fidelity_threshold_denominator",
    "fidelity_threshold",
    "distinctness_cutoff_numerator",
    "distinctness_cutoff_denominator",
    "distinctness_cutoff",
    "status",
    "stopping_reason",
    "family_size",
    "family_target",
    "right_censored",
    "exact_evaluations_used",
    "per_cell_budget",
    "budget_remaining",
    "restart_outcome_count",
    "scientifically_rerun",
    "raw_cell_directory",
    "cell_summary_path",
    "cell_summary_sha256",
)


@dataclass(frozen=True)
class SensitivityCellExecution:
    """One freshly executed sensitivity family-search cell."""

    cell: Stage14AnalysisCell
    source_context: Any
    execution: CheckpointFamilySearchExecution
    artifacts: Stage12CellArtifacts
    member_elapsed_seconds: Mapping[int, float]
    elapsed_seconds: float


@dataclass(frozen=True)
class SensitivityWorkloadResult:
    """Outputs from the frozen fidelity and distinctness grids."""

    analysis_run_id: str
    implementation_commit: str
    fidelity_table: Path
    distinctness_table: Path
    runtime_table: Path
    executed_cells: tuple[SensitivityCellExecution, ...]
    reference_cell_count: int


def sensitivity_cells(
    plan: Stage14ExecutionPlan,
    workload: SensitivityWorkload,
) -> tuple[Stage14AnalysisCell, ...]:
    """Return one frozen sensitivity grid in plan order."""

    if workload not in {
        "fidelity_sensitivity",
        "distinctness_sensitivity",
    }:
        raise ValueError(f"Unknown sensitivity workload: {workload}")

    cells = tuple(
        cell
        for cell in plan.cells
        if cell.workload == workload
    )
    expected_count = (
        6
        if workload == "fidelity_sensitivity"
        else 3
    )

    if len(cells) != expected_count:
        raise ValueError(
            f"{workload} expected {expected_count} cells, "
            f"found {len(cells)}."
        )

    if tuple(
        cell.checkpoint_step
        for cell in cells
    ) != (9_050,) * expected_count:
        raise ValueError(
            f"{workload} must use checkpoint 9050 only."
        )

    if sum(
        cell.execution_mode == "reference_primary"
        for cell in cells
    ) != 1:
        raise ValueError(
            f"{workload} must contain one primary reference."
        )

    return cells


def _sha256(file_name: Path) -> str:
    return hashlib.sha256(file_name.read_bytes()).hexdigest()


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
                for column in DIVERSITY_RUNTIME_COLUMNS
            }
            for row in csv.DictReader(handle)
        ]


def _read_primary_reference(
    *,
    family_summary_table: Path,
    checkpoint_step: int,
) -> dict[str, str]:
    with family_summary_table.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if int(row["checkpoint_step"])
            == checkpoint_step
        ]

    if len(matches) != 1:
        raise ValueError(
            "Primary reference family-summary row is not unique."
        )

    return matches[0]


def _result_row(
    *,
    analysis_run_id: str,
    output_root: Path,
    result: SensitivityCellExecution,
) -> dict[str, object]:
    cell = result.cell
    family = result.execution.result

    if cell.fidelity_threshold is None:
        raise ValueError("Sensitivity threshold is absent.")

    if cell.distinctness_cutoff is None:
        raise ValueError("Sensitivity cutoff is absent.")

    summary_path = (
        result.artifacts.output_directory
        / "cell_summary.json"
    )

    return {
        "analysis_run_id": analysis_run_id,
        "workload": cell.workload,
        "cell_id": cell.cell_id,
        "sequence_index": cell.sequence_index,
        "checkpoint_index": cell.checkpoint_index,
        "checkpoint_step": cell.checkpoint_step,
        "execution_mode": cell.execution_mode,
        "dependency_cell_id": (
            ""
            if cell.dependency_cell_id is None
            else cell.dependency_cell_id
        ),
        "fidelity_threshold_numerator": (
            cell.fidelity_threshold.numerator
        ),
        "fidelity_threshold_denominator": (
            cell.fidelity_threshold.denominator
        ),
        "fidelity_threshold": float(
            cell.fidelity_threshold
        ),
        "distinctness_cutoff_numerator": (
            cell.distinctness_cutoff.numerator
        ),
        "distinctness_cutoff_denominator": (
            cell.distinctness_cutoff.denominator
        ),
        "distinctness_cutoff": float(
            cell.distinctness_cutoff
        ),
        "status": family.status,
        "stopping_reason": family.stopping_reason,
        "family_size": family.family_size,
        "family_target": family.family_target,
        "right_censored": family.right_censored,
        "exact_evaluations_used": (
            family.exact_evaluations_used
        ),
        "per_cell_budget": family.per_cell_budget,
        "budget_remaining": family.budget_remaining,
        "restart_outcome_count": len(
            family.restart_outcomes
        ),
        "scientifically_rerun": True,
        "raw_cell_directory": _relative(
            output_root,
            result.artifacts.output_directory,
        ),
        "cell_summary_path": _relative(
            output_root,
            summary_path,
        ),
        "cell_summary_sha256": _sha256(
            summary_path
        ),
    }


def _reference_row(
    *,
    analysis_run_id: str,
    output_root: Path,
    cell: Stage14AnalysisCell,
    primary_row: Mapping[str, str],
    primary_raw_directory: Path,
) -> dict[str, object]:
    if cell.fidelity_threshold is None:
        raise ValueError("Sensitivity threshold is absent.")

    if cell.distinctness_cutoff is None:
        raise ValueError("Sensitivity cutoff is absent.")

    summary_path = (
        primary_raw_directory
        / f"step_{cell.checkpoint_step:08d}"
        / "cell_summary.json"
    )

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    return {
        "analysis_run_id": analysis_run_id,
        "workload": cell.workload,
        "cell_id": cell.cell_id,
        "sequence_index": cell.sequence_index,
        "checkpoint_index": cell.checkpoint_index,
        "checkpoint_step": cell.checkpoint_step,
        "execution_mode": cell.execution_mode,
        "dependency_cell_id": (
            ""
            if cell.dependency_cell_id is None
            else cell.dependency_cell_id
        ),
        "fidelity_threshold_numerator": (
            cell.fidelity_threshold.numerator
        ),
        "fidelity_threshold_denominator": (
            cell.fidelity_threshold.denominator
        ),
        "fidelity_threshold": float(
            cell.fidelity_threshold
        ),
        "distinctness_cutoff_numerator": (
            cell.distinctness_cutoff.numerator
        ),
        "distinctness_cutoff_denominator": (
            cell.distinctness_cutoff.denominator
        ),
        "distinctness_cutoff": float(
            cell.distinctness_cutoff
        ),
        "status": primary_row["status"],
        "stopping_reason": (
            primary_row["stopping_reason"]
        ),
        "family_size": int(
            primary_row["family_size"]
        ),
        "family_target": int(
            primary_row["family_target"]
        ),
        "right_censored": (
            primary_row["right_censored"]
        ),
        "exact_evaluations_used": int(
            primary_row["exact_evaluations_used"]
        ),
        "per_cell_budget": int(
            primary_row["per_cell_budget"]
        ),
        "budget_remaining": int(
            primary_row["budget_remaining"]
        ),
        "restart_outcome_count": int(
            primary_row["restart_outcome_count"]
        ),
        "scientifically_rerun": False,
        "raw_cell_directory": _relative(
            output_root,
            summary_path.parent,
        ),
        "cell_summary_path": _relative(
            output_root,
            summary_path,
        ),
        "cell_summary_sha256": _sha256(
            summary_path
        ),
    }


def execute_sensitivity_workload(
    *,
    repository_root: str | Path,
    expected_implementation_commit: str,
    input_root: str | Path | None = None,
    output_root: str | Path | None = None,
    device: str = "cpu",
    progress_callback: Callable[[str], None] | None = None,
) -> SensitivityWorkloadResult:
    """Run five fidelity and two distinctness sensitivity cells."""

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
            "Stage 15 artifacts exist before sensitivity execution."
        )

    configuration = load_frozen_analysis_configuration(
        repository_root=repository
    )
    plan = build_execution_plan(configuration)
    fidelity_cells = sensitivity_cells(
        plan,
        "fidelity_sensitivity",
    )
    distinctness_cells = sensitivity_cells(
        plan,
        "distinctness_sensitivity",
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
        "runtime_table",
    }
    observed_existing = {
        name
        for name, file_name in resolved.items()
        if file_name.exists()
    }

    if observed_existing != expected_existing:
        raise FileExistsError(
            "Sensitivity execution requires the exact "
            "completed primary output state."
        )

    raw_root = resolved["raw_output_directory"]
    fidelity_raw = raw_root / "fidelity_sensitivity"
    distinctness_raw = (
        raw_root / "distinctness_sensitivity"
    )
    fidelity_table = resolved[
        "fidelity_sensitivity_table"
    ]
    distinctness_table = resolved[
        "distinctness_sensitivity_table"
    ]
    runtime_table = resolved["runtime_table"]

    generated = (
        fidelity_table,
        distinctness_table,
    )

    for file_name in generated:
        if file_name.exists():
            raise FileExistsError(file_name)

    for directory in (
        fidelity_raw,
        distinctness_raw,
    ):
        if directory.exists():
            raise FileExistsError(directory)

    runtime_bytes = runtime_table.read_bytes()
    existing_runtime_rows = _read_runtime_rows(
        runtime_table
    )
    primary_reference = _read_primary_reference(
        family_summary_table=resolved[
            "family_summary_table"
        ],
        checkpoint_step=9_050,
    )
    primary_raw_directory = (
        raw_root / "primary_diversity"
    )
    search_configuration = configuration.payload[
        "search"
    ]
    model_seed = int(
        configuration.payload["source"]["model_seed"]
    )
    tie_tolerance = _tie_tolerance(
        search_configuration
    )
    completed: list[SensitivityCellExecution] = []

    try:
        fidelity_raw.mkdir(
            parents=True,
            exist_ok=False,
        )
        distinctness_raw.mkdir(
            parents=True,
            exist_ok=False,
        )

        execute_cells = tuple(
            cell
            for cell in (
                *fidelity_cells,
                *distinctness_cells,
            )
            if cell.execution_mode == "execute"
        )

        for index, cell in enumerate(
            execute_cells,
            start=1,
        ):
            if progress_callback is not None:
                progress_callback(
                    f"[{index:02d}/{len(execute_cells):02d}] "
                    f"{cell.workload} "
                    f"threshold={cell.fidelity_threshold} "
                    f"cutoff={cell.distinctness_cutoff}"
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
                    "Sensitivity threshold is absent."
                )

            if cell.distinctness_cutoff is None:
                raise ValueError(
                    "Sensitivity cutoff is absent."
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
            elapsed = time.perf_counter() - started

            workload_root = (
                fidelity_raw
                if cell.workload
                == "fidelity_sensitivity"
                else distinctness_raw
            )
            raw_cell_directory = (
                workload_root / cell.cell_id
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
                    "fidelity_threshold": float(
                        cell.fidelity_threshold
                    ),
                    "distinctness_cutoff": float(
                        cell.distinctness_cutoff
                    ),
                    "execution_mode": (
                        cell.execution_mode
                    ),
                },
            )
            completed_cell = SensitivityCellExecution(
                cell=cell,
                source_context=source_context,
                execution=execution,
                artifacts=artifacts,
                member_elapsed_seconds=dict(
                    member_elapsed
                ),
                elapsed_seconds=elapsed,
            )
            completed.append(completed_cell)

            if progress_callback is not None:
                progress_callback(
                    "completed "
                    f"cell={cell.cell_id} "
                    f"status={execution.result.status} "
                    f"family_size="
                    f"{execution.result.family_size} "
                    f"exact_evaluations="
                    f"{execution.result.exact_evaluations_used}"
                )

        completed_by_id = {
            result.cell.cell_id: result
            for result in completed
        }

        def rows_for(
            cells: tuple[Stage14AnalysisCell, ...],
        ) -> list[dict[str, object]]:
            rows: list[dict[str, object]] = []

            for cell in cells:
                if (
                    cell.execution_mode
                    == "reference_primary"
                ):
                    rows.append(
                        _reference_row(
                            analysis_run_id=(
                                configuration
                                .analysis_run_id
                            ),
                            output_root=(
                                selected_output_root
                            ),
                            cell=cell,
                            primary_row=primary_reference,
                            primary_raw_directory=(
                                primary_raw_directory
                            ),
                        )
                    )
                    continue

                rows.append(
                    _result_row(
                        analysis_run_id=(
                            configuration.analysis_run_id
                        ),
                        output_root=(
                            selected_output_root
                        ),
                        result=completed_by_id[
                            cell.cell_id
                        ],
                    )
                )

            return rows

        write_csv_records(
            fidelity_table,
            fieldnames=SENSITIVITY_COLUMNS,
            rows=rows_for(fidelity_cells),
        )
        write_csv_records(
            distinctness_table,
            fieldnames=SENSITIVITY_COLUMNS,
            rows=rows_for(distinctness_cells),
        )

        added_runtime_rows = [
            row
            for result in completed
            for row in primary_diversity_runtime_rows(
                analysis_run_id=(
                    configuration.analysis_run_id
                ),
                result=PrimaryDiversityCellExecution(
                    cell=result.cell,
                    source_context=(
                        result.source_context
                    ),
                    execution=result.execution,
                    artifacts=result.artifacts,
                    member_elapsed_seconds=(
                        result.member_elapsed_seconds
                    ),
                    cell_elapsed_seconds=(
                        result.elapsed_seconds
                    ),
                ),
            )
        ]
        write_csv_records(
            runtime_table,
            fieldnames=DIVERSITY_RUNTIME_COLUMNS,
            rows=[
                *existing_runtime_rows,
                *added_runtime_rows,
            ],
        )

        return SensitivityWorkloadResult(
            analysis_run_id=(
                configuration.analysis_run_id
            ),
            implementation_commit=(
                expected_implementation_commit
            ),
            fidelity_table=fidelity_table,
            distinctness_table=distinctness_table,
            runtime_table=runtime_table,
            executed_cells=tuple(completed),
            reference_cell_count=2,
        )

    except Exception:
        shutil.rmtree(
            fidelity_raw,
            ignore_errors=True,
        )
        shutil.rmtree(
            distinctness_raw,
            ignore_errors=True,
        )

        for file_name in generated:
            file_name.unlink(missing_ok=True)

        runtime_table.write_bytes(runtime_bytes)
        raise

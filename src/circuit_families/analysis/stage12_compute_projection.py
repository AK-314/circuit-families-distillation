"""Prospective compute projections from Stage 12 pilot effort."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from circuit_families.analysis.fidelity_calibration import (
    file_sha256,
    write_csv_records,
)
from circuit_families.interpretability.diversity_forced_search import (
    FAMILY_TARGET,
    PER_CELL_EXACT_EVALUATION_BUDGET,
)

FULL_CHECKPOINT_COUNT = 7
REDUCED_CHECKPOINT_COUNT = 3
FIDELITY_THRESHOLD_COUNT = 6
DISTINCTNESS_CUTOFF_COUNT = 3
MAIN_SEED_COUNT = 5
MINIMUM_CONTROL_COUNT = 2

FULL_GRID = "full_seven_checkpoint_grid"
REDUCED_GRID = "uniform_three_checkpoint_reduced_grid"
PRIMARY_ONLY = "primary_threshold_primary_cutoff"
SENSITIVITY = "fidelity_and_distinctness_sensitivity"
MAIN_ONLY = "main_seeds_only"
MAIN_PLUS_CONTROLS = "main_seeds_plus_minimum_controls"

FEASIBLE = "full_grid_technically_feasible"
INFEASIBLE = "full_grid_technically_infeasible"
UNRESOLVED = (
    "feasibility_unresolved_required_resource_assumption_absent"
)
UNRESOLVED_RANGE = (
    "feasibility_unresolved_between_observed_and_worst_case"
)

PROJECTION_COLUMNS = (
    "stage12_run_id",
    "scenario_id",
    "checkpoint_grid",
    "workload",
    "scope",
    "planned_checkpoint_count",
    "fidelity_threshold_count",
    "distinctness_cutoff_count",
    "main_seed_count",
    "control_count",
    "condition_count",
    "family_target",
    "circuits_requested_per_checkpoint",
    "projected_cell_count",
    "pilot_cell_count",
    "pilot_exact_evaluations",
    "pilot_runtime_seconds",
    "evaluations_per_recovered_circuit",
    "evaluations_per_failed_requested_alternative",
    "runtime_seconds_per_recovered_circuit",
    "runtime_seconds_per_failed_requested_alternative",
    "runtime_seconds_per_restart",
    "runtime_seconds_per_cell",
    "seconds_per_exact_evaluation",
    "observed_projection_exact_evaluations",
    "worst_case_budget_projection_exact_evaluations",
    "serial_observed_projection_seconds",
    "serial_worst_case_projection_seconds",
    "parallel_worker_count",
    "parallel_efficiency_assumption",
    "parallel_observed_projection_seconds",
    "parallel_worst_case_projection_seconds",
    "resource_ceiling_seconds",
    "technical_feasibility_conclusion",
    "checkpoint_grid_freeze_justified",
    "device",
    "ranking_batch_size",
    "evaluation_batch_size",
    "scientific_outcomes_used_for_projection",
)


@dataclass(frozen=True)
class PilotComputeProfile:
    """Observed Stage 12 effort used only for compute projection."""

    stage12_run_id: str
    pilot_cell_count: int
    pilot_exact_evaluations: int
    pilot_runtime_seconds: float
    recovered_circuit_count: int
    recovered_circuit_exact_evaluations: int
    recovered_circuit_runtime_seconds: float
    failed_requested_alternative_count: int
    failed_requested_alternative_exact_evaluations: int
    failed_requested_alternative_runtime_seconds: float
    restart_count: int
    restart_runtime_seconds: float
    device: str
    ranking_batch_size: int
    evaluation_batch_size: int


@dataclass(frozen=True)
class ProjectionScenario:
    """One fixed protocol-dimension projection scenario."""

    scenario_id: str
    checkpoint_grid: str
    workload: str
    scope: str
    planned_checkpoint_count: int
    fidelity_threshold_count: int
    distinctness_cutoff_count: int
    main_seed_count: int
    control_count: int


@dataclass(frozen=True)
class ComputeProjectionArtifacts:
    """Path and hash for the projection table."""

    table_path: Path
    table_sha256: str
    row_count: int


def _validate_positive_integer(
    value: int,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return value


def _validate_nonnegative_integer(
    value: int,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(
            f"{name} must be non-negative."
        )

    return value


def _validate_nonnegative_float(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(
            f"{name} must be a finite non-negative number."
        )

    return float(value)


def validate_pilot_compute_profile(
    profile: PilotComputeProfile,
) -> PilotComputeProfile:
    """Validate one observed pilot-effort summary."""

    if not isinstance(profile, PilotComputeProfile):
        raise TypeError(
            "profile must be a PilotComputeProfile."
        )

    if not profile.stage12_run_id:
        raise ValueError(
            "stage12_run_id must not be empty."
        )

    _validate_positive_integer(
        profile.pilot_cell_count,
        "pilot_cell_count",
    )
    _validate_positive_integer(
        profile.pilot_exact_evaluations,
        "pilot_exact_evaluations",
    )
    _validate_nonnegative_float(
        profile.pilot_runtime_seconds,
        "pilot_runtime_seconds",
    )
    _validate_nonnegative_integer(
        profile.recovered_circuit_count,
        "recovered_circuit_count",
    )
    _validate_nonnegative_integer(
        profile.recovered_circuit_exact_evaluations,
        "recovered_circuit_exact_evaluations",
    )
    _validate_nonnegative_float(
        profile.recovered_circuit_runtime_seconds,
        "recovered_circuit_runtime_seconds",
    )
    _validate_nonnegative_integer(
        profile.failed_requested_alternative_count,
        "failed_requested_alternative_count",
    )
    _validate_nonnegative_integer(
        profile.failed_requested_alternative_exact_evaluations,
        "failed_requested_alternative_exact_evaluations",
    )
    _validate_nonnegative_float(
        profile.failed_requested_alternative_runtime_seconds,
        "failed_requested_alternative_runtime_seconds",
    )
    _validate_nonnegative_integer(
        profile.restart_count,
        "restart_count",
    )
    _validate_nonnegative_float(
        profile.restart_runtime_seconds,
        "restart_runtime_seconds",
    )
    _validate_positive_integer(
        profile.ranking_batch_size,
        "ranking_batch_size",
    )
    _validate_positive_integer(
        profile.evaluation_batch_size,
        "evaluation_batch_size",
    )

    if not profile.device:
        raise ValueError("device must not be empty.")

    if (
        profile.recovered_circuit_exact_evaluations
        > profile.pilot_exact_evaluations
    ):
        raise ValueError(
            "Recovered-circuit evaluations exceed "
            "the pilot total."
        )

    if (
        profile.failed_requested_alternative_exact_evaluations
        > profile.pilot_exact_evaluations
    ):
        raise ValueError(
            "Failed-alternative evaluations exceed "
            "the pilot total."
        )

    return profile


def protocol_projection_scenarios(
) -> tuple[ProjectionScenario, ...]:
    """Return the eight Cartesian protocol scenarios."""

    grids = (
        (
            FULL_GRID,
            FULL_CHECKPOINT_COUNT,
        ),
        (
            REDUCED_GRID,
            REDUCED_CHECKPOINT_COUNT,
        ),
    )
    workloads = (
        (
            PRIMARY_ONLY,
            1,
            1,
        ),
        (
            SENSITIVITY,
            FIDELITY_THRESHOLD_COUNT,
            DISTINCTNESS_CUTOFF_COUNT,
        ),
    )
    scopes = (
        (
            MAIN_ONLY,
            MAIN_SEED_COUNT,
            0,
        ),
        (
            MAIN_PLUS_CONTROLS,
            MAIN_SEED_COUNT,
            MINIMUM_CONTROL_COUNT,
        ),
    )

    scenarios: list[ProjectionScenario] = []

    for (
        checkpoint_grid,
        checkpoint_count,
    ) in grids:
        for (
            workload,
            threshold_count,
            cutoff_count,
        ) in workloads:
            for (
                scope,
                main_seed_count,
                control_count,
            ) in scopes:
                scenarios.append(
                    ProjectionScenario(
                        scenario_id=(
                            f"{checkpoint_grid}__"
                            f"{workload}__{scope}"
                        ),
                        checkpoint_grid=checkpoint_grid,
                        workload=workload,
                        scope=scope,
                        planned_checkpoint_count=(
                            checkpoint_count
                        ),
                        fidelity_threshold_count=(
                            threshold_count
                        ),
                        distinctness_cutoff_count=(
                            cutoff_count
                        ),
                        main_seed_count=(
                            main_seed_count
                        ),
                        control_count=control_count,
                    )
                )

    return tuple(scenarios)


def _safe_average(
    total: int | float,
    count: int,
) -> float | str:
    if count == 0:
        return ""

    return float(total) / count


def _feasibility_conclusion(
    *,
    observed_parallel_seconds: float,
    worst_parallel_seconds: float,
    resource_ceiling_seconds: float | None,
) -> tuple[str, bool]:
    if resource_ceiling_seconds is None:
        return UNRESOLVED, False

    ceiling = _validate_nonnegative_float(
        resource_ceiling_seconds,
        "resource_ceiling_seconds",
    )

    if worst_parallel_seconds <= ceiling:
        return FEASIBLE, True

    if observed_parallel_seconds > ceiling:
        return INFEASIBLE, True

    return UNRESOLVED_RANGE, False


def projection_row(
    *,
    profile: PilotComputeProfile,
    scenario: ProjectionScenario,
    parallel_worker_count: int,
    parallel_efficiency_assumption: float,
    resource_ceiling_seconds: float | None,
) -> dict[str, Any]:
    """Project one fixed protocol scenario."""

    profile = validate_pilot_compute_profile(
        profile
    )
    workers = _validate_positive_integer(
        parallel_worker_count,
        "parallel_worker_count",
    )
    efficiency = _validate_nonnegative_float(
        parallel_efficiency_assumption,
        "parallel_efficiency_assumption",
    )

    if not 0.0 < efficiency <= 1.0:
        raise ValueError(
            "parallel_efficiency_assumption must be "
            "greater than zero and at most one."
        )

    condition_count = (
        scenario.main_seed_count
        + scenario.control_count
    )
    projected_cell_count = (
        scenario.planned_checkpoint_count
        * scenario.fidelity_threshold_count
        * scenario.distinctness_cutoff_count
        * condition_count
    )
    circuits_requested_per_checkpoint = (
        FAMILY_TARGET
        * scenario.fidelity_threshold_count
        * scenario.distinctness_cutoff_count
    )

    evaluations_per_cell = (
        profile.pilot_exact_evaluations
        / profile.pilot_cell_count
    )
    runtime_per_cell = (
        profile.pilot_runtime_seconds
        / profile.pilot_cell_count
    )
    seconds_per_exact_evaluation = (
        profile.pilot_runtime_seconds
        / profile.pilot_exact_evaluations
    )

    observed_evaluations = (
        evaluations_per_cell
        * projected_cell_count
    )
    worst_case_evaluations = (
        PER_CELL_EXACT_EVALUATION_BUDGET
        * projected_cell_count
    )

    serial_observed_seconds = (
        runtime_per_cell
        * projected_cell_count
    )
    serial_worst_case_seconds = (
        worst_case_evaluations
        * seconds_per_exact_evaluation
    )

    effective_workers = workers * efficiency
    parallel_observed_seconds = (
        serial_observed_seconds
        / effective_workers
    )
    parallel_worst_case_seconds = (
        serial_worst_case_seconds
        / effective_workers
    )

    conclusion, grid_freeze_justified = (
        _feasibility_conclusion(
            observed_parallel_seconds=(
                parallel_observed_seconds
            ),
            worst_parallel_seconds=(
                parallel_worst_case_seconds
            ),
            resource_ceiling_seconds=(
                resource_ceiling_seconds
            ),
        )
    )

    return {
        "stage12_run_id": profile.stage12_run_id,
        "scenario_id": scenario.scenario_id,
        "checkpoint_grid": (
            scenario.checkpoint_grid
        ),
        "workload": scenario.workload,
        "scope": scenario.scope,
        "planned_checkpoint_count": (
            scenario.planned_checkpoint_count
        ),
        "fidelity_threshold_count": (
            scenario.fidelity_threshold_count
        ),
        "distinctness_cutoff_count": (
            scenario.distinctness_cutoff_count
        ),
        "main_seed_count": scenario.main_seed_count,
        "control_count": scenario.control_count,
        "condition_count": condition_count,
        "family_target": FAMILY_TARGET,
        "circuits_requested_per_checkpoint": (
            circuits_requested_per_checkpoint
        ),
        "projected_cell_count": projected_cell_count,
        "pilot_cell_count": profile.pilot_cell_count,
        "pilot_exact_evaluations": (
            profile.pilot_exact_evaluations
        ),
        "pilot_runtime_seconds": (
            profile.pilot_runtime_seconds
        ),
        "evaluations_per_recovered_circuit": (
            _safe_average(
                profile
                .recovered_circuit_exact_evaluations,
                profile.recovered_circuit_count,
            )
        ),
        "evaluations_per_failed_requested_alternative": (
            _safe_average(
                profile
                .failed_requested_alternative_exact_evaluations,
                profile.failed_requested_alternative_count,
            )
        ),
        "runtime_seconds_per_recovered_circuit": (
            _safe_average(
                profile
                .recovered_circuit_runtime_seconds,
                profile.recovered_circuit_count,
            )
        ),
        "runtime_seconds_per_failed_requested_alternative": (
            _safe_average(
                profile
                .failed_requested_alternative_runtime_seconds,
                profile.failed_requested_alternative_count,
            )
        ),
        "runtime_seconds_per_restart": (
            _safe_average(
                profile.restart_runtime_seconds,
                profile.restart_count,
            )
        ),
        "runtime_seconds_per_cell": runtime_per_cell,
        "seconds_per_exact_evaluation": (
            seconds_per_exact_evaluation
        ),
        "observed_projection_exact_evaluations": (
            observed_evaluations
        ),
        "worst_case_budget_projection_exact_evaluations": (
            worst_case_evaluations
        ),
        "serial_observed_projection_seconds": (
            serial_observed_seconds
        ),
        "serial_worst_case_projection_seconds": (
            serial_worst_case_seconds
        ),
        "parallel_worker_count": workers,
        "parallel_efficiency_assumption": efficiency,
        "parallel_observed_projection_seconds": (
            parallel_observed_seconds
        ),
        "parallel_worst_case_projection_seconds": (
            parallel_worst_case_seconds
        ),
        "resource_ceiling_seconds": (
            ""
            if resource_ceiling_seconds is None
            else float(resource_ceiling_seconds)
        ),
        "technical_feasibility_conclusion": conclusion,
        "checkpoint_grid_freeze_justified": (
            grid_freeze_justified
        ),
        "device": profile.device,
        "ranking_batch_size": (
            profile.ranking_batch_size
        ),
        "evaluation_batch_size": (
            profile.evaluation_batch_size
        ),
        "scientific_outcomes_used_for_projection": (
            False
        ),
    }


def compute_projection_rows(
    *,
    profile: PilotComputeProfile,
    parallel_worker_count: int,
    parallel_efficiency_assumption: float,
    resource_ceiling_seconds: float | None,
    scenarios: Sequence[ProjectionScenario] | None = None,
) -> list[dict[str, Any]]:
    """Return projection rows in frozen scenario order."""

    active = (
        protocol_projection_scenarios()
        if scenarios is None
        else tuple(scenarios)
    )

    if not active:
        raise ValueError(
            "scenarios must not be empty."
        )

    identifiers = [
        scenario.scenario_id
        for scenario in active
    ]

    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "scenario identifiers must be unique."
        )

    return [
        projection_row(
            profile=profile,
            scenario=scenario,
            parallel_worker_count=(
                parallel_worker_count
            ),
            parallel_efficiency_assumption=(
                parallel_efficiency_assumption
            ),
            resource_ceiling_seconds=(
                resource_ceiling_seconds
            ),
        )
        for scenario in active
    ]


def write_compute_projection_table(
    path: str | Path,
    *,
    profile: PilotComputeProfile,
    parallel_worker_count: int,
    parallel_efficiency_assumption: float,
    resource_ceiling_seconds: float | None,
) -> ComputeProjectionArtifacts:
    """Write the prospective Stage 12 projection table."""

    output = write_csv_records(
        path,
        fieldnames=PROJECTION_COLUMNS,
        rows=compute_projection_rows(
            profile=profile,
            parallel_worker_count=(
                parallel_worker_count
            ),
            parallel_efficiency_assumption=(
                parallel_efficiency_assumption
            ),
            resource_ceiling_seconds=(
                resource_ceiling_seconds
            ),
        ),
    )

    return ComputeProjectionArtifacts(
        table_path=output,
        table_sha256=file_sha256(output),
        row_count=len(
            protocol_projection_scenarios()
        ),
    )

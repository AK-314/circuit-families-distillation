"""Tests for prospective Stage 12 compute projections."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from circuit_families.analysis.stage12_compute_projection import (
    FEASIBLE,
    FULL_GRID,
    INFEASIBLE,
    MAIN_ONLY,
    MAIN_PLUS_CONTROLS,
    PRIMARY_ONLY,
    REDUCED_GRID,
    SENSITIVITY,
    UNRESOLVED,
    UNRESOLVED_RANGE,
    PilotComputeProfile,
    compute_projection_rows,
    projection_row,
    protocol_projection_scenarios,
    write_compute_projection_table,
)


def pilot_profile() -> PilotComputeProfile:
    return PilotComputeProfile(
        stage12_run_id="fixture-run",
        pilot_cell_count=3,
        pilot_exact_evaluations=3_000,
        pilot_runtime_seconds=600.0,
        recovered_circuit_count=6,
        recovered_circuit_exact_evaluations=2_400,
        recovered_circuit_runtime_seconds=480.0,
        failed_requested_alternative_count=2,
        failed_requested_alternative_exact_evaluations=600,
        failed_requested_alternative_runtime_seconds=120.0,
        restart_count=8,
        restart_runtime_seconds=600.0,
        device="cpu",
        ranking_batch_size=256,
        evaluation_batch_size=256,
    )


def scenario(
    *,
    grid: str,
    workload: str,
    scope: str,
):
    matches = [
        value
        for value in protocol_projection_scenarios()
        if (
            value.checkpoint_grid == grid
            and value.workload == workload
            and value.scope == scope
        )
    ]

    assert len(matches) == 1
    return matches[0]


def test_protocol_projection_scenarios_are_frozen() -> None:
    scenarios = protocol_projection_scenarios()

    assert len(scenarios) == 8
    assert len(
        {
            value.scenario_id
            for value in scenarios
        }
    ) == 8

    full_primary_main = scenario(
        grid=FULL_GRID,
        workload=PRIMARY_ONLY,
        scope=MAIN_ONLY,
    )
    assert (
        full_primary_main.planned_checkpoint_count
        == 7
    )
    assert (
        full_primary_main.fidelity_threshold_count
        == 1
    )
    assert (
        full_primary_main.distinctness_cutoff_count
        == 1
    )
    assert full_primary_main.main_seed_count == 5
    assert full_primary_main.control_count == 0

    reduced_sensitivity_controls = scenario(
        grid=REDUCED_GRID,
        workload=SENSITIVITY,
        scope=MAIN_PLUS_CONTROLS,
    )
    assert (
        reduced_sensitivity_controls
        .planned_checkpoint_count
        == 3
    )
    assert (
        reduced_sensitivity_controls
        .fidelity_threshold_count
        == 6
    )
    assert (
        reduced_sensitivity_controls
        .distinctness_cutoff_count
        == 3
    )
    assert (
        reduced_sensitivity_controls.main_seed_count
        == 5
    )
    assert (
        reduced_sensitivity_controls.control_count
        == 2
    )


def test_projection_math_uses_only_compute_dimensions() -> None:
    row = projection_row(
        profile=pilot_profile(),
        scenario=scenario(
            grid=FULL_GRID,
            workload=PRIMARY_ONLY,
            scope=MAIN_ONLY,
        ),
        parallel_worker_count=4,
        parallel_efficiency_assumption=1.0,
        resource_ceiling_seconds=None,
    )

    assert row["projected_cell_count"] == 35
    assert (
        row["circuits_requested_per_checkpoint"]
        == 10
    )
    assert (
        row["observed_projection_exact_evaluations"]
        == 35_000
    )
    assert (
        row[
            "worst_case_budget_projection_exact_evaluations"
        ]
        == 1_750_000
    )
    assert (
        row["serial_observed_projection_seconds"]
        == 7_000.0
    )
    assert (
        row["parallel_observed_projection_seconds"]
        == 1_750.0
    )
    assert (
        row["technical_feasibility_conclusion"]
        == UNRESOLVED
    )
    assert (
        row["checkpoint_grid_freeze_justified"]
        is False
    )
    assert (
        row["scientific_outcomes_used_for_projection"]
        is False
    )


def test_feasibility_requires_explicit_resource_ceiling() -> None:
    active = scenario(
        grid=FULL_GRID,
        workload=PRIMARY_ONLY,
        scope=MAIN_ONLY,
    )

    feasible = projection_row(
        profile=pilot_profile(),
        scenario=active,
        parallel_worker_count=4,
        parallel_efficiency_assumption=1.0,
        resource_ceiling_seconds=100_000.0,
    )
    assert (
        feasible["technical_feasibility_conclusion"]
        == FEASIBLE
    )
    assert (
        feasible["checkpoint_grid_freeze_justified"]
        is True
    )

    infeasible = projection_row(
        profile=pilot_profile(),
        scenario=active,
        parallel_worker_count=4,
        parallel_efficiency_assumption=1.0,
        resource_ceiling_seconds=1_000.0,
    )
    assert (
        infeasible[
            "technical_feasibility_conclusion"
        ]
        == INFEASIBLE
    )
    assert (
        infeasible["checkpoint_grid_freeze_justified"]
        is True
    )

    unresolved = projection_row(
        profile=pilot_profile(),
        scenario=active,
        parallel_worker_count=4,
        parallel_efficiency_assumption=1.0,
        resource_ceiling_seconds=10_000.0,
    )
    assert (
        unresolved[
            "technical_feasibility_conclusion"
        ]
        == UNRESOLVED_RANGE
    )
    assert (
        unresolved[
            "checkpoint_grid_freeze_justified"
        ]
        is False
    )


def test_projection_table_is_deterministic(
    tmp_path: Path,
) -> None:
    first = write_compute_projection_table(
        tmp_path / "first.csv",
        profile=pilot_profile(),
        parallel_worker_count=4,
        parallel_efficiency_assumption=0.8,
        resource_ceiling_seconds=None,
    )
    second = write_compute_projection_table(
        tmp_path / "second.csv",
        profile=pilot_profile(),
        parallel_worker_count=4,
        parallel_efficiency_assumption=0.8,
        resource_ceiling_seconds=None,
    )

    assert (
        first.table_path.read_bytes()
        == second.table_path.read_bytes()
    )
    assert first.table_sha256 == second.table_sha256
    assert first.row_count == second.row_count == 8

    with first.table_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert {
        row["checkpoint_grid"]
        for row in rows
    } == {
        FULL_GRID,
        REDUCED_GRID,
    }
    assert {
        row["workload"]
        for row in rows
    } == {
        PRIMARY_ONLY,
        SENSITIVITY,
    }
    assert {
        row["scope"]
        for row in rows
    } == {
        MAIN_ONLY,
        MAIN_PLUS_CONTROLS,
    }


def test_projection_rejects_invalid_parallel_assumptions() -> None:
    with pytest.raises(
        ValueError,
        match="at most one",
    ):
        compute_projection_rows(
            profile=pilot_profile(),
            parallel_worker_count=4,
            parallel_efficiency_assumption=1.1,
            resource_ceiling_seconds=None,
        )

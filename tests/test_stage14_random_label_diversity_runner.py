"""Tests for Stage 14 primary diversity execution planning."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from circuit_families.analysis.random_label_circuit_analysis import (
    load_frozen_analysis_configuration,
)
from circuit_families.analysis.stage14_random_label_diversity import (
    DIVERSITY_RUNTIME_COLUMNS,
    primary_diversity_cells,
    primary_diversity_runtime_rows,
)
from circuit_families.analysis.stage14_random_label_runner import (
    build_execution_plan,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_primary_diversity_cells_are_exact() -> None:
    configuration = load_frozen_analysis_configuration(
        repository_root=repository_root()
    )
    plan = build_execution_plan(configuration)
    cells = primary_diversity_cells(plan)

    assert len(cells) == 7
    assert tuple(
        cell.checkpoint_step
        for cell in cells
    ) == (
        200,
        3_400,
        7_450,
        8_150,
        8_500,
        8_650,
        9_050,
    )
    assert all(
        float(cell.fidelity_threshold) == 0.99
        for cell in cells
    )
    assert all(
        float(cell.distinctness_cutoff) == 0.5
        for cell in cells
    )
    assert all(
        cell.execution_mode == "execute"
        for cell in cells
    )


def test_diversity_runtime_schema_is_unique() -> None:
    assert len(DIVERSITY_RUNTIME_COLUMNS) == 16
    assert len(DIVERSITY_RUNTIME_COLUMNS) == len(
        set(DIVERSITY_RUNTIME_COLUMNS)
    )
    assert "requested_member_index" in (
        DIVERSITY_RUNTIME_COLUMNS
    )
    assert "accepted_circuit" in (
        DIVERSITY_RUNTIME_COLUMNS
    )
    assert "restart_count" in (
        DIVERSITY_RUNTIME_COLUMNS
    )


def test_primary_diversity_runtime_rows_cover_cell_and_member() -> None:
    configuration = load_frozen_analysis_configuration(
        repository_root=repository_root()
    )
    plan = build_execution_plan(configuration)
    cell = primary_diversity_cells(plan)[0]

    search_result = SimpleNamespace(
        exact_evaluations_used=123,
    )
    outcome = SimpleNamespace(
        requested_member_index=1,
        execution=SimpleNamespace(
            result=search_result,
        ),
    )
    member = SimpleNamespace(
        member_index=1,
    )
    family = SimpleNamespace(
        restart_outcomes=(outcome,),
        members=(member,),
        exact_evaluations_used=123,
    )
    execution = SimpleNamespace(
        result=family,
    )
    result = SimpleNamespace(
        cell=cell,
        execution=execution,
        member_elapsed_seconds={1: 1.25},
        cell_elapsed_seconds=2.5,
    )

    rows = primary_diversity_runtime_rows(
        analysis_run_id="analysis-run",
        result=result,
    )

    assert len(rows) == 2
    assert rows[0]["record_type"] == "cell"
    assert rows[0]["restart_count"] == 1
    assert (
        rows[1]["record_type"]
        == "requested_member"
    )
    assert rows[1]["requested_member_index"] == 1
    assert rows[1]["accepted_circuit"] is True
    assert rows[1]["exact_evaluations_used"] == 123


def test_runner_help_exposes_primary_diversity_mode() -> None:
    completed = subprocess.run(
        [
            "/Users/alexkolesnikov/.local/bin/uv",
            "run",
            "python",
            "scripts/run_stage14_random_label_analysis.py",
            "--help",
        ],
        cwd=repository_root(),
        capture_output=True,
        text=True,
        check=True,
    )

    assert (
        "--execute-primary-diversity"
        in completed.stdout
    )



def test_tie_tolerance_uses_authoritative_engine_default() -> None:
    from circuit_families.analysis.stage14_random_label_diversity import (
        _tie_tolerance,
    )
    from circuit_families.interpretability.diversity_forced_search import (
        NUMERICALLY_INDISTINGUISHABLE_TOLERANCE,
    )

    assert _tie_tolerance({}) == (
        NUMERICALLY_INDISTINGUISHABLE_TOLERANCE
    )
    assert _tie_tolerance(
        {"tie_tolerance": 2.0e-12}
    ) == 2.0e-12
    assert _tie_tolerance(
        {
            "numerically_indistinguishable_tolerance": (
                3.0e-12
            )
        }
    ) == 3.0e-12



def test_primary_diversity_uses_frozen_restart_output_key() -> None:
    import inspect

    from circuit_families.analysis.random_label_circuit_analysis import (
        load_frozen_analysis_configuration,
    )
    from circuit_families.analysis.stage14_random_label_diversity import (
        execute_primary_diversity_workload,
    )
    from circuit_families.analysis.stage14_random_label_runner import (
        output_contract,
    )

    repository = repository_root()
    configuration = load_frozen_analysis_configuration(
        repository_root=repository
    )
    resolved = dict(
        output_contract(configuration).resolve(
            repository
        )
    )
    source = inspect.getsource(
        execute_primary_diversity_workload
    )

    assert "restart_table" in resolved
    assert "restarts_table" not in resolved
    assert 'resolved["restart_table"]' in source
    assert 'resolved["restarts_table"]' not in source



def test_primary_diversity_maps_plan_indices_to_positive_search_indices(
) -> None:
    from circuit_families.analysis.random_label_circuit_analysis import (
        load_frozen_analysis_configuration,
    )
    from circuit_families.analysis.stage14_random_label_diversity import (
        _search_checkpoint_index,
        primary_diversity_cells,
    )
    from circuit_families.analysis.stage14_random_label_runner import (
        build_execution_plan,
    )

    repository = repository_root()
    configuration = load_frozen_analysis_configuration(
        repository_root=repository
    )
    cells = primary_diversity_cells(
        build_execution_plan(configuration)
    )

    assert tuple(
        cell.checkpoint_index
        for cell in cells
    ) == tuple(range(7))
    assert tuple(
        _search_checkpoint_index(cell)
        for cell in cells
    ) == tuple(range(1, 8))

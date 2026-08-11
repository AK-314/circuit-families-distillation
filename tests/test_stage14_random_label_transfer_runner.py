"""Tests for Stage 14 transfer workload planning."""

from __future__ import annotations

import subprocess
from pathlib import Path

from circuit_families.analysis.random_label_circuit_analysis import (
    load_frozen_analysis_configuration,
)
from circuit_families.analysis.stage14_random_label_runner import (
    build_execution_plan,
)
from circuit_families.analysis.stage14_random_label_transfer import (
    TRANSFER_COLUMNS,
    transfer_cells,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_transfer_cell_counts_are_exact() -> None:
    configuration = load_frozen_analysis_configuration(
        repository_root=repository_root()
    )
    plan = build_execution_plan(configuration)

    assert len(
        transfer_cells(
            plan,
            "global_family_transfer",
        )
    ) == 7
    assert len(
        transfer_cells(
            plan,
            "subset_discovery",
        )
    ) == 28
    assert len(
        transfer_cells(
            plan,
            "transfer_grouping",
        )
    ) == 21


def test_transfer_schema_is_unique() -> None:
    assert len(TRANSFER_COLUMNS) == len(
        set(TRANSFER_COLUMNS)
    )
    assert "record_type" in TRANSFER_COLUMNS
    assert "groups_json" in TRANSFER_COLUMNS
    assert "scientifically_executed" in TRANSFER_COLUMNS


def test_runner_help_exposes_transfer_mode() -> None:
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

    assert "--execute-transfer" in completed.stdout

def test_transfer_rows_use_explicit_mask_retained_count() -> None:
    from types import SimpleNamespace

    from circuit_families.analysis.stage14_random_label_transfer import (
        _evaluation_rows,
    )

    configuration = load_frozen_analysis_configuration(
        repository_root=repository_root()
    )
    cell = transfer_cells(
        build_execution_plan(configuration),
        "global_family_transfer",
    )[0]
    metrics = SimpleNamespace(
        primary_fidelity=0.99,
        prediction_agreement_count=100,
        evaluated_example_count=101,
    )
    transfer = SimpleNamespace(
        evaluations=(
            SimpleNamespace(
                metrics=metrics,
                evaluation_subset="Q1",
                circuit_id="C1",
            ),
        )
    )

    rows = _evaluation_rows(
        analysis_run_id="analysis",
        cell=cell,
        transfer=transfer,
        source_member_index=1,
        retained_component_count=123,
        raw_cell_directory="results/raw/cell",
        record_type="global_family_evaluation",
    )

    assert rows[0]["retained_component_count"] == 123

def test_subset_discovery_uses_frozen_per_requested_circuit_budget(
) -> None:
    import inspect

    from circuit_families.analysis.stage14_random_label_transfer import (
        execute_transfer_workload,
    )

    source = inspect.getsource(
        execute_transfer_workload
    )

    assert (
        "per_requested_circuit_exact_evaluations"
        in source
    )
    assert "per_discovery_search_budget" not in source

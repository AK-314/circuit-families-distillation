"""Tests for Stage 14 sensitivity execution planning."""

from __future__ import annotations

import subprocess
from pathlib import Path

from circuit_families.analysis.random_label_circuit_analysis import (
    load_frozen_analysis_configuration,
)
from circuit_families.analysis.stage14_random_label_runner import (
    build_execution_plan,
)
from circuit_families.analysis.stage14_random_label_sensitivity import (
    SENSITIVITY_COLUMNS,
    sensitivity_cells,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_sensitivity_grids_are_exact() -> None:
    configuration = load_frozen_analysis_configuration(
        repository_root=repository_root()
    )
    plan = build_execution_plan(configuration)

    fidelity = sensitivity_cells(
        plan,
        "fidelity_sensitivity",
    )
    distinctness = sensitivity_cells(
        plan,
        "distinctness_sensitivity",
    )

    assert len(fidelity) == 6
    assert len(distinctness) == 3
    assert sum(
        cell.execution_mode == "execute"
        for cell in fidelity
    ) == 5
    assert sum(
        cell.execution_mode == "execute"
        for cell in distinctness
    ) == 2
    assert sum(
        cell.execution_mode == "reference_primary"
        for cell in (*fidelity, *distinctness)
    ) == 2


def test_sensitivity_schema_is_unique() -> None:
    assert len(SENSITIVITY_COLUMNS) == len(
        set(SENSITIVITY_COLUMNS)
    )
    assert "scientifically_rerun" in SENSITIVITY_COLUMNS
    assert "cell_summary_sha256" in SENSITIVITY_COLUMNS


def test_runner_help_exposes_sensitivity_mode() -> None:
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

    assert "--execute-sensitivity" in completed.stdout

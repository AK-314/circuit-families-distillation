"""Tests for deterministic Stage 16 genuine-task transfer orchestration."""

from __future__ import annotations

import inspect
import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from circuit_families.analysis.stage16_transfer import (
    DISTANCE_COLUMNS,
    GROUPING_TOLERANCES,
    JACCARD_CUTOFF,
    PRIMARY_CELL_ID,
    PRIMARY_GROUPING_TOLERANCE,
    SCIENTIFIC_TABLE_NAMES,
    _discovery_status,
    _group_maps,
    _pairwise_rows,
    _safe_archive_name,
    load_and_validate_stage12_family,
    load_stage16_configuration,
    null_subset_transfer_rows,
    validate_stage16_inputs,
)
from circuit_families.analysis.transfer import TransferProfile
from circuit_families.data.input_subsets import SUBSET_NAMES

ROOT = Path(__file__).resolve().parents[1]


def _profiles() -> tuple[TransferProfile, ...]:
    return tuple(
        TransferProfile(
            circuit_id=f"C{index}",
            q1_fidelity=0.80 + index * 0.01,
            q2_fidelity=0.81 + index * 0.01,
            q3_fidelity=0.82 + index * 0.01,
            q4_fidelity=0.83 + index * 0.01,
        )
        for index in range(1, 8)
    )


def _overlaps() -> dict[tuple[str, str], Fraction]:
    return {
        (f"C{left}", f"C{right}"): Fraction(left, left + right + 20)
        for left in range(1, 8)
        for right in range(left + 1, 8)
    }


def test_frozen_stage16_configuration_is_exact() -> None:
    configuration = load_stage16_configuration(ROOT)

    assert configuration.run_id.startswith("stage16-transfer-s1-")
    assert configuration.payload["primary_family_cell"] == PRIMARY_CELL_ID
    assert configuration.payload["source_family_size"] == 7
    assert configuration.payload["checkpoint_step"] == 9050
    assert configuration.payload["subset_order"] == list(SUBSET_NAMES)
    assert configuration.payload["stage17_started"] is False
    assert JACCARD_CUTOFF == Fraction(1, 2)


def test_real_stage12_primary_family_loads_exactly_seven_circuits() -> None:
    configuration = load_stage16_configuration(ROOT)
    circuits, overlaps = load_and_validate_stage12_family(ROOT, configuration)

    assert [circuit.circuit_id for circuit in circuits] == [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
    ]
    assert len({circuit.mask_sha256 for circuit in circuits}) == 7
    assert all(circuit.mask.retained_component_count <= 258 for circuit in circuits)
    assert len(overlaps) == 21
    assert all(overlap <= Fraction(1, 2) for overlap in overlaps.values())


def test_grouping_tolerances_are_exact_and_frozen() -> None:
    assert GROUPING_TOLERANCES == (
        Fraction(1, 40),
        Fraction(1, 20),
        Fraction(1, 10),
    )
    assert PRIMARY_GROUPING_TOLERANCE == Fraction(1, 20)


def test_group_maps_use_deterministic_labels_and_validate_linkage() -> None:
    maps, rows = _group_maps(_profiles())

    assert tuple(maps) == GROUPING_TOLERANCES
    assert set(maps[Fraction(1, 20)]) == {f"C{index}" for index in range(1, 8)}
    assert all(label.startswith("G") for label in maps[Fraction(1, 20)].values())
    assert all(row["complete_linkage_valid"] is True for row in rows)


def test_pairwise_join_contains_exactly_21_unique_pairs() -> None:
    profiles = _profiles()
    maps, _ = _group_maps(profiles)
    rows = _pairwise_rows(
        run_id="run",
        profiles=profiles,
        overlaps=_overlaps(),
        group_maps=maps,
    )

    assert len(rows) == 21
    assert len({(row["circuit_i"], row["circuit_j"]) for row in rows}) == 21
    assert all(set(row) == set(DISTANCE_COLUMNS) for row in rows)


def test_maximum_distance_subset_tie_uses_q1_to_q4_order() -> None:
    profiles = (
        TransferProfile("C1", 0.0, 0.0, 0.0, 0.0),
        TransferProfile("C2", 0.1, 0.1, 0.1, 0.1),
        *tuple(
            TransferProfile(f"C{index}", 0.2 + index / 100, 0.3, 0.4, 0.5)
            for index in range(3, 8)
        ),
    )
    maps, _ = _group_maps(profiles)
    rows = _pairwise_rows(
        run_id="run",
        profiles=profiles,
        overlaps=_overlaps(),
        group_maps=maps,
    )

    assert rows[0]["maximum_distance_subset"] == "Q1"


def test_failed_discovery_rows_are_explicit_nulls_not_zero() -> None:
    rows = null_subset_transfer_rows(
        run_id="run",
        discovery_subset="Q2",
        discovery_status="budget_exhaustion",
    )

    assert [row["evaluation_subset"] for row in rows] == list(SUBSET_NAMES)
    assert all(row["discovery_subset"] == "Q2" for row in rows)
    assert all(row["fidelity"] == "" for row in rows)
    assert all(row["prediction_agreement_count"] == "" for row in rows)
    assert not any(row["fidelity"] == 0 for row in rows)


def test_unknown_null_discovery_subset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown discovery subset"):
        null_subset_transfer_rows(
            run_id="run",
            discovery_subset="Q5",
            discovery_status="search_failure",
        )


@pytest.mark.parametrize(
    ("search_status", "stage16_status"),
    [
        ("valid_sparse_circuit", "valid_meaningfully_sparse"),
        ("valid_but_not_meaningfully_sparse", "valid_but_not_meaningfully_sparse"),
        ("ranking_failure", "search_failure"),
        ("budget_exhaustion", "budget_exhaustion"),
        ("fidelity_failure", "fidelity_failure"),
        ("invalid_masking_output", "invalid_masking_output"),
    ],
)
def test_discovery_status_classification(
    search_status: str,
    stage16_status: str,
) -> None:
    execution = SimpleNamespace(result=SimpleNamespace(status=search_status))
    assert _discovery_status(execution) == stage16_status


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../b"])
def test_stage12_archive_unsafe_paths_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_archive_name(name)


def test_stage12_archive_safe_path_is_canonical() -> None:
    assert _safe_archive_name("run/cell/final_mask.json") == (
        "run/cell/final_mask.json"
    )


def test_scientific_tables_exclude_runtime() -> None:
    assert "runtime" not in SCIENTIFIC_TABLE_NAMES
    assert len(SCIENTIFIC_TABLE_NAMES) == 7


def test_validate_only_can_ignore_existing_stage16_outputs() -> None:
    signature = inspect.signature(validate_stage16_inputs)
    assert "require_outputs_absent" in signature.parameters
    source = inspect.getsource(validate_stage16_inputs)
    assert "if require_outputs_absent" in source


def test_definitive_execution_requires_output_absence() -> None:
    from circuit_families.analysis.stage16_transfer import execute_stage16

    source = inspect.getsource(execute_stage16)
    assert "require_outputs_absent=True" in source
    assert "progress_callback" in source


def test_cli_exposes_validation_reproduction_and_clean_commit_modes() -> None:
    completed = subprocess.run(
        [
            "/Users/alexkolesnikov/.local/bin/uv",
            "run",
            "python",
            "scripts/run_stage16_transfer.py",
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--validate-inputs-only" in completed.stdout
    assert "--expected-implementation-commit" in completed.stdout
    assert "--output-root" in completed.stdout
    assert "--compare-reference-root" in completed.stdout
    assert "{cpu,cuda}" in completed.stdout


def test_cli_validate_only_is_output_state_independent() -> None:
    source = (ROOT / "scripts/run_stage16_transfer.py").read_text(encoding="utf-8")
    assert "require_outputs_absent=False" in source


def test_stage17_is_never_executed_by_stage16_module() -> None:
    source = inspect.getsource(
        __import__(
            "circuit_families.analysis.stage16_transfer",
            fromlist=["stage16_transfer"],
        )
    )
    assert "execute_stage17" not in source

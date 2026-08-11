"""Tests for the frozen Stage 17 registry and pipeline contract."""

import csv
import inspect
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from circuit_families.analysis.stage17_execution import (
    RUNTIME_COLUMNS,
    CellSearchResult,
    _empty_transfer,
    _failure_category,
    _matrix_rows,
    _scientific_rows,
    _write_figures,
    _write_tables,
    build_output_paths,
    compare_reproduction,
    deterministic_stage17_run_id,
    execute_stage17,
    normalize_reference_search,
    reference_primary_transfer,
    robustness_classification,
    validate_absent_outputs,
    validate_archive_member_contract,
)
from circuit_families.analysis.stage17_sensitivity import (
    CHECKPOINT_STEP,
    FAMILY_TARGET,
    PER_CELL_BUDGET,
    PRIMARY_CELL_KEY,
    PRIMARY_TRANSFER_TOLERANCE,
    Stage17InputValidation,
    _safe_archive_name,
    build_stage17_registry,
    cell_for,
    circuit_size_summary,
    load_stage17_configuration,
    structural_overlap_summary,
    validate_stage17_inputs,
)
from circuit_families.analysis.transfer import TransferProfile, transfer_grouping

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_validation() -> Stage17InputValidation:
    return validate_stage17_inputs(REPOSITORY)


def _zero_search(cell) -> CellSearchResult:
    return CellSearchResult(
        cell=cell,
        status="fidelity_failure",
        stopping_reason="requested_member_1_fidelity_failure",
        family_size=0,
        right_censored=False,
        exact_evaluations_used=100,
        budget_remaining=PER_CELL_BUDGET - 100,
        circuits=(),
        overlaps=(),
        restart_rows=(),
        ranking_passes=1,
        restarts_attempted=1,
        failed_requested_member_count=1,
        terminal_requested_member_index=1,
        invalid_output_count=0,
        raw_cell_directory=Path("unused"),
        search_integrity={},
    )


def _zero_surface() -> tuple[CellSearchResult, ...]:
    return tuple(_zero_search(cell) for cell in build_stage17_registry())


def _classification_surface(
    default: int,
    overrides: dict[tuple[Fraction, Fraction], int] | None = None,
):
    values = overrides or {}
    return tuple(
        SimpleNamespace(cell=cell, family_size=values.get(cell.key, default))
        for cell in build_stage17_registry()
    )


def test_registry_has_exactly_eighteen_unique_cells() -> None:
    registry = build_stage17_registry()
    assert len(registry) == 18
    assert len({cell.key for cell in registry}) == 18
    assert len({cell.cell_id for cell in registry}) == 18


def test_registry_execution_mode_identity() -> None:
    registry = build_stage17_registry()
    assert sum(cell.search_execution_mode == "fresh_execution" for cell in registry) == 15
    assert sum(cell.search_execution_mode == "reference_existing_result" for cell in registry) == 3
    assert (
        sum(cell.transfer_execution_mode == "reference_existing_result" for cell in registry) == 1
    )


def test_every_fresh_cell_has_an_independent_full_budget() -> None:
    fresh = [
        cell for cell in build_stage17_registry() if cell.search_execution_mode == "fresh_execution"
    ]
    assert len(fresh) == 15
    assert all(cell.expected_search_budget == 50_000 for cell in fresh)
    assert len({cell.cell_id for cell in fresh}) == 15


def test_only_point_nine_nine_cells_reference_stage12() -> None:
    referenced = tuple(cell for cell in build_stage17_registry() if cell.search_source_stage == 12)
    assert {cell.key for cell in referenced} == {
        (Fraction(99, 100), Fraction(1, 4)),
        (Fraction(99, 100), Fraction(1, 2)),
        (Fraction(99, 100), Fraction(3, 4)),
    }


def test_only_primary_cell_references_stage16_transfer() -> None:
    referenced = tuple(
        cell
        for cell in build_stage17_registry()
        if cell.transfer_execution_mode == "reference_existing_result"
    )
    assert len(referenced) == 1
    assert referenced[0].key == PRIMARY_CELL_KEY


def test_primary_and_budget_constants_are_frozen() -> None:
    assert PRIMARY_CELL_KEY == (Fraction(99, 100), Fraction(1, 2))
    assert PRIMARY_TRANSFER_TOLERANCE == Fraction(1, 20)
    assert PER_CELL_BUDGET == 50_000
    assert FAMILY_TARGET == 10


def test_unplanned_cell_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unplanned"):
        cell_for("0.70", "0.50")


def test_registry_records_preserve_exact_rationals() -> None:
    primary = cell_for("0.99", "0.5").to_record()
    assert primary["fidelity_numerator"] == 99
    assert primary["fidelity_denominator"] == 100
    assert primary["distinctness_numerator"] == 1
    assert primary["distinctness_denominator"] == 2
    assert primary["displayed_fidelity"] == "0.990"
    assert primary["displayed_jaccard_cutoff"] == "0.50"


def test_configuration_repeats_exact_frozen_science() -> None:
    configuration = load_stage17_configuration(REPOSITORY).payload
    assert configuration["search"] == {
        "exact_evaluation_examples": 12_769,
        "candidate_removal_batch_size_maximum": 16,
        "ranking_batch_size": 256,
        "evaluation_batch_size": 256,
        "reuse_coefficient": 0.5,
        "maximum_restarts_per_alternative": 5,
        "per_requested_circuit_budget": 10_000,
        "per_cell_budget": 50_000,
        "family_target": 10,
        "cell_order": "fidelity_grid_then_distinctness_grid",
        "tie_tolerance": 1.0e-12,
    }
    assert configuration["component_universe"]["total_searchable_components"] == 516
    assert configuration["sparsity"]["maximum_retained_components"] == 258


def test_real_reference_family_sizes_and_source_cells(
    real_validation: Stage17InputValidation,
) -> None:
    assert [family.source_cell_id for family in real_validation.reference_families] == [
        "cutoff-0.25",
        "cutoff-0.50",
        "cutoff-0.75",
    ]
    assert [len(family.circuits) for family in real_validation.reference_families] == [3, 7, 7]


def test_real_reference_circuit_counts_match_summary(
    real_validation: Stage17InputValidation,
) -> None:
    for family in real_validation.reference_families:
        assert len(family.circuits) == int(family.summary_row["family_size"])
        assert [circuit.member_index for circuit in family.circuits] == list(
            range(1, len(family.circuits) + 1)
        )


def test_real_reference_masks_and_fidelities_are_exact(
    real_validation: Stage17InputValidation,
) -> None:
    for family in real_validation.reference_families:
        for circuit in family.circuits:
            assert len(circuit.mask_sha256) == 64
            assert circuit.mask.retained_component_count <= 258
            agreement = int(circuit.circuit_row["prediction_agreement_count"])
            evaluated = int(circuit.circuit_row["evaluated_example_count"])
            assert agreement * 100 >= evaluated * 99


def test_real_reference_overlaps_obey_each_cutoff(
    real_validation: Stage17InputValidation,
) -> None:
    for family in real_validation.reference_families:
        cutoff = family.stage17_cell.distinctness_cutoff
        assert all(
            Fraction(int(row["jaccard_numerator"]), int(row["jaccard_denominator"])) <= cutoff
            for row in family.pairwise_rows
        )


def test_real_primary_transfer_reference_matches_stage16(
    real_validation: Stage17InputValidation,
) -> None:
    reference = real_validation.primary_transfer_reference
    assert len(reference.profiles) == 7
    assert reference.group_count == 1
    assert reference.groups == (("C1", "C2", "C3", "C4", "C5", "C6", "C7"),)
    assert all(len(profile.values) == 4 for profile in reference.profiles)


def test_real_primary_transfer_profile_order_is_q1_to_q4(
    real_validation: Stage17InputValidation,
) -> None:
    first = real_validation.primary_transfer_reference.profile_rows[0]
    profile = real_validation.primary_transfer_reference.profiles[0]
    assert profile.values == tuple(
        float(first[f"{subset}_fidelity"]) for subset in ("q1", "q2", "q3", "q4")
    )


def test_reference_normalization_writes_only_a_reference_record(
    real_validation: Stage17InputValidation,
    tmp_path: Path,
) -> None:
    family = real_validation.reference_families[0]
    result = normalize_reference_search(
        "stage17-test",
        family,
        tmp_path / "cell",
        real_validation.source_hashes,
    )
    assert result.family_size == 3
    assert result.cell.search_execution_mode == "reference_existing_result"
    assert [path.name for path in (tmp_path / "cell").iterdir()] == ["search_reference.json"]


def test_primary_transfer_normalization_reproduces_reference(
    real_validation: Stage17InputValidation,
    tmp_path: Path,
) -> None:
    family = next(
        family for family in real_validation.reference_families if family.stage17_cell.is_primary
    )
    search = normalize_reference_search(
        "stage17-test",
        family,
        tmp_path / "primary",
        real_validation.source_hashes,
    )
    transfer = reference_primary_transfer("stage17-test", real_validation, search)
    assert transfer.group_count == real_validation.primary_transfer_reference.group_count
    assert tuple(profile.values for profile in transfer.profiles) == tuple(
        profile.values for profile in real_validation.primary_transfer_reference.profiles
    )


def test_zero_family_size_summary_uses_nulls() -> None:
    summary = circuit_size_summary(())
    assert summary["circuit_count"] == 0
    assert all(value is None for key, value in summary.items() if key != "circuit_count")


def test_size_258_passes_and_259_fails() -> None:
    assert circuit_size_summary((258,))["maximum_retained_components"] == 258
    with pytest.raises(ValueError, match="invalid retained count"):
        circuit_size_summary((259,))


@pytest.mark.parametrize("family_size", (0, 1))
def test_empty_and_singleton_overlap_summaries_are_null(family_size: int) -> None:
    summary = structural_overlap_summary((), family_size=family_size, cutoff=Fraction(1, 2))
    assert summary["pair_count"] == 0
    assert all(value is None for key, value in summary.items() if key != "pair_count")


def test_maximum_overlap_determines_cutoff_compliance() -> None:
    summary = structural_overlap_summary(
        (Fraction(1, 10), Fraction(3, 5), Fraction(1, 10)),
        family_size=3,
        cutoff=Fraction(1, 2),
    )
    assert summary["mean_pairwise_overlap"] < 0.5
    assert summary["maximum_pairwise_jaccard_overlap"] == 0.6
    assert summary["cutoff_compliance"] is False


def test_zero_family_transfer_group_count_is_null() -> None:
    cell = build_stage17_registry()[0]
    transfer = _empty_transfer(cell)
    assert transfer.group_count is None
    assert transfer.profiles == ()


def test_singleton_transfer_group_count_is_one() -> None:
    grouping = transfer_grouping(
        (TransferProfile("C1", 0.8, 0.8, 0.8, 0.8),),
        tolerance=Fraction(1, 20),
    )
    assert grouping.group_count == 1
    assert grouping.groups == (("C1",),)


def test_zero_surface_matrix_contains_all_zero_cells() -> None:
    rows = _matrix_rows("stage17-test", _zero_surface())
    assert len(rows) == 6
    assert all(
        row[column] == 0
        for row in rows
        for column in (
            "family_size_cutoff_0_25",
            "family_size_cutoff_0_50",
            "family_size_cutoff_0_75",
        )
    )


def test_zero_surface_figure_sources_keep_all_eighteen_cells() -> None:
    searches = _zero_surface()
    transfers = {search.cell.cell_id: _empty_transfer(search.cell) for search in searches}
    rows = _scientific_rows("stage17-test", searches, transfers)
    assert len(rows["family_size_heatmap_source_table"]) == 18
    assert len(rows["family_size_curves_source_table"]) == 18
    assert all(row["zero_family"] for row in rows["family_size_heatmap_source_table"])


def test_zero_surface_tables_preserve_nulls_and_completeness(
    real_validation: Stage17InputValidation,
    tmp_path: Path,
) -> None:
    searches = _zero_surface()
    transfers = {search.cell.cell_id: _empty_transfer(search.cell) for search in searches}
    paths = build_output_paths(real_validation, output_root=tmp_path)
    _write_tables(paths, "stage17-test", searches, transfers, ())
    with paths.tables["cells_table"].open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 18
    with paths.tables["circuit_size_summary_table"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    assert all(row["circuit_count"] == "0" for row in rows)
    assert all(row["mean_retained_components"] == "" for row in rows)


def test_actual_references_write_all_nonempty_table_schemas(
    real_validation: Stage17InputValidation,
    tmp_path: Path,
) -> None:
    reference_by_key = {
        family.stage17_cell.key: family for family in real_validation.reference_families
    }
    searches = []
    for cell in build_stage17_registry():
        if cell.key in reference_by_key:
            searches.append(
                normalize_reference_search(
                    "stage17-test",
                    reference_by_key[cell.key],
                    tmp_path / "raw" / cell.cell_id,
                    real_validation.source_hashes,
                )
            )
        else:
            searches.append(_zero_search(cell))
    transfers = {search.cell.cell_id: _empty_transfer(search.cell) for search in searches}
    primary = next(search for search in searches if search.cell.is_primary)
    transfers[primary.cell.cell_id] = reference_primary_transfer(
        "stage17-test", real_validation, primary
    )
    paths = build_output_paths(real_validation, output_root=tmp_path / "outputs")
    _write_tables(paths, "stage17-test", searches, transfers, ())
    expected_counts = {
        "circuits_table": 17,
        "pairwise_overlap_table": 45,
        "transfer_profiles_table": 7,
        "transfer_distances_table": 21,
        "transfer_groups_table": 1,
    }
    for key, expected in expected_counts.items():
        with paths.tables[key].open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == expected


def test_synthetic_figures_are_byte_deterministic(
    real_validation: Stage17InputValidation,
    tmp_path: Path,
) -> None:
    searches = _zero_surface()
    first = build_output_paths(real_validation, output_root=tmp_path / "first")
    second = build_output_paths(real_validation, output_root=tmp_path / "second")
    _write_figures(first, searches)
    _write_figures(second, searches)
    assert all(
        first.figures[key].read_bytes() == second.figures[key].read_bytes() for key in first.figures
    )


def test_heatmap_explicitly_disables_interpolation() -> None:
    assert 'interpolation="none"' in inspect.getsource(_write_figures)


def test_robust_across_grid_classification() -> None:
    assert (
        robustness_classification(_classification_surface(2))
        == "robust across the frozen sensitivity grid"
    )


def test_limited_neighbourhood_classification() -> None:
    surface = _classification_surface(
        0,
        {
            (fidelity, cutoff): 2
            for fidelity in (Fraction(39, 40), Fraction(99, 100))
            for cutoff in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
        },
    )
    assert (
        robustness_classification(surface)
        == "robust only within a limited neighbourhood of the primary cell"
    )


def test_fragile_classification_on_immediate_reversal() -> None:
    surface = _classification_surface(2, {(Fraction(39, 40), Fraction(1, 2)): 0})
    assert robustness_classification(surface) == "threshold-sensitive or fragile"


def test_mixed_classification_on_non_immediate_neighbourhood_reversal() -> None:
    surface = _classification_surface(2, {(Fraction(39, 40), Fraction(1, 4)): 0})
    assert robustness_classification(surface) == "mixed"


def test_unresolved_classification_when_primary_is_not_multiple() -> None:
    surface = _classification_surface(2, {PRIMARY_CELL_KEY: 1})
    assert robustness_classification(surface) == "unresolved"


@pytest.mark.parametrize(
    ("status", "reason", "expected"),
    (
        ("fidelity_failure", "requested_member_1_fidelity_failure", "fidelity_failure"),
        ("sparsity_failure", "requested_member_1_sparsity_failure", "sparsity_failure"),
        (
            "distinctness_failure",
            "requested_member_2_distinctness_failure",
            "distinctness_failure",
        ),
        ("budget_exhaustion", "requested_member_2_budget_exhaustion", "budget_exhaustion"),
        ("search_failure", "requested_member_1_search_failure", "search_failure"),
        (
            "no_feasible_sparse_candidate_discovered_within_budget",
            "requested_member_1_no_feasible",
            "no_feasible_candidate_discovered_within_tested_search",
        ),
        (
            "invalid_masking_output",
            "requested_member_1_invalid_masking_output",
            "invalid_masking_output",
        ),
    ),
)
def test_search_failure_categories_are_preserved(status: str, reason: str, expected: str) -> None:
    search = _zero_search(build_stage17_registry()[0])
    altered = SimpleNamespace(
        right_censored=False,
        status=status,
        stopping_reason=reason,
    )
    assert search.failed_requested_member_count == 1
    assert _failure_category(altered) == expected


def test_output_contract_contains_every_required_stage17_table(
    real_validation: Stage17InputValidation,
    tmp_path: Path,
) -> None:
    paths = build_output_paths(real_validation, output_root=tmp_path)
    expected = {
        "cells_table",
        "family_size_matrix_table",
        "family_summary_table",
        "circuits_table",
        "pairwise_overlap_table",
        "restarts_table",
        "search_failures_table",
        "search_failure_summary_table",
        "circuit_size_summary_table",
        "transfer_profiles_table",
        "transfer_distances_table",
        "transfer_groups_table",
        "frontier_table",
        "family_size_curves_source_table",
        "family_size_distinctness_source_table",
        "family_size_heatmap_source_table",
        "runtime_table",
    }
    assert set(paths.tables) == expected


def test_runtime_is_separate_from_deterministic_table_contract(
    real_validation: Stage17InputValidation,
    tmp_path: Path,
) -> None:
    paths = build_output_paths(real_validation, output_root=tmp_path)
    assert paths.tables["runtime_table"].name == "seed_1_stage17_runtime.csv"
    assert "included_in_deterministic_scientific_hashes" in RUNTIME_COLUMNS


def test_definitive_output_paths_refuse_overwrite(
    real_validation: Stage17InputValidation,
    tmp_path: Path,
) -> None:
    paths = build_output_paths(real_validation, output_root=tmp_path)
    paths.tables["cells_table"].parent.mkdir(parents=True)
    paths.tables["cells_table"].write_text("occupied\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        validate_absent_outputs(paths)


def test_run_id_is_deterministic_and_commit_sensitive(
    real_validation: Stage17InputValidation,
) -> None:
    digest = real_validation.configuration.sha256
    first = deterministic_stage17_run_id(digest, "a" * 40)
    assert first == deterministic_stage17_run_id(digest, "a" * 40)
    assert first != deterministic_stage17_run_id(digest, "b" * 40)


@pytest.mark.parametrize("name", ("../bad", "/absolute", "a/../b", ""))
def test_unsafe_archive_member_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_archive_name(name)


def test_safe_archive_member_name_is_canonical() -> None:
    assert _safe_archive_name("run/cell/final_mask.json") == "run/cell/final_mask.json"


def test_archive_member_contract_rejects_duplicates(tmp_path: Path) -> None:
    member = tmp_path / "member.json"
    member.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        validate_archive_member_contract(tmp_path, (member, member))


def test_archive_member_contract_rejects_missing_member(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing"):
        validate_archive_member_contract(tmp_path, (tmp_path / "missing.json",))


def test_archive_member_contract_rejects_out_of_root(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "member.json"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside output root"):
        validate_archive_member_contract(tmp_path, (outside,))


def test_reproduction_comparison_detects_byte_mismatch(tmp_path: Path) -> None:
    run_id = "stage17-sensitivity-s1-test"
    reference = tmp_path / "reference"
    reproduction = tmp_path / "reproduction"
    relative = Path("results/tables/seed_1_stage17_family_summary.csv")
    for root in (reference, reproduction):
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_text("same\n", encoding="utf-8")
    matching = compare_reproduction(reference, reproduction, run_id=run_id)
    assert matching["byte_identical"]
    assert matching["passed"]
    (reproduction / relative).write_text("different\n", encoding="utf-8")
    mismatch = compare_reproduction(reference, reproduction, run_id=run_id)
    assert not mismatch["byte_identical"]
    assert not mismatch["passed"]


def test_validate_only_mode_is_output_state_independent(tmp_path: Path) -> None:
    occupied = tmp_path / "results/tables/seed_1_stage17_sensitivity_cells.csv"
    occupied.parent.mkdir(parents=True)
    occupied.write_text("occupied\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage17_sensitivity.py",
            "--repository-root",
            str(REPOSITORY),
            "--output-root",
            str(tmp_path),
            "--validate-inputs-only",
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "stage17_validate_inputs_only: passed" in completed.stdout
    assert occupied.read_text(encoding="utf-8") == "occupied\n"


def test_cli_exposes_validation_reproduction_and_cpu_only() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_stage17_sensitivity.py", "--help"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--validate-inputs-only" in completed.stdout
    assert "--compare-reproduction-root" in completed.stdout
    assert "--output-root" in completed.stdout
    assert "--device {cpu}" in completed.stdout
    assert "overwrite" not in completed.stdout.lower()


def test_stage15_unavailability_is_not_a_grid_cell(
    real_validation: Stage17InputValidation,
) -> None:
    assert len(real_validation.registry) == 18
    assert all("stage15" not in cell.cell_id for cell in real_validation.registry)


def test_checkpoint_grid_freeze_precedes_the_stage18_training_transition() -> None:
    expected_freeze_paths = {
        REPOSITORY / "manifests/post_stage17_checkpoint_grid_and_concurrency_freeze.json",
        REPOSITORY / "results/notes/post_stage17_checkpoint_grid_and_concurrency_freeze.md",
        REPOSITORY / "results/tables/post_stage17_concurrency_benchmark_summary.csv",
    }
    assert all(path.is_file() for path in expected_freeze_paths)

    training_manifest = REPOSITORY / "manifests/stage18_training.json"
    if training_manifest.is_file():
        payload = json.loads(training_manifest.read_text(encoding="utf-8"))
        assert payload["experiment_stage"] == 18
        assert payload["checkpoint_count"] == 35
        assert payload["stage19_started"] is False
        return

    permitted_stage18 = {
        REPOSITORY / "results/tables/stage18_main_seed_registry_pre_execution.csv",
        REPOSITORY / "results/tables/stage18_cell_registry_pre_execution.csv",
        REPOSITORY / "results/tables/stage18_worker_shards_pre_execution.csv",
        *{
            REPOSITORY / f"manifests/stage18_worker_shards/worker_{index:02d}.json"
            for index in range(12)
        },
    }
    for directory_name in ("manifests", "results", "figures"):
        for path in (REPOSITORY / directory_name).rglob("*"):
            if path.is_file():
                lowered = path.name.lower()
                if "stage18" in path.as_posix().lower():
                    assert path in permitted_stage18
                if "checkpoint_grid" in lowered or "scaled_checkpoint" in lowered:
                    assert path in expected_freeze_paths


def test_stage17_runner_does_not_execute_stage18() -> None:
    source = (REPOSITORY / "src/circuit_families/analysis/stage17_execution.py").read_text(
        encoding="utf-8"
    )
    assert "run_stage18" not in source
    assert '"stage18_started": False' in source


def test_runner_reuses_existing_family_search_and_records_empty_start() -> None:
    source = inspect.getsource(execute_stage17)
    assert "run_checkpoint_family_search" in source
    assert '"initial_family_size": 0' in source
    assert "per_cell_budget=PER_CELL_BUDGET" in source
    assert "per_requested_circuit_budget=PER_REQUESTED_CIRCUIT_BUDGET" in source


def test_validate_only_rejects_wrong_checkpoint() -> None:
    with pytest.raises(ValueError, match="9050"):
        validate_stage17_inputs(REPOSITORY, checkpoint_step=CHECKPOINT_STEP - 1)

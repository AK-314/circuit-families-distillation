"""Tests for the validate-only integrated Stage 14 runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from circuit_families.analysis.random_label_circuit_analysis import (
    ANALYSIS_RUN_ID,
    load_frozen_analysis_configuration,
)
from circuit_families.analysis.stage14_random_label_runner import (
    OUTPUT_RECORD_NAMES,
    Stage14OutputContract,
    build_execution_plan,
    current_git_commit,
    existing_output_paths,
    find_stage15_artifacts,
    output_contract,
    validate_analysis_inputs,
)


def repository_root() -> Path:
    """Return the repository root used by the test process."""

    return Path.cwd().resolve()


def test_execution_plan_has_exact_frozen_matrix() -> None:
    configuration = load_frozen_analysis_configuration(
        repository_root=repository_root()
    )
    plan = build_execution_plan(configuration)

    assert plan.analysis_run_id == ANALYSIS_RUN_ID
    assert len(plan.cells) == 79
    assert plan.execute_cell_count == 77
    assert plan.reference_cell_count == 2

    assert plan.workload_count("primary_sparse") == 7
    assert plan.workload_count("primary_diversity") == 7
    assert plan.workload_count("fidelity_sensitivity") == 6
    assert plan.workload_count("distinctness_sensitivity") == 3
    assert plan.workload_count("global_family_transfer") == 7
    assert plan.workload_count("subset_discovery") == 28
    assert plan.workload_count("transfer_grouping") == 21

    assert tuple(
        cell.sequence_index
        for cell in plan.cells
    ) == tuple(range(1, 80))

    assert len(
        {
            cell.cell_id
            for cell in plan.cells
        }
    ) == 79


def test_primary_search_cells_follow_checkpoint_order() -> None:
    configuration = load_frozen_analysis_configuration(
        repository_root=repository_root()
    )
    plan = build_execution_plan(configuration)

    sparse_cells = tuple(
        cell
        for cell in plan.cells
        if cell.workload == "primary_sparse"
    )
    family_cells = tuple(
        cell
        for cell in plan.cells
        if cell.workload == "primary_diversity"
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

    assert tuple(
        cell.checkpoint_step
        for cell in sparse_cells
    ) == expected_steps
    assert tuple(
        cell.checkpoint_step
        for cell in family_cells
    ) == expected_steps

    assert all(
        cell.execution_mode == "execute"
        for cell in sparse_cells
    )
    assert all(
        cell.execution_mode == "execute"
        for cell in family_cells
    )


def test_primary_sensitivity_duplicates_are_references() -> None:
    configuration = load_frozen_analysis_configuration(
        repository_root=repository_root()
    )
    plan = build_execution_plan(configuration)

    reference_cells = tuple(
        cell
        for cell in plan.cells
        if cell.execution_mode == "reference_primary"
    )

    assert len(reference_cells) == 2

    assert {
        cell.workload
        for cell in reference_cells
    } == {
        "fidelity_sensitivity",
        "distinctness_sensitivity",
    }

    assert {
        cell.dependency_cell_id
        for cell in reference_cells
    } == {
        "primary-family-step-00009050",
    }


def test_subset_discovery_matrix_is_seven_by_four() -> None:
    configuration = load_frozen_analysis_configuration(
        repository_root=repository_root()
    )
    plan = build_execution_plan(configuration)

    cells = tuple(
        cell
        for cell in plan.cells
        if cell.workload == "subset_discovery"
    )

    assert len(cells) == 28
    assert {
        cell.discovery_subset
        for cell in cells
    } == {"Q1", "Q2", "Q3", "Q4"}

    for checkpoint_step in (
        200,
        3_400,
        7_450,
        8_150,
        8_500,
        8_650,
        9_050,
    ):
        matching = tuple(
            cell
            for cell in cells
            if cell.checkpoint_step == checkpoint_step
        )

        assert tuple(
            cell.discovery_subset
            for cell in matching
        ) == ("Q1", "Q2", "Q3", "Q4")


def test_output_contract_matches_frozen_configuration() -> None:
    configuration = load_frozen_analysis_configuration(
        repository_root=repository_root()
    )
    contract = output_contract(configuration)

    assert tuple(
        name
        for name, _ in contract.records
    ) == OUTPUT_RECORD_NAMES

    mapping = contract.as_mapping()

    assert mapping["sparse_search_table"] == (
        "results/tables/"
        "seed_0_stage14_random_label_sparse_search.csv"
    )
    assert mapping["archive"] == (
        "results/archives/"
        "stage14-random-label-analysis-s0-7b472aa5163a.tar.gz"
    )
    assert mapping["manifest"] == (
        "manifests/"
        "stage14_random_label_analysis_"
        "stage14-random-label-analysis-s0-7b472aa5163a.json"
    )


def test_existing_output_detection_is_read_only(
    tmp_path: Path,
) -> None:
    contract = Stage14OutputContract(
        records=(
            (
                "first",
                Path("results/tables/first.csv"),
            ),
            (
                "second",
                Path("results/tables/second.csv"),
            ),
        )
    )

    first_file = (
        tmp_path
        / "results"
        / "tables"
        / "first.csv"
    )
    first_file.parent.mkdir(parents=True)
    first_file.write_text("value\n", encoding="utf-8")

    existing = existing_output_paths(
        contract,
        output_root=tmp_path,
    )

    assert existing == (("first", first_file.resolve()),)
    assert not (
        tmp_path
        / "results"
        / "tables"
        / "second.csv"
    ).exists()


def test_stage15_scan_is_confined_to_standard_roots(
    tmp_path: Path,
) -> None:
    result_file = (
        tmp_path
        / "results"
        / "tables"
        / "stage15_example.csv"
    )
    result_file.parent.mkdir(parents=True)
    result_file.write_text("value\n", encoding="utf-8")

    unrelated_file = (
        tmp_path
        / "scratch"
        / "stage15_unrelated.txt"
    )
    unrelated_file.parent.mkdir(parents=True)
    unrelated_file.write_text("value\n", encoding="utf-8")

    assert find_stage15_artifacts(tmp_path) == (
        result_file.resolve(),
    )


def test_stage15_scan_ignores_only_administrative_resolution_files(
    tmp_path: Path,
) -> None:
    administrative_files = (
        tmp_path / "manifests/stage15_no_generalisation_unavailable.json",
        tmp_path / "results/notes/stage15_no_generalisation_unavailable.md",
    )
    scientific_file = tmp_path / "results/tables/stage15_circuits.csv"

    for file_name in (*administrative_files, scientific_file):
        file_name.parent.mkdir(parents=True, exist_ok=True)
        file_name.write_text("record\n", encoding="utf-8")

    assert find_stage15_artifacts(tmp_path) == (scientific_file.resolve(),)


def test_validation_report_is_json_serialisable() -> None:
    repository = repository_root()
    head = current_git_commit(repository)
    configuration = load_frozen_analysis_configuration(
        repository_root=repository
    )
    expected_existing_outputs = {
        output_name
        for output_name, _ in existing_output_paths(
            output_contract(configuration),
            output_root=repository,
        )
    }

    report = validate_analysis_inputs(
        repository_root=repository,
        expected_implementation_commit=head,
        require_clean_repository=False,
        require_outputs_absent=False,
        verify_checkpoint_hashes=False,
    )
    record = report.to_record()

    assert record["analysis_run_id"] == ANALYSIS_RUN_ID
    assert record["current_commit"] == head
    assert len(record["verified_sources"]) == 11
    assert len(record["verified_checkpoints"]) == 7
    assert {
        output["output_name"]
        for output in record["existing_outputs"]
    } == expected_existing_outputs
    assert record["stage15_artifacts"] == []

    json.dumps(
        record,
        sort_keys=True,
        allow_nan=False,
    )


def test_runner_script_rejects_execution_without_validate_flag() -> None:
    repository = repository_root()
    head = current_git_commit(repository)
    configuration = load_frozen_analysis_configuration(
        repository_root=repository
    )
    contract = output_contract(configuration)

    def output_snapshot() -> tuple[object, ...]:
        records: list[object] = []

        for output_name, file_name in contract.resolve(
            repository
        ):
            if file_name.is_file():
                stat = file_name.stat()
                records.append(
                    (
                        output_name,
                        "file",
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )
                continue

            if file_name.is_dir():
                members = tuple(
                    (
                        member.relative_to(
                            file_name
                        ).as_posix(),
                        member.stat().st_size,
                        member.stat().st_mtime_ns,
                    )
                    for member in sorted(
                        item
                        for item in file_name.rglob("*")
                        if item.is_file()
                    )
                )
                records.append(
                    (
                        output_name,
                        "directory",
                        members,
                    )
                )
                continue

            records.append(
                (
                    output_name,
                    "absent",
                )
            )

        return tuple(records)

    before = output_snapshot()

    completed = subprocess.run(
        [
            "/Users/alexkolesnikov/.local/bin/uv",
            "run",
            "python",
            "scripts/run_stage14_random_label_analysis.py",
            "--repository-root",
            str(repository),
            "--expected-implementation-commit",
            head,
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    after = output_snapshot()

    assert completed.returncode == 2
    assert "Choose exactly one of" in completed.stderr
    assert after == before
    assert {
        output_name
        for output_name, state, *_ in after
        if state != "absent"
    } == {
        output_name
        for output_name, state, *_ in before
        if state != "absent"
    }


def test_random_label_context_adapts_to_search_context() -> None:
    from circuit_families.analysis.random_label_circuit_analysis import (
        load_random_label_checkpoint_context,
    )
    from circuit_families.analysis.stage14_random_label_runner import (
        adapt_random_label_search_context,
    )
    from circuit_families.interpretability.fidelity import (
        CheckpointEvaluationContext,
    )

    repository = repository_root()
    configuration = load_frozen_analysis_configuration(
        repository_root=repository
    )
    source_context = load_random_label_checkpoint_context(
        repository_root=repository,
        configuration=configuration,
        checkpoint_step=200,
        device="cpu",
    )
    adapted = adapt_random_label_search_context(
        source_context
    )

    assert isinstance(
        adapted,
        CheckpointEvaluationContext,
    )
    assert adapted.model is source_context.model
    assert adapted.inputs is source_context.inputs
    assert adapted.targets is source_context.targets
    assert (
        adapted.model_state_sha256
        == source_context.model_state_sha256
    )


def test_primary_sparse_cells_are_exact() -> None:
    from circuit_families.analysis.stage14_random_label_runner import (
        primary_sparse_cells,
    )

    configuration = load_frozen_analysis_configuration(
        repository_root=repository_root()
    )
    plan = build_execution_plan(configuration)
    cells = primary_sparse_cells(plan)

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
        cell.execution_mode == "execute"
        for cell in cells
    )
    assert all(
        cell.fidelity_threshold is not None
        for cell in cells
    )


def test_primary_sparse_and_runtime_schemas_are_unique() -> None:
    from circuit_families.analysis.stage14_random_label_runner import (
        PRIMARY_SPARSE_COLUMNS,
        RUNTIME_COLUMNS,
    )

    assert len(PRIMARY_SPARSE_COLUMNS) == len(
        set(PRIMARY_SPARSE_COLUMNS)
    )
    assert len(RUNTIME_COLUMNS) == len(
        set(RUNTIME_COLUMNS)
    )
    assert "analysis_run_id" in PRIMARY_SPARSE_COLUMNS
    assert "checkpoint_step" in PRIMARY_SPARSE_COLUMNS
    assert "primary_fidelity" in PRIMARY_SPARSE_COLUMNS
    assert "final_mask_sha256" in PRIMARY_SPARSE_COLUMNS
    assert (
        "included_in_deterministic_scientific_hashes"
        in RUNTIME_COLUMNS
    )


def test_runner_help_exposes_primary_sparse_mode() -> None:
    repository = repository_root()

    completed = subprocess.run(
        [
            "/Users/alexkolesnikov/.local/bin/uv",
            "run",
            "python",
            "scripts/run_stage14_random_label_analysis.py",
            "--help",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--validate-inputs-only" in completed.stdout
    assert "--execute-primary-sparse" in completed.stdout
    assert "--device {cpu,cuda}" in completed.stdout

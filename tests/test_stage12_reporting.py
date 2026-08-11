"""Tests for deterministic Stage 12 run-level reports."""

from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path

import pytest

from circuit_families.analysis.stage12_reporting import (
    Stage12ReportCell,
    write_stage12_report_artifacts,
)
from circuit_families.interpretability.diversity_forced_search import (
    CheckpointFamilySearchExecution,
    run_sequential_family_search,
)
from circuit_families.interpretability.fidelity import (
    MaskEvaluationMetrics,
)
from circuit_families.interpretability.masks import (
    SEARCHABLE_COMPONENT_COUNT,
    ComponentMask,
    component_location,
)
from circuit_families.interpretability.sparse_search import (
    MEANINGFULLY_SPARSE_MAX_COMPONENTS,
    ComponentRanking,
    RankingResult,
)

COMPONENT_INDEX = {
    identifier: index
    for index, identifier in enumerate(
        ComponentMask.all_retained().retained_component_ids
    )
}


def synthetic_metrics(
    mask: ComponentMask,
    *,
    fidelity: float,
) -> MaskEvaluationMetrics:
    return MaskEvaluationMetrics(
        primary_fidelity=fidelity,
        prediction_agreement_count=(
            100 if fidelity == 1.0 else 0
        ),
        full_accuracy=1.0,
        masked_accuracy=fidelity,
        accuracy_change=fidelity - 1.0,
        full_cross_entropy=0.0,
        masked_cross_entropy=1.0 - fidelity,
        cross_entropy_change=1.0 - fidelity,
        mean_kl_divergence=1.0 - fidelity,
        mean_jensen_shannon_divergence=(
            1.0 - fidelity
        ),
        maximum_absolute_logit_difference=(
            1.0 - fidelity
        ),
        retained_attention_head_count=(
            mask.retained_attention_head_count
        ),
        retained_mlp_neuron_count=(
            mask.retained_mlp_neuron_count
        ),
        retained_component_count=(
            mask.retained_component_count
        ),
        retained_component_proportion=(
            mask.retained_component_proportion
        ),
        evaluated_example_count=100,
        evaluation_batch_size=100,
    )


def synthetic_ranking(
    mask: ComponentMask,
) -> RankingResult:
    rankings = tuple(
        ComponentRanking(
            component_identifier=identifier,
            component_index=COMPONENT_INDEX[
                identifier
            ],
            component_class=(
                component_location(
                    identifier
                ).component_class
            ),
            gate_gradient=0.0,
            estimated_removal_damage=float(
                COMPONENT_INDEX[identifier]
            ),
            ranking_position=position,
        )
        for position, identifier in enumerate(
            mask.retained_component_ids,
            start=1,
        )
    )

    return RankingResult(
        mean_pseudo_target_loss=0.0,
        mean_gate_gradients=(
            (0.0,) * SEARCHABLE_COMPONENT_COUNT
        ),
        ranked_components=rankings,
        evaluated_example_count=100,
        ranking_batch_size=100,
        retained_component_count=(
            mask.retained_component_count
        ),
        model_state_sha256_before="state",
        model_state_sha256_after="state",
        hook_counts_before=(),
        hook_counts_after=(),
    )


def boundary_evaluator(
    mask: ComponentMask,
) -> MaskEvaluationMetrics:
    return synthetic_metrics(
        mask,
        fidelity=(
            1.0
            if mask.retained_component_count
            >= MEANINGFULLY_SPARSE_MAX_COMPONENTS
            else 0.0
        ),
    )


def synthetic_execution(
    cutoff: Fraction,
) -> CheckpointFamilySearchExecution:
    initial = ComponentMask.all_retained()
    family = run_sequential_family_search(
        base_ranking_function=synthetic_ranking,
        exact_evaluation_function=boundary_evaluator,
        initial_metrics=synthetic_metrics(
            initial,
            fidelity=1.0,
        ),
        fidelity_threshold=0.99,
        distinctness_cutoff=cutoff,
        model_seed=1,
        checkpoint_index=7,
        family_target=2,
        max_restarts_per_alternative=5,
        per_requested_circuit_budget=5_000,
        per_cell_budget=10_000,
        reuse_coefficient=1.0,
    )

    assert family.family_size == 2

    return CheckpointFamilySearchExecution(
        result=family,
        pseudo_target_sha256="a" * 64,
        pseudo_target_count=100,
        ranking_batch_size=100,
        evaluation_batch_size=100,
        full_model_reference_sha256="b" * 64,
        full_model_reference_example_count=100,
        full_model_reference_batch_size=100,
        model_state_sha256_before="state",
        model_state_sha256_after="state",
        hook_counts_before=(),
        hook_counts_after=(),
    )


def report_cells() -> tuple[Stage12ReportCell, ...]:
    return (
        Stage12ReportCell(
            cell_id="cutoff-0.25",
            checkpoint_step=9050,
            distinctness_cutoff=Fraction(1, 4),
            execution=synthetic_execution(
                Fraction(1, 4)
            ),
            raw_cell_directory=(
                "results/raw/fixture/cutoff-0.25"
            ),
        ),
        Stage12ReportCell(
            cell_id="cutoff-0.50",
            checkpoint_step=9050,
            distinctness_cutoff=Fraction(1, 2),
            execution=synthetic_execution(
                Fraction(1, 2)
            ),
            raw_cell_directory=(
                "results/raw/fixture/cutoff-0.50"
            ),
        ),
    )


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes()
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def test_stage12_reports_are_byte_deterministic(
    tmp_path: Path,
) -> None:
    cells = report_cells()
    order = (
        Fraction(1, 2),
        Fraction(1, 4),
        Fraction(3, 4),
    )

    first = write_stage12_report_artifacts(
        tmp_path / "first",
        stage12_run_id="fixture-run",
        seed=1,
        cells=tuple(reversed(cells)),
        execution_order=order,
    )
    second = write_stage12_report_artifacts(
        tmp_path / "second",
        stage12_run_id="fixture-run",
        seed=1,
        cells=cells,
        execution_order=order,
    )

    assert snapshot(
        first.output_directory
    ) == snapshot(
        second.output_directory
    )

    summaries = read_csv(
        first.family_summary_path
    )
    assert [
        row["cell_id"]
        for row in summaries
    ] == [
        "cutoff-0.50",
        "cutoff-0.25",
    ]

    circuits = read_csv(first.circuit_path)
    assert len(circuits) == 4
    assert {
        row["member_label"]
        for row in circuits
    } == {"C1", "C2"}

    overlaps = read_csv(
        first.pairwise_overlap_path
    )
    assert len(overlaps) == 2
    assert all(
        row["passes_active_cutoff"] == "True"
        for row in overlaps
    )

    restarts = read_csv(first.restart_path)
    c2_rows = [
        row
        for row in restarts
        if row["requested_member_label"] == "C2"
    ]
    assert len(c2_rows) == 10
    assert sum(
        row["restart_used"] == "True"
        for row in c2_rows
    ) == 2
    assert sum(
        row["unused_reason"]
        == "earlier_valid_distinct_candidate"
        for row in c2_rows
    ) == 8

    note = first.validation_note_path.read_text(
        encoding="utf-8"
    )
    assert "Stage 13 has not begun" in note
    assert "cutoff-0.50" in note
    assert "cutoff-0.25" in note


def test_stage12_report_rejects_nonempty_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    stale = output / "stale.txt"
    stale.write_text("stale", encoding="utf-8")

    with pytest.raises(
        FileExistsError,
        match="must be empty",
    ):
        write_stage12_report_artifacts(
            output,
            stage12_run_id="fixture-run",
            seed=1,
            cells=report_cells()[:1],
            execution_order=(
                Fraction(1, 2),
                Fraction(1, 4),
                Fraction(3, 4),
            ),
        )

    assert stale.read_text(
        encoding="utf-8"
    ) == "stale"


def test_stage12_report_rejects_cutoff_mismatch(
    tmp_path: Path,
) -> None:
    cell = report_cells()[0]
    invalid = Stage12ReportCell(
        cell_id=cell.cell_id,
        checkpoint_step=cell.checkpoint_step,
        distinctness_cutoff=Fraction(3, 4),
        execution=cell.execution,
        raw_cell_directory=cell.raw_cell_directory,
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        write_stage12_report_artifacts(
            tmp_path / "invalid",
            stage12_run_id="fixture-run",
            seed=1,
            cells=(invalid,),
            execution_order=(
                Fraction(1, 2),
                Fraction(1, 4),
                Fraction(3, 4),
            ),
        )

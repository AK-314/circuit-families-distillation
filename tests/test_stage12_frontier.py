"""Tests for the Stage 12 descriptive frontier."""

from __future__ import annotations

import csv
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from circuit_families.analysis.stage12_frontier import (
    Stage12FrontierRuntime,
    frontier_rows,
    write_frontier_table,
)
from circuit_families.analysis.stage12_reporting import (
    Stage12ReportCell,
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
    components = tuple(
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
        ranked_components=components,
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


def accepted_execution(
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
        distinctness_cutoff=Fraction(1, 2),
        model_seed=1,
        checkpoint_index=7,
        family_target=1,
        max_restarts_per_alternative=1,
        per_requested_circuit_budget=5_000,
        per_cell_budget=5_000,
        reuse_coefficient=0.5,
    )

    assert family.family_size == 1
    assert len(family.restart_outcomes) == 1

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


def execution_with_failed_alternative(
) -> CheckpointFamilySearchExecution:
    execution = accepted_execution()
    family = execution.result
    first_outcome = family.restart_outcomes[0]
    failed_outcome = replace(
        first_outcome,
        requested_member_index=2,
        restart_index=0,
        pairwise_overlaps=(Fraction(1, 1),),
        maximum_pairwise_overlap=Fraction(1, 1),
        outcome_status="distinctness_failure",
        accepted_candidate=False,
    )
    total_evaluations = (
        first_outcome.execution.result
        .exact_evaluations_used
        + failed_outcome.execution.result
        .exact_evaluations_used
    )
    failed_family = replace(
        family,
        status="distinctness_failure",
        family_target=2,
        per_cell_budget=total_evaluations,
        members=family.members,
        restart_outcomes=(
            first_outcome,
            failed_outcome,
        ),
        exact_evaluations_used=total_evaluations,
        budget_remaining=0,
        right_censored=False,
        stopping_reason=(
            "requested_member_2_distinctness_failure"
        ),
    )

    return replace(
        execution,
        result=failed_family,
    )


def report_cell(
    execution: CheckpointFamilySearchExecution,
) -> Stage12ReportCell:
    return Stage12ReportCell(
        cell_id="cutoff-0.50",
        checkpoint_step=9050,
        distinctness_cutoff=Fraction(1, 2),
        execution=execution,
        raw_cell_directory=(
            "results/raw/fixture/cutoff-0.50"
        ),
    )


def runtime_records(
    count: int,
) -> tuple[Stage12FrontierRuntime, ...]:
    return tuple(
        Stage12FrontierRuntime(
            cell_id="cutoff-0.50",
            requested_member_index=index,
            runtime_seconds=float(index * 10),
        )
        for index in range(1, count + 1)
    )


def test_frontier_records_accepted_and_failed_members() -> None:
    execution = execution_with_failed_alternative()
    rows = frontier_rows(
        stage12_run_id="fixture-run",
        cells=(report_cell(execution),),
        execution_order=(
            Fraction(1, 2),
            Fraction(1, 4),
            Fraction(3, 4),
        ),
        runtimes=runtime_records(2),
    )

    assert len(rows) == 2

    accepted, failed = rows

    assert accepted["member_label"] == "C1"
    assert accepted["accepted_circuit"] is True
    assert (
        accepted["scientific_family_result"]
        is True
    )
    assert (
        accepted["retained_component_count"]
        == 258
    )
    assert accepted["mean_prior_overlap"] == ""
    assert accepted["runtime_seconds"] == 10.0
    assert accepted["frontier_scalar_score"] == ""

    assert failed["member_label"] == "C2"
    assert failed["accepted_circuit"] is False
    assert (
        failed["scientific_family_result"]
        is False
    )
    assert failed["status"] == (
        "distinctness_failure"
    )
    assert failed["maximum_prior_overlap"] == 1.0
    assert failed["mean_prior_overlap"] == 1.0
    assert (
        failed[
            "most_overlapping_prior_member_label"
        ]
        == "C1"
    )
    assert failed["runtime_seconds"] == 20.0
    assert (
        failed["cumulative_exact_evaluations"]
        > accepted["cumulative_exact_evaluations"]
    )
    assert (
        failed[
            "runtime_included_in_"
            "deterministic_scientific_hashes"
        ]
        is False
    )
    assert (
        failed["runtime_comparison_policy"]
        == "semantic"
    )


def test_frontier_requires_exact_runtime_coverage() -> None:
    execution = execution_with_failed_alternative()

    with pytest.raises(
        ValueError,
        match="Missing frontier runtime",
    ):
        frontier_rows(
            stage12_run_id="fixture-run",
            cells=(report_cell(execution),),
            execution_order=(Fraction(1, 2),),
            runtimes=runtime_records(1),
        )

    with pytest.raises(
        ValueError,
        match="do not correspond",
    ):
        frontier_rows(
            stage12_run_id="fixture-run",
            cells=(
                report_cell(
                    accepted_execution()
                ),
            ),
            execution_order=(Fraction(1, 2),),
            runtimes=runtime_records(2),
        )


def test_frontier_table_is_stable_for_fixed_runtime(
    tmp_path: Path,
) -> None:
    execution = execution_with_failed_alternative()
    cell = report_cell(execution)
    runtimes = runtime_records(2)
    order = (
        Fraction(1, 2),
        Fraction(1, 4),
        Fraction(3, 4),
    )

    first = write_frontier_table(
        tmp_path / "first.csv",
        stage12_run_id="fixture-run",
        cells=(cell,),
        execution_order=order,
        runtimes=runtimes,
    )
    second = write_frontier_table(
        tmp_path / "second.csv",
        stage12_run_id="fixture-run",
        cells=(cell,),
        execution_order=order,
        runtimes=tuple(reversed(runtimes)),
    )

    assert (
        first.table_path.read_bytes()
        == second.table_path.read_bytes()
    )
    assert first.table_sha256 == second.table_sha256
    assert first.row_count == second.row_count == 2
    assert first.runtime_bearing is True

    with first.table_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert [
        row["member_label"]
        for row in rows
    ] == ["C1", "C2"]
    assert all(
        row["frontier_scalar_score"] == ""
        for row in rows
    )

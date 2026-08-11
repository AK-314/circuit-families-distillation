"""Tests for executable Stage 12 controls."""

from __future__ import annotations

from fractions import Fraction

from circuit_families.analysis.stage12_control_execution import (
    HARD_EXCLUSION_FIXTURE_BUDGET,
    canonical_ranking,
    hard_overlap_stress_result,
    run_distinctness_impossible_fixture,
    run_fidelity_impossible_fixture,
    run_hard_exclusion_fixture,
    run_shuffled_ranking_search,
    soft_reuse_stress_result,
    synthetic_metrics,
    target_sparse_mask,
)
from circuit_families.interpretability.diversity_forced_search import (
    DISTINCTNESS_FAILURE,
    NO_FEASIBLE_CANDIDATE,
)
from circuit_families.interpretability.masks import (
    ComponentMask,
)


def test_fidelity_impossible_fixture_executes_failure_path() -> None:
    result, family = (
        run_fidelity_impossible_fixture()
    )

    assert result.validation_passed
    assert family.family_size == 0
    assert family.status == (
        NO_FEASIBLE_CANDIDATE
    )
    assert (
        family.exact_evaluations_used
        == 516
    )


def test_distinctness_impossible_fixture_executes_status() -> None:
    result, family = (
        run_distinctness_impossible_fixture()
    )

    assert result.validation_passed
    assert family.family_size == 1
    assert family.status == (
        DISTINCTNESS_FAILURE
    )
    assert result.jaccard_overlap == 1.0


def test_hard_exclusion_fixture_retains_component() -> None:
    result, search = (
        run_hard_exclusion_fixture()
    )

    assert result.validation_passed
    assert "H0" in (
        search.final_mask
        .retained_component_ids
    )
    assert (
        search.exact_evaluations_used
        == HARD_EXCLUSION_FIXTURE_BUDGET
    )


def test_hard_overlap_and_soft_reuse_validations() -> None:
    target = target_sparse_mask()
    overlap = hard_overlap_stress_result(
        mask=target,
        distinctness_cutoff=Fraction(1, 2),
    )
    reuse = soft_reuse_stress_result(
        base_ranking=canonical_ranking(
            ComponentMask.all_retained()
        ),
        accepted_mask=target,
        reuse_coefficient=0.5,
    )

    assert overlap.validation_passed
    assert overlap.jaccard_overlap == 1.0
    assert reuse.validation_passed
    assert reuse.qualifying_count == 258


def test_shuffled_control_runs_actual_search() -> None:
    initial = ComponentMask.all_retained()
    original = canonical_ranking(initial)

    def always_valid(mask: ComponentMask):
        return synthetic_metrics(
            mask,
            fidelity=1.0,
        )

    result, search = (
        run_shuffled_ranking_search(
            original_ranking=original,
            base_ranking_function=(
                canonical_ranking
            ),
            exact_evaluation_function=(
                always_valid
            ),
            initial_metrics=synthetic_metrics(
                initial,
                fidelity=1.0,
            ),
            integer_seed=12345,
            fidelity_threshold=0.99,
            exact_evaluation_budget=16,
        )
    )

    assert result.validation_passed
    assert search.exact_evaluations_used == 16
    assert search.ranking_passes_used == 1
    assert len(search.accepted_removals) == 1

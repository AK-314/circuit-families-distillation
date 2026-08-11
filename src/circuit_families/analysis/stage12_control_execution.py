"""Executable negative controls and stress tests for Stage 12."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from circuit_families.analysis.stage12_negative_controls import (
    DEGRADED_C1_CONTROL,
    DISTINCTNESS_IMPOSSIBLE_CONTROL,
    FIDELITY_IMPOSSIBLE_CONTROL,
    SHUFFLED_RANKING_CONTROL,
    STAGE11_RANDOM_MASK_CONTROL,
    NegativeControlResult,
    degraded_c1_control_result,
    load_stage11_random_mask_controls,
    shuffled_ranking,
    shuffled_ranking_control_result,
    stage11_random_mask_control_result,
)
from circuit_families.interpretability.diversity_forced_search import (
    DISTINCTNESS_FAILURE,
    NO_FEASIBLE_CANDIDATE,
    FamilySearchResult,
    build_diversity_ranking,
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
from circuit_families.interpretability.overlap_constraints import (
    is_structurally_distinct,
    jaccard_fraction,
)
from circuit_families.interpretability.sparse_search import (
    ComponentRanking,
    RankingResult,
    SparseSearchResult,
    greedy_sparse_search,
)

SHUFFLED_SEARCH_EXACT_EVALUATION_BUDGET = 160
FIDELITY_IMPOSSIBLE_FIXTURE_BUDGET = 517
HARD_EXCLUSION_FIXTURE_BUDGET = 32

HARD_OVERLAP_STRESS = "stress_hard_overlap_constraint"
HARD_EXCLUSION_STRESS = "stress_hard_component_exclusion"
SOFT_REUSE_STRESS = "stress_soft_overlap_penalty"

RankingFunction = Callable[[ComponentMask], RankingResult]
ExactEvaluationFunction = Callable[
    [ComponentMask],
    MaskEvaluationMetrics,
]


@dataclass(frozen=True)
class Stage12ControlExecution:
    """Results of the executable Stage 12 control suite."""

    negative_controls: tuple[
        NegativeControlResult,
        ...,
    ]
    stress_tests: tuple[
        NegativeControlResult,
        ...,
    ]
    shuffled_search: SparseSearchResult
    fidelity_impossible_family: FamilySearchResult
    distinctness_impossible_family: FamilySearchResult
    hard_exclusion_search: SparseSearchResult
    model_exact_evaluations_used: int
    synthetic_fixture_evaluations_used: int

    def to_record(self) -> dict[str, object]:
        return {
            "negative_control_names": [
                result.control_name
                for result in self.negative_controls
            ],
            "stress_test_names": [
                result.control_name
                for result in self.stress_tests
            ],
            "all_negative_controls_passed": all(
                result.validation_passed
                for result in self.negative_controls
            ),
            "all_stress_tests_passed": all(
                result.validation_passed
                for result in self.stress_tests
            ),
            "shuffled_search_status": (
                self.shuffled_search.status
            ),
            "shuffled_search_exact_evaluations": (
                self.shuffled_search
                .exact_evaluations_used
            ),
            "fidelity_impossible_status": (
                self.fidelity_impossible_family.status
            ),
            "distinctness_impossible_status": (
                self.distinctness_impossible_family.status
            ),
            "hard_exclusion_search_status": (
                self.hard_exclusion_search.status
            ),
            "model_exact_evaluations_used": (
                self.model_exact_evaluations_used
            ),
            "synthetic_fixture_evaluations_used": (
                self.synthetic_fixture_evaluations_used
            ),
            "scientific_family_results": 0,
        }


def synthetic_metrics(
    mask: ComponentMask,
    *,
    fidelity: float,
    evaluated_example_count: int = 12_769,
) -> MaskEvaluationMetrics:
    """Create internally consistent deterministic fixture metrics."""

    if not isinstance(mask, ComponentMask):
        raise TypeError(
            "mask must be a ComponentMask."
        )

    if (
        isinstance(fidelity, bool)
        or not isinstance(fidelity, (int, float))
        or not math.isfinite(float(fidelity))
        or not 0.0 <= float(fidelity) <= 1.0
    ):
        raise ValueError(
            "fidelity must be finite and in [0, 1]."
        )

    value = float(fidelity)
    agreement = int(
        round(value * evaluated_example_count)
    )

    if agreement not in {
        0,
        evaluated_example_count,
    }:
        raise ValueError(
            "Fixture fidelity must currently be zero or one."
        )

    return MaskEvaluationMetrics(
        primary_fidelity=value,
        prediction_agreement_count=agreement,
        full_accuracy=1.0,
        masked_accuracy=value,
        accuracy_change=value - 1.0,
        full_cross_entropy=0.0,
        masked_cross_entropy=1.0 - value,
        cross_entropy_change=1.0 - value,
        mean_kl_divergence=1.0 - value,
        mean_jensen_shannon_divergence=(
            1.0 - value
        ),
        maximum_absolute_logit_difference=(
            1.0 - value
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
        evaluated_example_count=(
            evaluated_example_count
        ),
        evaluation_batch_size=(
            evaluated_example_count
        ),
    )


_GLOBAL_COMPONENT_INDEX = {
    identifier: index
    for index, identifier in enumerate(
        ComponentMask.all_retained()
        .retained_component_ids
    )
}


def canonical_ranking(
    mask: ComponentMask,
) -> RankingResult:
    """Return a complete deterministic ranking for one fixture mask."""

    if not isinstance(mask, ComponentMask):
        raise TypeError(
            "mask must be a ComponentMask."
        )

    rankings = tuple(
        ComponentRanking(
            component_identifier=identifier,
            component_index=(
                _GLOBAL_COMPONENT_INDEX[
                    identifier
                ]
            ),
            component_class=(
                component_location(
                    identifier
                ).component_class
            ),
            gate_gradient=0.0,
            estimated_removal_damage=float(
                _GLOBAL_COMPONENT_INDEX[
                    identifier
                ]
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
        evaluated_example_count=12_769,
        ranking_batch_size=256,
        retained_component_count=(
            mask.retained_component_count
        ),
        model_state_sha256_before="fixture-state",
        model_state_sha256_after="fixture-state",
        hook_counts_before=(),
        hook_counts_after=(),
        gradient_source=(
            "deterministic_stage12_fixture"
        ),
        score_definition=(
            "global_component_index"
        ),
    )


def target_aware_ranking(
    mask: ComponentMask,
    *,
    target: ComponentMask,
) -> RankingResult:
    """Rank removable fixture components before protected ones."""

    target_ids = set(
        target.retained_component_ids
    )
    removable = [
        identifier
        for identifier in mask.retained_component_ids
        if identifier not in target_ids
    ]
    protected = [
        identifier
        for identifier in mask.retained_component_ids
        if identifier in target_ids
    ]
    ordered = (
        *removable,
        *protected,
    )

    rankings = tuple(
        ComponentRanking(
            component_identifier=identifier,
            component_index=(
                _GLOBAL_COMPONENT_INDEX[
                    identifier
                ]
            ),
            component_class=(
                component_location(
                    identifier
                ).component_class
            ),
            gate_gradient=0.0,
            estimated_removal_damage=float(
                position
            ),
            ranking_position=position,
        )
        for position, identifier in enumerate(
            ordered,
            start=1,
        )
    )

    return RankingResult(
        mean_pseudo_target_loss=0.0,
        mean_gate_gradients=(
            (0.0,) * SEARCHABLE_COMPONENT_COUNT
        ),
        ranked_components=rankings,
        evaluated_example_count=12_769,
        ranking_batch_size=256,
        retained_component_count=(
            mask.retained_component_count
        ),
        model_state_sha256_before="fixture-state",
        model_state_sha256_after="fixture-state",
        hook_counts_before=(),
        hook_counts_after=(),
        gradient_source=(
            "deterministic_stage12_fixture"
        ),
        score_definition=(
            "removable_components_before_"
            "protected_target_components"
        ),
    )


def target_sparse_mask() -> ComponentMask:
    """Return a fixed 258-component fixture target."""

    mask = ComponentMask(
        attention_head_mask=(1, 1, 1, 1),
        mlp_neuron_mask=(
            (1,) * 254
            + (0,) * 258
        ),
    )

    if mask.retained_component_count != 258:
        raise RuntimeError(
            "Fixture target is not meaningfully sparse."
        )

    return mask


def _target_preserving_evaluator(
    target: ComponentMask,
) -> ExactEvaluationFunction:
    target_ids = set(
        target.retained_component_ids
    )

    def evaluate(
        mask: ComponentMask,
    ) -> MaskEvaluationMetrics:
        retained = set(
            mask.retained_component_ids
        )
        return synthetic_metrics(
            mask,
            fidelity=(
                1.0
                if target_ids.issubset(retained)
                else 0.0
            ),
        )

    return evaluate


def run_shuffled_ranking_search(
    *,
    original_ranking: RankingResult,
    base_ranking_function: RankingFunction,
    exact_evaluation_function: ExactEvaluationFunction,
    initial_metrics: MaskEvaluationMetrics,
    integer_seed: int,
    fidelity_threshold: float,
    exact_evaluation_budget: int = (
        SHUFFLED_SEARCH_EXACT_EVALUATION_BUDGET
    ),
) -> tuple[
    NegativeControlResult,
    SparseSearchResult,
]:
    """Run a deterministic shuffled-order small-budget search."""

    base_results: list[RankingResult] = []
    shuffled_results: list[RankingResult] = []

    def ranking_function(
        mask: ComponentMask,
    ) -> RankingResult:
        base = base_ranking_function(mask)
        shuffled = shuffled_ranking(
            base,
            integer_seed=integer_seed,
        )

        search_rankings = tuple(
            replace(
                ranking,
                estimated_removal_damage=float(
                    position
                ),
                ranking_position=position,
            )
            for position, ranking in enumerate(
                shuffled.ranked_components,
                start=1,
            )
        )
        search_compatible = replace(
            shuffled,
            ranked_components=search_rankings,
            score_definition=(
                "deterministic shuffled-ranking "
                "negative control with ordering-compatible "
                "search scores"
            ),
        )

        base_results.append(base)
        shuffled_results.append(
            search_compatible
        )
        return search_compatible

    search = greedy_sparse_search(
        ranking_function=ranking_function,
        exact_evaluation_function=(
            exact_evaluation_function
        ),
        initial_metrics=initial_metrics,
        fidelity_threshold=fidelity_threshold,
        exact_evaluation_budget=(
            exact_evaluation_budget
        ),
    )

    if not base_results or not shuffled_results:
        raise RuntimeError(
            "Shuffled search performed no ranking pass."
        )

    permutation = (
        shuffled_ranking_control_result(
            original_ranking,
            shuffled_results[0],
            integer_seed=integer_seed,
        )
    )

    allowed_statuses = {
        "budget_exhaustion",
        "valid_sparse_circuit",
        "valid_but_not_meaningfully_sparse",
        (
            "no_feasible_sparse_candidate_"
            "discovered_within_budget"
        ),
    }
    passed = (
        permutation.validation_passed
        and search.status in allowed_statuses
        and 0
        < search.exact_evaluations_used
        <= exact_evaluation_budget
        and search.ranking_passes_used
        == len(shuffled_results)
    )

    result = NegativeControlResult(
        control_name=SHUFFLED_RANKING_CONTROL,
        control_scope=(
            "deterministic_small_budget_search"
        ),
        expected_outcome=(
            "small_budget_search_completed"
        ),
        observed_outcome=search.status,
        validation_passed=passed,
        record_count=(
            search.exact_evaluations_used
        ),
        qualifying_count=len(
            search.accepted_removals
        ),
        retained_component_count=(
            search.final_mask
            .retained_component_count
        ),
        primary_fidelity=(
            search.final_metrics
            .primary_fidelity
        ),
        fidelity_threshold=(
            fidelity_threshold
        ),
        seed_integer=integer_seed,
        bit_generator="numpy.random.PCG64",
        details={
            "exact_evaluation_budget": (
                exact_evaluation_budget
            ),
            "exact_evaluations_used": (
                search.exact_evaluations_used
            ),
            "ranking_passes_used": (
                search.ranking_passes_used
            ),
            "accepted_removals": len(
                search.accepted_removals
            ),
            "ordering_changed": (
                permutation.validation_passed
            ),
            "scientific_family_result": False,
        },
    )

    return result, search


def run_fidelity_impossible_fixture(
    *,
    fidelity_threshold: float = 0.99,
) -> tuple[
    NegativeControlResult,
    FamilySearchResult,
]:
    """Exercise the controller when every deletion fails fidelity."""

    initial = ComponentMask.all_retained()

    def impossible_evaluator(
        mask: ComponentMask,
    ) -> MaskEvaluationMetrics:
        return synthetic_metrics(
            mask,
            fidelity=0.0,
        )

    family = run_sequential_family_search(
        base_ranking_function=canonical_ranking,
        exact_evaluation_function=(
            impossible_evaluator
        ),
        initial_metrics=synthetic_metrics(
            initial,
            fidelity=1.0,
        ),
        fidelity_threshold=(
            fidelity_threshold
        ),
        distinctness_cutoff=Fraction(1, 2),
        model_seed=1,
        checkpoint_index=7,
        family_target=1,
        max_restarts_per_alternative=1,
        per_requested_circuit_budget=(
            FIDELITY_IMPOSSIBLE_FIXTURE_BUDGET
        ),
        per_cell_budget=(
            FIDELITY_IMPOSSIBLE_FIXTURE_BUDGET
        ),
    )

    statuses = tuple(
        outcome.outcome_status
        for outcome in family.restart_outcomes
    )
    passed = (
        family.family_size == 0
        and family.status
        == NO_FEASIBLE_CANDIDATE
        and statuses
        == (NO_FEASIBLE_CANDIDATE,)
    )

    result = NegativeControlResult(
        control_name=FIDELITY_IMPOSSIBLE_CONTROL,
        control_scope=(
            "executable_fixture_all_candidate_"
            "deletions_fail_fidelity"
        ),
        expected_outcome=NO_FEASIBLE_CANDIDATE,
        observed_outcome=family.status,
        validation_passed=passed,
        record_count=(
            family.exact_evaluations_used
        ),
        qualifying_count=family.family_size,
        retained_component_count=516,
        fidelity_threshold=(
            fidelity_threshold
        ),
        details={
            "restart_outcome_statuses": list(
                statuses
            ),
            "exact_evaluations_used": (
                family.exact_evaluations_used
            ),
            "scientific_family_result": False,
        },
    )

    return result, family


def run_distinctness_impossible_fixture(
    *,
    fidelity_threshold: float = 0.99,
    distinctness_cutoff: Fraction = Fraction(
        1,
        2,
    ),
) -> tuple[
    NegativeControlResult,
    FamilySearchResult,
]:
    """Exercise the exact distinctness-failure controller path."""

    target = target_sparse_mask()
    evaluator = _target_preserving_evaluator(
        target
    )
    def ranking_function(
        mask: ComponentMask,
    ) -> RankingResult:
        return target_aware_ranking(
            mask,
            target=target,
        )

    family = run_sequential_family_search(
        base_ranking_function=ranking_function,
        exact_evaluation_function=evaluator,
        initial_metrics=synthetic_metrics(
            ComponentMask.all_retained(),
            fidelity=1.0,
        ),
        fidelity_threshold=(
            fidelity_threshold
        ),
        distinctness_cutoff=(
            distinctness_cutoff
        ),
        model_seed=1,
        checkpoint_index=7,
        family_target=2,
        max_restarts_per_alternative=1,
        per_requested_circuit_budget=10_000,
        per_cell_budget=20_000,
        reuse_coefficient=0.0,
    )

    second_outcomes = tuple(
        outcome
        for outcome in family.restart_outcomes
        if outcome.requested_member_index == 2
    )
    passed = (
        family.family_size == 1
        and family.status
        == DISTINCTNESS_FAILURE
        and len(second_outcomes) == 1
        and second_outcomes[0].outcome_status
        == DISTINCTNESS_FAILURE
        and second_outcomes[0]
        .maximum_pairwise_overlap
        == Fraction(1, 1)
    )

    result = NegativeControlResult(
        control_name=(
            DISTINCTNESS_IMPOSSIBLE_CONTROL
        ),
        control_scope=(
            "executable_two_member_fixture_"
            "identical_terminal_masks"
        ),
        expected_outcome=(
            DISTINCTNESS_FAILURE
        ),
        observed_outcome=family.status,
        validation_passed=passed,
        record_count=(
            family.exact_evaluations_used
        ),
        qualifying_count=family.family_size,
        retained_component_count=(
            target.retained_component_count
        ),
        primary_fidelity=1.0,
        fidelity_threshold=(
            fidelity_threshold
        ),
        jaccard_overlap=1.0,
        distinctness_cutoff=float(
            distinctness_cutoff
        ),
        details={
            "requested_family_target": 2,
            "accepted_family_size": (
                family.family_size
            ),
            "second_member_outcomes": [
                outcome.outcome_status
                for outcome in second_outcomes
            ],
            "exact_evaluations_used": (
                family.exact_evaluations_used
            ),
            "scientific_family_result": False,
        },
    )

    return result, family


def run_hard_exclusion_fixture(
    *,
    excluded_component: str = "H0",
    fidelity_threshold: float = 0.99,
) -> tuple[
    NegativeControlResult,
    SparseSearchResult,
]:
    """Exercise an exact hard-exclusion gate in the greedy engine."""

    initial = ComponentMask.all_retained()

    def exclusion_evaluator(
        mask: ComponentMask,
    ) -> MaskEvaluationMetrics:
        retained = set(
            mask.retained_component_ids
        )
        return synthetic_metrics(
            mask,
            fidelity=(
                1.0
                if excluded_component in retained
                else 0.0
            ),
        )

    search = greedy_sparse_search(
        ranking_function=canonical_ranking,
        exact_evaluation_function=(
            exclusion_evaluator
        ),
        initial_metrics=synthetic_metrics(
            initial,
            fidelity=1.0,
        ),
        fidelity_threshold=(
            fidelity_threshold
        ),
        exact_evaluation_budget=(
            HARD_EXCLUSION_FIXTURE_BUDGET
        ),
    )

    rejected_excluded = [
        evaluation
        for evaluation in (
            search.candidate_evaluations
        )
        if (
            evaluation.candidate_component
            == excluded_component
            and not evaluation.accepted
            and not evaluation.passed_threshold
        )
    ]
    retained = (
        excluded_component
        in search.final_mask.retained_component_ids
    )
    passed = (
        retained
        and bool(rejected_excluded)
        and search.exact_evaluations_used
        == HARD_EXCLUSION_FIXTURE_BUDGET
    )

    result = NegativeControlResult(
        control_name=HARD_EXCLUSION_STRESS,
        control_scope=(
            "executable_exact_candidate_gate"
        ),
        expected_outcome=(
            "excluded_component_never_removed"
        ),
        observed_outcome=(
            "excluded_component_retained"
            if retained
            else "excluded_component_removed"
        ),
        validation_passed=passed,
        record_count=(
            search.exact_evaluations_used
        ),
        selected_component=(
            excluded_component
        ),
        selected_component_index=(
            _GLOBAL_COMPONENT_INDEX[
                excluded_component
            ]
        ),
        retained_component_count=(
            search.final_mask
            .retained_component_count
        ),
        primary_fidelity=(
            search.final_metrics
            .primary_fidelity
        ),
        fidelity_threshold=(
            fidelity_threshold
        ),
        details={
            "rejected_excluded_candidate_count": (
                len(rejected_excluded)
            ),
            "search_status": search.status,
            "scientific_family_result": False,
        },
    )

    return result, search


def hard_overlap_stress_result(
    *,
    mask: ComponentMask,
    distinctness_cutoff: Fraction,
) -> NegativeControlResult:
    """Validate an exact hard Jaccard constraint."""

    overlap = jaccard_fraction(
        mask,
        mask,
    )
    distinct = is_structurally_distinct(
        mask,
        (mask,),
        cutoff=distinctness_cutoff,
    )
    passed = (
        overlap == Fraction(1, 1)
        and not distinct
    )

    return NegativeControlResult(
        control_name=HARD_OVERLAP_STRESS,
        control_scope=(
            "exact_hard_jaccard_gate"
        ),
        expected_outcome=(
            "candidate_rejected_above_cutoff"
        ),
        observed_outcome=(
            "candidate_rejected"
            if not distinct
            else "candidate_accepted"
        ),
        validation_passed=passed,
        mask_id=mask.mask_id,
        retained_component_count=(
            mask.retained_component_count
        ),
        jaccard_overlap=float(overlap),
        distinctness_cutoff=float(
            distinctness_cutoff
        ),
        details={
            "jaccard_numerator": (
                overlap.numerator
            ),
            "jaccard_denominator": (
                overlap.denominator
            ),
            "scientific_family_result": False,
        },
    )


def soft_reuse_stress_result(
    *,
    base_ranking: RankingResult,
    accepted_mask: ComponentMask,
    reuse_coefficient: float,
) -> NegativeControlResult:
    """Validate the frozen soft reuse-penalty transformation."""

    transformed = build_diversity_ranking(
        base_ranking,
        (accepted_mask,),
        reuse_coefficient=(
            reuse_coefficient
        ),
    )

    if not transformed.entries:
        raise RuntimeError(
            "Soft-reuse validation produced no entries."
        )

    formula_matches = all(
        math.isclose(
            entry.removal_score,
            (
                entry.damage_percentile
                - reuse_coefficient
                * entry.reuse_rate
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for entry in transformed.entries
    )
    reused = sum(
        entry.reuse_rate > 0.0
        for entry in transformed.entries
    )
    unused = sum(
        entry.reuse_rate == 0.0
        for entry in transformed.entries
    )
    passed = (
        formula_matches
        and reused > 0
        and unused > 0
    )

    return NegativeControlResult(
        control_name=SOFT_REUSE_STRESS,
        control_scope=(
            "frozen_reuse_cost_ranking_formula"
        ),
        expected_outcome=(
            "damage_percentile_minus_reuse_cost"
        ),
        observed_outcome=(
            "formula_verified"
            if passed
            else "formula_mismatch"
        ),
        validation_passed=passed,
        record_count=len(
            transformed.entries
        ),
        qualifying_count=reused,
        details={
            "reuse_coefficient": (
                reuse_coefficient
            ),
            "formula_matches": (
                formula_matches
            ),
            "reused_component_count": reused,
            "unused_component_count": unused,
            "scientific_family_result": False,
        },
    )


def execute_stage12_control_suite(
    *,
    stage11_archive_path: str | Path,
    primary_c1_search: SparseSearchResult,
    primary_c1_mask: ComponentMask,
    original_ranking: RankingResult,
    base_ranking_function: RankingFunction,
    exact_evaluation_function: ExactEvaluationFunction,
    initial_metrics: MaskEvaluationMetrics,
    shuffle_seed_integer: int,
    fidelity_threshold: float,
    distinctness_cutoff: Fraction,
    reuse_coefficient: float,
) -> Stage12ControlExecution:
    """Execute all frozen controls and method stress tests."""

    random_control = (
        stage11_random_mask_control_result(
            load_stage11_random_mask_controls(
                stage11_archive_path
            )
        )
    )
    degraded_control = (
        degraded_c1_control_result(
            primary_c1_search
        )
    )
    (
        shuffled_control,
        shuffled_search,
    ) = run_shuffled_ranking_search(
        original_ranking=original_ranking,
        base_ranking_function=(
            base_ranking_function
        ),
        exact_evaluation_function=(
            exact_evaluation_function
        ),
        initial_metrics=initial_metrics,
        integer_seed=shuffle_seed_integer,
        fidelity_threshold=(
            fidelity_threshold
        ),
    )
    (
        fidelity_control,
        fidelity_family,
    ) = run_fidelity_impossible_fixture(
        fidelity_threshold=fidelity_threshold
    )
    (
        distinctness_control,
        distinctness_family,
    ) = run_distinctness_impossible_fixture(
        fidelity_threshold=(
            fidelity_threshold
        ),
        distinctness_cutoff=(
            distinctness_cutoff
        ),
    )
    (
        exclusion_stress,
        exclusion_search,
    ) = run_hard_exclusion_fixture(
        fidelity_threshold=(
            fidelity_threshold
        )
    )
    overlap_stress = (
        hard_overlap_stress_result(
            mask=primary_c1_mask,
            distinctness_cutoff=(
                distinctness_cutoff
            ),
        )
    )
    reuse_stress = soft_reuse_stress_result(
        base_ranking=original_ranking,
        accepted_mask=primary_c1_mask,
        reuse_coefficient=(
            reuse_coefficient
        ),
    )

    negative_controls = (
        random_control,
        degraded_control,
        shuffled_control,
        fidelity_control,
        distinctness_control,
    )
    stress_tests = (
        exclusion_stress,
        overlap_stress,
        reuse_stress,
    )

    expected_negative_names = {
        STAGE11_RANDOM_MASK_CONTROL,
        DEGRADED_C1_CONTROL,
        SHUFFLED_RANKING_CONTROL,
        FIDELITY_IMPOSSIBLE_CONTROL,
        DISTINCTNESS_IMPOSSIBLE_CONTROL,
    }

    if {
        result.control_name
        for result in negative_controls
    } != expected_negative_names:
        raise RuntimeError(
            "Negative-control set is incomplete."
        )

    if not all(
        result.validation_passed
        for result in (
            *negative_controls,
            *stress_tests,
        )
    ):
        raise RuntimeError(
            "At least one executable Stage 12 "
            "control or stress test failed."
        )

    return Stage12ControlExecution(
        negative_controls=negative_controls,
        stress_tests=stress_tests,
        shuffled_search=shuffled_search,
        fidelity_impossible_family=(
            fidelity_family
        ),
        distinctness_impossible_family=(
            distinctness_family
        ),
        hard_exclusion_search=(
            exclusion_search
        ),
        model_exact_evaluations_used=(
            shuffled_search
            .exact_evaluations_used
        ),
        synthetic_fixture_evaluations_used=(
            fidelity_family
            .exact_evaluations_used
            + distinctness_family
            .exact_evaluations_used
            + exclusion_search
            .exact_evaluations_used
        ),
    )

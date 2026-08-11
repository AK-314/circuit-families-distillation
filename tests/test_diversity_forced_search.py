"""Tests for Stage 12 diversity-forced ranking utilities."""

import math
from pathlib import Path

import pytest
import torch

import circuit_families.interpretability.diversity_forced_search as diversity_search_module
from circuit_families.config import load_model_config
from circuit_families.interpretability.diversity_forced_search import (
    VALID_DISTINCT_CANDIDATE,
    build_diversity_ranking,
    component_reuse_rates,
    derive_search_seed,
    diversity_removal_score,
    run_checkpoint_family_search,
    run_diversity_sparse_search,
    run_sequential_family_search,
    stable_damage_percentiles,
)
from circuit_families.interpretability.fidelity import (
    CheckpointEvaluationContext,
    MaskEvaluationMetrics,
)
from circuit_families.interpretability.masks import (
    SEARCHABLE_COMPONENT_COUNT,
    SEARCHABLE_COMPONENT_IDS,
    ComponentMask,
    component_location,
)
from circuit_families.interpretability.sparse_search import (
    MEANINGFULLY_SPARSE_MAX_COMPONENTS,
    ComponentRanking,
    RankingResult,
    greedy_sparse_search,
)
from circuit_families.models import build_transformer
from circuit_families.training import canonical_state_hash


def ranking_component(
    identifier: str,
    index: int,
    damage: float,
) -> ComponentRanking:
    return ComponentRanking(
        component_identifier=identifier,
        component_index=index,
        component_class="attention_head",
        gate_gradient=-damage,
        estimated_removal_damage=damage,
        ranking_position=index + 1,
    )


def ranking_result(
    damages: tuple[float, ...],
) -> RankingResult:
    rankings = tuple(
        ranking_component(
            f"H{index}",
            index,
            damage,
        )
        for index, damage in enumerate(damages)
    )

    return RankingResult(
        mean_pseudo_target_loss=0.0,
        mean_gate_gradients=(
            (0.0,) * SEARCHABLE_COMPONENT_COUNT
        ),
        ranked_components=rankings,
        evaluated_example_count=12_769,
        ranking_batch_size=256,
        retained_component_count=len(rankings),
        model_state_sha256_before="state",
        model_state_sha256_after="state",
        hook_counts_before=(),
        hook_counts_after=(),
    )


def mask(*identifiers: str) -> ComponentMask:
    return ComponentMask.from_retained_identifiers(
        identifiers
    )


def test_percentile_endpoints_and_middle() -> None:
    result = ranking_result((2.0, 4.0, 9.0))

    assert stable_damage_percentiles(
        result.ranked_components
    ) == {
        "H0": 0.0,
        "H1": 0.5,
        "H2": 1.0,
    }


def test_singleton_percentile_is_zero() -> None:
    result = ranking_result((5.0,))

    assert stable_damage_percentiles(
        result.ranked_components
    ) == {
        "H0": 0.0,
    }


def test_exact_damage_ties_use_component_index() -> None:
    result = ranking_result((1.0, 1.0, 1.0))

    assert stable_damage_percentiles(
        result.ranked_components
    ) == {
        "H0": 0.0,
        "H1": 0.5,
        "H2": 1.0,
    }


def test_percentiles_are_positive_affine_invariant() -> None:
    first = ranking_result((2.0, 4.0, 9.0))
    transformed = ranking_result((7.0, 13.0, 28.0))

    assert stable_damage_percentiles(
        first.ranked_components
    ) == stable_damage_percentiles(
        transformed.ranked_components
    )


def test_nonfinite_damage_fails() -> None:
    result = ranking_result((0.0, 1.0))

    damaged = (
        result.ranked_components[0],
        ranking_component("H1", 1, math.nan),
    )

    with pytest.raises(ValueError, match="finite"):
        stable_damage_percentiles(damaged)


def test_reuse_rates_for_one_prior_circuit() -> None:
    rates = component_reuse_rates(
        ("H0", "H1", "H2"),
        (mask("H0", "H2"),),
    )

    assert rates == {
        "H0": 1.0,
        "H1": 0.0,
        "H2": 1.0,
    }


def test_reuse_rates_use_all_prior_circuits() -> None:
    rates = component_reuse_rates(
        ("H0", "H1", "H2"),
        (
            mask("H0", "H1"),
            mask("H0"),
            mask("H2"),
        ),
    )

    assert rates["H0"] == pytest.approx(2 / 3)
    assert rates["H1"] == pytest.approx(1 / 3)
    assert rates["H2"] == pytest.approx(1 / 3)


def test_primary_removal_score() -> None:
    assert diversity_removal_score(
        damage_percentile=0.4,
        reuse_rate=0.6,
    ) == pytest.approx(0.1)


def test_c1_ranking_is_returned_unchanged() -> None:
    base = ranking_result((0.0, 1.0, 2.0))

    result = build_diversity_ranking(base, ())

    assert result.ranking_result is base
    assert result.entries == ()


def test_reuse_penalty_changes_candidate_order_only() -> None:
    base = ranking_result((0.0, 0.1, 0.2, 0.3))

    result = build_diversity_ranking(
        base,
        (mask("H2"),),
    )

    ordered_ids = tuple(
        value.component_identifier
        for value in result.ranking_result.ranked_components
    )

    assert ordered_ids == (
        "H0",
        "H2",
        "H1",
        "H3",
    )

    diagnostics = {
        value.component_identifier: value
        for value in result.entries
    }

    assert (
        diagnostics["H2"].raw_estimated_removal_damage
        == 0.2
    )
    assert diagnostics["H2"].reuse_rate == 1.0
    assert (
        diagnostics["H2"].removal_score
        == pytest.approx(1 / 6)
    )


def test_transformed_stage9_score_is_removal_score() -> None:
    base = ranking_result((0.0, 1.0, 2.0))

    result = build_diversity_ranking(
        base,
        (mask("H1"),),
    )

    transformed = {
        value.component_identifier: (
            value.estimated_removal_damage
        )
        for value in result.ranking_result.ranked_components
    }
    diagnostics = {
        value.component_identifier: value.removal_score
        for value in result.entries
    }

    assert transformed == diagnostics



def restart_tie_family() -> tuple[ComponentMask, ...]:
    """Create exact removal-score ties for H0, H1 and H2."""

    return (
        mask("H1", "H2"),
        mask("H1", "H2"),
        mask("H2"),
        mask("H2"),
        ComponentMask.all_ablated(),
    )


def test_seed_derivation_is_reproducible() -> None:
    first = derive_search_seed(
        model_seed=1,
        checkpoint_index=7,
        family_member_index=2,
        restart_index=1,
    )
    repeated = derive_search_seed(
        model_seed=1,
        checkpoint_index=7,
        family_member_index=2,
        restart_index=1,
    )

    assert first == repeated
    assert first.bit_generator == "numpy.random.PCG64"
    assert 0 <= first.integer_seed <= 2**32 - 1
    assert (
        first.canonical_material
        == "circuit-families|stage12-diversity-search|"
        "model_seed=1|checkpoint_index=7|"
        "family_member_index=2|restart_index=1"
    )


def test_seed_depends_on_every_required_index() -> None:
    base = derive_search_seed(
        model_seed=1,
        checkpoint_index=7,
        family_member_index=2,
        restart_index=1,
    )

    alternatives = (
        derive_search_seed(
            model_seed=2,
            checkpoint_index=7,
            family_member_index=2,
            restart_index=1,
        ),
        derive_search_seed(
            model_seed=1,
            checkpoint_index=6,
            family_member_index=2,
            restart_index=1,
        ),
        derive_search_seed(
            model_seed=1,
            checkpoint_index=7,
            family_member_index=3,
            restart_index=1,
        ),
        derive_search_seed(
            model_seed=1,
            checkpoint_index=7,
            family_member_index=2,
            restart_index=2,
        ),
    )

    assert all(
        value.integer_seed != base.integer_seed
        for value in alternatives
    )
    assert all(
        value.sha256_digest != base.sha256_digest
        for value in alternatives
    )


def test_c1_rejects_restart_seed() -> None:
    base = ranking_result((0.0, 1.0))

    seed = derive_search_seed(
        model_seed=1,
        checkpoint_index=7,
        family_member_index=2,
        restart_index=1,
    )

    with pytest.raises(ValueError, match="C1"):
        build_diversity_ranking(
            base,
            (),
            restart_index=1,
            seed_record=seed,
        )


def test_restart_zero_preserves_unperturbed_scores() -> None:
    base = ranking_result(
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    )

    result = build_diversity_ranking(
        base,
        restart_tie_family(),
        restart_index=0,
    )

    assert all(
        entry.candidate_ordering_score
        == entry.removal_score
        for entry in result.entries
    )

    assert tuple(
        entry.component_identifier
        for entry in result.entries[:3]
    ) == ("H0", "H1", "H2")


def test_later_restart_is_reproducible() -> None:
    base = ranking_result(
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    )
    seed = derive_search_seed(
        model_seed=1,
        checkpoint_index=7,
        family_member_index=2,
        restart_index=1,
    )

    first = build_diversity_ranking(
        base,
        restart_tie_family(),
        restart_index=1,
        seed_record=seed,
    )
    repeated = build_diversity_ranking(
        base,
        restart_tie_family(),
        restart_index=1,
        seed_record=seed,
    )

    assert first == repeated


def test_restart_only_perturbs_indistinguishable_group() -> None:
    base = ranking_result(
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    )
    seed = derive_search_seed(
        model_seed=1,
        checkpoint_index=7,
        family_member_index=2,
        restart_index=1,
    )

    result = build_diversity_ranking(
        base,
        restart_tie_family(),
        restart_index=1,
        seed_record=seed,
    )

    ordered_ids = tuple(
        entry.component_identifier
        for entry in result.entries
    )

    assert ordered_ids[:3] == ("H2", "H0", "H1")
    assert ordered_ids[3:] == ("H3", "H4", "H5")

    diagnostics = {
        entry.component_identifier: entry
        for entry in result.entries
    }

    for identifier in ("H3", "H4", "H5"):
        assert (
            diagnostics[identifier].candidate_ordering_score
            == diagnostics[identifier].removal_score
        )

    for identifier in ("H0", "H1", "H2"):
        assert abs(
            diagnostics[identifier].candidate_ordering_score
            - diagnostics[identifier].removal_score
        ) <= 1.0e-12



_COMPONENT_INDEX = {
    identifier: index
    for index, identifier in enumerate(
        SEARCHABLE_COMPONENT_IDS
    )
}


def synthetic_metrics(
    mask_value: ComponentMask,
    *,
    fidelity: float,
) -> MaskEvaluationMetrics:
    """Return internally consistent exact synthetic metrics."""

    agreement_count = (
        100
        if fidelity == 1.0
        else 0
    )

    return MaskEvaluationMetrics(
        primary_fidelity=fidelity,
        prediction_agreement_count=agreement_count,
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
            mask_value.retained_attention_head_count
        ),
        retained_mlp_neuron_count=(
            mask_value.retained_mlp_neuron_count
        ),
        retained_component_count=(
            mask_value.retained_component_count
        ),
        retained_component_proportion=(
            mask_value.retained_component_proportion
        ),
        evaluated_example_count=100,
        evaluation_batch_size=100,
    )


def synthetic_full_ranking(
    mask_value: ComponentMask,
) -> RankingResult:
    """Rank every retained component by stable global index."""

    rankings = tuple(
        ComponentRanking(
            component_identifier=identifier,
            component_index=_COMPONENT_INDEX[identifier],
            component_class=(
                component_location(
                    identifier
                ).component_class
            ),
            gate_gradient=0.0,
            estimated_removal_damage=float(
                _COMPONENT_INDEX[identifier]
            ),
            ranking_position=position,
        )
        for position, identifier in enumerate(
            mask_value.retained_component_ids,
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
            mask_value.retained_component_count
        ),
        model_state_sha256_before="state",
        model_state_sha256_after="state",
        hook_counts_before=(),
        hook_counts_after=(),
    )


def single_valid_deletion_evaluator(
    target_component: str,
):
    """Allow exactly one specified deletion and no later deletion."""

    def evaluate(
        mask_value: ComponentMask,
    ) -> MaskEvaluationMetrics:
        ablated = set(mask_value.ablated_component_ids)

        fidelity = (
            1.0
            if ablated in (
                set(),
                {target_component},
            )
            else 0.0
        )

        return synthetic_metrics(
            mask_value,
            fidelity=fidelity,
        )

    return evaluate


def test_c1_wrapper_exactly_matches_stage9_engine() -> None:
    initial_mask = ComponentMask.all_retained()
    initial_metrics = synthetic_metrics(
        initial_mask,
        fidelity=1.0,
    )
    evaluator = single_valid_deletion_evaluator(
        "N0"
    )

    direct = greedy_sparse_search(
        ranking_function=synthetic_full_ranking,
        exact_evaluation_function=evaluator,
        initial_metrics=initial_metrics,
        fidelity_threshold=0.99,
        exact_evaluation_budget=600,
    )

    wrapped = run_diversity_sparse_search(
        base_ranking_function=synthetic_full_ranking,
        exact_evaluation_function=evaluator,
        initial_metrics=initial_metrics,
        accepted_family=(),
        fidelity_threshold=0.99,
        exact_evaluation_budget=600,
    )

    assert wrapped.result == direct
    assert wrapped.restart_index == 0
    assert wrapped.seed_record is None
    assert all(
        ranking.entries == ()
        for ranking in wrapped.ranking_results
    )


def test_c2_reuse_changes_effort_not_exact_acceptance() -> None:
    initial_mask = ComponentMask.all_retained()
    initial_metrics = synthetic_metrics(
        initial_mask,
        fidelity=1.0,
    )
    target = "N100"
    evaluator = single_valid_deletion_evaluator(
        target
    )

    direct = greedy_sparse_search(
        ranking_function=synthetic_full_ranking,
        exact_evaluation_function=evaluator,
        initial_metrics=initial_metrics,
        fidelity_threshold=0.99,
        exact_evaluation_budget=700,
    )

    diversity = run_diversity_sparse_search(
        base_ranking_function=synthetic_full_ranking,
        exact_evaluation_function=evaluator,
        initial_metrics=initial_metrics,
        accepted_family=(mask(target),),
        fidelity_threshold=0.99,
        exact_evaluation_budget=700,
    )

    assert (
        direct.accepted_removals[0].removed_component
        == target
    )
    assert (
        diversity.result.accepted_removals[
            0
        ].removed_component
        == target
    )

    assert (
        diversity.result.exact_evaluations_used
        < direct.exact_evaluations_used
    )

    first_ranking = diversity.ranking_results[0]

    assert (
        first_ranking.ranking_result.ranked_components[
            0
        ].component_identifier
        == target
    )

    target_entry = next(
        entry
        for entry in first_ranking.entries
        if entry.component_identifier == target
    )

    assert target_entry.reuse_rate == 1.0
    assert target_entry.removal_score < 0.0



def sparsity_boundary_evaluator(
    mask_value: ComponentMask,
) -> MaskEvaluationMetrics:
    """Permit every mask down to the exact sparsity boundary."""

    fidelity = (
        1.0
        if mask_value.retained_component_count
        >= MEANINGFULLY_SPARSE_MAX_COMPONENTS
        else 0.0
    )

    return synthetic_metrics(
        mask_value,
        fidelity=fidelity,
    )


def test_family_controller_recovers_distinct_c1_and_c2() -> None:
    initial_mask = ComponentMask.all_retained()
    initial_metrics = synthetic_metrics(
        initial_mask,
        fidelity=1.0,
    )

    result = run_sequential_family_search(
        base_ranking_function=synthetic_full_ranking,
        exact_evaluation_function=(
            sparsity_boundary_evaluator
        ),
        initial_metrics=initial_metrics,
        fidelity_threshold=0.99,
        distinctness_cutoff=0.5,
        model_seed=1,
        checkpoint_index=7,
        family_target=2,
        max_restarts_per_alternative=1,
        per_requested_circuit_budget=5_000,
        per_cell_budget=10_000,
        reuse_coefficient=1.0,
    )

    assert result.family_size == 2
    assert result.status == (
        "right_censored_at_family_target"
    )
    assert result.right_censored
    assert result.stopping_reason == "family_target_reached"

    first, second = result.members

    assert first.member_index == 1
    assert second.member_index == 2
    assert first.mask.retained_component_count == 258
    assert second.mask.retained_component_count == 258
    assert second.maximum_pairwise_overlap <= 0.5
    assert first.mask != second.mask

    assert result.exact_evaluations_used <= 10_000
    assert result.budget_remaining >= 0


def test_exact_distinctness_failure_stops_family() -> None:
    initial_mask = ComponentMask.all_retained()
    initial_metrics = synthetic_metrics(
        initial_mask,
        fidelity=1.0,
    )

    result = run_sequential_family_search(
        base_ranking_function=synthetic_full_ranking,
        exact_evaluation_function=(
            sparsity_boundary_evaluator
        ),
        initial_metrics=initial_metrics,
        fidelity_threshold=0.99,
        distinctness_cutoff=0.5,
        model_seed=1,
        checkpoint_index=7,
        family_target=2,
        max_restarts_per_alternative=1,
        per_requested_circuit_budget=5_000,
        per_cell_budget=10_000,
        reuse_coefficient=0.0,
    )

    assert result.family_size == 1
    assert result.status == "distinctness_failure"
    assert not result.right_censored

    alternative = result.restart_outcomes[-1]

    assert (
        alternative.outcome_status
        == "distinctness_failure"
    )
    assert alternative.maximum_pairwise_overlap == 1
    assert not alternative.accepted_candidate


def test_member_budget_is_shared_and_never_exceeded() -> None:
    initial_mask = ComponentMask.all_retained()
    initial_metrics = synthetic_metrics(
        initial_mask,
        fidelity=1.0,
    )

    result = run_sequential_family_search(
        base_ranking_function=synthetic_full_ranking,
        exact_evaluation_function=(
            sparsity_boundary_evaluator
        ),
        initial_metrics=initial_metrics,
        fidelity_threshold=0.99,
        distinctness_cutoff=0.5,
        model_seed=1,
        checkpoint_index=7,
        family_target=1,
        per_requested_circuit_budget=4_000,
        per_cell_budget=4_000,
    )

    assert result.family_size == 0
    assert result.status == "budget_exhaustion"
    assert result.exact_evaluations_used == 4_000
    assert result.budget_remaining == 0


def test_family_controller_rejects_budget_increases() -> None:
    initial_mask = ComponentMask.all_retained()
    initial_metrics = synthetic_metrics(
        initial_mask,
        fidelity=1.0,
    )

    with pytest.raises(ValueError, match="10,000"):
        run_sequential_family_search(
            base_ranking_function=synthetic_full_ranking,
            exact_evaluation_function=(
                sparsity_boundary_evaluator
            ),
            initial_metrics=initial_metrics,
            fidelity_threshold=0.99,
            distinctness_cutoff=0.5,
            model_seed=1,
            checkpoint_index=7,
            per_requested_circuit_budget=10_001,
        )

    with pytest.raises(ValueError, match="50,000"):
        run_sequential_family_search(
            base_ranking_function=synthetic_full_ranking,
            exact_evaluation_function=(
                sparsity_boundary_evaluator
            ),
            initial_metrics=initial_metrics,
            fidelity_threshold=0.99,
            distinctness_cutoff=0.5,
            model_seed=1,
            checkpoint_index=7,
            per_cell_budget=50_001,
        )



def checkpoint_fixture_model():
    return build_transformer(
        load_model_config("configs/model.yaml"),
        seed=0,
        device="cpu",
    )


def checkpoint_fixture_examples(
    count: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [
        (left, right)
        for left in range(113)
        for right in range(113)
    ][:count]

    inputs = torch.tensor(
        [
            [left, right, 113]
            for left, right in pairs
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            (left + right) % 113
            for left, right in pairs
        ],
        dtype=torch.long,
    )

    return inputs, targets


def checkpoint_fixture_hook_counts(
    model,
) -> dict[str, int]:
    from circuit_families.interpretability.masks import (
        ATTENTION_HEAD_HOOK_NAME,
        MLP_NEURON_HOOK_NAME,
    )

    return {
        name: len(model.hook_dict[name]._forward_hooks)
        for name in (
            ATTENTION_HEAD_HOOK_NAME,
            MLP_NEURON_HOOK_NAME,
        )
    }


def checkpoint_fixture_context(
    *,
    example_count: int = 7,
) -> CheckpointEvaluationContext:
    model = checkpoint_fixture_model()
    inputs, targets = checkpoint_fixture_examples(
        count=example_count
    )
    model_hash = canonical_state_hash(
        model.state_dict()
    )

    return CheckpointEvaluationContext(
        run_id="fixture-run",
        checkpoint_phase="fixture",
        checkpoint_step=0,
        checkpoint_path=Path("fixture.pt"),
        checkpoint_sha256="a" * 64,
        checkpoint_manifest_path=Path(
            "fixture-checkpoints.json"
        ),
        checkpoint_manifest_sha256="b" * 64,
        training_manifest_path=Path(
            "fixture-training.json"
        ),
        training_manifest_sha256="c" * 64,
        model_state_sha256=model_hash,
        task_config_sha256="d" * 64,
        model_config_sha256="e" * 64,
        training_config_sha256="f" * 64,
        combined_config_sha256="1" * 64,
        dataset_sha256="2" * 64,
        split_sha256="3" * 64,
        dataset_archive_sha256="4" * 64,
        dataset_metadata_sha256="5" * 64,
        example_ordering="fixture_order",
        model=model,
        inputs=inputs,
        targets=targets,
        device=torch.device("cpu"),
    )


def test_checkpoint_family_search_preserves_integrity() -> None:
    context = checkpoint_fixture_context(
        example_count=7
    )
    state_before = canonical_state_hash(
        context.model.state_dict()
    )
    hooks_before = checkpoint_fixture_hook_counts(
        context.model
    )

    execution = run_checkpoint_family_search(
        context,
        fidelity_threshold=1.0e-12,
        distinctness_cutoff=0.5,
        model_seed=0,
        checkpoint_index=1,
        ranking_batch_size=3,
        evaluation_batch_size=4,
        family_target=1,
        per_requested_circuit_budget=16,
        per_cell_budget=16,
    )

    assert execution.result.status == "budget_exhaustion"
    assert execution.result.family_size == 0
    assert execution.result.exact_evaluations_used == 16
    assert execution.pseudo_target_count == 7
    assert len(execution.pseudo_target_sha256) == 64
    assert len(
        execution.full_model_reference_sha256
    ) == 64
    assert (
        execution.full_model_reference_example_count
        == 7
    )
    assert (
        execution.full_model_reference_batch_size
        == 4
    )

    assert (
        execution.model_state_sha256_before
        == execution.model_state_sha256_after
        == state_before
    )
    assert dict(
        execution.hook_counts_before
    ) == hooks_before
    assert dict(
        execution.hook_counts_after
    ) == hooks_before
    assert canonical_state_hash(
        context.model.state_dict()
    ) == state_before
    assert checkpoint_fixture_hook_counts(
        context.model
    ) == hooks_before
    assert all(
        parameter.grad is None
        for parameter in context.model.parameters()
    )


def test_checkpoint_family_search_is_deterministic() -> None:
    first_context = checkpoint_fixture_context(
        example_count=7
    )
    second_context = checkpoint_fixture_context(
        example_count=7
    )

    arguments = {
        "fidelity_threshold": 1.0e-12,
        "distinctness_cutoff": 0.5,
        "model_seed": 0,
        "checkpoint_index": 1,
        "ranking_batch_size": 7,
        "evaluation_batch_size": 7,
        "family_target": 1,
        "per_requested_circuit_budget": 16,
        "per_cell_budget": 16,
    }

    first = run_checkpoint_family_search(
        first_context,
        **arguments,
    )
    second = run_checkpoint_family_search(
        second_context,
        **arguments,
    )

    assert first == second


def test_checkpoint_family_search_reuses_one_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = checkpoint_fixture_context(
        example_count=7
    )
    original = (
        diversity_search_module.evaluate_component_mask
    )
    reference_ids: list[int] = []

    def wrapped_evaluator(
        *args,
        full_model_reference=None,
        **kwargs,
    ):
        assert full_model_reference is not None
        reference_ids.append(id(full_model_reference))

        return original(
            *args,
            full_model_reference=full_model_reference,
            **kwargs,
        )

    monkeypatch.setattr(
        diversity_search_module,
        "evaluate_component_mask",
        wrapped_evaluator,
    )

    run_checkpoint_family_search(
        context,
        fidelity_threshold=1.0e-12,
        distinctness_cutoff=0.5,
        model_seed=0,
        checkpoint_index=1,
        ranking_batch_size=3,
        evaluation_batch_size=4,
        family_target=1,
        per_requested_circuit_budget=1,
        per_cell_budget=1,
    )

    assert len(reference_ids) == 2
    assert len(set(reference_ids)) == 1


def test_checkpoint_family_search_rejects_state_mismatch() -> None:
    context = checkpoint_fixture_context(
        example_count=3
    )
    invalid = CheckpointEvaluationContext(
        **{
            **context.__dict__,
            "model_state_sha256": "0" * 64,
        }
    )

    with pytest.raises(
        ValueError,
        match="model-state hash",
    ):
        run_checkpoint_family_search(
            invalid,
            fidelity_threshold=0.99,
            distinctness_cutoff=0.5,
            model_seed=0,
            checkpoint_index=1,
            ranking_batch_size=3,
            evaluation_batch_size=3,
            family_target=1,
            per_requested_circuit_budget=1,
            per_cell_budget=1,
        )



def test_family_controller_stops_after_first_valid_alternative(
) -> None:
    initial_mask = ComponentMask.all_retained()
    initial_metrics = synthetic_metrics(
        initial_mask,
        fidelity=1.0,
    )

    result = run_sequential_family_search(
        base_ranking_function=synthetic_full_ranking,
        exact_evaluation_function=(
            sparsity_boundary_evaluator
        ),
        initial_metrics=initial_metrics,
        fidelity_threshold=0.99,
        distinctness_cutoff=0.5,
        model_seed=1,
        checkpoint_index=7,
        family_target=2,
        max_restarts_per_alternative=5,
        per_requested_circuit_budget=5_000,
        per_cell_budget=10_000,
        reuse_coefficient=1.0,
    )

    assert result.family_size == 2

    alternative_outcomes = [
        outcome
        for outcome in result.restart_outcomes
        if outcome.requested_member_index == 2
    ]

    assert len(alternative_outcomes) == 1
    assert alternative_outcomes[0].restart_index == 0
    assert alternative_outcomes[0].accepted_candidate
    assert (
        alternative_outcomes[0].outcome_status
        == VALID_DISTINCT_CANDIDATE
    )



def test_family_controller_reports_member_boundaries() -> None:
    initial_mask = ComponentMask.all_retained()
    initial_metrics = synthetic_metrics(
        initial_mask,
        fidelity=1.0,
    )
    started: list[int] = []
    finished: list[int] = []

    result = run_sequential_family_search(
        base_ranking_function=synthetic_full_ranking,
        exact_evaluation_function=(
            sparsity_boundary_evaluator
        ),
        initial_metrics=initial_metrics,
        fidelity_threshold=0.99,
        distinctness_cutoff=0.5,
        model_seed=1,
        checkpoint_index=7,
        family_target=2,
        max_restarts_per_alternative=1,
        per_requested_circuit_budget=5_000,
        per_cell_budget=10_000,
        reuse_coefficient=1.0,
        member_started_callback=started.append,
        member_finished_callback=finished.append,
    )

    assert result.family_size == 2
    assert started == [1, 2]
    assert finished == [1, 2]

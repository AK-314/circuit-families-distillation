"""Tests for deterministic Stage 9 gate-gradient ranking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import circuit_families.interpretability.sparse_search as sparse_search_module
from circuit_families.config import load_model_config
from circuit_families.interpretability.fidelity import (
    CheckpointEvaluationContext,
    MaskEvaluationMetrics,
    compute_full_model_reference,
    evaluate_component_mask,
)
from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_HOOK_NAME,
    MLP_NEURON_HOOK_NAME,
    SEARCHABLE_COMPONENT_COUNT,
    SEARCHABLE_COMPONENT_IDS,
    ComponentMask,
    component_location,
    load_component_mask,
)
from circuit_families.interpretability.sparse_search import (
    CANDIDATE_BATCH_SIZE,
    ComponentRanking,
    RankingResult,
    freeze_full_model_pseudo_targets,
    greedy_sparse_search,
    is_meaningfully_sparse,
    partition_ranked_candidates,
    rank_retained_components,
    remove_component,
    run_checkpoint_sparse_search,
    sort_component_rankings,
    write_sparse_search_artifacts,
)
from circuit_families.models import build_transformer
from circuit_families.training import canonical_state_hash
from circuit_families.training.metrics import final_position_logits


def _model():
    return build_transformer(
        load_model_config("configs/model.yaml"),
        seed=0,
        device="cpu",
    )


def _examples(
    count: int = 17,
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


def _hook_counts(model) -> dict[str, int]:
    return {
        name: len(model.hook_dict[name]._forward_hooks)
        for name in (
            ATTENTION_HEAD_HOOK_NAME,
            MLP_NEURON_HOOK_NAME,
        )
    }


def _ranking(
    identifier: str,
    index: int,
    score: float,
) -> ComponentRanking:
    return ComponentRanking(
        component_identifier=identifier,
        component_index=index,
        component_class="test",
        gate_gradient=-score,
        estimated_removal_damage=score,
        ranking_position=0,
    )


def test_full_model_pseudo_targets_use_only_final_position_logits() -> None:
    model = _model()
    inputs, _ = _examples()

    frozen = freeze_full_model_pseudo_targets(
        model,
        inputs,
        batch_size=5,
    )

    with torch.inference_mode():
        expected = final_position_logits(model(inputs)).argmax(dim=-1)

    assert torch.equal(frozen, expected)
    assert frozen.shape == (17,)
    assert frozen.dtype == torch.long
    assert int(frozen.min().item()) >= 0
    assert int(frozen.max().item()) <= 112


def test_all_retained_ranking_contains_all_516_components() -> None:
    model = _model()
    inputs, _ = _examples(count=7)
    pseudo_targets = freeze_full_model_pseudo_targets(
        model,
        inputs,
        batch_size=7,
    )

    result = rank_retained_components(
        model,
        inputs,
        pseudo_targets,
        ComponentMask.all_retained(),
        batch_size=3,
    )

    assert result.retained_component_count == 516
    assert len(result.ranked_components) == 516
    assert len(result.mean_gate_gradients) == 516
    assert result.gradient_source == "component_gates"
    assert result.evaluated_example_count == 7
    assert {
        value.component_identifier
        for value in result.ranked_components
    } == set(ComponentMask.all_retained().retained_component_ids)


def test_only_retained_components_are_ranked() -> None:
    model = _model()
    inputs, _ = _examples(count=7)
    pseudo_targets = freeze_full_model_pseudo_targets(
        model,
        inputs,
        batch_size=7,
    )
    mask = ComponentMask.from_ablated_identifiers(
        ["H0", "N0", "N511"]
    )

    result = rank_retained_components(
        model,
        inputs,
        pseudo_targets,
        mask,
        batch_size=4,
    )

    identifiers = {
        value.component_identifier
        for value in result.ranked_components
    }

    assert len(identifiers) == 513
    assert "H0" not in identifiers
    assert "N0" not in identifiers
    assert "N511" not in identifiers

    assert result.mean_gate_gradients[0] == 0.0
    assert result.mean_gate_gradients[4] == 0.0
    assert result.mean_gate_gradients[-1] == 0.0


def test_exact_score_ties_use_lower_stable_component_index() -> None:
    rankings = (
        _ranking("N1", 5, 0.25),
        _ranking("H1", 1, 0.25),
        _ranking("N0", 4, -0.5),
        _ranking("H0", 0, 0.25),
    )

    ordered = sort_component_rankings(rankings)

    assert [
        value.component_identifier
        for value in ordered
    ] == ["N0", "H0", "H1", "N1"]

    assert [
        value.ranking_position
        for value in ordered
    ] == [1, 2, 3, 4]


def test_gradient_accumulation_matches_unbatched_calculation() -> None:
    model = _model()
    inputs, _ = _examples(count=17)
    pseudo_targets = freeze_full_model_pseudo_targets(
        model,
        inputs,
        batch_size=17,
    )
    mask = ComponentMask.from_ablated_identifiers(
        ["H3", "N17", "N255"]
    )

    unbatched = rank_retained_components(
        model,
        inputs,
        pseudo_targets,
        mask,
        batch_size=17,
    )
    batched = rank_retained_components(
        model,
        inputs,
        pseudo_targets,
        mask,
        batch_size=5,
    )

    assert batched.mean_pseudo_target_loss == pytest.approx(
        unbatched.mean_pseudo_target_loss,
        abs=1.0e-6,
        rel=1.0e-6,
    )

    assert batched.mean_gate_gradients == pytest.approx(
        unbatched.mean_gate_gradients,
        abs=1.0e-6,
        rel=1.0e-6,
    )


def test_parameter_gradients_do_not_drive_ranking() -> None:
    model = _model()
    inputs, _ = _examples(count=7)
    pseudo_targets = freeze_full_model_pseudo_targets(
        model,
        inputs,
        batch_size=7,
    )

    clean = rank_retained_components(
        model,
        inputs,
        pseudo_targets,
        ComponentMask.all_retained(),
        batch_size=4,
    )

    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 1.0e9)

    contaminated = rank_retained_components(
        model,
        inputs,
        pseudo_targets,
        ComponentMask.all_retained(),
        batch_size=4,
    )

    assert contaminated.mean_gate_gradients == pytest.approx(
        clean.mean_gate_gradients,
        abs=0.0,
        rel=0.0,
    )
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )


def test_ranking_preserves_model_state_mode_and_hooks() -> None:
    model = _model()
    inputs, _ = _examples(count=7)
    pseudo_targets = freeze_full_model_pseudo_targets(
        model,
        inputs,
        batch_size=7,
    )

    model.train()
    state_before = canonical_state_hash(model.state_dict())
    hooks_before = _hook_counts(model)

    result = rank_retained_components(
        model,
        inputs,
        pseudo_targets,
        ComponentMask.all_retained(),
        batch_size=3,
    )

    assert model.training
    assert canonical_state_hash(model.state_dict()) == state_before
    assert _hook_counts(model) == hooks_before
    assert (
        result.model_state_sha256_before
        == result.model_state_sha256_after
    )
    assert result.hook_counts_before == result.hook_counts_after


def test_ranking_exception_cleans_up_hooks_and_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    inputs, _ = _examples(count=7)
    pseudo_targets = freeze_full_model_pseudo_targets(
        model,
        inputs,
        batch_size=7,
    )

    model.train()
    hooks_before = _hook_counts(model)
    state_before = canonical_state_hash(model.state_dict())

    def fail_cross_entropy(*args, **kwargs):
        raise RuntimeError("deliberate ranking failure")

    monkeypatch.setattr(
        "circuit_families.interpretability.sparse_search."
        "functional.cross_entropy",
        fail_cross_entropy,
    )

    with pytest.raises(
        RuntimeError,
        match="deliberate ranking failure",
    ):
        rank_retained_components(
            model,
            inputs,
            pseudo_targets,
            ComponentMask.all_retained(),
            batch_size=3,
        )

    assert model.training
    assert _hook_counts(model) == hooks_before
    assert canonical_state_hash(model.state_dict()) == state_before
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )


def test_candidate_partitioning_uses_consecutive_batches_of_16() -> None:
    rankings = tuple(
        _ranking(f"N{index}", index, float(index))
        for index in range(33)
    )

    batches = partition_ranked_candidates(rankings)

    assert [len(batch) for batch in batches] == [16, 16, 1]
    assert all(
        len(batch) <= CANDIDATE_BATCH_SIZE
        for batch in batches
    )
    assert [
        value.component_index
        for batch in batches
        for value in batch
    ] == list(range(33))


def test_single_component_removal_preserves_stable_mask_format() -> None:
    initial = ComponentMask.all_retained()
    after_head = remove_component(initial, "H0")
    after_neuron = remove_component(after_head, "N511")

    assert after_head.retained_component_count == 515
    assert after_head.ablated_component_ids == ("H0",)

    assert after_neuron.retained_component_count == 514
    assert after_neuron.ablated_component_ids == (
        "H0",
        "N511",
    )

    with pytest.raises(ValueError, match="not currently retained"):
        remove_component(after_neuron, "H0")


def test_frozen_meaningful_sparsity_boundary() -> None:
    retained_258 = ComponentMask.from_retained_identifiers(
        ComponentMask.all_retained().retained_component_ids[:258]
    )
    retained_259 = ComponentMask.from_retained_identifiers(
        ComponentMask.all_retained().retained_component_ids[:259]
    )
    empty = ComponentMask.all_ablated()

    assert is_meaningfully_sparse(retained_258)
    assert not is_meaningfully_sparse(retained_259)
    assert is_meaningfully_sparse(empty)
    assert not is_meaningfully_sparse(
        ComponentMask.all_retained()
    )

    assert SEARCHABLE_COMPONENT_COUNT == 516


_COMPONENT_INDEX = {
    identifier: index
    for index, identifier in enumerate(SEARCHABLE_COMPONENT_IDS)
}


def _exact_metrics(
    mask: ComponentMask,
    fidelity: float,
    *,
    example_count: int = 101,
) -> MaskEvaluationMetrics:
    agreement_count = round(fidelity * example_count)

    return MaskEvaluationMetrics(
        primary_fidelity=fidelity,
        prediction_agreement_count=agreement_count,
        full_accuracy=1.0,
        masked_accuracy=fidelity,
        accuracy_change=fidelity - 1.0,
        full_cross_entropy=0.1,
        masked_cross_entropy=0.2,
        cross_entropy_change=0.1,
        mean_kl_divergence=0.01,
        mean_jensen_shannon_divergence=0.005,
        maximum_absolute_logit_difference=0.25,
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
        evaluated_example_count=example_count,
        evaluation_batch_size=example_count,
    )


def _synthetic_ranking_result(
    mask: ComponentMask,
    *,
    preferred_order: tuple[str, ...] = (),
) -> RankingResult:
    retained = set(mask.retained_component_ids)

    ordered_identifiers = tuple(
        identifier
        for identifier in preferred_order
        if identifier in retained
    ) + tuple(
        identifier
        for identifier in mask.retained_component_ids
        if identifier not in preferred_order
    )

    rankings = tuple(
        ComponentRanking(
            component_identifier=identifier,
            component_index=_COMPONENT_INDEX[identifier],
            component_class=component_location(
                identifier
            ).component_class,
            gate_gradient=-float(position),
            estimated_removal_damage=float(position),
            ranking_position=position,
        )
        for position, identifier in enumerate(
            ordered_identifiers,
            start=1,
        )
    )

    return RankingResult(
        mean_pseudo_target_loss=0.1,
        mean_gate_gradients=(0.0,) * SEARCHABLE_COMPONENT_COUNT,
        ranked_components=rankings,
        evaluated_example_count=101,
        ranking_batch_size=101,
        retained_component_count=mask.retained_component_count,
        model_state_sha256_before="a" * 64,
        model_state_sha256_after="a" * 64,
        hook_counts_before=(),
        hook_counts_after=(),
    )


def test_known_synthetic_removal_sequence_is_recovered() -> None:
    ranking_masks: list[ComponentMask] = []

    def ranker(mask: ComponentMask) -> RankingResult:
        ranking_masks.append(mask)
        return _synthetic_ranking_result(mask)

    fidelity_by_ablated = {
        frozenset({"H1"}): 0.92,
        frozenset({"H2"}): 0.95,
        frozenset({"H2", "H1"}): 0.94,
        frozenset({"H2", "H3"}): 0.93,
    }

    def evaluator(mask: ComponentMask) -> MaskEvaluationMetrics:
        fidelity = fidelity_by_ablated.get(
            frozenset(mask.ablated_component_ids),
            0.0,
        )
        return _exact_metrics(mask, fidelity)

    result = greedy_sparse_search(
        ranking_function=ranker,
        exact_evaluation_function=evaluator,
        initial_metrics=_exact_metrics(
            ComponentMask.all_retained(),
            1.0,
        ),
        fidelity_threshold=0.90,
        exact_evaluation_budget=1_000,
    )

    assert [
        removal.removed_component
        for removal in result.accepted_removals
    ] == ["H2", "H1"]

    assert [
        removal.retained_count_before
        for removal in result.accepted_removals
    ] == [516, 515]

    assert [
        removal.retained_count_after
        for removal in result.accepted_removals
    ] == [515, 514]

    assert result.final_mask.ablated_component_ids == (
        "H1",
        "H2",
    )
    assert result.final_metrics.primary_fidelity == 0.94
    assert result.status == "valid_but_not_meaningfully_sparse"
    assert result.locally_single_deletion_minimal
    assert not result.meaningfully_sparse

    assert result.ranking_passes_used == 3
    assert len(ranking_masks) == 3
    assert ranking_masks[0] == ComponentMask.all_retained()
    assert ranking_masks[1].ablated_component_ids == ("H2",)
    assert ranking_masks[2].ablated_component_ids == (
        "H1",
        "H2",
    )

    assert result.exact_evaluations_used == 546
    assert result.candidate_batches_tested == 35
    assert result.rejected_candidate_count == 544

    for removal in result.accepted_removals:
        assert (
            removal.exact_fidelity_after_removal
            >= result.fidelity_threshold
        )
        assert (
            removal.accepted_mask.retained_component_count
            == removal.retained_count_after
        )


def test_first_valid_batch_is_used_and_highest_fidelity_wins() -> None:
    def evaluator(mask: ComponentMask) -> MaskEvaluationMetrics:
        ablated = frozenset(mask.ablated_component_ids)

        fidelity = {
            frozenset({"N12"}): 0.93,
            frozenset({"N13"}): 0.97,
            frozenset({"N14"}): 0.95,
        }.get(ablated, 0.0)

        return _exact_metrics(mask, fidelity)

    result = greedy_sparse_search(
        ranking_function=_synthetic_ranking_result,
        exact_evaluation_function=evaluator,
        initial_metrics=_exact_metrics(
            ComponentMask.all_retained(),
            1.0,
        ),
        fidelity_threshold=0.90,
        exact_evaluation_budget=32,
    )

    assert len(result.accepted_removals) == 1
    accepted = result.accepted_removals[0]

    assert accepted.removed_component == "N13"
    assert accepted.candidate_batch_index == 2
    assert accepted.candidates_exactly_tested_in_iteration == 32
    assert accepted.exact_fidelity_after_removal == 0.97

    assert result.exact_evaluations_used == 32
    assert result.status == "budget_exhaustion"
    assert not result.locally_single_deletion_minimal

    first_batch = result.candidate_evaluations[:16]
    second_batch = result.candidate_evaluations[16:32]

    assert not any(record.passed_threshold for record in first_batch)
    assert sum(record.accepted for record in second_batch) == 1


def test_lower_component_index_resolves_exact_fidelity_tie() -> None:
    def evaluator(mask: ComponentMask) -> MaskEvaluationMetrics:
        ablated = frozenset(mask.ablated_component_ids)

        fidelity = {
            frozenset({"N12"}): 0.95,
            frozenset({"N13"}): 0.95,
        }.get(ablated, 0.0)

        return _exact_metrics(mask, fidelity)

    result = greedy_sparse_search(
        ranking_function=_synthetic_ranking_result,
        exact_evaluation_function=evaluator,
        initial_metrics=_exact_metrics(
            ComponentMask.all_retained(),
            1.0,
        ),
        fidelity_threshold=0.90,
        exact_evaluation_budget=32,
    )

    assert result.accepted_removals[0].removed_component == "N12"

    tied_loser = next(
        record
        for record in result.candidate_evaluations
        if record.candidate_component == "N13"
    )
    assert tied_loser.passed_threshold
    assert not tied_loser.accepted
    assert tied_loser.rejection_reason == (
        "exact_fidelity_tie_broken_by_component_index"
    )


def test_threshold_equality_is_valid_and_below_threshold_is_rejected() -> None:
    def evaluator(mask: ComponentMask) -> MaskEvaluationMetrics:
        ablated = frozenset(mask.ablated_component_ids)

        fidelity = {
            frozenset({"H0"}): 0.899999,
            frozenset({"H1"}): 0.90,
        }.get(ablated, 0.0)

        return _exact_metrics(mask, fidelity)

    result = greedy_sparse_search(
        ranking_function=_synthetic_ranking_result,
        exact_evaluation_function=evaluator,
        initial_metrics=_exact_metrics(
            ComponentMask.all_retained(),
            1.0,
        ),
        fidelity_threshold=0.90,
        exact_evaluation_budget=16,
    )

    assert result.accepted_removals[0].removed_component == "H1"

    rejected = next(
        record
        for record in result.candidate_evaluations
        if record.candidate_component == "H0"
    )
    assert not rejected.passed_threshold
    assert not rejected.accepted
    assert rejected.rejection_reason == (
        "below_fidelity_threshold"
    )


def test_incomplete_candidate_batch_cannot_accept_valid_candidate() -> None:
    def evaluator(mask: ComponentMask) -> MaskEvaluationMetrics:
        fidelity = (
            0.99
            if mask.ablated_component_ids == ("H1",)
            else 0.0
        )
        return _exact_metrics(mask, fidelity)

    result = greedy_sparse_search(
        ranking_function=_synthetic_ranking_result,
        exact_evaluation_function=evaluator,
        initial_metrics=_exact_metrics(
            ComponentMask.all_retained(),
            1.0,
        ),
        fidelity_threshold=0.90,
        exact_evaluation_budget=10,
    )

    assert result.status == "budget_exhaustion"
    assert result.exact_evaluations_used == 10
    assert result.budget_remaining == 0
    assert not result.accepted_removals
    assert result.final_mask == ComponentMask.all_retained()
    assert not result.locally_single_deletion_minimal

    h1_record = next(
        record
        for record in result.candidate_evaluations
        if record.candidate_component == "H1"
    )
    assert h1_record.passed_threshold
    assert not h1_record.accepted
    assert h1_record.rejection_reason == (
        "incomplete_candidate_batch_due_to_budget"
    )


def test_terminal_check_tests_every_retained_deletion() -> None:
    def evaluator(mask: ComponentMask) -> MaskEvaluationMetrics:
        return _exact_metrics(mask, 0.0)

    result = greedy_sparse_search(
        ranking_function=_synthetic_ranking_result,
        exact_evaluation_function=evaluator,
        initial_metrics=_exact_metrics(
            ComponentMask.all_retained(),
            1.0,
        ),
        fidelity_threshold=0.90,
        exact_evaluation_budget=516,
    )

    assert result.status == (
        "no_feasible_sparse_candidate_discovered_within_budget"
    )
    assert result.exact_evaluations_used == 516
    assert result.ranking_passes_used == 1
    assert result.candidate_batches_tested == 33
    assert result.locally_single_deletion_minimal
    assert result.final_mask == ComponentMask.all_retained()
    assert not result.meaningfully_sparse
    assert not result.accepted_removals

    assert {
        record.candidate_component
        for record in result.candidate_evaluations
    } == set(SEARCHABLE_COMPONENT_IDS)


def test_insufficient_terminal_budget_is_budget_exhaustion() -> None:
    def evaluator(mask: ComponentMask) -> MaskEvaluationMetrics:
        return _exact_metrics(mask, 0.0)

    result = greedy_sparse_search(
        ranking_function=_synthetic_ranking_result,
        exact_evaluation_function=evaluator,
        initial_metrics=_exact_metrics(
            ComponentMask.all_retained(),
            1.0,
        ),
        fidelity_threshold=0.90,
        exact_evaluation_budget=515,
    )

    assert result.status == "budget_exhaustion"
    assert result.exact_evaluations_used == 515
    assert not result.locally_single_deletion_minimal
    assert result.final_mask == ComponentMask.all_retained()
    assert result.stopping_reason == (
        "budget_exhausted_inside_candidate_batch"
    )


def test_exact_evaluation_budget_is_never_exceeded() -> None:
    for budget in (0, 1, 15, 16, 17, 100, 515, 516):
        result = greedy_sparse_search(
            ranking_function=_synthetic_ranking_result,
            exact_evaluation_function=lambda mask: _exact_metrics(
                mask,
                0.0,
            ),
            initial_metrics=_exact_metrics(
                ComponentMask.all_retained(),
                1.0,
            ),
            fidelity_threshold=0.90,
            exact_evaluation_budget=budget,
        )

        assert result.exact_evaluations_used <= budget
        assert result.budget_remaining >= 0


def test_invalid_ranking_has_explicit_failure_status() -> None:
    def invalid_ranker(mask: ComponentMask) -> RankingResult:
        valid = _synthetic_ranking_result(mask)
        return RankingResult(
            **{
                **valid.__dict__,
                "ranked_components": valid.ranked_components[:-1],
            }
        )

    result = greedy_sparse_search(
        ranking_function=invalid_ranker,
        exact_evaluation_function=lambda mask: _exact_metrics(
            mask,
            1.0,
        ),
        initial_metrics=_exact_metrics(
            ComponentMask.all_retained(),
            1.0,
        ),
        fidelity_threshold=0.90,
        exact_evaluation_budget=1_000,
    )

    assert result.status == "ranking_failure"
    assert result.exact_evaluations_used == 0
    assert result.ranking_passes_used == 1
    assert not result.locally_single_deletion_minimal
    assert result.failure_detail is not None


def test_invalid_exact_metrics_have_explicit_failure_status() -> None:
    wrong_mask = ComponentMask.all_retained()

    def evaluator(mask: ComponentMask) -> MaskEvaluationMetrics:
        return _exact_metrics(wrong_mask, 1.0)

    result = greedy_sparse_search(
        ranking_function=_synthetic_ranking_result,
        exact_evaluation_function=evaluator,
        initial_metrics=_exact_metrics(
            ComponentMask.all_retained(),
            1.0,
        ),
        fidelity_threshold=0.90,
        exact_evaluation_budget=1_000,
    )

    assert result.status == "invalid_masking_output"
    assert result.exact_evaluations_used == 0
    assert result.failure_detail is not None
    assert result.final_mask == ComponentMask.all_retained()


def test_search_is_deterministic_under_identical_inputs() -> None:
    fidelity_by_ablated = {
        frozenset({"H2"}): 0.95,
        frozenset({"H2", "H1"}): 0.94,
    }

    def evaluator(mask: ComponentMask) -> MaskEvaluationMetrics:
        return _exact_metrics(
            mask,
            fidelity_by_ablated.get(
                frozenset(mask.ablated_component_ids),
                0.0,
            ),
        )

    arguments = {
        "ranking_function": _synthetic_ranking_result,
        "exact_evaluation_function": evaluator,
        "initial_metrics": _exact_metrics(
            ComponentMask.all_retained(),
            1.0,
        ),
        "fidelity_threshold": 0.90,
        "exact_evaluation_budget": 1_000,
    }

    first = greedy_sparse_search(**arguments)
    second = greedy_sparse_search(**arguments)

    assert first == second



def _checkpoint_context(
    *,
    example_count: int = 7,
) -> CheckpointEvaluationContext:
    model = _model()
    inputs, targets = _examples(count=example_count)
    model_hash = canonical_state_hash(model.state_dict())

    return CheckpointEvaluationContext(
        run_id="fixture-run",
        checkpoint_phase="fixture",
        checkpoint_step=0,
        checkpoint_path=Path("fixture.pt"),
        checkpoint_sha256="a" * 64,
        checkpoint_manifest_path=Path("fixture-checkpoints.json"),
        checkpoint_manifest_sha256="b" * 64,
        training_manifest_path=Path("fixture-training.json"),
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


def test_checkpoint_search_uses_real_stage8_evaluator() -> None:
    context = _checkpoint_context(example_count=7)
    state_before = canonical_state_hash(
        context.model.state_dict()
    )
    hooks_before = _hook_counts(context.model)

    execution = run_checkpoint_sparse_search(
        context,
        fidelity_threshold=1.0e-12,
        ranking_batch_size=3,
        evaluation_batch_size=4,
        exact_evaluation_budget=16,
    )

    result = execution.result

    assert result.status == "budget_exhaustion"
    assert result.exact_evaluations_used == 16
    assert result.exact_evaluations_used <= (
        result.exact_evaluation_budget
    )
    assert result.ranking_passes_used == 1
    assert result.budget_remaining == 0
    assert not result.locally_single_deletion_minimal

    assert result.initial_mask == ComponentMask.all_retained()
    assert result.final_metrics.evaluated_example_count == 7
    assert all(
        record.metrics.evaluated_example_count == 7
        for record in result.candidate_evaluations
    )
    assert all(
        record.metrics.evaluation_batch_size == 4
        for record in result.candidate_evaluations
    )

    assert execution.pseudo_target_count == 7
    assert len(execution.pseudo_target_sha256) == 64
    assert execution.ranking_batch_size == 3
    assert execution.evaluation_batch_size == 4

    assert (
        execution.model_state_sha256_before
        == execution.model_state_sha256_after
        == state_before
    )
    assert dict(execution.hook_counts_before) == hooks_before
    assert dict(execution.hook_counts_after) == hooks_before
    assert canonical_state_hash(
        context.model.state_dict()
    ) == state_before
    assert _hook_counts(context.model) == hooks_before
    assert all(
        parameter.grad is None
        for parameter in context.model.parameters()
    )


def test_checkpoint_search_is_deterministic_on_real_fixture() -> None:
    first_context = _checkpoint_context(example_count=7)
    second_context = _checkpoint_context(example_count=7)

    arguments = {
        "fidelity_threshold": 1.0e-12,
        "ranking_batch_size": 7,
        "evaluation_batch_size": 7,
        "exact_evaluation_budget": 16,
    }

    first = run_checkpoint_sparse_search(
        first_context,
        **arguments,
    )
    second = run_checkpoint_sparse_search(
        second_context,
        **arguments,
    )

    assert first == second
    assert (
        first.pseudo_target_sha256
        == second.pseudo_target_sha256
    )


def test_checkpoint_search_rejects_context_state_mismatch() -> None:
    context = _checkpoint_context(example_count=3)
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
        run_checkpoint_sparse_search(
            invalid,
            fidelity_threshold=0.90,
            ranking_batch_size=3,
            evaluation_batch_size=3,
            exact_evaluation_budget=16,
        )



def _two_removal_fixture_result():
    fidelity_by_ablated = {
        frozenset({"H2"}): 0.95,
        frozenset({"H2", "H1"}): 0.94,
    }

    def evaluator(mask: ComponentMask) -> MaskEvaluationMetrics:
        return _exact_metrics(
            mask,
            fidelity_by_ablated.get(
                frozenset(mask.ablated_component_ids),
                0.0,
            ),
        )

    return greedy_sparse_search(
        ranking_function=_synthetic_ranking_result,
        exact_evaluation_function=evaluator,
        initial_metrics=_exact_metrics(
            ComponentMask.all_retained(),
            1.0,
        ),
        fidelity_threshold=0.90,
        exact_evaluation_budget=1_000,
    )


def test_search_artifacts_are_complete_and_self_consistent(
    tmp_path: Path,
) -> None:
    result = _two_removal_fixture_result()

    artifacts = write_sparse_search_artifacts(
        tmp_path / "cell",
        result,
        cell_metadata={
            "checkpoint_step": 9050,
            "fidelity_threshold": 0.90,
            "fixture": True,
        },
    )

    assert artifacts.final_mask_path.is_file()
    assert artifacts.accepted_removal_trajectory_path.is_file()
    assert artifacts.candidate_evaluation_log_path.is_file()
    assert artifacts.cell_summary_path.is_file()
    assert artifacts.hashes_path.is_file()

    assert len(artifacts.accepted_mask_paths) == 2
    assert len(artifacts.accepted_mask_sha256s) == 2

    assert load_component_mask(
        artifacts.final_mask_path
    ) == result.final_mask

    for removal, mask_path in zip(
        result.accepted_removals,
        artifacts.accepted_mask_paths,
        strict=True,
    ):
        assert load_component_mask(mask_path) == (
            removal.accepted_mask
        )

    trajectory_lines = (
        artifacts.accepted_removal_trajectory_path
        .read_text(encoding="utf-8")
        .splitlines()
    )
    candidate_lines = (
        artifacts.candidate_evaluation_log_path
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert len(trajectory_lines) == 2
    assert len(candidate_lines) == (
        result.exact_evaluations_used
    )

    trajectory = [
        json.loads(line)
        for line in trajectory_lines
    ]

    assert [
        record["removed_component"]
        for record in trajectory
    ] == ["H2", "H1"]

    assert [
        record["retained_count_after"]
        for record in trajectory
    ] == [515, 514]

    for record, mask_path, mask_sha256 in zip(
        trajectory,
        artifacts.accepted_mask_paths,
        artifacts.accepted_mask_sha256s,
        strict=True,
    ):
        assert record["accepted_mask_path"] == (
            f"accepted_masks/{mask_path.name}"
        )
        assert record["accepted_mask_sha256"] == mask_sha256

    summary = json.loads(
        artifacts.cell_summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert summary["search"]["status"] == (
        result.status
    )
    assert summary["search"]["accepted_removal_count"] == 2
    assert summary["search"]["exact_evaluations_used"] == (
        result.exact_evaluations_used
    )
    assert summary["final_mask"]["sha256"] == (
        artifacts.final_mask_sha256
    )
    assert not summary["runtime_telemetry"][
        "included_in_deterministic_artifacts"
    ]


def test_search_artifact_bytes_and_hashes_reproduce(
    tmp_path: Path,
) -> None:
    result = _two_removal_fixture_result()
    metadata = {
        "checkpoint_step": 9050,
        "fidelity_threshold": 0.90,
        "fixture": True,
    }

    first = write_sparse_search_artifacts(
        tmp_path / "first",
        result,
        cell_metadata=metadata,
    )
    second = write_sparse_search_artifacts(
        tmp_path / "second",
        result,
        cell_metadata=metadata,
    )

    pairs = (
        (first.final_mask_path, second.final_mask_path),
        (
            first.accepted_removal_trajectory_path,
            second.accepted_removal_trajectory_path,
        ),
        (
            first.candidate_evaluation_log_path,
            second.candidate_evaluation_log_path,
        ),
        (first.cell_summary_path, second.cell_summary_path),
        (first.hashes_path, second.hashes_path),
    )

    for first_path, second_path in pairs:
        assert first_path.read_bytes() == second_path.read_bytes()

    assert first.final_mask_sha256 == second.final_mask_sha256
    assert (
        first.accepted_removal_trajectory_sha256
        == second.accepted_removal_trajectory_sha256
    )
    assert (
        first.candidate_evaluation_log_sha256
        == second.candidate_evaluation_log_sha256
    )
    assert first.cell_summary_sha256 == second.cell_summary_sha256
    assert first.hashes_sha256 == second.hashes_sha256
    assert (
        first.accepted_mask_sha256s
        == second.accepted_mask_sha256s
    )


def test_search_artifact_writer_rejects_nonempty_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cell"
    output.mkdir()
    (output / "stale.txt").write_text(
        "stale",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="must be empty",
    ):
        write_sparse_search_artifacts(
            output,
            _two_removal_fixture_result(),
            cell_metadata={"fixture": True},
        )



def test_cached_full_model_reference_matches_live_stage8_metrics(
) -> None:
    model = _model()
    inputs, targets = _examples(count=7)
    mask = ComponentMask.one_head_ablated("H0")

    reference = compute_full_model_reference(
        model,
        inputs,
        targets,
        batch_size=3,
    )
    live = evaluate_component_mask(
        model,
        inputs,
        targets,
        mask,
        batch_size=4,
    )
    cached = evaluate_component_mask(
        model,
        inputs,
        targets,
        mask,
        batch_size=4,
        full_model_reference=reference,
    )

    assert cached == live


def test_checkpoint_search_reuses_one_full_model_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _checkpoint_context(example_count=7)
    original = sparse_search_module.evaluate_component_mask
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
        sparse_search_module,
        "evaluate_component_mask",
        wrapped_evaluator,
    )

    execution = (
        sparse_search_module.run_checkpoint_sparse_search(
            context,
            fidelity_threshold=1.0e-12,
            ranking_batch_size=3,
            evaluation_batch_size=4,
            exact_evaluation_budget=1,
        )
    )

    assert len(reference_ids) == 2
    assert len(set(reference_ids)) == 1
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

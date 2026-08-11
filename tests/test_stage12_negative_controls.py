"""Tests for deterministic Stage 12 negative controls."""

from __future__ import annotations

import csv
from dataclasses import replace
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
    distinctness_impossible_control_result,
    fidelity_impossible_control_result,
    load_stage11_random_mask_controls,
    shuffled_ranking,
    shuffled_ranking_control_result,
    stage11_random_mask_control_result,
    write_negative_control_table,
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
    CandidateEvaluation,
    ComponentRanking,
    RankingResult,
    SparseSearchResult,
)

STAGE11_ARCHIVE = Path(
    "results/archives/"
    "stage11-calibration-s1-c2856467c00f.tar.gz"
)


def metrics(
    mask: ComponentMask,
    *,
    fidelity: float,
) -> MaskEvaluationMetrics:
    return MaskEvaluationMetrics(
        primary_fidelity=fidelity,
        prediction_agreement_count=int(
            round(fidelity * 100)
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


def ranking() -> RankingResult:
    mask = ComponentMask.all_retained()
    components = tuple(
        ComponentRanking(
            component_identifier=identifier,
            component_index=index,
            component_class=(
                component_location(
                    identifier
                ).component_class
            ),
            gate_gradient=float(index),
            estimated_removal_damage=float(index),
            ranking_position=index + 1,
        )
        for index, identifier in enumerate(
            mask.retained_component_ids
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


def remove_component(
    mask: ComponentMask,
    identifier: str,
) -> ComponentMask:
    """Return a canonical mask with one component ablated."""

    heads = list(mask.attention_head_mask)
    neurons = list(mask.mlp_neuron_mask)

    if identifier.startswith("H"):
        heads[int(identifier[1:])] = 0
    elif identifier.startswith("N"):
        neurons[int(identifier[1:])] = 0
    else:
        raise ValueError(
            f"Unknown component identifier: {identifier}"
        )

    return ComponentMask(
        attention_head_mask=tuple(heads),
        mlp_neuron_mask=tuple(neurons),
    )


def degraded_result() -> SparseSearchResult:
    final_mask = ComponentMask(
        attention_head_mask=(1, 1, 1, 1),
        mlp_neuron_mask=(
            (1,) * 142
            + (0,) * 370
        ),
    )
    first_component = (
        final_mask.retained_component_ids[0]
    )
    second_component = (
        final_mask.retained_component_ids[1]
    )

    first_mask = remove_component(
        final_mask,
        first_component,
    )
    second_mask = remove_component(
        final_mask,
        second_component,
    )

    candidates = (
        CandidateEvaluation(
            iteration=371,
            candidate_component=second_component,
            component_index=1,
            component_class="attention_head",
            ranking_score=1.0,
            ranking_position=1,
            candidate_batch_index=0,
            exact_fidelity=0.97,
            passed_threshold=False,
            accepted=False,
            rejection_reason="below_fidelity_threshold",
            cumulative_exact_evaluations=1,
            candidate_mask=second_mask,
            metrics=metrics(
                second_mask,
                fidelity=0.97,
            ),
        ),
        CandidateEvaluation(
            iteration=371,
            candidate_component=first_component,
            component_index=0,
            component_class="attention_head",
            ranking_score=2.0,
            ranking_position=2,
            candidate_batch_index=0,
            exact_fidelity=0.98,
            passed_threshold=False,
            accepted=False,
            rejection_reason="below_fidelity_threshold",
            cumulative_exact_evaluations=2,
            candidate_mask=first_mask,
            metrics=metrics(
                first_mask,
                fidelity=0.98,
            ),
        ),
    )

    return SparseSearchResult(
        status="valid_sparse_circuit",
        fidelity_threshold=0.99,
        exact_evaluation_budget=10_000,
        initial_mask=ComponentMask.all_retained(),
        final_mask=final_mask,
        final_metrics=metrics(
            final_mask,
            fidelity=0.995,
        ),
        accepted_removals=(),
        candidate_evaluations=candidates,
        exact_evaluations_used=2,
        ranking_passes_used=1,
        candidate_batches_tested=1,
        rejected_candidate_count=2,
        budget_remaining=9_998,
        budget_exhausted=False,
        locally_single_deletion_minimal=True,
        meaningfully_sparse=True,
        stopping_reason="local_minimum",
        failure_detail=None,
    )


def test_stage11_primary_random_masks_all_fail() -> None:
    records = load_stage11_random_mask_controls(
        STAGE11_ARCHIVE
    )
    result = stage11_random_mask_control_result(
        records
    )

    assert len(records) == 100
    assert result.control_name == (
        STAGE11_RANDOM_MASK_CONTROL
    )
    assert result.validation_passed
    assert result.qualifying_count == 0
    assert result.retained_component_count == 146
    assert result.fidelity_threshold == 0.99


def test_degraded_c1_uses_lowest_stable_index() -> None:
    result = degraded_c1_control_result(
        degraded_result()
    )

    assert result.control_name == DEGRADED_C1_CONTROL
    assert result.validation_passed
    assert result.selected_component_index == 0
    assert result.primary_fidelity == 0.98
    assert result.retained_component_count == 145


def test_shuffled_ranking_is_deterministic() -> None:
    original = ranking()
    first = shuffled_ranking(
        original,
        integer_seed=12345,
    )
    second = shuffled_ranking(
        original,
        integer_seed=12345,
    )

    assert first == second
    assert first != original

    result = shuffled_ranking_control_result(
        original,
        first,
        integer_seed=12345,
    )

    assert result.control_name == (
        SHUFFLED_RANKING_CONTROL
    )
    assert result.validation_passed
    assert result.seed_integer == 12345
    assert (
        result.bit_generator
        == "numpy.random.PCG64"
    )


def test_impossible_controls_reject() -> None:
    empty = ComponentMask.all_ablated()
    fidelity = fidelity_impossible_control_result(
        mask=empty,
        metrics=metrics(
            empty,
            fidelity=0.01,
        ),
        fidelity_threshold=0.99,
    )

    assert fidelity.control_name == (
        FIDELITY_IMPOSSIBLE_CONTROL
    )
    assert fidelity.validation_passed

    accepted = ComponentMask.all_retained()
    distinctness = (
        distinctness_impossible_control_result(
            accepted_mask=accepted,
            candidate_mask=accepted,
            distinctness_cutoff=Fraction(1, 2),
        )
    )

    assert distinctness.control_name == (
        DISTINCTNESS_IMPOSSIBLE_CONTROL
    )
    assert distinctness.validation_passed
    assert distinctness.jaccard_overlap == 1.0


def test_negative_control_table_is_deterministic(
    tmp_path: Path,
) -> None:
    base = NegativeControlResult(
        control_name="fixture_a",
        control_scope="fixture",
        expected_outcome="rejection",
        observed_outcome="rejection",
        validation_passed=True,
        details={"b": 2, "a": 1},
    )
    other = replace(
        base,
        control_name="fixture_b",
    )

    first = write_negative_control_table(
        tmp_path / "first.csv",
        stage12_run_id="fixture-run",
        results=(other, base),
    )
    second = write_negative_control_table(
        tmp_path / "second.csv",
        stage12_run_id="fixture-run",
        results=(base, other),
    )

    assert (
        first.table_path.read_bytes()
        == second.table_path.read_bytes()
    )
    assert first.table_sha256 == second.table_sha256
    assert first.row_count == second.row_count == 2

    with first.table_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert [
        row["control_name"]
        for row in rows
    ] == ["fixture_a", "fixture_b"]
    assert all(
        row["scientific_family_result"] == "False"
        for row in rows
    )
    assert rows[0]["details_json"] == (
        '{"a":1,"b":2}'
    )

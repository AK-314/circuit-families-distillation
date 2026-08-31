from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from circuit_families.stage12r2.contracts import canonical_sha256
from circuit_families.stage12r3.combinatorial import (
    CombinatorialFloorError,
    CombinatorialFloorProfile,
    CombinatorialPackingCandidate,
    CombinatorialPackingRule,
    SizeTypeMatchingRule,
    derive_combinatorial_seed,
    exact_sampling_distribution,
    packing_statistic_for_masks,
    run_combinatorial_floor,
)


def _profile(
    *,
    ids: tuple[str, ...] = ("a0", "a1", "m0", "m1"),
    types: tuple[str, ...] = (
        "attention",
        "attention",
        "mlp",
        "mlp",
    ),
    rule: SizeTypeMatchingRule | None = None,
    batches: int = 3,
    draws: int = 8,
    root_seed: int = 17,
    stream: str = "test-stream",
) -> CombinatorialFloorProfile:
    basis_hash = canonical_sha256(
        {
            "ids": ids,
            "types": types,
            "fixture": "stage12r3-combinatorial",
        }
    )
    return CombinatorialFloorProfile(
        profile_id="stage12r3-combinatorial-test",
        basis_hash=basis_hash,
        ordered_component_ids=ids,
        component_types=types,
        matching_rule=rule or SizeTypeMatchingRule(retained_sizes=(2,)),
        batch_count=batches,
        draws_per_batch=draws,
        root_seed=root_seed,
        seed_stream_id=stream,
    )


def _policy(
    profile: CombinatorialFloorProfile,
    *,
    cutoff: float = 0.5,
) -> CombinatorialPackingRule:
    return CombinatorialPackingRule(
        component_basis_reference=profile.basis_hash,
        component_basis_size=len(profile.ordered_component_ids),
        max_pairwise_overlap=cutoff,
    )


def _fractions(records):
    return {
        record.retained_components: Fraction(
            record.numerator,
            record.denominator,
        )
        for record in records
    }


def test_exact_small_universe_fixed_size_distribution_is_uniform() -> None:
    profile = _profile()
    probabilities = _fractions(exact_sampling_distribution(profile))

    assert len(probabilities) == 6
    assert set(probabilities.values()) == {Fraction(1, 6)}
    assert sum(probabilities.values()) == 1


def test_exact_type_matched_distribution_is_uniform() -> None:
    profile = _profile(
        rule=SizeTypeMatchingRule(
            retained_sizes=(2,),
            component_type_counts=(("attention", 1), ("mlp", 1)),
        )
    )
    probabilities = _fractions(exact_sampling_distribution(profile))

    assert len(probabilities) == 4
    assert set(probabilities.values()) == {Fraction(1, 4)}


def test_declared_size_distribution_preserves_repeated_size_weight() -> None:
    profile = _profile(
        rule=SizeTypeMatchingRule(retained_sizes=(1, 2, 2))
    )
    probabilities = _fractions(exact_sampling_distribution(profile))

    for mask, probability in probabilities.items():
        if len(mask) == 1:
            assert probability == Fraction(1, 12)
        elif len(mask) == 2:
            assert probability == Fraction(1, 9)
        else:
            raise AssertionError("unexpected mask size")
    assert sum(probabilities.values()) == 1


def test_type_matching_is_preserved_for_every_draw() -> None:
    profile = _profile(
        rule=SizeTypeMatchingRule(
            retained_sizes=(2,),
            component_type_counts=(("attention", 1), ("mlp", 1)),
        ),
        batches=4,
        draws=20,
    )
    result = run_combinatorial_floor(profile, _policy(profile))

    assert all(draw.retained_size == 2 for draw in result.draws)
    assert all(
        draw.retained_type_counts == (("attention", 1), ("mlp", 1))
        for draw in result.draws
    )


def test_seed_and_draw_order_are_deterministic() -> None:
    profile = _profile(batches=2, draws=12)

    first = run_combinatorial_floor(profile, _policy(profile))
    second = run_combinatorial_floor(profile, _policy(profile))

    assert first.identity == second.identity
    assert first.draws == second.draws
    assert first.batches == second.batches


def test_complete_identity_changes_seed_stream() -> None:
    profile = _profile()
    changed = replace(profile, seed_stream_id="different-stream")

    assert derive_combinatorial_seed(
        profile, batch_index=0, draw_index=0
    ) != derive_combinatorial_seed(
        changed, batch_index=0, draw_index=0
    )


def test_impossible_type_composition_is_rejected() -> None:
    with pytest.raises(
        CombinatorialFloorError,
        match="impossible component-type composition",
    ):
        _profile(
            rule=SizeTypeMatchingRule(
                retained_sizes=(3,),
                component_type_counts=(("attention", 3),),
            )
        )


def test_unknown_component_type_is_rejected() -> None:
    with pytest.raises(
        CombinatorialFloorError,
        match="absent",
    ):
        _profile(
            rule=SizeTypeMatchingRule(
                retained_sizes=(1,),
                component_type_counts=(("residual", 1),),
            )
        )


def test_type_count_total_must_match_declared_size() -> None:
    with pytest.raises(CombinatorialFloorError):
        SizeTypeMatchingRule(
            retained_sizes=(2,),
            component_type_counts=(("attention", 1),),
        )


def test_duplicate_draws_are_preserved_but_packed_uniquely() -> None:
    profile = _profile(
        ids=("only",),
        types=("mlp",),
        rule=SizeTypeMatchingRule(retained_sizes=(1,)),
        batches=1,
        draws=5,
    )
    result = run_combinatorial_floor(profile, _policy(profile))

    assert result.raw_draw_count == 5
    assert len(result.draws) == 5
    assert result.batches[0].unique_mask_count == 1
    assert result.batches[0].duplicate_draw_count == 4
    assert result.duplicate_draw_count == 4
    assert result.batches[0].packing_statistic == 1
    assert result.draws[0].duplicate_of_draw_index is None
    assert all(
        draw.duplicate_of_draw_index == 0 for draw in result.draws[1:]
    )


def test_basis_mismatch_is_rejected() -> None:
    profile = _profile()
    bad_policy = replace(
        _policy(profile),
        component_basis_reference="different-basis",
    )

    with pytest.raises(CombinatorialFloorError, match="basis identity"):
        run_combinatorial_floor(profile, bad_policy)


def test_basis_size_mismatch_is_rejected() -> None:
    profile = _profile()
    bad_policy = replace(
        _policy(profile),
        component_basis_size=99,
    )

    with pytest.raises(CombinatorialFloorError, match="basis size"):
        run_combinatorial_floor(profile, bad_policy)


def test_overlap_extremes_reuse_stage6e_graph_and_solver() -> None:
    profile = _profile(
        ids=("x0", "x1", "x2"),
        types=("x", "x", "x"),
        rule=SizeTypeMatchingRule(retained_sizes=(2,)),
    )
    masks = ((0, 1), (0, 2))

    low, _, _ = packing_statistic_for_masks(
        profile,
        _policy(profile, cutoff=0.0),
        masks,
    )
    high, _, _ = packing_statistic_for_masks(
        profile,
        _policy(profile, cutoff=1.0),
        masks,
    )

    assert low == 1
    assert high == 2


def test_empty_batch_has_zero_packing() -> None:
    profile = _profile(batches=2, draws=0)
    result = run_combinatorial_floor(profile, _policy(profile))

    assert [batch.packing_statistic for batch in result.batches] == [0, 0]
    assert result.raw_draw_count == 0
    assert result.unique_mask_draw_outcomes == 0
    assert result.packing_distribution[0].packing_statistic == 0
    assert result.packing_distribution[0].batch_count == 2


def test_explicit_empty_mask_collection_has_zero_packing() -> None:
    profile = _profile()
    packing, selected, _ = packing_statistic_for_masks(
        profile,
        _policy(profile),
        (),
    )

    assert packing == 0
    assert selected == ()


def test_invalid_explicit_mask_is_rejected() -> None:
    profile = _profile()

    with pytest.raises(CombinatorialFloorError):
        packing_statistic_for_masks(
            profile,
            _policy(profile),
            ((2, 1),),
        )

    with pytest.raises(CombinatorialFloorError):
        packing_statistic_for_masks(
            profile,
            _policy(profile),
            ((0, 99),),
        )


def test_combinatorial_floor_makes_no_fidelity_or_endpoint2_claim() -> None:
    profile = _profile()
    result = run_combinatorial_floor(profile, _policy(profile))

    assert result.fidelity_claim is False
    assert result.exact_evaluation_count == 0
    assert result.endpoint2_claim is False
    assert result.packing_semantics == (
        "combinatorial_overlap_packing_statistic_not_endpoint2"
    )


def test_profile_and_result_reject_scientific_or_production_state() -> None:
    profile = _profile()

    with pytest.raises(CombinatorialFloorError):
        replace(profile, scientific_data=True)

    with pytest.raises(CombinatorialFloorError):
        replace(profile, production_eligible=True)

    result = run_combinatorial_floor(profile, _policy(profile))
    with pytest.raises(CombinatorialFloorError):
        replace(result, fidelity_claim=True)


def test_sampling_rule_does_not_contain_model_or_fidelity_inputs() -> None:
    fields = set(SizeTypeMatchingRule.__dataclass_fields__)
    forbidden = {
        "fidelity",
        "model_output",
        "discovery_score",
        "packing_member",
        "effect_direction",
    }
    assert fields.isdisjoint(forbidden)


def test_structural_candidate_has_no_fidelity_or_ledger_semantics() -> None:
    fields = set(CombinatorialPackingCandidate.__dataclass_fields__)

    assert "exact_fidelity" not in fields
    assert "fidelity" not in fields
    assert "qualifies" not in fields
    assert "exact_evaluation_references" not in fields
    assert "source_ledger_references" not in fields


def test_structural_policy_is_not_endpoint2_policy() -> None:
    profile = _profile()
    policy = _policy(profile)

    assert policy.endpoint2_policy is False
    assert policy.scientific_data is False
    assert policy.production_eligible is False

    with pytest.raises(CombinatorialFloorError):
        replace(policy, endpoint2_policy=True)


def test_stage6e_graph_and_exact_solver_accept_structural_adapters() -> None:
    profile = _profile(
        ids=("x0", "x1", "x2"),
        types=("x", "x", "x"),
        rule=SizeTypeMatchingRule(retained_sizes=(1,)),
    )

    packing, selected, graph_hash = packing_statistic_for_masks(
        profile,
        _policy(profile, cutoff=0.0),
        ((0,), (1,), (2,)),
    )

    assert packing == 3
    assert len(selected) == 3
    assert len(graph_hash) == 64

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from circuit_families.stage6a import COMPONENT_COUNT, ExactLedgerBuilder
from circuit_families.stage6a.models import (
    SealedLedger,
    TechnicalLedgerProfile,
)
from circuit_families.stage6e.records import load_technical_policy
from circuit_families.stage12r3.local_exact import (
    LocalExactError,
    LocalExactProfile,
    run_local_exact_perturbations,
    validate_seed_masks,
)


def _policy():
    return load_technical_policy(
        Path(__file__).resolve().parents[1]
        / "followup/configs/stage6e/technical_endpoint2_policy_v1.json"
    )


def _mask(*indices: int) -> tuple[int, ...]:
    values = [0] * COMPONENT_COUNT
    for index in indices:
        values[index] = 1
    return tuple(values)


def _types() -> tuple[str, ...]:
    return tuple(
        "attention" if index < 4 else "mlp"
        for index in range(COMPONENT_COUNT)
    )


def _profile(
    *,
    exact: int = 100,
    operations: tuple[str, ...] = ("add", "drop"),
    radius: int = 1,
) -> LocalExactProfile:
    policy = _policy()
    return LocalExactProfile(
        profile_id="stage12r3-local-test",
        run_id="stage12r3-local-test-run",
        model_id="stage12r3-synthetic-technical-model",
        discovery_method_id="stage12r3-local-perturbation-v1",
        discovery_config_id="stage12r3-local-config-v1",
        component_basis_reference=policy.component_basis_reference,
        component_types=_types(),
        fidelity_threshold=policy.fidelity_threshold,
        exact_evaluation_allowance=exact,
        enabled_operations=operations,
        max_hamming_distance=radius,
    )


def _seed_ledger(
    seeds: tuple[tuple[int, ...], ...],
    *,
    seed_fidelity: float = 1.0,
) -> SealedLedger:
    builder = ExactLedgerBuilder(
        evaluator=lambda mask: (
            seed_fidelity if mask in seeds else 1.0
        ),
        fidelity_threshold=_policy().fidelity_threshold,
    )
    builder.add_mask((1,) * COMPONENT_COUNT, proposal_index=0)
    for index, seed in enumerate(seeds, start=1):
        builder.add_mask(seed, proposal_index=index)
    entries = builder.seal()

    return SealedLedger(
        profile=TechnicalLedgerProfile(
            profile_version="stage12r3-seed-test/v1",
            name="stage12r3-seed-test",
            synthetic_only=True,
            scientific_data=False,
            production_eligible=False,
            unresolved_decisions=("RD-006", "RD-008", "RD-009"),
        ),
        evaluations=entries,
        proposals=tuple(builder.proposals),
        has_intact_baseline=True,
        sealed=True,
    )


def test_seed_must_be_present_and_exact_qualified_in_sealed_ledger() -> None:
    seed = _mask(0)
    ledger = _seed_ledger((seed,))

    validated = validate_seed_masks(
        seed_ledger=ledger,
        seed_masks=(seed,),
    )
    assert len(validated) == 1
    assert validated[0].source_exact_fidelity == 1.0

    with pytest.raises(LocalExactError, match="absent"):
        validate_seed_masks(
            seed_ledger=ledger,
            seed_masks=(_mask(1),),
        )


def test_unsealed_or_non_intact_seed_ledger_is_rejected() -> None:
    seed = _mask(0)
    ledger = _seed_ledger((seed,))

    with pytest.raises(LocalExactError, match="sealed"):
        validate_seed_masks(
            seed_ledger=replace(ledger, sealed=False),
            seed_masks=(seed,),
        )

    with pytest.raises(LocalExactError, match="intact"):
        validate_seed_masks(
            seed_ledger=replace(ledger, has_intact_baseline=False),
            seed_masks=(seed,),
        )


def test_nonqualifying_source_seed_is_rejected() -> None:
    seed = _mask(0)
    ledger = _seed_ledger((seed,), seed_fidelity=0.0)

    with pytest.raises(LocalExactError, match="not exact-qualified"):
        validate_seed_masks(
            seed_ledger=ledger,
            seed_masks=(seed,),
        )


def test_seed_requires_source_proposal_provenance() -> None:
    seed = _mask(0)
    ledger = _seed_ledger((seed,))
    seed_identity = validate_seed_masks(
        seed_ledger=ledger,
        seed_masks=(seed,),
    )[0].mask_identity

    stripped = replace(
        ledger,
        proposals=tuple(
            event
            for event in ledger.proposals
            if event.mask_identity != seed_identity
        ),
    )

    with pytest.raises(LocalExactError, match="proposal provenance"):
        validate_seed_masks(
            seed_ledger=stripped,
            seed_masks=(seed,),
        )


def test_isolated_optimum_keeps_only_fresh_exact_seed() -> None:
    seed = _mask(0)
    ledger = _seed_ledger((seed,))
    profile = _profile(exact=600, operations=("add", "drop"), radius=1)

    def evaluator(mask: tuple[int, ...]) -> float:
        if mask == (1,) * COMPONENT_COUNT:
            return 1.0
        return 1.0 if mask == seed else 0.0

    result = run_local_exact_perturbations(
        profile=profile,
        policy=_policy(),
        seed_ledger=ledger,
        seed_ledger_reference="seed-ledger-isolated",
        seed_masks=(seed,),
        evaluator=evaluator,
    )

    assert result.qualification.qualified_candidate_count == 1
    assert result.validated_seeds[0].source_exact_fidelity == 1.0
    seed_record = next(
        item for item in result.proposals if item.operation == "self"
    )
    assert seed_record.exact_fidelity == 1.0
    assert result.inherited_fidelity_used is False


def test_exact_plateau_recovers_multiple_local_qualifiers() -> None:
    seed = _mask(0)
    neighbor = _mask(0, 1)
    ledger = _seed_ledger((seed,))
    profile = _profile(exact=600, operations=("add",), radius=1)

    def evaluator(mask: tuple[int, ...]) -> float:
        if mask == (1,) * COMPONENT_COUNT:
            return 1.0
        return 1.0 if mask in (seed, neighbor) else 0.0

    result = run_local_exact_perturbations(
        profile=profile,
        policy=_policy(),
        seed_ledger=ledger,
        seed_ledger_reference="seed-ledger-plateau",
        seed_masks=(seed,),
        evaluator=evaluator,
    )

    assert result.qualification.qualified_candidate_count == 2


def test_deceptive_surrogate_cannot_enter_api_or_override_exact_evaluator() -> None:
    assert "surrogate" not in inspect.signature(
        run_local_exact_perturbations
    ).parameters

    seed = _mask(0)
    ledger = _seed_ledger((seed,))
    profile = _profile(exact=600, operations=("add",), radius=1)

    deceptive_surrogate = {
        _mask(0, 1): 1.0,
        seed: 0.0,
    }
    assert deceptive_surrogate[_mask(0, 1)] > deceptive_surrogate[seed]

    def exact_evaluator(mask: tuple[int, ...]) -> float:
        if mask == (1,) * COMPONENT_COUNT:
            return 1.0
        return 1.0 if mask == seed else 0.0

    result = run_local_exact_perturbations(
        profile=profile,
        policy=_policy(),
        seed_ledger=ledger,
        seed_ledger_reference="seed-ledger-surrogate",
        seed_masks=(seed,),
        evaluator=exact_evaluator,
    )

    assert result.qualification.qualified_candidate_count == 1
    assert result.surrogate_fidelity_used is False


def test_type_preserving_swap_never_crosses_component_types() -> None:
    seed = _mask(0, 4)
    ledger = _seed_ledger((seed,))
    profile = _profile(
        exact=1200,
        operations=("type_preserving_swap",),
        radius=2,
    )

    result = run_local_exact_perturbations(
        profile=profile,
        policy=_policy(),
        seed_ledger=ledger,
        seed_ledger_reference="seed-ledger-type-swap",
        seed_masks=(seed,),
        evaluator=lambda mask: 1.0,
    )

    swaps = [
        item
        for item in result.proposals
        if item.operation == "type_preserving_swap"
    ]
    assert swaps

    for proposal in swaps:
        left, right = proposal.changed_indices
        assert profile.component_types[left] == profile.component_types[right]
        assert proposal.hamming_distance == 2


def test_general_swap_can_cross_types_but_stays_radius_two() -> None:
    seed = _mask(0)
    ledger = _seed_ledger((seed,))
    profile = _profile(
        exact=1200,
        operations=("swap",),
        radius=2,
    )

    result = run_local_exact_perturbations(
        profile=profile,
        policy=_policy(),
        seed_ledger=ledger,
        seed_ledger_reference="seed-ledger-swap",
        seed_masks=(seed,),
        evaluator=lambda mask: 1.0,
    )

    swaps = [item for item in result.proposals if item.operation == "swap"]
    assert swaps
    assert all(item.hamming_distance == 2 for item in swaps)
    assert any(
        profile.component_types[item.changed_indices[0]]
        != profile.component_types[item.changed_indices[1]]
        for item in swaps
    )


def test_failed_exact_neighbor_is_preserved_and_not_charged() -> None:
    seed = _mask(0)
    failing = _mask(0, 1)
    ledger = _seed_ledger((seed,))
    profile = _profile(exact=600, operations=("add",), radius=1)

    def evaluator(mask: tuple[int, ...]) -> float:
        if mask == failing:
            raise RuntimeError("constructed local evaluator failure")
        return 1.0

    result = run_local_exact_perturbations(
        profile=profile,
        policy=_policy(),
        seed_ledger=ledger,
        seed_ledger_reference="seed-ledger-failure",
        seed_masks=(seed,),
        evaluator=evaluator,
    )

    failed = [
        item
        for item in result.proposals
        if item.exact_status == "evaluation_failed"
    ]
    assert len(failed) == 1
    assert result.exact_request_failure_count == 1
    assert result.procedure_censored is True
    assert (
        result.exact_budget_charged
        == result.exact_ledger_evaluation_count
    )


def test_exact_budget_censoring_preserves_unexamined_neighbors() -> None:
    seed = _mask(0)
    ledger = _seed_ledger((seed,))
    profile = _profile(exact=2, operations=("add",), radius=1)

    result = run_local_exact_perturbations(
        profile=profile,
        policy=_policy(),
        seed_ledger=ledger,
        seed_ledger_reference="seed-ledger-budget",
        seed_masks=(seed,),
        evaluator=lambda mask: 1.0,
    )

    assert result.exact_budget_charged == 2  # intact + freshly evaluated seed
    assert result.exact_request_censored_count > 0
    assert result.procedure_censored is True


def test_duplicate_neighbors_preserve_parent_provenance_but_reuse_exact_eval() -> None:
    seed_a = _mask(0)
    seed_b = _mask(1)
    ledger = _seed_ledger((seed_a, seed_b))
    profile = _profile(exact=700, operations=("add",), radius=1)

    result = run_local_exact_perturbations(
        profile=profile,
        policy=_policy(),
        seed_ledger=ledger,
        seed_ledger_reference="seed-ledger-duplicates",
        seed_masks=(seed_a, seed_b),
        evaluator=lambda mask: 1.0,
    )

    shared_records = [
        item for item in result.proposals if item.mask_identity == next(
            candidate.mask_identity
            for candidate in result.qualification.qualified_candidates
            if candidate.retained_components == (0, 1)
        )
    ]
    assert len(shared_records) == 2
    assert {item.parent_mask_identity for item in shared_records} == {
        result.validated_seeds[0].mask_identity,
        result.validated_seeds[1].mask_identity,
    }
    assert {
        item.exact_status for item in shared_records
    } == {"evaluated", "duplicate_evaluation_reused"}


def test_profile_rejects_independent_discovery_or_scientific_claims() -> None:
    profile = _profile()

    with pytest.raises(LocalExactError):
        replace(profile, independent_discovery_claim=True)
    with pytest.raises(LocalExactError):
        replace(profile, scientific_data=True)
    with pytest.raises(LocalExactError):
        replace(profile, production_eligible=True)

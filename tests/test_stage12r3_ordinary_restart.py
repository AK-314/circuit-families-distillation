from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from circuit_families.stage6a import COMPONENT_COUNT
from circuit_families.stage6e.records import load_technical_policy
from circuit_families.stage12r1 import ALGORITHM_FAMILY
from circuit_families.stage12r3.ordinary_restart import (
    OrdinaryRestartContext,
    OrdinaryRestartError,
    OrdinaryRestartProfile,
    RestartDiscoveryOutput,
    derive_restart_seed,
    run_ordinary_restart_baseline,
)


def _mask(*indices: int) -> tuple[int, ...]:
    values = [0] * COMPONENT_COUNT
    for index in indices:
        values[index] = 1
    return tuple(values)


def _policy():
    return load_technical_policy(
        Path(__file__).resolve().parents[1]
        / "followup/configs/stage6e/technical_endpoint2_policy_v1.json"
    )


def _profile(
    *,
    restarts: int = 3,
    native: int = 10,
    exact: int = 10,
    root_seed: int = 17,
) -> OrdinaryRestartProfile:
    policy = _policy()
    return OrdinaryRestartProfile(
        profile_id="stage12r3-ordinary-test",
        run_id="ordinary-test-run",
        method_name="technical-ordinary-discovery",
        method_version="v1",
        discovery_config_id="technical-config-v1",
        model_id="stage12r3-synthetic-technical-model",
        component_basis_reference=policy.component_basis_reference,
        fidelity_threshold=policy.fidelity_threshold,
        restart_count=restarts,
        root_seed=root_seed,
        native_budget_per_restart=native,
        exact_evaluation_allowance=exact,
    )


def _evaluator(mask: tuple[int, ...]) -> float:
    return 1.0 if sum(mask) <= COMPONENT_COUNT else 0.0


def test_repeated_masks_deduplicate_only_at_common_exact_boundary() -> None:
    profile = _profile(restarts=3, exact=5)
    same = _mask(0)

    def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
        return RestartDiscoveryOutput(
            proposals=(same,),
            native_work_consumed=3,
        )

    result = run_ordinary_restart_baseline(
        profile=profile,
        policy=_policy(),
        evaluator=_evaluator,
        discovery_procedure=procedure,
    )

    assert result.raw_restart_proposal_count == 3
    assert result.exact_ledger_evaluation_count == 2  # intact + one unique
    assert result.exact_budget.charged == 2
    assert result.qualification.unique_candidate_count == 1
    assert result.qualification.qualified_candidate_count == 1

    statuses = [
        record.proposals[0].exact_status
        for record in result.restart_records
    ]
    assert statuses[0] == "evaluated"
    assert statuses[1:] == [
        "duplicate_evaluation_reused",
        "duplicate_evaluation_reused",
    ]


def test_disjoint_restart_masks_can_increase_procedure_relative_packing() -> None:
    profile = _profile(restarts=3, exact=5)
    masks = (_mask(0), _mask(1), _mask(2))

    def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
        return RestartDiscoveryOutput(
            proposals=(masks[context.restart_index],),
            native_work_consumed=2,
        )

    result = run_ordinary_restart_baseline(
        profile=profile,
        policy=_policy(),
        evaluator=_evaluator,
        discovery_procedure=procedure,
    )

    assert result.qualification.qualified_candidate_count == 3
    assert result.packing_lower_bound == 3
    assert result.discovery_family == ALGORITHM_FAMILY
    assert result.discovery_relationship == (
        "same_discovery_family_ordinary_restart"
    )
    assert result.mechanism_count_claim is False


def test_restart_seed_uses_complete_restart_identity() -> None:
    profile = _profile(root_seed=17)
    changed = replace(profile, root_seed=18)

    assert derive_restart_seed(
        profile,
        restart_index=0,
    ) != derive_restart_seed(
        changed,
        restart_index=0,
    )
    assert derive_restart_seed(
        profile,
        restart_index=0,
    ) != derive_restart_seed(
        profile,
        restart_index=1,
    )


def test_context_exposes_no_prior_restart_or_packing_state() -> None:
    fields = set(OrdinaryRestartContext.__dataclass_fields__)

    assert "prior_masks" not in fields
    assert "prior_restart_results" not in fields
    assert "packing_members" not in fields
    assert "packing_feedback" not in fields


@pytest.mark.parametrize(
    "field",
    [
        "uses_diversity_pressure",
        "uses_packing_feedback",
        "uses_prior_restart_mask_exclusion",
    ],
)
def test_profile_rejects_cross_restart_proposal_modification(field: str) -> None:
    profile = _profile()

    with pytest.raises(OrdinaryRestartError):
        replace(profile, **{field: True})


def test_output_rejects_declared_cross_restart_information_use() -> None:
    with pytest.raises(OrdinaryRestartError):
        RestartDiscoveryOutput(
            proposals=(),
            native_work_consumed=0,
            used_cross_restart_information=True,
        )


def test_exact_budget_exhaustion_preserves_censored_proposals() -> None:
    profile = _profile(restarts=3, exact=2)
    masks = (_mask(0), _mask(1), _mask(2))

    def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
        return RestartDiscoveryOutput(
            proposals=(masks[context.restart_index],),
            native_work_consumed=1,
        )

    result = run_ordinary_restart_baseline(
        profile=profile,
        policy=_policy(),
        evaluator=_evaluator,
        discovery_procedure=procedure,
    )

    assert result.exact_budget.charged == 2  # intact + first unique proposal
    assert result.exact_budget.exhausted is True
    assert result.exact_request_censored_count == 2
    assert result.procedure_censored is True

    censored = [
        proposal
        for restart in result.restart_records
        for proposal in restart.proposals
        if proposal.exact_status == "exact_budget_censored"
    ]
    assert len(censored) == 2


def test_duplicate_does_not_consume_extra_exact_budget() -> None:
    profile = _profile(restarts=4, exact=2)
    same = _mask(0)

    def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
        return RestartDiscoveryOutput(
            proposals=(same,),
            native_work_consumed=1,
        )

    result = run_ordinary_restart_baseline(
        profile=profile,
        policy=_policy(),
        evaluator=_evaluator,
        discovery_procedure=procedure,
    )

    assert result.exact_budget.charged == 2
    assert result.exact_request_censored_count == 0
    assert result.raw_restart_proposal_count == 4


def test_procedure_exception_is_retained_and_other_restarts_continue() -> None:
    profile = _profile(restarts=3)

    def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
        if context.restart_index == 1:
            raise RuntimeError("constructed failure")
        return RestartDiscoveryOutput(
            proposals=(_mask(context.restart_index),),
            native_work_consumed=2,
        )

    result = run_ordinary_restart_baseline(
        profile=profile,
        policy=_policy(),
        evaluator=_evaluator,
        discovery_procedure=procedure,
    )

    assert result.failed_restart_count == 1
    assert result.completed_restart_count == 2
    assert result.procedure_censored is True
    assert result.restart_records[1].failure_kind == "procedure_exception"
    assert "constructed failure" in (
        result.restart_records[1].failure_reason or ""
    )


def test_native_budget_exhaustion_is_recorded_without_new_method_semantics() -> None:
    profile = _profile(restarts=2, native=4)

    def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
        return RestartDiscoveryOutput(
            proposals=(_mask(context.restart_index),),
            native_work_consumed=4,
            terminal_state="budget_exhausted",
        )

    result = run_ordinary_restart_baseline(
        profile=profile,
        policy=_policy(),
        evaluator=_evaluator,
        discovery_procedure=procedure,
    )

    assert result.native_exhausted_restart_count == 2
    assert all(
        record.native_budget.exhausted
        for record in result.restart_records
    )
    assert result.discovery_family == ALGORITHM_FAMILY


def test_native_budget_overrun_is_rejected() -> None:
    profile = _profile(native=3)

    def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
        return RestartDiscoveryOutput(
            proposals=(),
            native_work_consumed=4,
        )

    with pytest.raises(OrdinaryRestartError, match="native budget"):
        run_ordinary_restart_baseline(
            profile=profile,
            policy=_policy(),
            evaluator=_evaluator,
            discovery_procedure=procedure,
        )


def test_failed_exact_evaluation_remains_recorded() -> None:
    profile = _profile(restarts=2, exact=5)
    failing = _mask(0)
    succeeding = _mask(1)

    def evaluator(mask: tuple[int, ...]) -> float:
        if mask == failing:
            raise RuntimeError("constructed evaluator failure")
        return 1.0

    def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
        return RestartDiscoveryOutput(
            proposals=(
                failing if context.restart_index == 0 else succeeding,
            ),
            native_work_consumed=1,
        )

    result = run_ordinary_restart_baseline(
        profile=profile,
        policy=_policy(),
        evaluator=evaluator,
        discovery_procedure=procedure,
    )

    assert result.exact_request_failure_count == 1
    assert (
        result.restart_records[0].proposals[0].exact_status
        == "evaluation_failed"
    )
    assert result.qualification.unique_candidate_count == 1
    assert result.exact_budget.charged == result.exact_ledger_evaluation_count
    assert result.procedure_censored is True


def test_execution_order_does_not_change_recovered_set_or_packing() -> None:
    profile = _profile(restarts=3, exact=6)
    masks = (_mask(0), _mask(1), _mask(2))

    def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
        return RestartDiscoveryOutput(
            proposals=(masks[context.restart_index],),
            native_work_consumed=2,
        )

    forward = run_ordinary_restart_baseline(
        profile=profile,
        policy=_policy(),
        evaluator=_evaluator,
        discovery_procedure=procedure,
        execution_order=(0, 1, 2),
    )
    reverse = run_ordinary_restart_baseline(
        profile=profile,
        policy=_policy(),
        evaluator=_evaluator,
        discovery_procedure=procedure,
        execution_order=(2, 1, 0),
    )

    assert set(forward.qualified_mask_identities) == set(
        reverse.qualified_mask_identities
    )
    assert forward.packing_lower_bound == reverse.packing_lower_bound
    assert (
        forward.endpoint1.retained_proportion
        == reverse.endpoint1.retained_proportion
    )


def test_only_exact_stage6a_qualifiers_feed_stage6e() -> None:
    profile = _profile(restarts=2, exact=4)
    good = _mask(0)
    bad = _mask(1)

    def evaluator(mask: tuple[int, ...]) -> float:
        if mask == bad:
            return 0.0
        return 1.0

    def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
        return RestartDiscoveryOutput(
            proposals=(good if context.restart_index == 0 else bad,),
            native_work_consumed=1,
        )

    result = run_ordinary_restart_baseline(
        profile=profile,
        policy=_policy(),
        evaluator=evaluator,
        discovery_procedure=procedure,
    )

    assert result.qualification.unique_candidate_count == 2
    assert result.qualification.qualified_candidate_count == 1
    assert result.qualified_mask_identities == (
        result.restart_records[0].proposals[0].mask_identity,
    )


def test_intact_baseline_preserves_endpoint1_when_discovery_recovers_nothing() -> None:
    profile = _profile(restarts=2, exact=1)

    def procedure(context: OrdinaryRestartContext) -> RestartDiscoveryOutput:
        return RestartDiscoveryOutput(
            proposals=(_mask(context.restart_index),),
            native_work_consumed=1,
        )

    result = run_ordinary_restart_baseline(
        profile=profile,
        policy=_policy(),
        evaluator=_evaluator,
        discovery_procedure=procedure,
    )

    assert result.endpoint1.retained_proportion == 1.0
    assert result.exact_request_censored_count == 2
    assert result.packing_lower_bound == 0


def test_profile_is_explicitly_technical_only() -> None:
    profile = _profile()

    with pytest.raises(OrdinaryRestartError):
        replace(profile, scientific_data=True)

    with pytest.raises(OrdinaryRestartError):
        replace(profile, production_eligible=True)

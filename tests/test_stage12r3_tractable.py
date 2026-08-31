from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from circuit_families.stage6a import COMPONENT_COUNT
from circuit_families.stage6e.records import load_technical_policy
from circuit_families.stage12r3.contracts import (
    CompletenessCertificate,
    Stage12R3ContractError,
)
from circuit_families.stage12r3.tractable import (
    TractableCalibrationError,
    TractableCalibrationResult,
    TractableFixtureProfile,
    TractableSearchContext,
    TractableSearchOutput,
    run_tractable_calibration,
)


def _policy():
    return load_technical_policy(
        Path(__file__).resolve().parents[1]
        / "followup/configs/stage6e/technical_endpoint2_policy_v1.json"
    )


def _profile(
    *,
    free: tuple[int, ...] = (0, 1, 2),
    feasible: tuple[int, ...] = (0, 1, 2),
    exact: int = 20,
) -> TractableFixtureProfile:
    policy = _policy()
    return TractableFixtureProfile(
        profile_id="stage12r3-tractable-test",
        run_id="stage12r3-tractable-test-run",
        model_id="stage12r3-tractable-synthetic-model",
        discovery_method_id="stage12r3-tractable-search-v1",
        discovery_config_id="stage12r3-tractable-config-v1",
        component_basis_reference=policy.component_basis_reference,
        free_component_indices=free,
        qualifying_free_patterns=feasible,
        fidelity_threshold=policy.fidelity_threshold,
        search_exact_evaluation_allowance=exact,
    )


def _mask(profile: TractableFixtureProfile, pattern: int) -> tuple[int, ...]:
    values = [0] * COMPONENT_COUNT
    for bit_index, component_index in enumerate(profile.free_component_indices):
        if pattern & (1 << bit_index):
            values[component_index] = 1
    return tuple(values)


def _run(
    profile: TractableFixtureProfile,
    patterns: tuple[int, ...],
) -> TractableCalibrationResult:
    def search(context: TractableSearchContext) -> TractableSearchOutput:
        assert context.qualifying_patterns_exposed is False
        return TractableSearchOutput(
            proposals=tuple(_mask(profile, pattern) for pattern in patterns)
        )

    return run_tractable_calibration(
        profile=profile,
        policy=_policy(),
        search_procedure=search,
    )


def test_exhaustive_reference_enumerates_complete_declared_universe() -> None:
    profile = _profile()
    result = _run(profile, ())

    assert result.admissible_mask_count == 8
    assert len(result.admissible_mask_identities) == 8
    assert result.exhaustive_exact_evaluation_count == 9  # intact + 8
    assert result.certificate.exactness_claim == "exact"
    assert result.certificate.exhaustive is True
    assert result.certificate.lower_bound == result.feasible_mask_count
    assert result.certificate.upper_bound == result.feasible_mask_count


def test_exact_feasible_inventory_matches_declared_qualifying_patterns() -> None:
    profile = _profile(feasible=(0, 1, 2))
    result = _run(profile, ())

    assert result.feasible_mask_count == 3
    assert result.exhaustive_qualification.qualified_candidate_count == 3
    assert result.exhaustive_endpoint1.retained_proportion == 0.0
    assert result.certified_packing_optimum == 3


def test_complete_search_has_unit_recall_and_zero_gaps() -> None:
    profile = _profile(exact=20)
    result = _run(profile, tuple(range(profile.admissible_mask_count)))

    assert result.feasible_recall == 1.0
    assert result.missed_feasible_count == 0
    assert result.endpoint1_retained_proportion_gap == 0.0
    assert result.packing_gap == 0
    assert result.search_exact_evaluation_coverage == 1.0
    assert result.search_procedure_censored is False


def test_search_can_miss_feasible_masks_despite_feasible_region_existing() -> None:
    profile = _profile(feasible=(0, 1, 2), exact=10)
    result = _run(profile, (1, 2))

    assert result.feasible_mask_count == 3
    assert result.recovered_feasible_count == 2
    assert result.missed_feasible_count == 1
    assert result.feasible_recall == 2 / 3
    assert result.endpoint1_retained_proportion_gap == 1 / COMPONENT_COUNT
    assert result.packing_gap == 1


def test_duplicate_search_outputs_are_preserved_and_exact_deduplicated() -> None:
    profile = _profile(exact=10)
    result = _run(profile, (1, 1, 1))

    assert result.search_raw_proposal_count == 3
    assert result.search_valid_proposal_count == 3
    assert result.search_duplicate_proposal_count == 2
    assert result.search_exact_evaluation_count == 2  # intact + pattern 1
    assert result.search_exact_budget_charged == 2
    assert [
        item.status for item in result.search_proposals
    ] == [
        "evaluated",
        "duplicate_evaluation_reused",
        "duplicate_evaluation_reused",
    ]


def test_invalid_search_outputs_are_retained_without_exact_evaluation() -> None:
    profile = _profile(exact=10)
    bad_length = (0, 1)
    outside = [0] * COMPONENT_COUNT
    outside[100] = 1

    def search(context: TractableSearchContext) -> TractableSearchOutput:
        return TractableSearchOutput(
            proposals=(
                bad_length,
                tuple(outside),
                _mask(profile, 1),
            )
        )

    result = run_tractable_calibration(
        profile=profile,
        policy=_policy(),
        search_procedure=search,
    )

    assert result.search_invalid_proposal_count == 2
    assert result.search_valid_proposal_count == 1
    assert result.search_exact_evaluation_count == 2
    assert [item.status for item in result.search_proposals[:2]] == [
        "invalid",
        "invalid",
    ]


def test_exact_budget_censoring_is_measured_against_admissible_universe() -> None:
    profile = _profile(exact=2)
    result = _run(profile, (0, 1, 2, 3))

    assert result.search_exact_budget_charged == 2
    assert result.search_exact_censored_count == 3
    assert result.search_procedure_censored is True
    assert result.search_exact_evaluation_coverage == 1 / 8


def test_empty_search_still_has_stage6a_intact_baseline_but_zero_recall() -> None:
    profile = _profile()
    result = _run(profile, ())

    assert result.search_exact_evaluation_count == 1
    assert result.search_endpoint1.retained_proportion == 1.0
    assert result.feasible_recall == 0.0
    assert result.search_endpoint2.packing_lower_bound == 0


def test_search_exception_is_preserved_as_failed_censored_procedure() -> None:
    profile = _profile()

    def search(context: TractableSearchContext) -> TractableSearchOutput:
        raise RuntimeError("constructed search failure")

    result = run_tractable_calibration(
        profile=profile,
        policy=_policy(),
        search_procedure=search,
    )

    assert result.search_procedure_failed is True
    assert result.search_procedure_censored is True
    assert "constructed search failure" in (result.search_procedure_error or "")
    assert result.feasible_recall == 0.0


def test_search_context_does_not_expose_feasible_inventory_or_transfer_state() -> None:
    fields = set(TractableSearchContext.__dataclass_fields__)

    assert "qualifying_free_patterns" not in fields
    assert "feasible_mask_identities" not in fields
    assert "teacher_seed" not in fields
    assert "main_experiment_state" not in fields

    profile = _profile()
    seen = {}

    def search(context: TractableSearchContext) -> TractableSearchOutput:
        seen["context"] = context
        return TractableSearchOutput(proposals=())

    run_tractable_calibration(
        profile=profile,
        policy=_policy(),
        search_procedure=search,
    )

    context = seen["context"]
    assert context.qualifying_patterns_exposed is False
    assert context.teacher_seed_exposed is False
    assert context.main_experiment_state_exposed is False


def test_fixture_rejects_teacher_seed_or_main_experiment_transfer() -> None:
    profile = _profile()

    with pytest.raises(TractableCalibrationError):
        replace(profile, teacher_seed_transfer=True)
    with pytest.raises(TractableCalibrationError):
        replace(profile, main_experiment_transfer=True)


def test_runner_has_no_external_evaluator_injection() -> None:
    parameters = inspect.signature(run_tractable_calibration).parameters
    assert "evaluator" not in parameters
    assert "teacher" not in parameters
    assert "seed_masks" not in parameters


def test_tractable_bound_and_free_coordinate_contract_are_enforced() -> None:
    with pytest.raises(TractableCalibrationError):
        _profile(free=tuple(range(13)))

    with pytest.raises(TractableCalibrationError):
        _profile(free=(0, 0, 1))

    with pytest.raises(TractableCalibrationError):
        _profile(free=(0, COMPONENT_COUNT))


def test_qualifying_patterns_must_belong_to_complete_universe() -> None:
    with pytest.raises(TractableCalibrationError):
        _profile(free=(0, 1), feasible=(0, 4))

    with pytest.raises(TractableCalibrationError):
        _profile(feasible=())


def test_inventory_hash_corruption_is_rejected() -> None:
    result = _run(_profile(), (0, 1, 2))

    with pytest.raises(TractableCalibrationError, match="inventory hash"):
        replace(result, feasible_inventory_hash="0" * 64)


def test_universe_hash_corruption_is_rejected() -> None:
    result = _run(_profile(), (0, 1, 2))

    with pytest.raises(TractableCalibrationError, match="universe hash"):
        replace(result, admissible_universe_hash="0" * 64)


def test_certificate_corruption_is_rejected() -> None:
    result = _run(_profile(), (0, 1, 2))

    with pytest.raises(Stage12R3ContractError):
        replace(result.certificate, exhaustive=False)

    bad_certificate = CompletenessCertificate(
        exactness_claim="exact",
        exhaustive=True,
        lower_bound=result.feasible_mask_count + 1,
        upper_bound=result.feasible_mask_count + 1,
        gap=0,
        certificate_reference="corrupted-cardinality",
    )
    with pytest.raises(TractableCalibrationError, match="certificate bounds"):
        replace(result, certificate=bad_certificate)


def test_certified_packing_corruption_is_rejected() -> None:
    result = _run(_profile(), (0, 1, 2))

    with pytest.raises(TractableCalibrationError, match="packing optimum"):
        replace(
            result,
            certified_packing_optimum=result.certified_packing_optimum + 1,
        )


def test_deterministic_reference_hashes_and_metrics() -> None:
    profile = _profile()
    first = _run(profile, (0, 1, 2))
    second = _run(profile, (0, 1, 2))

    assert first.admissible_universe_hash == second.admissible_universe_hash
    assert first.feasible_inventory_hash == second.feasible_inventory_hash
    assert first.identity == second.identity


def test_result_is_explicitly_technical_and_not_mechanism_count() -> None:
    result = _run(_profile(), (0, 1, 2))

    assert result.scientific_data is False
    assert result.production_eligible is False
    assert result.teacher_seed_transfer is False
    assert result.main_experiment_transfer is False
    assert result.mechanism_count_claim is False

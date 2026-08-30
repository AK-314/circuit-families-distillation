from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from circuit_families.stage12r1 import (
    ALGORITHM_FAMILY,
    NATIVE_BUDGET_UNIT,
    TECHNICAL_PROFILE_VERSION,
    UNRESOLVED_PRODUCTION_DECISIONS,
    ExactBridgeSummary,
    ExactBudgetRecord,
    NativeBudgetRecord,
    ProposalProvenanceSummary,
    Stage12R1TechnicalProfile,
    build_lifecycle_record,
    lifecycle_record_from_mapping,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def profile() -> Stage12R1TechnicalProfile:
    return Stage12R1TechnicalProfile(
        profile_version=TECHNICAL_PROFILE_VERSION,
        profile_id="stage12r1-technical-fixture",
        method_name="stage12r1_hard_concrete",
        method_version="technical-v1",
        algorithm_family=ALGORITHM_FAMILY,
        configuration_reference="fixture://stage12r1-part-f",
        native_budget_unit=NATIVE_BUDGET_UNIT,
        native_budget_allowance=None,
        exact_evaluation_allowance=None,
        maximum_restarts=None,
        production_algorithm_selected=False,
        scientific_data=False,
        production_eligible=False,
        unresolved_decisions=UNRESOLVED_PRODUCTION_DECISIONS,
    )


def native() -> NativeBudgetRecord:
    return NativeBudgetRecord(
        unit="optimizer_step",
        allowance=5,
        consumed=5,
        exhausted=True,
    )


def proposals() -> ProposalProvenanceSummary:
    return ProposalProvenanceSummary(
        gate_state_sha256=SHA_A,
        extraction_config_sha256=SHA_B,
        proposal_count=3,
        unique_mask_count=2,
        duplicate_proposal_count=1,
    )


def exact_budget() -> ExactBudgetRecord:
    return ExactBudgetRecord(
        allowance=3,
        charged=3,
        evaluation_count=3,
        proposal_count=3,
        exhausted=True,
    )


def exact_bridge() -> ExactBridgeSummary:
    return ExactBridgeSummary(
        exact_ledger_sha256=SHA_C,
        exact_ledger_evaluation_count=3,
        exact_ledger_proposal_count=3,
        qualifying_count=2,
        minimum_exact_fidelity=-0.25,
        maximum_exact_fidelity=1.0,
    )


def lifecycle():
    return build_lifecycle_record(
        run_id="part-f-run",
        method_name="stage12r1_hard_concrete",
        method_version="technical-v1",
        configuration_reference="fixture://part-f",
        run_identity_sha256=SHA_A,
        gate_config_sha256=SHA_B,
        optimizer_config_sha256=SHA_C,
        optimizer_result_sha256=SHA_D,
        checkpoint_identity_sha256=SHA_E,
        native_budget=native(),
        proposals=proposals(),
        exact_budget=exact_budget(),
        exact_bridge=exact_bridge(),
        terminal_state="exhausted",
        failure_kind="none",
        intact_endpoint1_available=True,
    )


def test_technical_profile_keeps_production_choices_unresolved() -> None:
    value = profile()

    assert value.native_budget_allowance is None
    assert value.exact_evaluation_allowance is None
    assert value.maximum_restarts is None
    assert value.production_algorithm_selected is False
    assert value.scientific_data is False
    assert value.production_eligible is False
    assert value.unresolved_decisions == UNRESOLVED_PRODUCTION_DECISIONS


def test_native_budget_cannot_masquerade_as_other_method_unit() -> None:
    with pytest.raises(ValueError, match="optimizer_step"):
        replace(native(), unit="proposal")


def test_budget_transfer_corruption_rejects() -> None:
    with pytest.raises(ValueError, match="consumption exceeds"):
        NativeBudgetRecord(
            unit="optimizer_step",
            allowance=4,
            consumed=5,
            exhausted=True,
        )

    with pytest.raises(ValueError, match="exact charge exceeds"):
        ExactBudgetRecord(
            allowance=2,
            charged=3,
            evaluation_count=3,
            proposal_count=3,
            exhausted=True,
        )


def test_duplicate_proposal_accounting_is_reconstructable() -> None:
    value = proposals()

    assert value.proposal_count == 3
    assert value.unique_mask_count == 2
    assert value.duplicate_proposal_count == 1


def test_negative_exact_fidelity_is_preserved_not_clipped() -> None:
    value = exact_bridge()

    assert value.minimum_exact_fidelity == -0.25

    with pytest.raises(ValueError, match="bounds are reversed"):
        ExactBridgeSummary(
            exact_ledger_sha256=SHA_C,
            exact_ledger_evaluation_count=3,
            exact_ledger_proposal_count=3,
            qualifying_count=2,
            minimum_exact_fidelity=0.0,
            maximum_exact_fidelity=-0.25,
        )


def test_lifecycle_round_trip_and_hash_are_stable() -> None:
    original = lifecycle()
    restored = lifecycle_record_from_mapping(asdict(original))

    assert restored == original
    assert len(original.record_sha256) == 64


def test_stale_hash_corruption_rejects() -> None:
    record = asdict(lifecycle())
    record["run_id"] = "tampered-run"

    with pytest.raises(ValueError, match="hash mismatch"):
        lifecycle_record_from_mapping(record)


def test_result_relabeling_method_family_corruption_rejects() -> None:
    record = asdict(lifecycle())
    record["algorithm_family"] = "greedy_deletion"

    with pytest.raises(ValueError, match="algorithm-family"):
        lifecycle_record_from_mapping(record)


def test_production_eligibility_cannot_be_asserted() -> None:
    with pytest.raises(ValueError, match="production_eligible=false"):
        replace(profile(), production_eligible=True)


def test_production_algorithm_selection_cannot_be_asserted() -> None:
    with pytest.raises(ValueError, match="cannot select"):
        replace(profile(), production_algorithm_selected=True)


def test_required_decisions_cannot_be_silently_removed() -> None:
    with pytest.raises(ValueError, match="must remain unresolved"):
        replace(
            profile(),
            unresolved_decisions=(
                "RD-005",
                "RD-006",
            ),
        )


def test_private_paths_reject() -> None:
    with pytest.raises(ValueError, match="private filesystem paths"):
        replace(
            profile(),
            configuration_reference="/Users/private/technical.json",
        )


def test_failed_optimization_can_exist_without_sparse_proposals() -> None:
    record = build_lifecycle_record(
        run_id="failed-native-run",
        method_name="stage12r1_hard_concrete",
        method_version="technical-v1",
        configuration_reference="fixture://failed-native",
        run_identity_sha256=SHA_A,
        gate_config_sha256=SHA_B,
        optimizer_config_sha256=SHA_C,
        optimizer_result_sha256=SHA_D,
        checkpoint_identity_sha256=None,
        native_budget=NativeBudgetRecord(
            unit="optimizer_step",
            allowance=5,
            consumed=0,
            exhausted=False,
        ),
        proposals=None,
        exact_budget=ExactBudgetRecord(
            allowance=1,
            charged=1,
            evaluation_count=1,
            proposal_count=0,
            exhausted=True,
        ),
        exact_bridge=ExactBridgeSummary(
            exact_ledger_sha256=SHA_F,
            exact_ledger_evaluation_count=1,
            exact_ledger_proposal_count=0,
            qualifying_count=1,
            minimum_exact_fidelity=1.0,
            maximum_exact_fidelity=1.0,
        ),
        terminal_state="numerical_failure",
        failure_kind="nonfinite_objective",
        intact_endpoint1_available=True,
    )

    assert record.proposals is None
    assert record.intact_endpoint1_available is True


def test_exact_bridge_implies_intact_endpoint1_availability() -> None:
    with pytest.raises(ValueError, match="intact Endpoint 1"):
        build_lifecycle_record(
            run_id="bad-intact-claim",
            method_name="stage12r1_hard_concrete",
            method_version="technical-v1",
            configuration_reference="fixture://bad-intact",
            run_identity_sha256=SHA_A,
            gate_config_sha256=SHA_B,
            optimizer_config_sha256=SHA_C,
            optimizer_result_sha256=SHA_D,
            checkpoint_identity_sha256=None,
            native_budget=native(),
            proposals=proposals(),
            exact_budget=exact_budget(),
            exact_bridge=exact_bridge(),
            terminal_state="exhausted",
            failure_kind="none",
            intact_endpoint1_available=False,
        )


def test_exact_evaluation_count_must_equal_unique_charge() -> None:
    with pytest.raises(ValueError, match="must equal charged"):
        ExactBudgetRecord(
            allowance=3,
            charged=2,
            evaluation_count=3,
            proposal_count=4,
            exhausted=False,
        )


def test_records_do_not_contain_large_payload_fields() -> None:
    record = asdict(lifecycle())
    forbidden = {
        "logits",
        "examples",
        "per_example",
        "gate_values",
        "checkpoint_bytes",
        "model_state_dict",
    }

    assert not forbidden.intersection(record)
    assert not forbidden.intersection(record["exact_bridge"])
    assert not forbidden.intersection(record["proposals"])

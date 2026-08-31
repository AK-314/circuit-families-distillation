from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from circuit_families.stage12p5 import (
    CONDITIONS,
    ModelReference,
    PairContract,
    Stage12P5ContractError,
    account_payload,
    derive_condition_seed,
    deterministic_derangement,
    trial_from_mapping,
    validate_comparison_capacity,
)
from circuit_families.stage12p5.controls import ControlUnavailableError
from circuit_families.stage12p5.synthetic import SHA_A, build_synthetic_fixture


def test_trial_closed_schema_round_trip_and_stable_identity() -> None:
    fixture = build_synthetic_fixture()
    restored = trial_from_mapping(fixture.trial.to_mapping())
    assert restored == fixture.trial
    assert restored.trial_id == fixture.trial.trial_id
    assert restored.comparison_set_id == fixture.trial.comparison_set_id
    assert restored.conditions == CONDITIONS


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(extra=True), "closed schema"),
        (
            lambda value: value["pair"].__setitem__("candidate_outcomes_consulted", True),
            "without candidate outcomes",
        ),
        (lambda value: value["conditions"].pop(), "exactly aligned plus five controls"),
        (
            lambda value: value["conditions"].__setitem__(1, value["conditions"][0]),
            "exactly aligned plus five controls",
        ),
        (lambda value: value.__setitem__("scientific_data", True), "scientific_data=false"),
        (lambda value: value.__setitem__("production_eligible", True), "production_eligible=false"),
        (lambda value: value.__setitem__("source_input_id", "input-b"), "identity/hash"),
        (
            lambda value: value["alignment"].__setitem__("fit_data_sha256", "0" * 64),
            "alignment profile hash",
        ),
    ],
)
def test_trial_adversarial_mutations_reject(mutation, match: str) -> None:
    value = copy.deepcopy(build_synthetic_fixture().trial.to_mapping())
    mutation(value)
    with pytest.raises(Stage12P5ContractError, match=match):
        trial_from_mapping(value)


def test_pair_rejects_reversed_roles_and_hard_soft_confusion() -> None:
    fixture = build_synthetic_fixture()
    pair = fixture.trial.pair
    with pytest.raises(Stage12P5ContractError, match="roles"):
        PairContract(
            pair.recipient,
            pair.source,
            pair.selection_evidence_ref,
            pair.selection_evidence_sha256,
            pair.selection_rule_ref,
        )
    with pytest.raises(Stage12P5ContractError, match="hard/soft"):
        ModelReference(
            "recipient_student",
            "model/bad/v1",
            "architecture/bad/v1",
            SHA_A,
            "checkpoint/bad/v1",
            SHA_A,
            "direct_teacher",
        )


def test_seed_domains_are_deterministic_and_disjoint() -> None:
    trial = build_synthetic_fixture().trial
    domains = ("pair", "input", "alignment", *CONDITIONS, "shuffle", "mismatch", "retry")
    values = [derive_condition_seed(trial, domain) for domain in domains]
    assert values == [derive_condition_seed(trial, domain) for domain in domains]
    assert len({seed for seed, _ in values}) == len(domains)
    assert len({digest for _, digest in values}) == len(domains)


def test_derangement_is_canonical_and_single_input_is_unavailable() -> None:
    mapping = deterministic_derangement(("z", "a", "m"), seed=7)
    assert set(mapping) == {"a", "m", "z"}
    assert all(source != recipient for source, recipient in mapping.items())
    with pytest.raises(ControlUnavailableError, match="at least two"):
        deterministic_derangement(("only",), seed=7)


def test_all_payloads_have_distinct_condition_identity_and_control_evidence() -> None:
    fixture = build_synthetic_fixture()
    assert tuple(fixture.payloads) == CONDITIONS
    assert len({payload.payload_sha256 for payload in fixture.payloads.values()}) == 6
    wrong = fixture.payloads["wrong_fourier_mode"]
    assert wrong.mode_id != fixture.trial.source_mode_id
    shuffled = fixture.payloads["shuffled_coefficients"]
    assert shuffled.construction_evidence["marginal_values_preserved"] is True
    assert shuffled.construction_evidence["norm_preserved"] is True
    mismatched = fixture.payloads["mismatched_input"]
    assert mismatched.source_input_id != fixture.trial.recipient_input_id
    random_state = fixture.payloads["equal_norm_random_state"]
    assert np.isclose(
        random_state.construction_evidence["target_norm"],
        random_state.construction_evidence["observed_norm"],
        atol=random_state.construction_evidence["norm_tolerance"],
    )
    ordinary = fixture.payloads["unaligned_ordinary_activation_patching"]
    assert ordinary.alignment_plan_sha256 is None
    assert ordinary.construction_evidence["Fourier_alignment_applied"] is False


def test_zero_norm_and_repeated_coefficients_have_declared_deterministic_behavior() -> None:
    fixture = build_synthetic_fixture(source_a=np.zeros(4, dtype=np.float64))
    random_state = fixture.payloads["equal_norm_random_state"]
    assert random_state.construction_evidence["zero_norm_behavior"] == "deterministic_zero"
    assert np.count_nonzero(random_state.coordinate_values) == 0
    shuffled = fixture.payloads["shuffled_coefficients"]
    assert shuffled.construction_evidence["marginal_values_preserved"] is True
    assert shuffled.construction_evidence["norm_preserved"] is True


def test_capacity_matches_real_dof_rank_support_precision_and_side_information() -> None:
    fixture = build_synthetic_fixture()
    comparison_hash = validate_comparison_capacity(fixture.accounting, fixture.trial.capacity)
    assert len(comparison_hash) == 64
    for record in fixture.accounting.values():
        assert record.eligible
        assert record.real_degrees_of_freedom == 4
        assert record.scalar_precision_bits == 64
        assert record.write_budget_scalars == 4
        assert record.capacity_sha256 == fixture.trial.capacity.capacity_sha256


@pytest.mark.parametrize(
    "hidden",
    [
        {"mode_label": "hidden"},
        {"input_identity": "hidden"},
        {"alignment_matrix": "hidden"},
        {"coordinate_indices": [1, 3]},
        {"random_seed": 1},
        {"payload_length": 2},
    ],
)
def test_capacity_exposes_hidden_side_channels(hidden) -> None:
    fixture = build_synthetic_fixture()
    record = account_payload(
        fixture.payloads[CONDITIONS[0]],
        fixture.trial.capacity,
        scalar_precision_bits=64,
        hidden_metadata=hidden,
    )
    assert not record.eligible
    assert "hidden side channel" in record.ineligibility_reason


def test_capacity_rejects_complex_real_precision_and_rank_inflation() -> None:
    fixture = build_synthetic_fixture()
    payload = fixture.payloads[CONDITIONS[0]]
    real_payload = replace(payload, coordinate_values=payload.coordinate_values.real.copy())
    real_record = account_payload(real_payload, fixture.trial.capacity, scalar_precision_bits=64)
    assert not real_record.eligible
    assert "degree" in real_record.ineligibility_reason
    precision_record = account_payload(payload, fixture.trial.capacity, scalar_precision_bits=32)
    assert not precision_record.eligible
    assert "precision" in precision_record.ineligibility_reason


def test_comparison_capacity_rejects_missing_or_reordered_condition() -> None:
    fixture = build_synthetic_fixture()
    missing = dict(fixture.accounting)
    missing.pop(CONDITIONS[-1])
    with pytest.raises(Stage12P5ContractError, match="incomplete"):
        validate_comparison_capacity(missing, fixture.trial.capacity)
    reordered = {name: fixture.accounting[name] for name in reversed(CONDITIONS)}
    with pytest.raises(Stage12P5ContractError, match="reordered"):
        validate_comparison_capacity(reordered, fixture.trial.capacity)

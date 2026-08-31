from __future__ import annotations

from dataclasses import replace

import pytest

from circuit_families.stage12r1 import ALGORITHM_FAMILY
from circuit_families.stage12r2.contracts import canonical_sha256
from circuit_families.stage12r3 import (
    CLAIM_BOUNDARIES,
    EXPECTED_LAYER_KINDS,
    CalibrationLayerContract,
    CompletenessCertificate,
    Stage12R3CalibrationContract,
    Stage12R3ContractError,
    build_technical_calibration_contract,
)

COMPONENT_IDS = (
    "attn.h0",
    "attn.h1",
    "mlp.0",
    "mlp.1",
    "mlp.2",
    "mlp.3",
)
COMPONENT_TYPES = (
    "attention_head",
    "attention_head",
    "mlp_neuron",
    "mlp_neuron",
    "mlp_neuron",
    "mlp_neuron",
)
BASIS_HASH = canonical_sha256(
    {
        "fixture": "stage12r3-contract-test",
        "ordered_component_ids": COMPONENT_IDS,
        "component_types": COMPONENT_TYPES,
    }
)


def make_contract() -> Stage12R3CalibrationContract:
    return build_technical_calibration_contract(
        profile_id="stage12r3-test-profile",
        basis_hash=BASIS_HASH,
        ordered_component_ids=COMPONENT_IDS,
        component_types=COMPONENT_TYPES,
        combinatorial_native_allowance=20,
        restart_native_allowance=100,
        restart_exact_allowance=12,
        local_native_allowance=30,
        local_exact_allowance=15,
        tractable_native_allowance=64,
        tractable_exact_allowance=64,
    )


def test_contract_has_four_distinct_layers_and_stable_identities() -> None:
    left = make_contract()
    right = make_contract()

    assert tuple(layer.layer_kind for layer in left.layers) == EXPECTED_LAYER_KINDS
    assert len({layer.layer_id for layer in left.layers}) == 4
    assert left.identity == right.identity
    assert [layer.identity for layer in left.layers] == [
        layer.identity for layer in right.layers
    ]


def test_native_and_exact_budgets_are_separate() -> None:
    contract = make_contract()

    restart = contract.layers[1]
    assert restart.native_budget.unit == "optimizer_step"
    assert restart.native_budget.allowance == 100
    assert restart.exact_budget.allowance == 12

    combinatorial = contract.layers[0]
    assert combinatorial.native_budget.unit == "combinatorial_draw"
    assert combinatorial.exact_budget.allowance == 0


def test_boundaries_reuse_stage6a_stage6e_and_r1_family() -> None:
    contract = make_contract()

    for layer in contract.layers:
        assert layer.exact_evaluation_boundary == "stage6a_exact_evaluation_bridge"
        assert layer.endpoint_boundary == "shared_stage6a_stage6e_reducers"

    restart = contract.layers[1]
    assert restart.discovery_family == ALGORITHM_FAMILY
    assert restart.discovery_relationship == (
        "same_discovery_family_ordinary_restart"
    )


def test_combinatorial_floor_makes_no_fidelity_claim() -> None:
    layer = make_contract().layers[0]

    assert layer.qualification_source == "none"
    assert layer.claim_boundary == CLAIM_BOUNDARIES["combinatorial_floor"]


def test_ordinary_restart_cannot_be_relabelled_independent() -> None:
    layer = make_contract().layers[1]

    with pytest.raises(
        Stage12R3ContractError,
        match="cannot be relabelled",
    ):
        replace(layer, discovery_relationship="not_applicable")


@pytest.mark.parametrize(
    "field",
    [
        "uses_diversity_pressure",
        "uses_packing_feedback",
        "uses_prior_restart_mask_exclusion",
    ],
)
def test_ordinary_restart_rejects_cross_restart_feedback(field: str) -> None:
    layer = make_contract().layers[1]

    with pytest.raises(Stage12R3ContractError):
        replace(layer, **{field: True})


def test_local_layer_rejects_surrogate_or_inherited_fidelity() -> None:
    layer = make_contract().layers[2]

    with pytest.raises(
        Stage12R3ContractError,
        match="fresh exact common-ledger",
    ):
        replace(layer, qualification_source="none")


def test_tractable_exactness_requires_certificate() -> None:
    layer = make_contract().layers[3]

    with pytest.raises(
        Stage12R3ContractError,
        match="requires a certificate",
    ):
        replace(layer, completeness_certificate=None)


def test_exact_certificate_rejects_nonexhaustive_claim() -> None:
    with pytest.raises(
        Stage12R3ContractError,
        match="requires exhaustive",
    ):
        CompletenessCertificate(
            exactness_claim="exact",
            exhaustive=False,
            lower_bound=3,
            upper_bound=3,
            gap=0,
            certificate_reference="invalid",
        )


def test_exact_certificate_rejects_nonzero_gap() -> None:
    with pytest.raises(Stage12R3ContractError):
        CompletenessCertificate(
            exactness_claim="exact",
            exhaustive=True,
            lower_bound=3,
            upper_bound=4,
            gap=1,
            certificate_reference="invalid",
        )


def test_near_exact_certificate_requires_valid_nonzero_gap() -> None:
    certificate = CompletenessCertificate(
        exactness_claim="certified_near_exact",
        exhaustive=False,
        lower_bound=7,
        upper_bound=9,
        gap=2,
        certificate_reference="technical-bound-proof",
    )
    assert certificate.gap == 2

    with pytest.raises(Stage12R3ContractError):
        replace(certificate, gap=1)

    with pytest.raises(Stage12R3ContractError):
        replace(certificate, upper_bound=7, gap=0)


def test_outer_contract_rejects_collapsed_or_duplicate_layers() -> None:
    contract = make_contract()

    with pytest.raises(Stage12R3ContractError, match="exactly four"):
        replace(contract, layers=contract.layers[:3])

    duplicate = (
        contract.layers[0],
        contract.layers[1],
        contract.layers[2],
        contract.layers[2],
    )
    with pytest.raises(Stage12R3ContractError):
        replace(contract, layers=duplicate)


def test_outer_contract_rejects_basis_identity_mixing() -> None:
    contract = make_contract()
    bad_layer = replace(
        contract.layers[2],
        basis_hash=canonical_sha256({"different": "basis"}),
    )

    with pytest.raises(
        Stage12R3ContractError,
        match="mix basis identities",
    ):
        replace(
            contract,
            layers=(
                contract.layers[0],
                contract.layers[1],
                bad_layer,
                contract.layers[3],
            ),
        )


def test_outer_contract_rejects_component_type_mixing() -> None:
    contract = make_contract()
    bad_types = (
        "attention_head",
        "mlp_neuron",
        "mlp_neuron",
        "mlp_neuron",
        "mlp_neuron",
        "mlp_neuron",
    )
    bad_layer = replace(contract.layers[2], component_types=bad_types)

    with pytest.raises(
        Stage12R3ContractError,
        match="component-type identities",
    ):
        replace(
            contract,
            layers=(
                contract.layers[0],
                contract.layers[1],
                bad_layer,
                contract.layers[3],
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scientific_data", True),
        ("production_eligible", True),
        ("production_packing_policy_selected", True),
        ("tractable_calibration_production_choice", True),
    ],
)
def test_outer_contract_rejects_scientific_or_production_state(
    field: str,
    value: bool,
) -> None:
    contract = make_contract()

    with pytest.raises(Stage12R3ContractError):
        replace(contract, **{field: value})


def test_layer_rejects_scientific_or_production_state() -> None:
    layer = make_contract().layers[2]

    with pytest.raises(Stage12R3ContractError):
        replace(layer, scientific_data=True)

    with pytest.raises(Stage12R3ContractError):
        replace(layer, production_eligible=True)


def test_required_production_decisions_remain_open() -> None:
    contract = make_contract()

    for decision in ("RD-006", "RD-008", "RD-009"):
        assert decision in contract.unresolved_production_decisions

    with pytest.raises(Stage12R3ContractError):
        replace(
            contract,
            unresolved_production_decisions=("RD-006", "RD-008"),
        )


def test_claim_boundaries_are_layer_specific_and_no_transfer() -> None:
    contract = make_contract()

    assert len({layer.claim_boundary for layer in contract.layers}) == 4
    assert "no_main_scale_transfer" in contract.layers[3].claim_boundary


def test_contract_does_not_create_alternate_endpoint_or_fidelity_semantics() -> None:
    contract = make_contract()

    assert all(
        layer.exact_evaluation_boundary == "stage6a_exact_evaluation_bridge"
        for layer in contract.layers
    )
    assert all(
        layer.endpoint_boundary == "shared_stage6a_stage6e_reducers"
        for layer in contract.layers
    )


def test_explicit_layer_constructor_rejects_wrong_claim_boundary() -> None:
    source = make_contract().layers[0]

    with pytest.raises(Stage12R3ContractError, match="claim boundary"):
        CalibrationLayerContract(
            **{
                **source.__dict__,
                "claim_boundary": "mechanism_count",
            }
        )

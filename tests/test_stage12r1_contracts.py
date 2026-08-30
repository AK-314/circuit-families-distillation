from __future__ import annotations

from dataclasses import replace

import pytest

from circuit_families.stage12r1 import (
    ALGORITHM_FAMILY,
    NATIVE_BUDGET_UNIT,
    UNRESOLVED_PRODUCTION_DECISIONS,
    IndependentMethodContract,
    validate_algorithmic_independence,
)


def test_independent_contract_separates_native_and_exact_accounting() -> None:
    contract = IndependentMethodContract()

    assert contract.algorithm_family == ALGORITHM_FAMILY
    assert contract.native_budget_unit == NATIVE_BUDGET_UNIT
    assert contract.native_budget_unit != "proposal"
    assert contract.exact_evaluation_boundary == "stage6a_exact_evaluation_bridge"
    assert contract.endpoint_boundary == "shared_stage6a_stage6e_reducers"
    assert contract.scientific_data is False
    assert contract.production_eligible is False
    assert (
        contract.unresolved_production_decisions
        == UNRESOLVED_PRODUCTION_DECISIONS
    )


@pytest.mark.parametrize(
    "delegated",
    ["GreedyDeletionAdapter", "DiversityForcedAdapter"],
)
def test_relabelled_inherited_adapter_is_rejected(delegated: str) -> None:
    contract = IndependentMethodContract()

    with pytest.raises(
        ValueError,
        match="not algorithmically independent",
    ):
        validate_algorithmic_independence(
            contract,
            adapter_method_name="stochastic_l0_v1",
            delegated_adapter_name=delegated,
        )


def test_componentwise_greedy_deletion_cannot_be_enabled() -> None:
    with pytest.raises(ValueError, match="greedy deletion is prohibited"):
        replace(
            IndependentMethodContract(),
            uses_componentwise_greedy_deletion=True,
        )


def test_inherited_native_budget_unit_cannot_be_relabelled() -> None:
    with pytest.raises(ValueError, match="optimizer_step"):
        replace(
            IndependentMethodContract(),
            native_budget_unit="greedy_proposal",
        )


def test_valid_independent_method_name_passes_contract() -> None:
    contract = IndependentMethodContract()

    validate_algorithmic_independence(
        contract,
        adapter_method_name="stage12r1_hard_concrete",
    )

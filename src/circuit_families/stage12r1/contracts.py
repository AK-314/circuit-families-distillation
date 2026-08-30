"""Stage 12-R1 technical contract for an independent discovery family.

This module defines only the method boundary. It deliberately does not freeze
the production optimizer, regularization, thresholds, budgets, restart count,
or exact-evaluation allowance.

Algorithmic-independence rationale
----------------------------------
Stage 12-R1 shares Stage 6D request/result records and the Stage 6A exact-ledger
bridge, but its proposal dynamics are different from inherited discrete
component deletion:

* all component gates are represented by jointly optimized continuous
  parameters;
* sparsity pressure is differentiable and acts on the joint gate state;
* stochastic training samples are produced from a seeded reparameterized gate
  distribution;
* binary masks are extracted only after/alongside continuous optimization;
* no component-by-component greedy deletion may be hidden behind the adapter;
* native work is measured in optimizer steps, not inherited proposal counts;
* exact qualification is delegated exclusively to the shared Stage 6A bridge.

Therefore common ledgers may be reused without implying equivalent native
optimization budgets or equivalent discovery dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ALGORITHM_FAMILY = "continuous_stochastic_gates"
METHOD_CONTRACT_VERSION = "stage12r1-independent-method-contract/v1"
NATIVE_BUDGET_UNIT = "optimizer_step"

INHERITED_METHOD_NAMES = frozenset(
    {
        "greedy_deletion",
        "diversity_forced",
        "GreedyDeletionAdapter",
        "DiversityForcedAdapter",
    }
)

UNRESOLVED_PRODUCTION_DECISIONS = (
    "RD-005",
    "RD-006",
    "RD-007",
    "RD-008",
    "RD-009",
    "RD-012",
    "RD-014",
)


@dataclass(frozen=True)
class IndependentMethodContract:
    """Auditable technical contract for the Stage 12-R1 method family."""

    contract_version: str = METHOD_CONTRACT_VERSION
    algorithm_family: str = ALGORITHM_FAMILY
    proposal_dynamics: Literal[
        "joint_continuous_gate_optimization"
    ] = "joint_continuous_gate_optimization"
    sparsity_mechanism: Literal[
        "differentiable_expected_l0"
    ] = "differentiable_expected_l0"
    stochastic_mechanism: Literal[
        "seeded_reparameterized_gates"
    ] = "seeded_reparameterized_gates"
    proposal_extraction: Literal[
        "binary_masks_from_learned_gate_state"
    ] = "binary_masks_from_learned_gate_state"
    native_budget_unit: str = NATIVE_BUDGET_UNIT
    exact_evaluation_boundary: Literal[
        "stage6a_exact_evaluation_bridge"
    ] = "stage6a_exact_evaluation_bridge"
    endpoint_boundary: Literal[
        "shared_stage6a_stage6e_reducers"
    ] = "shared_stage6a_stage6e_reducers"
    uses_componentwise_greedy_deletion: bool = False
    inherited_adapter_delegate: str | None = None
    scientific_data: bool = False
    production_eligible: bool = False
    unresolved_production_decisions: tuple[str, ...] = (
        UNRESOLVED_PRODUCTION_DECISIONS
    )

    def __post_init__(self) -> None:
        if self.contract_version != METHOD_CONTRACT_VERSION:
            raise ValueError("unsupported Stage 12-R1 method contract version")
        if self.algorithm_family != ALGORITHM_FAMILY:
            raise ValueError("Stage 12-R1 algorithm family must remain continuous")
        if self.native_budget_unit != NATIVE_BUDGET_UNIT:
            raise ValueError(
                "Stage 12-R1 native accounting must use optimizer_step"
            )
        if self.uses_componentwise_greedy_deletion:
            raise ValueError(
                "component-by-component greedy deletion is prohibited"
            )
        if self.inherited_adapter_delegate is not None:
            raise ValueError(
                "Stage 12-R1 may not delegate discovery to an inherited adapter"
            )
        if self.scientific_data:
            raise ValueError("Stage 12-R1 technical profiles require scientific_data=false")
        if self.production_eligible:
            raise ValueError(
                "Stage 12-R1 technical profiles require production_eligible=false"
            )
        if self.unresolved_production_decisions != UNRESOLVED_PRODUCTION_DECISIONS:
            raise ValueError(
                "Stage 12-R1 must leave the required production decisions unresolved"
            )


def validate_algorithmic_independence(
    contract: IndependentMethodContract,
    *,
    adapter_method_name: str,
    delegated_adapter_name: str | None = None,
) -> None:
    """Reject a relabelled inherited discrete discovery implementation."""

    if adapter_method_name in INHERITED_METHOD_NAMES:
        raise ValueError("Stage 12-R1 cannot identify itself as an inherited method")
    if delegated_adapter_name in INHERITED_METHOD_NAMES:
        raise ValueError(
            "relabelled inherited discovery adapter is not algorithmically independent"
        )

    # Re-run invariant validation even when callers pass a reconstructed object.
    contract.__post_init__()

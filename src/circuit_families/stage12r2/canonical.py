"""Compatibility adapter for the frozen canonical head-plus-neuron basis."""

from __future__ import annotations

from circuit_families.interpretability.masks import (
    ATTENTION_HEAD_HOOK_NAME,
    ATTENTION_HEAD_IDS,
    MLP_NEURON_HOOK_NAME,
    MLP_NEURON_IDS,
    ComponentMask,
)
from circuit_families.stage12r2.contracts import (
    BasisComponentDescriptor,
    BasisContract,
    BasisMask,
)

CANONICAL_BASIS_FAMILY = "canonical_head_plus_individual_mlp_neurons"
CANONICAL_COORDINATE_DEFINITION = "stage4-frozen-head-plus-neuron-order/v1"


def canonical_basis_contract(
    *,
    parent_model_identity: str,
    parent_component_basis_identity: str,
    attention_parameter_weight: int,
    mlp_parameter_weight: int,
) -> BasisContract:
    """Wrap the existing frozen ordering without changing its semantics."""
    components: list[BasisComponentDescriptor] = []

    for identifier in ATTENTION_HEAD_IDS:
        components.append(
            BasisComponentDescriptor(
                component_id=identifier,
                component_type="attention_head",
                source_subspace=ATTENTION_HEAD_HOOK_NAME,
                intervention_location=ATTENTION_HEAD_HOOK_NAME,
                parameter_weight=attention_parameter_weight,
                coordinate_identity=f"{ATTENTION_HEAD_HOOK_NAME}:{identifier}",
            )
        )

    for identifier in MLP_NEURON_IDS:
        components.append(
            BasisComponentDescriptor(
                component_id=identifier,
                component_type="mlp_neuron",
                source_subspace=MLP_NEURON_HOOK_NAME,
                intervention_location=MLP_NEURON_HOOK_NAME,
                parameter_weight=mlp_parameter_weight,
                coordinate_identity=f"{MLP_NEURON_HOOK_NAME}:{identifier}",
            )
        )

    return BasisContract(
        parent_model_identity=parent_model_identity,
        parent_component_basis_identity=parent_component_basis_identity,
        basis_family=CANONICAL_BASIS_FAMILY,
        coordinate_definition=CANONICAL_COORDINATE_DEFINITION,
        components=tuple(components),
        intervention_location="canonical Stage 4 hooks",
        intervention_semantics="existing exact binary head-plus-neuron zero ablation",
        parameter_weight_denominator_definition="sum declared canonical component weights",
        raw_component_denominator_definition="4 attention heads + 512 MLP neurons",
        display_label="canonical head-plus-neuron basis",
    )


def component_mask_to_basis_mask(
    component_mask: ComponentMask,
    basis: BasisContract,
) -> BasisMask:
    """Translate only the existing canonical ordering into a basis-bound mask."""
    if basis.basis_family != CANONICAL_BASIS_FAMILY:
        raise ValueError("target basis is not the canonical compatibility basis")

    values = component_mask.attention_head_mask + component_mask.mlp_neuron_mask
    result = BasisMask(basis_hash=basis.basis_hash, values=values)
    result.validate_for(basis)
    return result


def basis_mask_to_component_mask(
    basis_mask: BasisMask,
    basis: BasisContract,
) -> ComponentMask:
    """Recover the exact existing ComponentMask from the canonical view."""
    if basis.basis_family != CANONICAL_BASIS_FAMILY:
        raise ValueError("source basis is not the canonical compatibility basis")
    basis_mask.validate_for(basis)

    return ComponentMask(
        attention_head_mask=basis_mask.values[:4],
        mlp_neuron_mask=basis_mask.values[4:],
    )

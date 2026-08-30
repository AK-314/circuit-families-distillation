"""Pre-output-projection attention-coordinate refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from circuit_families.interpretability.masks import ATTENTION_HEAD_HOOK_NAME
from circuit_families.stage12r2.contracts import (
    BasisComponentDescriptor,
    BasisContract,
    BasisMask,
    BasisRelationship,
)

ATTENTION_COORDINATE_FAMILY = "pre_output_projection_attention_coordinates"
ATTENTION_COORDINATE_DEFINITION = "hook_z-head-coordinate/v1"


@dataclass(frozen=True)
class AttentionCoordinateSpec:
    layer: int
    n_heads: int
    d_head: int
    hook_name: str = ATTENTION_HEAD_HOOK_NAME

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.n_heads <= 0 or self.d_head <= 0:
            raise ValueError("n_heads and d_head must be positive")
        if self.hook_name != ATTENTION_HEAD_HOOK_NAME:
            raise ValueError(
                "attention coordinates must be defined at the pre-output "
                f"projection hook {ATTENTION_HEAD_HOOK_NAME!r}"
            )


def attention_coordinate_basis(
    *,
    parent_basis: BasisContract,
    spec: AttentionCoordinateSpec,
    parameter_weight_per_coordinate: tuple[int, ...],
) -> BasisContract:
    """Refine each parent attention head into ordered pre-W_O coordinates."""
    parent_heads = [
        component
        for component in parent_basis.components
        if component.component_type == "attention_head"
    ]
    if len(parent_heads) != spec.n_heads:
        raise ValueError("parent basis head count does not match attention spec")

    expected = spec.n_heads * spec.d_head
    if len(parameter_weight_per_coordinate) != expected:
        raise ValueError("parameter-weight metadata length does not match coordinates")

    components: list[BasisComponentDescriptor] = []
    weight_index = 0
    for head_index, parent in enumerate(parent_heads):
        for coordinate_index in range(spec.d_head):
            components.append(
                BasisComponentDescriptor(
                    component_id=f"L{spec.layer}.H{head_index}.Z{coordinate_index}",
                    component_type="attention_coordinate",
                    source_subspace=(
                        f"{spec.hook_name}:layer={spec.layer}:head={head_index}"
                    ),
                    intervention_location=spec.hook_name,
                    parameter_weight=parameter_weight_per_coordinate[weight_index],
                    coordinate_identity=(
                        f"layer={spec.layer}/head={head_index}/"
                        f"coordinate={coordinate_index}"
                    ),
                    parent_component_identity=parent.component_id,
                )
            )
            weight_index += 1

    mapping_identity = (
        f"head-coordinate-map:layer={spec.layer}:"
        f"heads={spec.n_heads}:d_head={spec.d_head}"
    )

    return BasisContract(
        parent_model_identity=parent_basis.parent_model_identity,
        parent_component_basis_identity=parent_basis.parent_component_basis_identity,
        basis_family=ATTENTION_COORDINATE_FAMILY,
        coordinate_definition=ATTENTION_COORDINATE_DEFINITION,
        components=tuple(components),
        intervention_location=spec.hook_name,
        intervention_semantics=(
            "binary coordinate mask on hook_z before attention output projection; "
            "broadcast over batch and position"
        ),
        parameter_weight_denominator_definition=(
            "sum explicitly supplied per-coordinate parameter weights"
        ),
        raw_component_denominator_definition=(
            "number of pre-output-projection attention coordinates"
        ),
        relationship=BasisRelationship(
            kind="refinement",
            parent_basis_hash=parent_basis.basis_hash,
            mapping_identity=mapping_identity,
        ),
        grouping_partition_identity=mapping_identity,
        display_label="pre-output-projection attention coordinates",
    )


def apply_attention_coordinate_mask(
    activation: torch.Tensor,
    *,
    mask: BasisMask,
    basis: BasisContract,
    spec: AttentionCoordinateSpec,
) -> torch.Tensor:
    """Mask hook_z coordinates with broadcasting over batch and position."""
    mask.validate_for(basis)
    if basis.basis_family != ATTENTION_COORDINATE_FAMILY:
        raise ValueError("basis is not an attention-coordinate refinement")
    if activation.ndim != 4:
        raise ValueError(
            "attention activation must have shape (batch, position, head, d_head)"
        )
    if activation.shape[2:] != (spec.n_heads, spec.d_head):
        raise ValueError("attention activation shape does not match coordinate spec")

    values = torch.tensor(
        mask.values,
        dtype=activation.dtype,
        device=activation.device,
    ).view(1, 1, spec.n_heads, spec.d_head)
    return activation * values


def parent_head_mask_to_coordinate_mask(
    *,
    parent_attention_mask: tuple[int, ...],
    basis: BasisContract,
    spec: AttentionCoordinateSpec,
) -> BasisMask:
    """Expand each parent head bit over its complete d_head coordinate group."""
    if len(parent_attention_mask) != spec.n_heads:
        raise ValueError("parent attention mask length does not match n_heads")
    if any(value not in (0, 1) for value in parent_attention_mask):
        raise ValueError("parent attention mask must be binary")

    values = tuple(
        value
        for head_value in parent_attention_mask
        for value in (head_value,) * spec.d_head
    )
    result = BasisMask(basis_hash=basis.basis_hash, values=values)
    result.validate_for(basis)
    return result

from __future__ import annotations

import pytest
import torch

from circuit_families.interpretability.component_ablation import (
    apply_attention_head_mask,
)
from circuit_families.interpretability.masks import ComponentMask
from circuit_families.stage12r2.attention import (
    AttentionCoordinateSpec,
    apply_attention_coordinate_mask,
    attention_coordinate_basis,
    parent_head_mask_to_coordinate_mask,
)
from circuit_families.stage12r2.canonical import (
    basis_mask_to_component_mask,
    canonical_basis_contract,
    component_mask_to_basis_mask,
)
from circuit_families.stage12r2.contracts import BasisMask, validate_relationship


def canonical():
    return canonical_basis_contract(
        parent_model_identity="technical-model:fixture-v1",
        parent_component_basis_identity="stage4-component-basis:v1",
        attention_parameter_weight=32,
        mlp_parameter_weight=129,
    )


def refined():
    parent = canonical()
    spec = AttentionCoordinateSpec(layer=0, n_heads=4, d_head=3)
    child = attention_coordinate_basis(
        parent_basis=parent,
        spec=spec,
        parameter_weight_per_coordinate=(1,) * 12,
    )
    return parent, child, spec


def test_canonical_order_and_mask_round_trip_are_unchanged() -> None:
    parent = canonical()
    assert parent.component_count == 516
    assert tuple(c.component_id for c in parent.components[:4]) == (
        "H0", "H1", "H2", "H3"
    )
    assert parent.components[4].component_id == "N0"
    assert parent.components[-1].component_id == "N511"

    original = ComponentMask.one_head_ablated("H2")
    stage12_mask = component_mask_to_basis_mask(original, parent)
    assert basis_mask_to_component_mask(stage12_mask, parent) == original


def test_attention_refinement_relationship_is_valid() -> None:
    parent, child, _ = refined()
    validate_relationship(child, parent)
    assert {c.parent_component_identity for c in child.components[:3]} == {"H0"}


def test_all_on_attention_coordinates_are_identity() -> None:
    _, basis, spec = refined()
    activation = torch.arange(48, dtype=torch.float64).reshape(1, 4, 4, 3)
    mask = BasisMask(basis_hash=basis.basis_hash, values=(1,) * 12)
    result = apply_attention_coordinate_mask(
        activation, mask=mask, basis=basis, spec=spec
    )
    assert torch.equal(result, activation)


def test_all_off_attention_coordinates_zero_activation() -> None:
    _, basis, spec = refined()
    activation = torch.ones((2, 5, 4, 3), dtype=torch.float32)
    mask = BasisMask(basis_hash=basis.basis_hash, values=(0,) * 12)
    result = apply_attention_coordinate_mask(
        activation, mask=mask, basis=basis, spec=spec
    )
    assert torch.count_nonzero(result).item() == 0


def test_single_coordinate_changes_only_documented_coordinate() -> None:
    _, basis, spec = refined()
    activation = torch.ones((2, 2, 4, 3), dtype=torch.float32)
    values = [1] * 12
    values[5] = 0
    result = apply_attention_coordinate_mask(
        activation,
        mask=BasisMask(basis_hash=basis.basis_hash, values=tuple(values)),
        basis=basis,
        spec=spec,
    )
    assert torch.count_nonzero(result[:, :, 1, 2]).item() == 0
    assert result.sum().item() == pytest.approx(44.0)


def test_full_coordinate_group_reconstructs_parent_head_mask() -> None:
    _, basis, spec = refined()
    activation = torch.arange(48, dtype=torch.float64).reshape(1, 4, 4, 3)
    parent_mask = ComponentMask.one_head_ablated("H1")

    refined_mask = parent_head_mask_to_coordinate_mask(
        parent_attention_mask=parent_mask.attention_head_mask,
        basis=basis,
        spec=spec,
    )
    refined_result = apply_attention_coordinate_mask(
        activation,
        mask=refined_mask,
        basis=basis,
        spec=spec,
    )
    canonical_result = apply_attention_head_mask(activation, parent_mask)
    assert torch.equal(refined_result, canonical_result)


def test_identity_is_deterministic() -> None:
    _, first, _ = refined()
    _, second, _ = refined()
    assert first.basis_hash == second.basis_hash
    assert first.to_record() == second.to_record()


def test_post_projection_or_wrong_shape_rejects() -> None:
    _, basis, spec = refined()
    mask = BasisMask(basis_hash=basis.basis_hash, values=(1,) * 12)

    with pytest.raises(ValueError, match="shape"):
        apply_attention_coordinate_mask(
            torch.ones((2, 5, 12)),
            mask=mask,
            basis=basis,
            spec=spec,
        )

    with pytest.raises(ValueError, match="pre-output projection"):
        AttentionCoordinateSpec(
            layer=0,
            n_heads=4,
            d_head=3,
            hook_name="blocks.0.hook_attn_out",
        )


def test_parameter_weights_are_explicit_metadata() -> None:
    parent = canonical()
    spec = AttentionCoordinateSpec(layer=0, n_heads=4, d_head=2)
    basis = attention_coordinate_basis(
        parent_basis=parent,
        spec=spec,
        parameter_weight_per_coordinate=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    assert tuple(c.parameter_weight for c in basis.components) == (
        1, 2, 3, 4, 5, 6, 7, 8
    )


def test_wrong_basis_mask_rejects() -> None:
    _, basis, spec = refined()
    wrong = BasisMask(basis_hash="0" * 64, values=(1,) * 12)
    with pytest.raises(ValueError, match="basis identity"):
        apply_attention_coordinate_mask(
            torch.ones((1, 1, 4, 3)),
            mask=wrong,
            basis=basis,
            spec=spec,
        )

from __future__ import annotations

import copy

import pytest

from circuit_families.stage12r2.contracts import (
    BasisComponentDescriptor,
    BasisContract,
    BasisMask,
    BasisRelationship,
    validate_relationship,
    validate_technical_record_payload,
)


def component(
    component_id: str,
    *,
    coordinate: str | None = None,
    parent: str | None = None,
) -> BasisComponentDescriptor:
    return BasisComponentDescriptor(
        component_id=component_id,
        component_type="mlp_neuron",
        source_subspace="layer0.mlp.post",
        intervention_location="layer0.mlp.post",
        parameter_weight=129,
        coordinate_identity=coordinate or f"coord:{component_id}",
        parent_component_identity=parent,
    )


def basis(
    components: tuple[BasisComponentDescriptor, ...],
    *,
    family: str = "canonical_head_plus_neuron",
    label: str | None = "technical basis",
    relationship: BasisRelationship | None = None,
    grouping: str | None = None,
) -> BasisContract:
    return BasisContract(
        parent_model_identity="technical-model:fixture-v1",
        parent_component_basis_identity="stage4-component-basis:fixture-v1",
        basis_family=family,
        coordinate_definition="ordered technical coordinates",
        components=components,
        intervention_location="documented technical hook",
        intervention_semantics="binary zero-ablation at documented hook",
        parameter_weight_denominator_definition="sum declared component weights",
        raw_component_denominator_definition="component count in this basis only",
        relationship=relationship,
        grouping_partition_identity=grouping,
        display_label=label,
    )


def test_round_trip_preserves_basis_hash_and_order() -> None:
    original = basis((component("n0"), component("n1")))
    restored = BasisContract.from_record(original.to_record())
    assert restored == original
    assert restored.basis_hash == original.basis_hash


def test_reordered_components_under_stale_hash_reject() -> None:
    original = basis((component("n0"), component("n1")))
    record = original.to_record()
    record["components"] = list(reversed(record["components"]))
    with pytest.raises(ValueError, match="basis hash"):
        BasisContract.from_record(record)


def test_duplicate_component_identity_rejects() -> None:
    with pytest.raises(ValueError, match="unique"):
        basis((component("n0"), component("n0", coordinate="coord:other")))


def test_identical_display_names_do_not_imply_same_basis() -> None:
    first = basis((component("n0"),), label="same label")
    second = basis(
        (component("n0", coordinate="different-coordinate"),),
        label="same label",
    )
    assert first.display_label == second.display_label
    assert first.basis_hash != second.basis_hash


def test_wrong_basis_mask_rejects_even_when_lengths_match() -> None:
    first = basis((component("n0"), component("n1")))
    second = basis(
        (
            component("n0", coordinate="other:0"),
            component("n1", coordinate="other:1"),
        ),
    )
    mask = BasisMask(basis_hash=first.basis_hash, values=(1, 0))
    with pytest.raises(ValueError, match="basis identity"):
        mask.validate_for(second)


def test_malformed_refinement_relationship_rejects() -> None:
    parent = basis((component("n0"),))
    child = basis(
        (component("r0", parent="missing-parent"),),
        family="refined",
        relationship=BasisRelationship(
            kind="refinement",
            parent_basis_hash=parent.basis_hash,
            mapping_identity="refinement-map:v1",
        ),
        grouping="partition:v1",
    )
    with pytest.raises(ValueError, match="invalid parent"):
        validate_relationship(child, parent)


def test_parent_hash_tampering_rejects_relationship() -> None:
    parent = basis((component("n0"),))
    other_parent = basis((component("n1"),))
    child = basis(
        (component("r0", parent="n0"),),
        family="refined",
        relationship=BasisRelationship(
            kind="refinement",
            parent_basis_hash=parent.basis_hash,
            mapping_identity="refinement-map:v1",
        ),
        grouping="partition:v1",
    )
    with pytest.raises(ValueError, match="parent basis hash"):
        validate_relationship(child, other_parent)


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"artifact": "/Users/alex/private/result.json"}, "absolute private"),
        ({"scientific_data": True}, "scientific_data"),
        ({"production_eligible": True}, "production_eligible"),
    ],
)
def test_forbidden_technical_record_content_rejects(
    payload: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_technical_record_payload(payload)


def test_large_tensor_like_record_payload_rejects() -> None:
    with pytest.raises(ValueError, match="large tensor"):
        validate_technical_record_payload(
            {"matrix": [0.0] * 4097},
        )


def test_stale_hash_rejects_after_identity_mutation() -> None:
    original = basis((component("n0"),))
    record = copy.deepcopy(original.to_record())
    record["coordinate_definition"] = "mutated coordinates"
    with pytest.raises(ValueError, match="basis hash"):
        BasisContract.from_record(record)


def test_scientific_or_production_basis_contract_rejects() -> None:
    kwargs = dict(
        parent_model_identity="technical-model:fixture-v1",
        parent_component_basis_identity="stage4-component-basis:fixture-v1",
        basis_family="canonical",
        coordinate_definition="technical",
        components=(component("n0"),),
        intervention_location="technical hook",
        intervention_semantics="zero ablation",
        parameter_weight_denominator_definition="declared",
        raw_component_denominator_definition="basis count",
    )
    with pytest.raises(ValueError, match="scientific_data=false"):
        BasisContract(**kwargs, scientific_data=True)
    with pytest.raises(ValueError, match="production_eligible=false"):
        BasisContract(**kwargs, production_eligible=True)

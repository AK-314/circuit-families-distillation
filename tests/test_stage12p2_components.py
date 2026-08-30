"""Tests for Stage 12-P2 architecture-aware component accounting."""

from __future__ import annotations

from dataclasses import replace

import pytest

from circuit_families.stage12p2 import (
    ComponentContractError,
    ComponentInventory,
    ComponentMask,
    ComponentProportion,
    DenseOutputDescriptor,
    TechnicalTransformerBuilder,
    canonical_predecessor_record,
    compare_component_proportions,
    technical_transformer_record,
    transformer_component_inventory,
    validate_model_component_inventory,
)


def _compatibility() -> dict[str, object]:
    return {
        "implementation": "transformer_lens",
        "input_representation": "token_sequence",
        "output_class_count": 113,
    }


def _two_layer_record():
    return technical_transformer_record(
        family="technical",
        name="two-layer-component-fixture",
        version="v1",
        n_layers=2,
        n_ctx=3,
        d_model=32,
        n_heads=4,
        d_head=8,
        d_mlp=16,
        d_vocab=114,
        d_vocab_out=113,
        compatibility=_compatibility(),
    )


def _width_variant_record():
    return technical_transformer_record(
        family="technical",
        name="width-component-fixture",
        version="v1",
        n_layers=1,
        n_ctx=3,
        d_model=48,
        n_heads=3,
        d_head=16,
        d_mlp=24,
        d_vocab=114,
        d_vocab_out=113,
        compatibility=_compatibility(),
    )


def test_canonical_inventory_reproduces_predecessor_counts_and_hooks() -> None:
    record = canonical_predecessor_record()
    inventory = transformer_component_inventory(record)

    assert inventory.parameter_count == 227_313
    assert inventory.searchable_component_count == 516
    assert inventory.component_type_counts == {
        "attention_head": 4,
        "mlp_neuron": 512,
    }
    assert inventory.attention_head_count_by_layer == {0: 4}
    assert inventory.mlp_neuron_count_by_layer == {0: 512}

    assert inventory.component("L0:H0").hook_name == "blocks.0.attn.hook_z"
    assert inventory.component("L0:H3").index_within_layer == 3
    assert inventory.component("L0:N0").hook_name == "blocks.0.mlp.hook_post"
    assert inventory.component("L0:N511").index_within_layer == 511

    assert inventory.dense_output.output_class_count == 113
    assert inventory.dense_output.representation == "raw_final_position_logits"
    assert len(inventory.dense_output.identity_sha256) == 64
    assert inventory.dense_output.dense_output_ref.endswith("/v1")
    assert len(inventory.component_basis_compatibility_sha256) == 64


def test_multilayer_inventory_preserves_layer_identity() -> None:
    record = _two_layer_record()
    inventory = transformer_component_inventory(record)

    assert inventory.searchable_component_count == 40
    assert inventory.attention_head_count_by_layer == {
        0: 4,
        1: 4,
    }
    assert inventory.mlp_neuron_count_by_layer == {
        0: 16,
        1: 16,
    }
    assert inventory.component("L0:H0").hook_name == "blocks.0.attn.hook_z"
    assert inventory.component("L1:H0").hook_name == "blocks.1.attn.hook_z"
    assert inventory.component("L0:N0").hook_name == "blocks.0.mlp.hook_post"
    assert inventory.component("L1:N0").hook_name == "blocks.1.mlp.hook_post"

    assert "L0:H0" in inventory.component_ids
    assert "L1:H0" in inventory.component_ids
    assert inventory.component_ids.index("L0:H0") != inventory.component_ids.index("L1:H0")


def test_inventory_is_reproducible_from_same_architecture_record() -> None:
    record = _two_layer_record()

    first = transformer_component_inventory(record)
    second = transformer_component_inventory(record)

    assert first.to_mapping() == second.to_mapping()
    assert first.component_basis_compatibility_sha256 == second.component_basis_compatibility_sha256


def test_distinct_architectures_have_distinct_component_compatibility_hashes() -> None:
    first = transformer_component_inventory(_two_layer_record())
    second = transformer_component_inventory(_width_variant_record())

    assert first.architecture_ref != second.architecture_ref
    assert first.component_basis_compatibility_sha256 != second.component_basis_compatibility_sha256
    assert first.dense_output.identity_sha256 != second.dense_output.identity_sha256


def test_component_inventory_rejects_descriptor_from_another_architecture() -> None:
    first = transformer_component_inventory(_two_layer_record())
    second = transformer_component_inventory(_width_variant_record())

    foreign = replace(
        first.components[0],
        architecture_ref=second.architecture_ref,
    )
    components = (foreign,) + first.components[1:]

    with pytest.raises(
        ComponentContractError,
        match="another architecture",
    ):
        ComponentInventory(
            architecture_ref=first.architecture_ref,
            architecture_record_sha256=first.architecture_record_sha256,
            parameter_count=first.parameter_count,
            components=components,
            dense_output=first.dense_output,
            component_basis_compatibility_sha256=(first.component_basis_compatibility_sha256),
        )


def test_component_inventory_rejects_duplicate_components() -> None:
    source = transformer_component_inventory(_two_layer_record())
    duplicate = source.components[0]
    components = (
        duplicate,
        duplicate,
        *source.components[2:],
    )

    with pytest.raises(
        ComponentContractError,
        match="duplicate components|stable layer/type/index ordering",
    ):
        ComponentInventory(
            architecture_ref=source.architecture_ref,
            architecture_record_sha256=source.architecture_record_sha256,
            parameter_count=source.parameter_count,
            components=tuple(components),
            dense_output=source.dense_output,
            component_basis_compatibility_sha256=(source.component_basis_compatibility_sha256),
        )


def test_component_inventory_rejects_compatibility_hash_mismatch() -> None:
    source = transformer_component_inventory(_two_layer_record())

    with pytest.raises(
        ComponentContractError,
        match="compatibility hash mismatch",
    ):
        replace(
            source,
            component_basis_compatibility_sha256="f" * 64,
        )


def test_dense_output_descriptor_rejects_cross_architecture_inventory() -> None:
    source = transformer_component_inventory(_two_layer_record())
    foreign_dense = DenseOutputDescriptor(
        architecture_ref=_width_variant_record().architecture_ref,
        output_class_count=113,
    )

    with pytest.raises(
        ComponentContractError,
        match="dense-output descriptor belongs to another architecture",
    ):
        replace(
            source,
            dense_output=foreign_dense,
        )


def test_inventory_generation_rejects_record_component_count_tampering() -> None:
    record = _two_layer_record()
    tampered = replace(
        record,
        searchable_component_count=record.searchable_component_count + 1,
        component_type_counts={
            "attention_head": 8,
            "mlp_neuron": 33,
        },
    )

    with pytest.raises(
        ComponentContractError,
        match="searchable-component count",
    ):
        transformer_component_inventory(tampered)


@pytest.mark.parametrize(
    "values",
    [
        (1, 1, 2),
        (1, True, 0),
        (1, -1, 0),
    ],
)
def test_component_mask_rejects_nonbinary_values(
    values: tuple[object, ...],
) -> None:
    inventory = transformer_component_inventory(_width_variant_record())

    with pytest.raises(
        ComponentContractError,
        match="binary integer",
    ):
        ComponentMask(
            architecture_ref=inventory.architecture_ref,
            component_basis_compatibility_sha256=(inventory.component_basis_compatibility_sha256),
            values=values,
        )


def test_component_mask_rejects_duplicate_retained_identifiers() -> None:
    inventory = transformer_component_inventory(_width_variant_record())

    with pytest.raises(
        ComponentContractError,
        match="duplicates",
    ):
        ComponentMask.from_retained_component_ids(
            inventory,
            ["L0:H0", "L0:H0"],
        )


def test_component_mask_rejects_unknown_retained_identifier() -> None:
    inventory = transformer_component_inventory(_width_variant_record())

    with pytest.raises(
        ComponentContractError,
        match="unknown retained",
    ):
        ComponentMask.from_retained_component_ids(
            inventory,
            ["L0:H0", "L9:N999"],
        )


def test_component_mask_rejects_cross_architecture_use() -> None:
    first = transformer_component_inventory(_two_layer_record())
    second = transformer_component_inventory(_width_variant_record())
    mask = ComponentMask.all_retained(first)

    with pytest.raises(
        ComponentContractError,
        match="another architecture",
    ):
        mask.validate_against(second)


def test_component_mask_rejects_count_mismatch() -> None:
    inventory = transformer_component_inventory(_two_layer_record())
    mask = ComponentMask(
        architecture_ref=inventory.architecture_ref,
        component_basis_compatibility_sha256=(inventory.component_basis_compatibility_sha256),
        values=(1,),
    )

    with pytest.raises(
        ComponentContractError,
        match="component count mismatch",
    ):
        mask.validate_against(inventory)


def test_component_mask_rejects_basis_hash_mismatch() -> None:
    inventory = transformer_component_inventory(_two_layer_record())
    mask = ComponentMask(
        architecture_ref=inventory.architecture_ref,
        component_basis_compatibility_sha256="e" * 64,
        values=(1,) * inventory.searchable_component_count,
    )

    with pytest.raises(
        ComponentContractError,
        match="compatibility hash mismatch",
    ):
        mask.validate_against(inventory)


def test_component_proportion_carries_explicit_denominator_metadata() -> None:
    inventory = transformer_component_inventory(_two_layer_record())
    retained = inventory.component_ids[:10]
    mask = ComponentMask.from_retained_component_ids(
        inventory,
        retained,
    )
    proportion = mask.proportion(inventory)
    mapping = proportion.to_mapping()

    assert proportion.retained_component_count == 10
    assert proportion.denominator_component_count == 40
    assert proportion.value == 0.25
    assert mapping["denominator"] == {
        "component_count": 40,
        "architecture_ref": inventory.architecture_ref,
        "component_basis_compatibility_sha256": (inventory.component_basis_compatibility_sha256),
    }


def test_raw_float_component_proportions_cannot_be_compared() -> None:
    with pytest.raises(
        ComponentContractError,
        match="explicit denominator metadata",
    ):
        compare_component_proportions(0.5, 0.25)


def test_component_proportion_comparison_uses_exact_counts() -> None:
    first = ComponentProportion(
        retained_component_count=1,
        denominator_component_count=3,
        denominator_architecture_ref="technical-a/v1",
        denominator_component_basis_compatibility_sha256="a" * 64,
    )
    second = ComponentProportion(
        retained_component_count=2,
        denominator_component_count=6,
        denominator_architecture_ref="technical-b/v1",
        denominator_component_basis_compatibility_sha256="b" * 64,
    )
    third = ComponentProportion(
        retained_component_count=3,
        denominator_component_count=6,
        denominator_architecture_ref="technical-c/v1",
        denominator_component_basis_compatibility_sha256="c" * 64,
    )

    assert compare_component_proportions(first, second) == 0
    assert compare_component_proportions(first, third) == -1
    assert compare_component_proportions(third, first) == 1


@pytest.mark.parametrize(
    "record_factory",
    [
        canonical_predecessor_record,
        _two_layer_record,
        _width_variant_record,
    ],
)
def test_built_models_validate_against_reproducible_inventory(
    record_factory,
) -> None:
    record = record_factory()
    inventory = transformer_component_inventory(record)

    if record.family == "predecessor":
        from circuit_families.stage12p2 import CanonicalPredecessorBuilder

        model = CanonicalPredecessorBuilder().build(
            record=record,
            seed=7,
            device="cpu",
        )
    else:
        model = TechnicalTransformerBuilder().build(
            record=record,
            seed=7,
            device="cpu",
        )

    validate_model_component_inventory(
        model=model,
        record=record,
        inventory=inventory,
    )


def test_model_inventory_validation_rejects_another_architecture() -> None:
    record = _two_layer_record()
    model = TechnicalTransformerBuilder().build(
        record=record,
        seed=2,
        device="cpu",
    )
    foreign_inventory = transformer_component_inventory(_width_variant_record())

    with pytest.raises(
        ComponentContractError,
        match="another architecture",
    ):
        validate_model_component_inventory(
            model=model,
            record=record,
            inventory=foreign_inventory,
        )


def test_component_identifiers_are_not_flattened_across_layers() -> None:
    inventory = transformer_component_inventory(_two_layer_record())

    descriptors = {component.component_id: component for component in inventory.components}

    assert descriptors["L0:H0"].layer_index == 0
    assert descriptors["L1:H0"].layer_index == 1
    assert descriptors["L0:N0"].layer_index == 0
    assert descriptors["L1:N0"].layer_index == 1

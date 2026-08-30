from __future__ import annotations

import copy

import pytest

from circuit_families.stage12r2.accounting import account_basis_mask
from circuit_families.stage12r2.blocks import (
    BalancedBlockPartition,
    balanced_block_basis,
    build_balanced_partition,
    expand_block_mask_to_parent_values,
)
from circuit_families.stage12r2.canonical import canonical_basis_contract
from circuit_families.stage12r2.contracts import BasisMask


def parent_basis():
    return canonical_basis_contract(
        parent_model_identity="technical-model:fixture-v1",
        parent_component_basis_identity="stage4-component-basis:v1",
        attention_parameter_weight=32,
        mlp_parameter_weight=129,
    )


def partition(seed: int = 17, blocks: int = 7):
    parent = parent_basis()
    return parent, build_balanced_partition(
        parent_basis=parent,
        layer_identity="blocks.0.mlp.hook_post",
        partition_seed=seed,
        block_count=blocks,
    )


def test_partition_is_complete_nonoverlapping_and_balanced() -> None:
    _, part = partition()
    flattened = [item for block in part.blocks for item in block]
    assert len(flattened) == 512
    assert len(set(flattened)) == 512
    assert set(flattened) == set(part.eligible_neuron_ids)
    sizes = [len(block) for block in part.blocks]
    assert max(sizes) - min(sizes) <= 1


def test_partition_is_deterministic() -> None:
    _, first = partition(seed=123, blocks=11)
    _, second = partition(seed=123, blocks=11)
    assert first.partition_hash == second.partition_hash
    assert first.blocks == second.blocks


def test_seed_block_count_layer_and_parent_affect_identity() -> None:
    parent = parent_basis()
    first = build_balanced_partition(
        parent_basis=parent,
        layer_identity="blocks.0.mlp.hook_post",
        partition_seed=1,
        block_count=8,
    )
    second = build_balanced_partition(
        parent_basis=parent,
        layer_identity="blocks.0.mlp.hook_post",
        partition_seed=2,
        block_count=8,
    )
    third = build_balanced_partition(
        parent_basis=parent,
        layer_identity="blocks.0.mlp.hook_post",
        partition_seed=1,
        block_count=9,
    )
    assert len({first.partition_hash, second.partition_hash, third.partition_hash}) == 3


@pytest.mark.parametrize("count", [0, 513])
def test_impossible_block_count_rejects(count: int) -> None:
    parent = parent_basis()
    with pytest.raises(ValueError, match="impossible"):
        build_balanced_partition(
            parent_basis=parent,
            layer_identity="blocks.0.mlp.hook_post",
            partition_seed=1,
            block_count=count,
        )


def test_tampered_partition_membership_hash_rejects() -> None:
    _, original = partition()
    record = copy.deepcopy(original.to_record())
    record["blocks"][0][0], record["blocks"][1][0] = (
        record["blocks"][1][0],
        record["blocks"][0][0],
    )
    with pytest.raises(ValueError, match="partition hash"):
        BalancedBlockPartition.from_record(record)


def test_all_on_block_mask_expands_to_all_on_parent() -> None:
    parent, part = partition()
    grouped = balanced_block_basis(parent_basis=parent, partition=part)
    mask = BasisMask(
        basis_hash=grouped.basis_hash,
        values=(1,) * grouped.component_count,
    )
    assert expand_block_mask_to_parent_values(
        block_mask=mask,
        block_basis=grouped,
        parent_basis=parent,
        partition=part,
    ) == (1,) * 516


def test_block_mask_expansion_matches_exact_membership() -> None:
    parent, part = partition(blocks=4)
    grouped = balanced_block_basis(parent_basis=parent, partition=part)

    values = (1, 1, 1, 1, 0, 1, 0, 1)
    expanded = expand_block_mask_to_parent_values(
        block_mask=BasisMask(basis_hash=grouped.basis_hash, values=values),
        block_basis=grouped,
        parent_basis=parent,
        partition=part,
    )

    assert expanded[:4] == (1, 1, 1, 1)
    retained_neurons = {
        neuron
        for block_index, block in enumerate(part.blocks)
        if values[4 + block_index] == 1
        for neuron in block
    }
    actual_retained = {
        component.component_id
        for component, bit in zip(parent.components[4:], expanded[4:], strict=True)
        if bit
    }
    assert actual_retained == retained_neurons


def test_raw_parameter_type_and_parent_accounting() -> None:
    parent, part = partition(blocks=4)
    grouped = balanced_block_basis(parent_basis=parent, partition=part)

    values = (1, 0, 1, 0, 1, 0, 0, 0)
    result = account_basis_mask(
        basis=grouped,
        mask=BasisMask(basis_hash=grouped.basis_hash, values=values),
        partition=part,
    )

    assert result.raw_retained_count == 3
    assert result.raw_total_count == 8
    assert result.raw_retained_proportion == pytest.approx(3 / 8)

    rows = {row.component_type: row for row in result.type_accounting}
    assert rows["attention_head"].retained_count == 2
    assert rows["attention_head"].total_count == 4
    assert rows["mlp_block"].retained_count == 1
    assert rows["mlp_block"].total_count == 4

    assert result.parameter_weight_total == parent.parameter_weight_denominator
    assert result.parameter_weight_retained_proportion != result.raw_retained_proportion
    assert result.parent_neuron_total_count == 512
    assert result.parent_neuron_retained_count == len(part.blocks[0])


def test_accounting_rejects_stale_partition_identity() -> None:
    parent, first = partition(seed=1, blocks=4)
    _, second = partition(seed=2, blocks=4)
    grouped = balanced_block_basis(parent_basis=parent, partition=first)
    mask = BasisMask(
        basis_hash=grouped.basis_hash,
        values=(1,) * grouped.component_count,
    )
    with pytest.raises(ValueError, match="partition identity"):
        account_basis_mask(basis=grouped, mask=mask, partition=second)

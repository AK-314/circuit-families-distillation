"""Basis-aware technical retention accounting."""

from __future__ import annotations

from dataclasses import dataclass

from circuit_families.stage12r2.blocks import BalancedBlockPartition
from circuit_families.stage12r2.contracts import BasisContract, BasisMask


@dataclass(frozen=True)
class TypeAccounting:
    component_type: str
    retained_count: int
    total_count: int

    @property
    def retained_proportion(self) -> float:
        return self.retained_count / self.total_count


@dataclass(frozen=True)
class BasisAccounting:
    basis_hash: str
    raw_retained_count: int
    raw_total_count: int
    parameter_weight_retained: int
    parameter_weight_total: int
    parameter_weight_denominator_definition: str
    type_accounting: tuple[TypeAccounting, ...]
    parent_neuron_retained_count: int | None = None
    parent_neuron_total_count: int | None = None

    @property
    def raw_retained_proportion(self) -> float:
        return self.raw_retained_count / self.raw_total_count

    @property
    def parameter_weight_retained_proportion(self) -> float:
        if self.parameter_weight_total <= 0:
            raise ValueError("parameter-weight denominator must be positive")
        return self.parameter_weight_retained / self.parameter_weight_total

    @property
    def parent_neuron_retained_proportion(self) -> float | None:
        if self.parent_neuron_total_count is None:
            return None
        if self.parent_neuron_retained_count is None:
            raise ValueError("parent-neuron accounting is incomplete")
        return self.parent_neuron_retained_count / self.parent_neuron_total_count


def account_basis_mask(
    *,
    basis: BasisContract,
    mask: BasisMask,
    partition: BalancedBlockPartition | None = None,
) -> BasisAccounting:
    """Compute raw, parameter, type, and optional parent-neuron accounting."""
    mask.validate_for(basis)

    retained_count = sum(mask.values)
    retained_weight = sum(
        component.parameter_weight * value
        for component, value in zip(basis.components, mask.values, strict=True)
    )
    total_weight = sum(component.parameter_weight for component in basis.components)

    component_types = sorted({component.component_type for component in basis.components})
    type_rows: list[TypeAccounting] = []
    for component_type in component_types:
        indices = [
            index
            for index, component in enumerate(basis.components)
            if component.component_type == component_type
        ]
        type_rows.append(
            TypeAccounting(
                component_type=component_type,
                retained_count=sum(mask.values[index] for index in indices),
                total_count=len(indices),
            )
        )

    parent_retained: int | None = None
    parent_total: int | None = None
    if partition is not None:
        if basis.grouping_partition_identity != partition.partition_hash:
            raise ValueError("partition identity does not match grouped basis")

        block_indices = [
            index
            for index, component in enumerate(basis.components)
            if component.component_type == "mlp_block"
        ]
        if len(block_indices) != partition.block_count:
            raise ValueError("grouped basis block count does not match partition")

        parent_total = len(partition.eligible_neuron_ids)
        parent_retained = sum(
            len(partition.blocks[block_number]) * mask.values[basis_index]
            for block_number, basis_index in enumerate(block_indices)
        )

    return BasisAccounting(
        basis_hash=basis.basis_hash,
        raw_retained_count=retained_count,
        raw_total_count=basis.component_count,
        parameter_weight_retained=retained_weight,
        parameter_weight_total=total_weight,
        parameter_weight_denominator_definition=(
            basis.parameter_weight_denominator_definition
        ),
        type_accounting=tuple(type_rows),
        parent_neuron_retained_count=parent_retained,
        parent_neuron_total_count=parent_total,
    )

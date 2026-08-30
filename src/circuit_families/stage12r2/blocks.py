"""Deterministic balanced coarsenings of eligible MLP-neuron coordinates."""

from __future__ import annotations

from dataclasses import dataclass

from circuit_families.stage12r2.contracts import (
    BasisComponentDescriptor,
    BasisContract,
    BasisMask,
    BasisRelationship,
    canonical_sha256,
)

BLOCK_PARTITION_VERSION = "stage12r2-balanced-block-partition/v1"
BLOCK_BASIS_FAMILY = "canonical_heads_plus_balanced_mlp_blocks"


@dataclass(frozen=True)
class BalancedBlockPartition:
    parent_basis_hash: str
    parent_model_identity: str
    layer_identity: str
    partition_seed: int
    block_count: int
    eligible_neuron_ids: tuple[str, ...]
    blocks: tuple[tuple[str, ...], ...]
    policy_version: str = BLOCK_PARTITION_VERSION

    def __post_init__(self) -> None:
        if self.policy_version != BLOCK_PARTITION_VERSION:
            raise ValueError("unsupported balanced-block partition version")
        if not isinstance(self.partition_seed, int) or self.partition_seed < 0:
            raise ValueError("partition_seed must be a non-negative integer")
        if self.block_count <= 0:
            raise ValueError("block_count must be positive")
        if self.block_count > len(self.eligible_neuron_ids):
            raise ValueError("block_count cannot exceed eligible neuron count")
        if len(self.blocks) != self.block_count:
            raise ValueError("block count does not match partition membership")
        if any(not block for block in self.blocks):
            raise ValueError("balanced blocks must not be empty")

        flattened = tuple(item for block in self.blocks for item in block)
        if len(flattened) != len(set(flattened)):
            raise ValueError("block membership must not overlap or duplicate")
        if set(flattened) != set(self.eligible_neuron_ids):
            raise ValueError("blocks must cover every eligible neuron exactly once")

        sizes = [len(block) for block in self.blocks]
        if max(sizes) - min(sizes) > 1:
            raise ValueError("balanced block sizes must differ by at most one")

    def identity_payload(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "parent_basis_hash": self.parent_basis_hash,
            "parent_model_identity": self.parent_model_identity,
            "layer_identity": self.layer_identity,
            "partition_seed": self.partition_seed,
            "block_count": self.block_count,
            "eligible_neuron_ids": list(self.eligible_neuron_ids),
            "blocks": [list(block) for block in self.blocks],
        }

    @property
    def partition_hash(self) -> str:
        return canonical_sha256(self.identity_payload())

    def to_record(self) -> dict[str, object]:
        record = self.identity_payload()
        record["partition_hash"] = self.partition_hash
        return record

    @classmethod
    def from_record(cls, record: dict[str, object]) -> BalancedBlockPartition:
        supplied_hash = str(record["partition_hash"])
        blocks_raw = record["blocks"]
        eligible_raw = record["eligible_neuron_ids"]
        if not isinstance(blocks_raw, list) or not isinstance(eligible_raw, list):
            raise ValueError("partition membership must be JSON arrays")

        obj = cls(
            policy_version=str(record["policy_version"]),
            parent_basis_hash=str(record["parent_basis_hash"]),
            parent_model_identity=str(record["parent_model_identity"]),
            layer_identity=str(record["layer_identity"]),
            partition_seed=int(record["partition_seed"]),
            block_count=int(record["block_count"]),
            eligible_neuron_ids=tuple(str(x) for x in eligible_raw),
            blocks=tuple(tuple(str(x) for x in block) for block in blocks_raw),
        )
        if supplied_hash != obj.partition_hash:
            raise ValueError("partition hash does not match partition contents")
        return obj


def build_balanced_partition(
    *,
    parent_basis: BasisContract,
    layer_identity: str,
    partition_seed: int,
    block_count: int,
) -> BalancedBlockPartition:
    """Build a stable seeded partition from parent-basis neuron identities."""
    eligible = tuple(
        component.component_id
        for component in parent_basis.components
        if component.component_type == "mlp_neuron"
        and component.source_subspace == layer_identity
    )
    if not eligible:
        raise ValueError("no eligible MLP neurons found for layer identity")
    if block_count <= 0 or block_count > len(eligible):
        raise ValueError("impossible block_count for eligible neuron set")

    ranked = sorted(
        eligible,
        key=lambda component_id: (
            canonical_sha256(
                {
                    "policy_version": BLOCK_PARTITION_VERSION,
                    "parent_model_identity": parent_basis.parent_model_identity,
                    "parent_basis_hash": parent_basis.basis_hash,
                    "layer_identity": layer_identity,
                    "partition_seed": partition_seed,
                    "block_count": block_count,
                    "component_id": component_id,
                }
            ),
            component_id,
        ),
    )

    quotient, remainder = divmod(len(ranked), block_count)
    blocks: list[tuple[str, ...]] = []
    start = 0
    for block_index in range(block_count):
        size = quotient + (1 if block_index < remainder else 0)
        blocks.append(tuple(ranked[start : start + size]))
        start += size

    return BalancedBlockPartition(
        parent_basis_hash=parent_basis.basis_hash,
        parent_model_identity=parent_basis.parent_model_identity,
        layer_identity=layer_identity,
        partition_seed=partition_seed,
        block_count=block_count,
        eligible_neuron_ids=eligible,
        blocks=tuple(blocks),
    )


def balanced_block_basis(
    *,
    parent_basis: BasisContract,
    partition: BalancedBlockPartition,
) -> BasisContract:
    """Keep parent heads intact while coarsening MLP neurons into fixed blocks."""
    if partition.parent_basis_hash != parent_basis.basis_hash:
        raise ValueError("partition parent basis hash does not match parent basis")
    if partition.parent_model_identity != parent_basis.parent_model_identity:
        raise ValueError("partition model identity does not match parent basis")

    parent_by_id = {component.component_id: component for component in parent_basis.components}
    components: list[BasisComponentDescriptor] = []

    for component in parent_basis.components:
        if component.component_type == "attention_head":
            components.append(
                BasisComponentDescriptor(
                    component_id=component.component_id,
                    component_type="attention_head",
                    source_subspace=component.source_subspace,
                    intervention_location=component.intervention_location,
                    parameter_weight=component.parameter_weight,
                    coordinate_identity=component.coordinate_identity,
                    parent_component_identity=component.component_id,
                )
            )

    for index, membership in enumerate(partition.blocks):
        weight = sum(parent_by_id[item].parameter_weight for item in membership)
        components.append(
            BasisComponentDescriptor(
                component_id=f"MLP_BLOCK_{index:04d}",
                component_type="mlp_block",
                source_subspace=partition.layer_identity,
                intervention_location=partition.layer_identity,
                parameter_weight=weight,
                coordinate_identity=(
                    f"partition={partition.partition_hash}/block={index}/"
                    f"members={','.join(membership)}"
                ),
            )
        )

    return BasisContract(
        parent_model_identity=parent_basis.parent_model_identity,
        parent_component_basis_identity=parent_basis.parent_component_basis_identity,
        basis_family=BLOCK_BASIS_FAMILY,
        coordinate_definition="canonical heads plus seeded balanced MLP-neuron blocks/v1",
        components=tuple(components),
        intervention_location="canonical attention hook plus canonical MLP hook",
        intervention_semantics=(
            "attention heads retain canonical mask semantics; each MLP block bit "
            "expands to all member parent-neuron bits"
        ),
        parameter_weight_denominator_definition=(
            "sum declared retained head weights and member-neuron weights"
        ),
        raw_component_denominator_definition=(
            "canonical attention-head count plus balanced-block count"
        ),
        relationship=BasisRelationship(
            kind="coarsening",
            parent_basis_hash=parent_basis.basis_hash,
            mapping_identity=partition.partition_hash,
        ),
        grouping_partition_identity=partition.partition_hash,
        display_label="canonical heads plus balanced MLP blocks",
    )


def expand_block_mask_to_parent_values(
    *,
    block_mask: BasisMask,
    block_basis: BasisContract,
    parent_basis: BasisContract,
    partition: BalancedBlockPartition,
) -> tuple[int, ...]:
    """Expand grouped-basis bits into the exact ordered parent-basis mask."""
    block_mask.validate_for(block_basis)
    if block_basis.grouping_partition_identity != partition.partition_hash:
        raise ValueError("block basis carries a stale partition identity")
    if partition.parent_basis_hash != parent_basis.basis_hash:
        raise ValueError("partition does not belong to supplied parent basis")

    head_components = [
        component
        for component in parent_basis.components
        if component.component_type == "attention_head"
    ]
    if len(block_mask.values) != len(head_components) + partition.block_count:
        raise ValueError("block mask shape is incompatible with partition")

    retained: dict[str, int] = {}
    for component, value in zip(
        head_components,
        block_mask.values[: len(head_components)],
        strict=True,
    ):
        retained[component.component_id] = value

    block_values = block_mask.values[len(head_components) :]
    for block, value in zip(partition.blocks, block_values, strict=True):
        for neuron_id in block:
            retained[neuron_id] = value

    return tuple(retained[component.component_id] for component in parent_basis.components)

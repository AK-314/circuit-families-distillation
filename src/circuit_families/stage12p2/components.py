"""Architecture-aware searchable-component accounting for Stage 12-P2."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import torch

from circuit_families.models.transformer import parameter_count
from circuit_families.stage12p2.architecture import ArchitectureRecord

COMPONENT_DESCRIPTOR_SCHEMA_VERSION = "stage12p2-component-descriptor/v1"
COMPONENT_INVENTORY_SCHEMA_VERSION = "stage12p2-component-inventory/v1"
COMPONENT_MASK_SCHEMA_VERSION = "stage12p2-component-mask/v1"
DENSE_OUTPUT_DESCRIPTOR_SCHEMA_VERSION = "stage12p2-dense-output/v1"
COMPONENT_PROPORTION_SCHEMA_VERSION = "stage12p2-component-proportion/v1"

_COMPONENT_TYPE_VALUES = frozenset({"attention_head", "mlp_neuron"})
_INTERVENTION_KIND = "activation_zeroing"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ComponentContractError(ValueError):
    """Raised when Stage 12-P2 component accounting is inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ComponentContractError(
            "component metadata must contain only finite JSON values"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ComponentContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComponentContractError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    value = _require_nonnegative_int(value, name=name)
    if value == 0:
        raise ComponentContractError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class DenseOutputDescriptor:
    """Identity of the architecture's intact dense-output representation."""

    architecture_ref: str
    output_class_count: int
    representation: str = "raw_final_position_logits"
    position_semantics: str = "final_sequence_position"
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.architecture_ref, str) or not self.architecture_ref:
            raise ComponentContractError("dense-output architecture_ref must be non-empty")
        _require_positive_int(
            self.output_class_count,
            name="dense-output output_class_count",
        )
        if self.representation != "raw_final_position_logits":
            raise ComponentContractError("unsupported dense-output representation")
        if self.position_semantics != "final_sequence_position":
            raise ComponentContractError("unsupported dense-output position semantics")
        if self.scientific_data is not False:
            raise ComponentContractError(
                "dense-output descriptor must declare scientific_data=false"
            )
        if self.production_eligible is not False:
            raise ComponentContractError(
                "dense-output descriptor must declare production_eligible=false"
            )

    def identity_material(self) -> dict[str, Any]:
        return {
            "schema_version": DENSE_OUTPUT_DESCRIPTOR_SCHEMA_VERSION,
            "architecture_ref": self.architecture_ref,
            "output_class_count": self.output_class_count,
            "representation": self.representation,
            "position_semantics": self.position_semantics,
            "scientific_data": False,
            "production_eligible": False,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.identity_material())

    @property
    def dense_output_ref(self) -> str:
        return f"stage12p2-dense-output-{self.identity_sha256[:16]}/v1"

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.identity_material(),
            "dense_output_ref": self.dense_output_ref,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True)
class ComponentDescriptor:
    """One searchable component with explicit architecture and layer identity."""

    architecture_ref: str
    component_type: str
    layer_index: int
    index_within_layer: int
    hook_name: str
    activation_axis: int
    intervention_kind: str = _INTERVENTION_KIND
    parameters_per_component: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.architecture_ref, str) or not self.architecture_ref:
            raise ComponentContractError("component architecture_ref must be non-empty")
        if self.component_type not in _COMPONENT_TYPE_VALUES:
            raise ComponentContractError(f"unsupported component_type: {self.component_type!r}")
        _require_nonnegative_int(
            self.layer_index,
            name="component layer_index",
        )
        _require_nonnegative_int(
            self.index_within_layer,
            name="component index_within_layer",
        )
        if not isinstance(self.hook_name, str) or not self.hook_name:
            raise ComponentContractError("component hook_name must be non-empty")
        _require_nonnegative_int(
            self.activation_axis,
            name="component activation_axis",
        )
        if self.intervention_kind != _INTERVENTION_KIND:
            raise ComponentContractError("unsupported component intervention_kind")
        if self.parameters_per_component is not None:
            _require_positive_int(
                self.parameters_per_component,
                name="parameters_per_component",
            )

    @property
    def component_id(self) -> str:
        prefix = "H" if self.component_type == "attention_head" else "N"
        return f"L{self.layer_index}:{prefix}{self.index_within_layer}"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
            "architecture_ref": self.architecture_ref,
            "component_id": self.component_id,
            "component_type": self.component_type,
            "layer_index": self.layer_index,
            "index_within_layer": self.index_within_layer,
            "hook_name": self.hook_name,
            "activation_axis": self.activation_axis,
            "intervention_kind": self.intervention_kind,
            "parameters_per_component": self.parameters_per_component,
        }


@dataclass(frozen=True)
class ComponentInventory:
    """Reproducible ordered searchable-component inventory for one architecture."""

    architecture_ref: str
    architecture_record_sha256: str
    parameter_count: int
    components: tuple[ComponentDescriptor, ...]
    dense_output: DenseOutputDescriptor
    component_basis_compatibility_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.architecture_ref, str) or not self.architecture_ref:
            raise ComponentContractError("inventory architecture_ref must be non-empty")
        _require_sha256(
            self.architecture_record_sha256,
            name="architecture_record_sha256",
        )
        _require_positive_int(
            self.parameter_count,
            name="inventory parameter_count",
        )

        if not isinstance(self.components, tuple) or not self.components:
            raise ComponentContractError("inventory components must be a non-empty tuple")

        identifiers: list[str] = []
        previous_key: tuple[int, int, int] | None = None

        type_order = {
            "attention_head": 0,
            "mlp_neuron": 1,
        }

        for component in self.components:
            if not isinstance(component, ComponentDescriptor):
                raise ComponentContractError(
                    "inventory components must be ComponentDescriptor values"
                )
            if component.architecture_ref != self.architecture_ref:
                raise ComponentContractError("component descriptor belongs to another architecture")

            identifiers.append(component.component_id)
            key = (
                component.layer_index,
                type_order[component.component_type],
                component.index_within_layer,
            )
            if previous_key is not None and key <= previous_key:
                raise ComponentContractError("components must use stable layer/type/index ordering")
            previous_key = key

        if len(set(identifiers)) != len(identifiers):
            raise ComponentContractError("inventory contains duplicate components")

        if self.dense_output.architecture_ref != self.architecture_ref:
            raise ComponentContractError("dense-output descriptor belongs to another architecture")

        declared_hash = _require_sha256(
            self.component_basis_compatibility_sha256,
            name="component_basis_compatibility_sha256",
        )
        expected_hash = _inventory_compatibility_sha256(
            architecture_ref=self.architecture_ref,
            architecture_record_sha256=self.architecture_record_sha256,
            parameter_count=self.parameter_count,
            components=self.components,
            dense_output=self.dense_output,
        )
        if declared_hash != expected_hash:
            raise ComponentContractError("component-basis compatibility hash mismatch")

    @property
    def searchable_component_count(self) -> int:
        return len(self.components)

    @property
    def component_type_counts(self) -> dict[str, int]:
        counts = Counter(component.component_type for component in self.components)
        return {
            component_type: counts.get(component_type, 0)
            for component_type in sorted(_COMPONENT_TYPE_VALUES)
        }

    @property
    def attention_head_count_by_layer(self) -> dict[int, int]:
        return _count_by_layer(
            self.components,
            component_type="attention_head",
        )

    @property
    def mlp_neuron_count_by_layer(self) -> dict[int, int]:
        return _count_by_layer(
            self.components,
            component_type="mlp_neuron",
        )

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(component.component_id for component in self.components)

    def component(self, component_id: str) -> ComponentDescriptor:
        matches = [
            component for component in self.components if component.component_id == component_id
        ]
        if not matches:
            raise ComponentContractError(f"unknown component identifier: {component_id}")
        if len(matches) != 1:
            raise ComponentContractError(f"duplicate component identifier: {component_id}")
        return matches[0]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": COMPONENT_INVENTORY_SCHEMA_VERSION,
            "architecture_ref": self.architecture_ref,
            "architecture_record_sha256": self.architecture_record_sha256,
            "parameter_count": self.parameter_count,
            "searchable_component_count": self.searchable_component_count,
            "component_type_counts": self.component_type_counts,
            "attention_head_count_by_layer": {
                str(key): value for key, value in self.attention_head_count_by_layer.items()
            },
            "mlp_neuron_count_by_layer": {
                str(key): value for key, value in self.mlp_neuron_count_by_layer.items()
            },
            "components": [component.to_mapping() for component in self.components],
            "dense_output": self.dense_output.to_mapping(),
            "component_basis_compatibility_sha256": (self.component_basis_compatibility_sha256),
            "scientific_data": False,
            "production_eligible": False,
        }


def _count_by_layer(
    components: Sequence[ComponentDescriptor],
    *,
    component_type: str,
) -> dict[int, int]:
    counts: Counter[int] = Counter(
        component.layer_index
        for component in components
        if component.component_type == component_type
    )
    return {layer: counts[layer] for layer in sorted(counts)}


def _inventory_compatibility_material(
    *,
    architecture_ref: str,
    architecture_record_sha256: str,
    parameter_count: int,
    components: Sequence[ComponentDescriptor],
    dense_output: DenseOutputDescriptor,
) -> dict[str, Any]:
    return {
        "schema_version": COMPONENT_INVENTORY_SCHEMA_VERSION,
        "architecture_ref": architecture_ref,
        "architecture_record_sha256": architecture_record_sha256,
        "parameter_count": parameter_count,
        "components": [component.to_mapping() for component in components],
        "dense_output": dense_output.to_mapping(),
    }


def _inventory_compatibility_sha256(
    *,
    architecture_ref: str,
    architecture_record_sha256: str,
    parameter_count: int,
    components: Sequence[ComponentDescriptor],
    dense_output: DenseOutputDescriptor,
) -> str:
    return _canonical_sha256(
        _inventory_compatibility_material(
            architecture_ref=architecture_ref,
            architecture_record_sha256=architecture_record_sha256,
            parameter_count=parameter_count,
            components=components,
            dense_output=dense_output,
        )
    )


def transformer_component_inventory(
    record: ArchitectureRecord,
) -> ComponentInventory:
    """Construct canonical layer-aware head/neuron metadata from a record."""
    if not isinstance(record, ArchitectureRecord):
        raise ComponentContractError("record must be ArchitectureRecord")

    required_dimensions = {
        "n_layers",
        "n_ctx",
        "d_model",
        "n_heads",
        "d_head",
        "d_mlp",
        "d_vocab",
        "d_vocab_out",
    }
    if set(record.dimensions) != required_dimensions:
        raise ComponentContractError(
            "transformer component inventory requires exact transformer dimensions"
        )

    dimensions = record.dimensions
    n_layers = dimensions["n_layers"]
    n_heads = dimensions["n_heads"]
    d_mlp = dimensions["d_mlp"]

    components: list[ComponentDescriptor] = []

    for layer_index in range(n_layers):
        for head_index in range(n_heads):
            components.append(
                ComponentDescriptor(
                    architecture_ref=record.architecture_ref,
                    component_type="attention_head",
                    layer_index=layer_index,
                    index_within_layer=head_index,
                    hook_name=f"blocks.{layer_index}.attn.hook_z",
                    activation_axis=2,
                )
            )

        for neuron_index in range(d_mlp):
            components.append(
                ComponentDescriptor(
                    architecture_ref=record.architecture_ref,
                    component_type="mlp_neuron",
                    layer_index=layer_index,
                    index_within_layer=neuron_index,
                    hook_name=f"blocks.{layer_index}.mlp.hook_post",
                    activation_axis=2,
                )
            )

    component_tuple = tuple(components)

    if len(component_tuple) != record.searchable_component_count:
        raise ComponentContractError(
            "record searchable-component count does not match generated inventory"
        )

    generated_counts = Counter(component.component_type for component in component_tuple)
    if dict(record.component_type_counts) != {
        key: generated_counts.get(key, 0) for key in record.component_type_counts
    }:
        raise ComponentContractError(
            "record component-type counts do not match generated inventory"
        )

    architecture_mapping = record.to_mapping()
    architecture_record_sha256 = architecture_mapping["record_sha256"]
    dense_output = DenseOutputDescriptor(
        architecture_ref=record.architecture_ref,
        output_class_count=dimensions["d_vocab_out"],
    )

    compatibility_hash = _inventory_compatibility_sha256(
        architecture_ref=record.architecture_ref,
        architecture_record_sha256=architecture_record_sha256,
        parameter_count=record.parameter_count,
        components=component_tuple,
        dense_output=dense_output,
    )

    return ComponentInventory(
        architecture_ref=record.architecture_ref,
        architecture_record_sha256=architecture_record_sha256,
        parameter_count=record.parameter_count,
        components=component_tuple,
        dense_output=dense_output,
        component_basis_compatibility_sha256=compatibility_hash,
    )


def validate_model_component_inventory(
    *,
    model: torch.nn.Module,
    record: ArchitectureRecord,
    inventory: ComponentInventory,
) -> None:
    """Validate an intact built model against architecture/component metadata."""
    if not isinstance(model, torch.nn.Module):
        raise ComponentContractError("model must be torch.nn.Module")
    if not isinstance(record, ArchitectureRecord):
        raise ComponentContractError("record must be ArchitectureRecord")
    if not isinstance(inventory, ComponentInventory):
        raise ComponentContractError("inventory must be ComponentInventory")

    if inventory.architecture_ref != record.architecture_ref:
        raise ComponentContractError("inventory belongs to another architecture")
    if inventory.architecture_record_sha256 != record.to_mapping()["record_sha256"]:
        raise ComponentContractError("inventory architecture-record hash mismatch")
    if inventory.parameter_count != record.parameter_count:
        raise ComponentContractError("inventory parameter count mismatch")
    if parameter_count(model) != record.parameter_count:
        raise ComponentContractError("model parameter count mismatch")

    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise ComponentContractError("model does not expose architecture cfg")

    for field in (
        "n_layers",
        "n_ctx",
        "d_model",
        "n_heads",
        "d_head",
        "d_mlp",
        "d_vocab",
        "d_vocab_out",
    ):
        actual = getattr(cfg, field, None)
        expected = record.dimensions[field]
        if actual != expected:
            raise ComponentContractError(f"model architecture mismatch for {field}")

    hook_dict = getattr(model, "hook_dict", None)
    if not isinstance(hook_dict, Mapping):
        raise ComponentContractError("model does not expose hook_dict")

    required_hooks = {component.hook_name for component in inventory.components}
    missing = sorted(required_hooks.difference(hook_dict))
    if missing:
        raise ComponentContractError("model is missing component hooks: " + ", ".join(missing))


@dataclass(frozen=True)
class ComponentMask:
    """Binary mask bound to one exact architecture/component inventory."""

    architecture_ref: str
    component_basis_compatibility_sha256: str
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.architecture_ref, str) or not self.architecture_ref:
            raise ComponentContractError("mask architecture_ref must be non-empty")
        _require_sha256(
            self.component_basis_compatibility_sha256,
            name="mask component_basis_compatibility_sha256",
        )
        if not isinstance(self.values, tuple) or not self.values:
            raise ComponentContractError("mask values must be a non-empty tuple")
        for index, value in enumerate(self.values):
            if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
                raise ComponentContractError(
                    f"mask value at index {index} must be binary integer 0 or 1"
                )

    @classmethod
    def all_retained(
        cls,
        inventory: ComponentInventory,
    ) -> ComponentMask:
        return cls(
            architecture_ref=inventory.architecture_ref,
            component_basis_compatibility_sha256=(inventory.component_basis_compatibility_sha256),
            values=(1,) * inventory.searchable_component_count,
        )

    @classmethod
    def from_retained_component_ids(
        cls,
        inventory: ComponentInventory,
        retained_component_ids: Sequence[str],
    ) -> ComponentMask:
        if isinstance(retained_component_ids, (str, bytes)) or not isinstance(
            retained_component_ids,
            Sequence,
        ):
            raise ComponentContractError("retained_component_ids must be a sequence")

        values = tuple(retained_component_ids)
        if any(not isinstance(value, str) for value in values):
            raise ComponentContractError("retained component identifiers must be strings")
        if len(set(values)) != len(values):
            raise ComponentContractError("retained component identifiers contain duplicates")

        known = set(inventory.component_ids)
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ComponentContractError(
                "unknown retained component identifiers: " + ", ".join(unknown)
            )

        retained = set(values)
        return cls(
            architecture_ref=inventory.architecture_ref,
            component_basis_compatibility_sha256=(inventory.component_basis_compatibility_sha256),
            values=tuple(int(component_id in retained) for component_id in inventory.component_ids),
        )

    def validate_against(
        self,
        inventory: ComponentInventory,
    ) -> None:
        if self.architecture_ref != inventory.architecture_ref:
            raise ComponentContractError("mask belongs to another architecture")
        if (
            self.component_basis_compatibility_sha256
            != inventory.component_basis_compatibility_sha256
        ):
            raise ComponentContractError("mask component-basis compatibility hash mismatch")
        if len(self.values) != inventory.searchable_component_count:
            raise ComponentContractError("mask component count mismatch")

    def proportion(
        self,
        inventory: ComponentInventory,
    ) -> ComponentProportion:
        self.validate_against(inventory)
        return ComponentProportion(
            retained_component_count=sum(self.values),
            denominator_component_count=inventory.searchable_component_count,
            denominator_architecture_ref=inventory.architecture_ref,
            denominator_component_basis_compatibility_sha256=(
                inventory.component_basis_compatibility_sha256
            ),
        )

    def to_mapping(
        self,
        inventory: ComponentInventory,
    ) -> dict[str, Any]:
        self.validate_against(inventory)
        proportion = self.proportion(inventory)
        return {
            "schema_version": COMPONENT_MASK_SCHEMA_VERSION,
            "architecture_ref": self.architecture_ref,
            "component_basis_compatibility_sha256": (self.component_basis_compatibility_sha256),
            "component_ids": list(inventory.component_ids),
            "values": list(self.values),
            "retained_component_ids": [
                component_id
                for component_id, retained in zip(
                    inventory.component_ids,
                    self.values,
                    strict=True,
                )
                if retained
            ],
            "component_proportion": proportion.to_mapping(),
        }


@dataclass(frozen=True)
class ComponentProportion:
    """A component fraction carrying its exact denominator identity."""

    retained_component_count: int
    denominator_component_count: int
    denominator_architecture_ref: str
    denominator_component_basis_compatibility_sha256: str

    def __post_init__(self) -> None:
        retained = _require_nonnegative_int(
            self.retained_component_count,
            name="retained_component_count",
        )
        denominator = _require_positive_int(
            self.denominator_component_count,
            name="denominator_component_count",
        )
        if retained > denominator:
            raise ComponentContractError("retained_component_count cannot exceed denominator")
        if (
            not isinstance(self.denominator_architecture_ref, str)
            or not self.denominator_architecture_ref
        ):
            raise ComponentContractError("denominator_architecture_ref must be non-empty")
        _require_sha256(
            self.denominator_component_basis_compatibility_sha256,
            name="denominator_component_basis_compatibility_sha256",
        )

    @property
    def exact_fraction(self) -> Fraction:
        return Fraction(
            self.retained_component_count,
            self.denominator_component_count,
        )

    @property
    def value(self) -> float:
        return float(self.exact_fraction)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": COMPONENT_PROPORTION_SCHEMA_VERSION,
            "retained_component_count": self.retained_component_count,
            "denominator": {
                "component_count": self.denominator_component_count,
                "architecture_ref": self.denominator_architecture_ref,
                "component_basis_compatibility_sha256": (
                    self.denominator_component_basis_compatibility_sha256
                ),
            },
            "value": self.value,
        }


def compare_component_proportions(
    left: ComponentProportion,
    right: ComponentProportion,
) -> int:
    """Compare fractions only when both carry explicit denominator metadata."""
    if not isinstance(left, ComponentProportion) or not isinstance(
        right,
        ComponentProportion,
    ):
        raise ComponentContractError("component proportions require explicit denominator metadata")

    left_fraction = left.exact_fraction
    right_fraction = right.exact_fraction
    if left_fraction < right_fraction:
        return -1
    if left_fraction > right_fraction:
        return 1
    return 0

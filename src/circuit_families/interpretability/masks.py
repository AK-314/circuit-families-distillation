"""Frozen searchable components and deterministic binary mask records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

MASK_SCHEMA_VERSION = 1

ATTENTION_HEAD_COUNT = 4
MLP_NEURON_COUNT = 512
SEARCHABLE_COMPONENT_COUNT = ATTENTION_HEAD_COUNT + MLP_NEURON_COUNT

ATTENTION_HEAD_HOOK_NAME = "blocks.0.attn.hook_z"
MLP_NEURON_HOOK_NAME = "blocks.0.mlp.hook_post"

ATTENTION_HEAD_IDS = tuple(
    f"H{index}"
    for index in range(ATTENTION_HEAD_COUNT)
)
MLP_NEURON_IDS = tuple(
    f"N{index}"
    for index in range(MLP_NEURON_COUNT)
)
SEARCHABLE_COMPONENT_IDS = ATTENTION_HEAD_IDS + MLP_NEURON_IDS
_SEARCHABLE_COMPONENT_ID_SET = frozenset(SEARCHABLE_COMPONENT_IDS)


@dataclass(frozen=True)
class ComponentLocation:
    """One stable searchable-component activation location."""

    identifier: str
    component_class: str
    hook_name: str
    activation_axis: int
    index: int


COMPONENT_LOCATIONS = tuple(
    ComponentLocation(
        identifier=identifier,
        component_class="attention_head",
        hook_name=ATTENTION_HEAD_HOOK_NAME,
        activation_axis=2,
        index=index,
    )
    for index, identifier in enumerate(ATTENTION_HEAD_IDS)
) + tuple(
    ComponentLocation(
        identifier=identifier,
        component_class="mlp_neuron",
        hook_name=MLP_NEURON_HOOK_NAME,
        activation_axis=2,
        index=index,
    )
    for index, identifier in enumerate(MLP_NEURON_IDS)
)

COMPONENT_LOCATION_BY_ID = {
    location.identifier: location
    for location in COMPONENT_LOCATIONS
}


def _validate_binary_mask(
    values: Sequence[object],
    *,
    expected_length: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of binary integers.")

    if len(values) != expected_length:
        raise ValueError(
            f"{name} must contain exactly {expected_length} values."
        )

    validated: list[int] = []

    for index, value in enumerate(values):
        if not isinstance(value, Integral) or int(value) not in (0, 1):
            raise ValueError(
                f"{name}[{index}] must be the binary integer 0 or 1."
            )
        validated.append(int(value))

    return tuple(validated)


def _validate_component_identifiers(
    identifiers: Iterable[str],
) -> tuple[str, ...]:
    if isinstance(identifiers, (str, bytes)):
        raise TypeError(
            "component identifiers must be an iterable of identifiers."
        )

    values = tuple(identifiers)

    if any(not isinstance(identifier, str) for identifier in values):
        raise TypeError("Every component identifier must be a string.")

    if len(set(values)) != len(values):
        raise ValueError("Component identifiers must not contain duplicates.")

    unknown = sorted(set(values).difference(_SEARCHABLE_COMPONENT_ID_SET))

    if unknown:
        raise ValueError(
            "Unknown component identifiers: " + ", ".join(unknown)
        )

    return values


def component_location(identifier: str) -> ComponentLocation:
    """Return the unique frozen activation location for an identifier."""

    if not isinstance(identifier, str):
        raise TypeError("component identifier must be a string.")

    try:
        return COMPONENT_LOCATION_BY_ID[identifier]
    except KeyError as exc:
        raise ValueError(
            f"Unknown component identifier: {identifier}"
        ) from exc


def validate_frozen_architecture(
    *,
    n_heads: int,
    d_mlp: int,
) -> None:
    """Reject an architecture incompatible with the frozen mask universe."""

    if n_heads != ATTENTION_HEAD_COUNT or d_mlp != MLP_NEURON_COUNT:
        raise ValueError(
            "Mask architecture is incompatible with the frozen model: "
            f"expected n_heads={ATTENTION_HEAD_COUNT} and "
            f"d_mlp={MLP_NEURON_COUNT}; received "
            f"n_heads={n_heads} and d_mlp={d_mlp}."
        )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class ComponentMask:
    """Binary retention mask over four heads and 512 MLP neurons."""

    attention_head_mask: tuple[int, ...]
    mlp_neuron_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attention_head_mask",
            _validate_binary_mask(
                self.attention_head_mask,
                expected_length=ATTENTION_HEAD_COUNT,
                name="attention_head_mask",
            ),
        )
        object.__setattr__(
            self,
            "mlp_neuron_mask",
            _validate_binary_mask(
                self.mlp_neuron_mask,
                expected_length=MLP_NEURON_COUNT,
                name="mlp_neuron_mask",
            ),
        )

    @classmethod
    def all_retained(cls) -> ComponentMask:
        """Return the identity mask."""

        return cls(
            attention_head_mask=(1,) * ATTENTION_HEAD_COUNT,
            mlp_neuron_mask=(1,) * MLP_NEURON_COUNT,
        )

    @classmethod
    def all_ablated(cls) -> ComponentMask:
        """Return the zero-ablation mask over every searchable component."""

        return cls(
            attention_head_mask=(0,) * ATTENTION_HEAD_COUNT,
            mlp_neuron_mask=(0,) * MLP_NEURON_COUNT,
        )

    @classmethod
    def one_head_ablated(cls, identifier: str) -> ComponentMask:
        """Return a mask ablating exactly one specified attention head."""

        location = component_location(identifier)

        if location.component_class != "attention_head":
            raise ValueError(
                f"{identifier} is not an attention-head identifier."
            )

        values = [1] * ATTENTION_HEAD_COUNT
        values[location.index] = 0

        return cls(
            attention_head_mask=tuple(values),
            mlp_neuron_mask=(1,) * MLP_NEURON_COUNT,
        )

    @classmethod
    def one_neuron_ablated(cls, identifier: str) -> ComponentMask:
        """Return a mask ablating exactly one specified MLP neuron."""

        location = component_location(identifier)

        if location.component_class != "mlp_neuron":
            raise ValueError(
                f"{identifier} is not an MLP-neuron identifier."
            )

        values = [1] * MLP_NEURON_COUNT
        values[location.index] = 0

        return cls(
            attention_head_mask=(1,) * ATTENTION_HEAD_COUNT,
            mlp_neuron_mask=tuple(values),
        )

    @classmethod
    def from_retained_identifiers(
        cls,
        identifiers: Iterable[str],
    ) -> ComponentMask:
        """Retain exactly the supplied component identifiers."""

        retained = set(_validate_component_identifiers(identifiers))

        return cls(
            attention_head_mask=tuple(
                int(identifier in retained)
                for identifier in ATTENTION_HEAD_IDS
            ),
            mlp_neuron_mask=tuple(
                int(identifier in retained)
                for identifier in MLP_NEURON_IDS
            ),
        )

    @classmethod
    def from_ablated_identifiers(
        cls,
        identifiers: Iterable[str],
    ) -> ComponentMask:
        """Ablate exactly the supplied component identifiers."""

        ablated = set(_validate_component_identifiers(identifiers))

        return cls(
            attention_head_mask=tuple(
                int(identifier not in ablated)
                for identifier in ATTENTION_HEAD_IDS
            ),
            mlp_neuron_mask=tuple(
                int(identifier not in ablated)
                for identifier in MLP_NEURON_IDS
            ),
        )

    @property
    def retained_attention_head_count(self) -> int:
        return sum(self.attention_head_mask)

    @property
    def retained_mlp_neuron_count(self) -> int:
        return sum(self.mlp_neuron_mask)

    @property
    def retained_component_count(self) -> int:
        return (
            self.retained_attention_head_count
            + self.retained_mlp_neuron_count
        )

    @property
    def retained_component_proportion(self) -> float:
        return self.retained_component_count / SEARCHABLE_COMPONENT_COUNT

    @property
    def retained_component_ids(self) -> tuple[str, ...]:
        values = self.attention_head_mask + self.mlp_neuron_mask
        return tuple(
            identifier
            for identifier, retained in zip(
                SEARCHABLE_COMPONENT_IDS,
                values,
                strict=True,
            )
            if retained
        )

    @property
    def ablated_component_ids(self) -> tuple[str, ...]:
        values = self.attention_head_mask + self.mlp_neuron_mask
        return tuple(
            identifier
            for identifier, retained in zip(
                SEARCHABLE_COMPONENT_IDS,
                values,
                strict=True,
            )
            if not retained
        )

    def _identity_record(self) -> dict[str, Any]:
        return {
            "schema_version": MASK_SCHEMA_VERSION,
            "architecture": {
                "n_heads": ATTENTION_HEAD_COUNT,
                "d_mlp": MLP_NEURON_COUNT,
            },
            "attention_head_mask": list(self.attention_head_mask),
            "mlp_neuron_mask": list(self.mlp_neuron_mask),
        }

    @property
    def mask_id(self) -> str:
        digest = hashlib.sha256(
            _canonical_json(self._identity_record()).encode("utf-8")
        )
        return f"component-mask-{digest.hexdigest()[:16]}"

    def to_record(self) -> dict[str, Any]:
        """Return a transparent JSON-safe mask record."""

        return {
            **self._identity_record(),
            "mask_id": self.mask_id,
            "mask_convention": {
                "retained": 1,
                "ablated": 0,
                "ablation_baseline": "zero",
            },
            "component_identifiers": list(SEARCHABLE_COMPONENT_IDS),
            "retained_component_ids": list(self.retained_component_ids),
            "ablated_component_ids": list(self.ablated_component_ids),
            "retained_attention_head_count": (
                self.retained_attention_head_count
            ),
            "retained_mlp_neuron_count": (
                self.retained_mlp_neuron_count
            ),
            "retained_component_count": self.retained_component_count,
            "retained_component_proportion": (
                self.retained_component_proportion
            ),
            "searchable_component_count": SEARCHABLE_COMPONENT_COUNT,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ComponentMask:
        """Validate and reconstruct a transparent mask record."""

        if not isinstance(record, Mapping):
            raise TypeError("Mask record must be a mapping.")

        required = {
            "schema_version",
            "architecture",
            "attention_head_mask",
            "mlp_neuron_mask",
            "mask_id",
            "mask_convention",
            "component_identifiers",
            "retained_component_ids",
            "ablated_component_ids",
            "retained_attention_head_count",
            "retained_mlp_neuron_count",
            "retained_component_count",
            "retained_component_proportion",
            "searchable_component_count",
        }
        missing = sorted(required.difference(record))

        if missing:
            raise ValueError(
                "Mask record is missing required fields: "
                + ", ".join(missing)
            )

        if record["schema_version"] != MASK_SCHEMA_VERSION:
            raise ValueError("Unsupported mask schema version.")

        architecture = record["architecture"]

        if not isinstance(architecture, Mapping):
            raise ValueError("Mask architecture must be a mapping.")

        validate_frozen_architecture(
            n_heads=architecture.get("n_heads"),
            d_mlp=architecture.get("d_mlp"),
        )

        mask = cls(
            attention_head_mask=record["attention_head_mask"],
            mlp_neuron_mask=record["mlp_neuron_mask"],
        )
        expected = mask.to_record()

        for field in required:
            if record[field] != expected[field]:
                raise ValueError(
                    f"Mask record field {field!r} is inconsistent."
                )

        return mask


def save_component_mask(
    path: str | Path,
    mask: ComponentMask,
) -> Path:
    """Write a deterministic human-readable JSON mask record."""

    if not isinstance(mask, ComponentMask):
        raise TypeError("mask must be a ComponentMask.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(
        mask.to_record(),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    output_path.write_text(serialised + "\n", encoding="utf-8")
    return output_path


def load_component_mask(path: str | Path) -> ComponentMask:
    """Load and validate a JSON mask record."""

    input_path = Path(path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Mask file does not exist: {input_path}")

    record = json.loads(input_path.read_text(encoding="utf-8"))
    return ComponentMask.from_record(record)

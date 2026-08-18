"""Stage 4 canonical condition identity builder/parser.

This module implements condition-identity/v1 only. It performs no scientific
computation and has no dependency on machine-specific paths.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote_to_bytes

IDENTITY_VERSION = "condition-identity/v1"
WIRE_PREFIX = "cfdid"
WIRE_VERSION_TOKEN = "v1"

CANONICAL_HIERARCHY = (
    "teacher_seed",
    "phase",
    "distillation_condition",
    "student_initialization",
    "discovery_method",
    "fidelity_setting",
    "component_cap",
    "overlap_setting",
)

PHASE_VALUES = (
    "pre-grokking",
    "50%",
    "stable post-grokking",
)

DISTILLATION_CONDITION_VALUES = (
    "direct_teacher",
    "hard_target",
    "soft_target",
)

VERSION_REFERENCE_FIELDS = frozenset(
    {
        "discovery_method",
        "fidelity_setting",
        "component_cap",
        "overlap_setting",
    }
)

VERSION_REFERENCE_RE = re.compile(
    r"^[a-z][a-z0-9._-]*/v[1-9][0-9]*$"
)

WIRE_HEADER_RE = re.compile(r"^cfdid:v1:d([2-8])$")

SAFE_ASCII = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789-._~"
)

_UPPER_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-F]{2}")
_ANY_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")


class ConditionIdentityError(ValueError):
    """Raised when a condition identity violates the Stage 4 contract."""


@dataclass(frozen=True)
class Stage3Cell:
    """Minimal Stage 3 availability information required by identity checks."""

    teacher_seed: int
    phase: str
    availability_state: str


@dataclass(frozen=True)
class Stage3AvailabilityIndex:
    """Portable Stage 3 teacher-phase availability index."""

    canonical_seed_order: tuple[int, ...]
    canonical_phase_order: tuple[str, ...]
    cells: Mapping[tuple[int, str], str]

    @classmethod
    def from_registry(cls, registry: Mapping[str, Any]) -> Stage3AvailabilityIndex:
        """Build the minimal index from stage3_teacher_registry_v1 data."""
        required_top = {
            "canonical_seed_order",
            "canonical_phase_order",
            "expected_cell_count",
            "selected_cell_count",
            "unavailable_cell_count",
            "records",
        }
        missing = required_top - set(registry)
        if missing:
            raise ConditionIdentityError(
                f"Stage 3 registry missing keys: {sorted(missing)!r}"
            )

        seeds = tuple(registry["canonical_seed_order"])
        phases = tuple(registry["canonical_phase_order"])
        records = registry["records"]

        if seeds != (0, 1, 2, 3, 4):
            raise ConditionIdentityError(
                f"unexpected Stage 3 seed order: {seeds!r}"
            )
        if phases != PHASE_VALUES:
            raise ConditionIdentityError(
                f"unexpected Stage 3 phase order: {phases!r}"
            )
        if registry["expected_cell_count"] != 15:
            raise ConditionIdentityError("Stage 3 expected_cell_count must be 15")
        if registry["selected_cell_count"] != 13:
            raise ConditionIdentityError("Stage 3 selected_cell_count must be 13")
        if registry["unavailable_cell_count"] != 2:
            raise ConditionIdentityError(
                "Stage 3 unavailable_cell_count must be 2"
            )
        if not isinstance(records, list) or len(records) != 15:
            raise ConditionIdentityError("Stage 3 registry must contain 15 records")

        cells: dict[tuple[int, str], str] = {}

        for record in records:
            try:
                seed = record["teacher_seed"]
                phase = record["phase_label"]
                availability = record["availability_status"]
            except KeyError as exc:
                raise ConditionIdentityError(
                    f"Stage 3 record missing required field: {exc.args[0]}"
                ) from exc

            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed not in seeds
            ):
                raise ConditionIdentityError(
                    f"invalid Stage 3 teacher_seed: {seed!r}"
                )
            if phase not in phases:
                raise ConditionIdentityError(
                    f"invalid Stage 3 phase_label: {phase!r}"
                )
            if availability not in {"selected", "unavailable"}:
                raise ConditionIdentityError(
                    f"invalid Stage 3 availability_status: {availability!r}"
                )

            key = (seed, phase)
            if key in cells:
                raise ConditionIdentityError(
                    f"duplicate Stage 3 teacher-phase cell: {key!r}"
                )
            cells[key] = availability

        if len(cells) != 15:
            raise ConditionIdentityError(
                f"Stage 3 cell count mismatch: {len(cells)} != 15"
            )

        selected = sum(value == "selected" for value in cells.values())
        unavailable = sum(value == "unavailable" for value in cells.values())

        if selected != 13 or unavailable != 2:
            raise ConditionIdentityError(
                "Stage 3 availability accounting must be selected=13, "
                "unavailable=2"
            )

        return cls(
            canonical_seed_order=seeds,
            canonical_phase_order=phases,
            cells=cells,
        )

    def availability(self, teacher_seed: int, phase: str) -> str:
        """Return selected/unavailable for a planned teacher-phase cell."""
        try:
            return self.cells[(teacher_seed, phase)]
        except KeyError as exc:
            raise ConditionIdentityError(
                "teacher-phase cell is not present in the canonical "
                f"Stage 3 registry: seed={teacher_seed!r}, phase={phase!r}"
            ) from exc


@dataclass(frozen=True)
class ConditionIdentity:
    """Typed canonical condition identity prefix or complete identity."""

    teacher_seed: int
    phase: str
    distillation_condition: str | None = None
    student_initialization: int | None = None
    discovery_method: str | None = None
    fidelity_setting: str | None = None
    component_cap: str | None = None
    overlap_setting: str | None = None

    def values(self) -> tuple[Any, ...]:
        """Return hierarchy values in canonical order."""
        return tuple(getattr(self, field) for field in CANONICAL_HIERARCHY)

    @property
    def depth(self) -> int:
        """Return validated prefix depth from contiguous populated fields."""
        values = self.values()
        first_none = next(
            (index for index, value in enumerate(values) if value is None),
            len(values),
        )
        if any(value is not None for value in values[first_none:]):
            raise ConditionIdentityError(
                "identity fields must form one contiguous canonical prefix"
            )
        if first_none < 2:
            raise ConditionIdentityError(
                "identity requires teacher_seed and phase"
            )
        return first_none

    def as_mapping(self) -> dict[str, Any]:
        """Return only the canonical prefix fields."""
        depth = self.depth
        return {
            field: getattr(self, field)
            for field in CANONICAL_HIERARCHY[:depth]
        }


def _encode_string(value: str) -> str:
    if not isinstance(value, str):
        raise ConditionIdentityError("string identity component required")
    if not value:
        raise ConditionIdentityError("empty identity component is forbidden")
    if unicodedata.normalize("NFC", value) != value:
        raise ConditionIdentityError(
            "string identity components must already be NFC-normalized"
        )
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ConditionIdentityError(
            "control characters are forbidden in identity components"
        )
    return quote(value, safe=SAFE_ASCII, encoding="utf-8", errors="strict")


def _decode_string(value: str) -> str:
    if not value:
        raise ConditionIdentityError("empty serialized component is forbidden")

    # Every percent escape must be complete and use uppercase hexadecimal.
    percent_positions = [
        index for index, char in enumerate(value) if char == "%"
    ]
    for index in percent_positions:
        token = value[index : index + 3]
        if len(token) != 3 or not _UPPER_PERCENT_ESCAPE_RE.fullmatch(token):
            raise ConditionIdentityError(
                "percent escapes must be complete uppercase hexadecimal"
            )

    try:
        decoded = unquote_to_bytes(value).decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ConditionIdentityError(
            "serialized component is not valid UTF-8"
        ) from exc

    if _encode_string(decoded) != value:
        raise ConditionIdentityError(
            "serialized component is not in canonical percent-encoded form"
        )

    return decoded


def _validate_uint(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConditionIdentityError(
            f"{field} must be a non-negative integer"
        )
    if value < 0:
        raise ConditionIdentityError(
            f"{field} must be a non-negative integer"
        )
    return value


def _parse_uint(field: str, value: str) -> int:
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
        raise ConditionIdentityError(
            f"{field} must use canonical unsigned base-10 encoding"
        )
    return int(value)


def _validate_fields(
    identity: ConditionIdentity,
    stage3: Stage3AvailabilityIndex,
) -> int:
    depth = identity.depth

    teacher_seed = _validate_uint("teacher_seed", identity.teacher_seed)
    if teacher_seed not in stage3.canonical_seed_order:
        raise ConditionIdentityError(
            f"teacher_seed outside Stage 3 canonical order: {teacher_seed}"
        )

    phase = identity.phase
    if phase not in stage3.canonical_phase_order:
        raise ConditionIdentityError(f"invalid phase: {phase!r}")
    _encode_string(phase)

    availability = stage3.availability(teacher_seed, phase)

    if depth >= 3:
        condition = identity.distillation_condition
        if condition not in DISTILLATION_CONDITION_VALUES:
            raise ConditionIdentityError(
                f"invalid distillation_condition: {condition!r}"
            )
        _encode_string(condition)

        if availability != "selected":
            raise ConditionIdentityError(
                "unavailable Stage 3 cell cannot form downstream "
                "condition identity"
            )

    if depth >= 4:
        if identity.distillation_condition == "direct_teacher":
            raise ConditionIdentityError(
                "direct_teacher cannot have student_initialization"
            )
        if identity.distillation_condition not in {
            "hard_target",
            "soft_target",
        }:
            raise ConditionIdentityError(
                "student identity requires hard_target or soft_target"
            )
        _validate_uint(
            "student_initialization",
            identity.student_initialization,
        )

    for field in CANONICAL_HIERARCHY[4:depth]:
        value = getattr(identity, field)
        if not isinstance(value, str):
            raise ConditionIdentityError(
                f"{field} must be a version reference string"
            )
        if not VERSION_REFERENCE_RE.fullmatch(value):
            raise ConditionIdentityError(
                f"{field} must match version-reference grammar"
            )
        _encode_string(value)

    return depth


def build_condition_id(
    identity: ConditionIdentity,
    stage3: Stage3AvailabilityIndex,
) -> str:
    """Build the unique canonical wire ID after Stage 3 validation."""
    depth = _validate_fields(identity, stage3)

    pieces = []
    for field in CANONICAL_HIERARCHY[:depth]:
        value = getattr(identity, field)
        if field in {"teacher_seed", "student_initialization"}:
            encoded = str(_validate_uint(field, value))
        else:
            encoded = _encode_string(value)
        pieces.append(f"{field}={encoded}")

    return f"{WIRE_PREFIX}:{WIRE_VERSION_TOKEN}:d{depth}|" + "|".join(pieces)


def parse_condition_id(
    condition_id: str,
    stage3: Stage3AvailabilityIndex,
) -> ConditionIdentity:
    """Parse, validate, and canonical-round-trip a condition identity."""
    if not isinstance(condition_id, str) or not condition_id:
        raise ConditionIdentityError("condition_id must be a non-empty string")

    segments = condition_id.split("|")
    header = segments[0]

    match = WIRE_HEADER_RE.fullmatch(header)
    if match is None:
        raise ConditionIdentityError(
            "condition_id header must match cfdid:v1:d<2..8>"
        )

    depth = int(match.group(1))
    assignments = segments[1:]

    if len(assignments) != depth:
        raise ConditionIdentityError(
            f"condition_id depth={depth} requires {depth} fields, "
            f"found {len(assignments)}"
        )

    values: dict[str, Any] = {}

    for expected_field, assignment in zip(
        CANONICAL_HIERARCHY[:depth],
        assignments,
        strict=True,
    ):
        if "=" not in assignment:
            raise ConditionIdentityError(
                f"missing '=' for field {expected_field}"
            )

        field, encoded = assignment.split("=", 1)

        if field != expected_field:
            raise ConditionIdentityError(
                "identity fields must appear once in canonical order: "
                f"expected {expected_field!r}, found {field!r}"
            )

        if expected_field in {"teacher_seed", "student_initialization"}:
            values[expected_field] = _parse_uint(expected_field, encoded)
        else:
            values[expected_field] = _decode_string(encoded)

    identity = ConditionIdentity(**values)
    rebuilt = build_condition_id(identity, stage3)

    if rebuilt != condition_id:
        raise ConditionIdentityError(
            "condition_id does not round-trip to its canonical representation"
        )

    return identity

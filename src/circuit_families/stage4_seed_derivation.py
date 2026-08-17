"""Deterministic Stage 4 seed derivation.

Implements seed-derivation/v1 exactly. The canonical condition ID is reused
verbatim; this module does not implement a second identity serializer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from circuit_families.stage4_condition_identity import (
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    parse_condition_id,
)

NAMESPACE = "circuit-families-distillation"
SEED_DERIVATION_VERSION = "seed-derivation/v1"
WIRE_HEADER = "cfdseed:v1"

PURPOSE_DEPTHS = {
    "training": 4,
    "tie_breaking": 4,
    "discovery": 8,
}

SEED_MASK = 0x7FFFFFFFFFFFFFFF
SEED_MIN = 0
SEED_MAX = SEED_MASK

_CANONICAL_UINT_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")
_SELECTED_BYTES_HEX_RE = re.compile(r"[0-9a-f]{16}\Z")


class SeedDerivationError(ValueError):
    """Raised when seed material or stored evidence violates the contract."""


@dataclass(frozen=True)
class SeedInputs:
    """Explicit stochastic coordinates used by seed-derivation/v1."""

    condition_id: str
    purpose: str
    attempt_index: int
    retry_index: int


@dataclass(frozen=True)
class SeedEvidence:
    """Required stored evidence for one deterministic seed."""

    seed_derivation_version: str
    seed_material: str
    digest_sha256: str
    selected_bytes_hex: str
    seed_value: int


def _canonical_uint(value: int, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SeedDerivationError(
            f"{field} must be a non-negative integer"
        )
    return str(value)


def _parse_canonical_uint(value: str, field: str) -> int:
    if not _CANONICAL_UINT_RE.fullmatch(value):
        raise SeedDerivationError(
            f"{field} must use canonical unsigned base-10 encoding"
        )
    return int(value)


def _validate_purpose_and_identity(
    *,
    condition_id: str,
    purpose: str,
    stage3: Stage3AvailabilityIndex,
) -> None:
    expected_depth = PURPOSE_DEPTHS.get(purpose)

    if expected_depth is None:
        raise SeedDerivationError(
            f"unsupported seed purpose: {purpose!r}"
        )

    try:
        identity = parse_condition_id(condition_id, stage3)
    except ConditionIdentityError as exc:
        raise SeedDerivationError(
            f"invalid canonical condition_id: {exc}"
        ) from exc

    actual_depth = identity.depth

    if actual_depth != expected_depth:
        raise SeedDerivationError(
            f"purpose {purpose!r} requires identity depth "
            f"{expected_depth}, found {actual_depth}"
        )


def canonical_seed_material(
    inputs: SeedInputs,
    stage3: Stage3AvailabilityIndex,
) -> str:
    """Return exact ASCII derivation material including the final LF."""
    _validate_purpose_and_identity(
        condition_id=inputs.condition_id,
        purpose=inputs.purpose,
        stage3=stage3,
    )

    attempt = _canonical_uint(inputs.attempt_index, "attempt_index")
    retry = _canonical_uint(inputs.retry_index, "retry_index")

    material = (
        f"{WIRE_HEADER}\n"
        f"namespace={NAMESPACE}\n"
        f"seed_derivation_version={SEED_DERIVATION_VERSION}\n"
        f"condition_id={inputs.condition_id}\n"
        f"purpose={inputs.purpose}\n"
        f"attempt_index={attempt}\n"
        f"retry_index={retry}\n"
    )

    try:
        material.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SeedDerivationError(
            "seed derivation material must be ASCII"
        ) from exc

    return material


def derive_seed(
    inputs: SeedInputs,
    stage3: Stage3AvailabilityIndex,
) -> SeedEvidence:
    """Derive and return all required seed evidence."""
    material = canonical_seed_material(inputs, stage3)
    digest = hashlib.sha256(material.encode("ascii")).digest()
    selected = digest[:8]
    raw_u64 = int.from_bytes(
        selected,
        byteorder="big",
        signed=False,
    )
    seed = raw_u64 & SEED_MASK

    return SeedEvidence(
        seed_derivation_version=SEED_DERIVATION_VERSION,
        seed_material=material,
        digest_sha256=digest.hex(),
        selected_bytes_hex=selected.hex(),
        seed_value=seed,
    )


def parse_seed_material(
    material: str,
    stage3: Stage3AvailabilityIndex,
) -> SeedInputs:
    """Parse exact canonical seed material and recover its explicit inputs."""
    if not isinstance(material, str):
        raise SeedDerivationError("seed_material must be a string")

    try:
        material.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SeedDerivationError(
            "seed_material must be ASCII"
        ) from exc

    if not material.endswith("\n"):
        raise SeedDerivationError(
            "seed_material must end with one final LF"
        )

    lines = material.splitlines()

    if len(lines) != 7:
        raise SeedDerivationError(
            f"seed_material must contain exactly 7 lines, found {len(lines)}"
        )

    if lines[0] != WIRE_HEADER:
        raise SeedDerivationError(
            f"invalid seed wire header: {lines[0]!r}"
        )

    expected_prefixes = (
        "namespace=",
        "seed_derivation_version=",
        "condition_id=",
        "purpose=",
        "attempt_index=",
        "retry_index=",
    )

    values: dict[str, str] = {}

    for line, prefix in zip(lines[1:], expected_prefixes, strict=True):
        if not line.startswith(prefix):
            raise SeedDerivationError(
                f"seed_material field order mismatch: expected {prefix!r}"
            )
        key = prefix[:-1]
        values[key] = line[len(prefix):]

    if values["namespace"] != NAMESPACE:
        raise SeedDerivationError(
            f"invalid seed namespace: {values['namespace']!r}"
        )

    if values["seed_derivation_version"] != SEED_DERIVATION_VERSION:
        raise SeedDerivationError(
            "invalid seed_derivation_version: "
            f"{values['seed_derivation_version']!r}"
        )

    inputs = SeedInputs(
        condition_id=values["condition_id"],
        purpose=values["purpose"],
        attempt_index=_parse_canonical_uint(
            values["attempt_index"],
            "attempt_index",
        ),
        retry_index=_parse_canonical_uint(
            values["retry_index"],
            "retry_index",
        ),
    )

    rebuilt = canonical_seed_material(inputs, stage3)

    if rebuilt != material:
        raise SeedDerivationError(
            "seed_material is not in canonical form"
        )

    return inputs


def verify_seed_evidence(
    evidence: SeedEvidence,
    stage3: Stage3AvailabilityIndex,
) -> SeedInputs:
    """Verify all stored evidence and return the recovered seed inputs."""
    if evidence.seed_derivation_version != SEED_DERIVATION_VERSION:
        raise SeedDerivationError(
            "stored seed_derivation_version mismatch"
        )

    if not _SHA256_HEX_RE.fullmatch(evidence.digest_sha256):
        raise SeedDerivationError(
            "digest_sha256 must be 64 lowercase hexadecimal characters"
        )

    if not _SELECTED_BYTES_HEX_RE.fullmatch(
        evidence.selected_bytes_hex
    ):
        raise SeedDerivationError(
            "selected_bytes_hex must be 16 lowercase hexadecimal characters"
        )

    if (
        isinstance(evidence.seed_value, bool)
        or not isinstance(evidence.seed_value, int)
        or not SEED_MIN <= evidence.seed_value <= SEED_MAX
    ):
        raise SeedDerivationError(
            f"seed_value must be in [{SEED_MIN}, {SEED_MAX}]"
        )

    inputs = parse_seed_material(
        evidence.seed_material,
        stage3,
    )
    expected = derive_seed(inputs, stage3)

    if evidence.digest_sha256 != expected.digest_sha256:
        raise SeedDerivationError(
            "stored digest_sha256 does not match derivation material"
        )

    if evidence.selected_bytes_hex != expected.selected_bytes_hex:
        raise SeedDerivationError(
            "stored selected_bytes_hex does not match digest"
        )

    if evidence.seed_value != expected.seed_value:
        raise SeedDerivationError(
            "stored seed_value does not match derivation material"
        )

    return inputs

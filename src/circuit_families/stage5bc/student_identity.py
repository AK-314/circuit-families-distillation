"""Canonical student-attempt identity and seed evidence for Stage 5B.

The student condition ID is built only by the frozen Stage 4 condition
identity implementation. Attempt/retry indices remain explicit coordinates
outside the condition ID and are consumed by frozen seed-derivation/v1.

No second condition serializer and no Python hash() based identity exists here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    build_condition_id,
    parse_condition_id,
)
from circuit_families.stage4_seed_derivation import (
    SeedDerivationError,
    SeedEvidence,
    SeedInputs,
    derive_seed,
    verify_seed_evidence,
)

STUDENT_SEED_PURPOSES = (
    "training",
    "tie_breaking",
)


class StudentIdentityError(ValueError):
    """Raised when Stage 5B attempt identity evidence is inconsistent."""


def _validate_index(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StudentIdentityError(
            f"{name} must be a non-negative integer"
        )
    return value


def _seed_evidence_mapping(evidence: SeedEvidence) -> dict[str, Any]:
    return {
        "seed_derivation_version": evidence.seed_derivation_version,
        "seed_material": evidence.seed_material,
        "digest_sha256": evidence.digest_sha256,
        "selected_bytes_hex": evidence.selected_bytes_hex,
        "seed_value": evidence.seed_value,
    }


@dataclass(frozen=True)
class StudentAttemptIdentity:
    """One depth-4 student condition plus explicit attempt/retry coordinates."""

    condition_id: str
    attempt_index: int
    retry_index: int
    training_seed: SeedEvidence
    tie_breaking_seed: SeedEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.condition_id, str) or not self.condition_id:
            raise StudentIdentityError(
                "condition_id must be a non-empty string"
            )
        _validate_index(self.attempt_index, name="attempt_index")
        _validate_index(self.retry_index, name="retry_index")

        if not isinstance(self.training_seed, SeedEvidence):
            raise StudentIdentityError(
                "training_seed must be SeedEvidence"
            )
        if not isinstance(self.tie_breaking_seed, SeedEvidence):
            raise StudentIdentityError(
                "tie_breaking_seed must be SeedEvidence"
            )

    def to_mapping(self) -> dict[str, Any]:
        """Return complete portable identity/seed evidence."""
        return {
            "condition_id": self.condition_id,
            "attempt_index": self.attempt_index,
            "retry_index": self.retry_index,
            "training_seed": _seed_evidence_mapping(
                self.training_seed
            ),
            "tie_breaking_seed": _seed_evidence_mapping(
                self.tie_breaking_seed
            ),
        }


def build_student_condition_id(
    *,
    stage3: Stage3AvailabilityIndex,
    teacher_seed: int,
    phase: str,
    distillation_condition: str,
    student_initialization: int,
) -> str:
    """Build the frozen Stage 4 depth-4 student condition ID."""
    identity = ConditionIdentity(
        teacher_seed=teacher_seed,
        phase=phase,
        distillation_condition=distillation_condition,
        student_initialization=student_initialization,
    )

    try:
        condition_id = build_condition_id(identity, stage3)
    except ConditionIdentityError as exc:
        raise StudentIdentityError(
            f"invalid student condition identity: {exc}"
        ) from exc

    try:
        parsed = parse_condition_id(condition_id, stage3)
    except ConditionIdentityError as exc:
        raise StudentIdentityError(
            f"student condition failed canonical round-trip: {exc}"
        ) from exc

    if parsed.depth != 4:
        raise StudentIdentityError(
            f"student condition must have depth 4, found {parsed.depth}"
        )

    return condition_id


def build_student_attempt_identity(
    *,
    stage3: Stage3AvailabilityIndex,
    teacher_seed: int,
    phase: str,
    distillation_condition: str,
    student_initialization: int,
    attempt_index: int,
    retry_index: int,
) -> StudentAttemptIdentity:
    """Build condition identity and both required deterministic seed records."""
    attempt = _validate_index(
        attempt_index,
        name="attempt_index",
    )
    retry = _validate_index(
        retry_index,
        name="retry_index",
    )

    condition_id = build_student_condition_id(
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        distillation_condition=distillation_condition,
        student_initialization=student_initialization,
    )

    try:
        training_seed = derive_seed(
            SeedInputs(
                condition_id=condition_id,
                purpose="training",
                attempt_index=attempt,
                retry_index=retry,
            ),
            stage3,
        )
        tie_breaking_seed = derive_seed(
            SeedInputs(
                condition_id=condition_id,
                purpose="tie_breaking",
                attempt_index=attempt,
                retry_index=retry,
            ),
            stage3,
        )
    except SeedDerivationError as exc:
        raise StudentIdentityError(
            f"seed derivation failed: {exc}"
        ) from exc

    return StudentAttemptIdentity(
        condition_id=condition_id,
        attempt_index=attempt,
        retry_index=retry,
        training_seed=training_seed,
        tie_breaking_seed=tie_breaking_seed,
    )


def verify_student_attempt_identity(
    identity: StudentAttemptIdentity,
    stage3: Stage3AvailabilityIndex,
) -> ConditionIdentity:
    """Verify canonical condition and all stored seed evidence."""
    if not isinstance(identity, StudentAttemptIdentity):
        raise StudentIdentityError(
            "identity must be StudentAttemptIdentity"
        )

    try:
        parsed = parse_condition_id(
            identity.condition_id,
            stage3,
        )
    except ConditionIdentityError as exc:
        raise StudentIdentityError(
            f"invalid stored condition_id: {exc}"
        ) from exc

    if parsed.depth != 4:
        raise StudentIdentityError(
            f"stored student condition must have depth 4, found {parsed.depth}"
        )

    try:
        training_inputs = verify_seed_evidence(
            identity.training_seed,
            stage3,
        )
        tie_inputs = verify_seed_evidence(
            identity.tie_breaking_seed,
            stage3,
        )
    except SeedDerivationError as exc:
        raise StudentIdentityError(
            f"stored seed evidence failed verification: {exc}"
        ) from exc

    expected_common = {
        "condition_id": identity.condition_id,
        "attempt_index": identity.attempt_index,
        "retry_index": identity.retry_index,
    }

    for purpose, inputs in (
        ("training", training_inputs),
        ("tie_breaking", tie_inputs),
    ):
        actual = {
            "condition_id": inputs.condition_id,
            "attempt_index": inputs.attempt_index,
            "retry_index": inputs.retry_index,
        }

        if actual != expected_common:
            raise StudentIdentityError(
                f"{purpose} seed coordinates disagree with attempt identity"
            )

        if inputs.purpose != purpose:
            raise StudentIdentityError(
                f"{purpose} seed evidence uses purpose {inputs.purpose!r}"
            )

    if (
        identity.training_seed.seed_material
        == identity.tie_breaking_seed.seed_material
    ):
        raise StudentIdentityError(
            "training and tie-breaking seed material must be distinct"
        )

    return copy.deepcopy(parsed)

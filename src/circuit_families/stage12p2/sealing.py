"""Architecture-aware sealing, accounting, and discovery release for Stage 12-P2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .eligibility import (
    AttemptFailureRecord,
    HardEligibilityRecord,
    SoftEligibilityRecord,
)
from .engine import StudentAttemptExecution
from .training import StudentTrainingIdentity

SEAL_SCHEMA_VERSION = "stage12p2-sealed-student/v1"
RELEASE_SCHEMA_VERSION = "stage12p2-discovery-release/v1"

EligibilityRecord = HardEligibilityRecord | SoftEligibilityRecord
AttemptRecord = EligibilityRecord | AttemptFailureRecord


class StudentSealingError(ValueError):
    """Raised when P2 sealing or release evidence is inconsistent."""


def _sha256_json(mapping: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            mapping,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StudentSealingError(f"{name} must be lowercase SHA-256")


def _terminal_checkpoint_sha256(
    execution: StudentAttemptExecution,
) -> str:
    if execution.status != "completed":
        raise StudentSealingError("only completed attempts may seal")
    terminal = tuple(entry for entry in execution.checkpoints.entries if entry.role == "terminal")
    if len(terminal) != 1:
        raise StudentSealingError("completed attempt must have exactly one terminal checkpoint")
    return terminal[0].file_sha256


@dataclass(frozen=True)
class SealedStudentModelRecord:
    schema_version: str
    estimand: Literal["hard", "soft"]
    identity_sha256: str
    architecture_ref: str
    architecture_record_sha256: str
    task_identity_sha256: str
    teacher_record_sha256: str
    target_cache_manifest_sha256: str
    training_config_sha256: str
    backend_qualification_sha256: str
    checkpoint_sha256: str
    dense_output_sha256: str
    eligibility_record_sha256: str
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SEAL_SCHEMA_VERSION:
            raise StudentSealingError("invalid seal schema version")
        if self.estimand not in {"hard", "soft"}:
            raise StudentSealingError("invalid sealed estimand")
        if not isinstance(self.architecture_ref, str) or not self.architecture_ref:
            raise StudentSealingError("architecture_ref must be non-empty")
        for name in (
            "identity_sha256",
            "architecture_record_sha256",
            "task_identity_sha256",
            "teacher_record_sha256",
            "target_cache_manifest_sha256",
            "training_config_sha256",
            "backend_qualification_sha256",
            "checkpoint_sha256",
            "dense_output_sha256",
            "eligibility_record_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.scientific_data is not False:
            raise StudentSealingError("scientific_data must be false")
        if self.production_eligible is not False:
            raise StudentSealingError("production_eligible must be false")

    def to_mapping(self) -> dict[str, object]:
        return dict(self.__dict__)

    @property
    def record_sha256(self) -> str:
        return _sha256_json(self.to_mapping())


@dataclass(frozen=True)
class DiscoveryReleaseRecord:
    schema_version: str
    estimand: Literal["hard", "soft"]
    identity_sha256: str
    architecture_ref: str
    architecture_record_sha256: str
    task_identity_sha256: str
    teacher_record_sha256: str
    target_cache_manifest_sha256: str
    training_config_sha256: str
    backend_qualification_sha256: str
    checkpoint_sha256: str
    dense_output_sha256: str
    eligibility_record_sha256: str
    sealed_model_record_sha256: str
    release_status: Literal["released"]
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != RELEASE_SCHEMA_VERSION:
            raise StudentSealingError("invalid release schema version")
        if self.estimand not in {"hard", "soft"}:
            raise StudentSealingError("invalid release estimand")
        if self.release_status != "released":
            raise StudentSealingError("release_status must be released")
        if not isinstance(self.architecture_ref, str) or not self.architecture_ref:
            raise StudentSealingError("architecture_ref must be non-empty")
        for name in (
            "identity_sha256",
            "architecture_record_sha256",
            "task_identity_sha256",
            "teacher_record_sha256",
            "target_cache_manifest_sha256",
            "training_config_sha256",
            "backend_qualification_sha256",
            "checkpoint_sha256",
            "dense_output_sha256",
            "eligibility_record_sha256",
            "sealed_model_record_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.scientific_data is not False:
            raise StudentSealingError("scientific_data must be false")
        if self.production_eligible is not False:
            raise StudentSealingError("production_eligible must be false")

    def to_mapping(self) -> dict[str, object]:
        return dict(self.__dict__)

    @property
    def record_sha256(self) -> str:
        return _sha256_json(self.to_mapping())


@dataclass(frozen=True)
class AttemptAccounting:
    """Counts one estimand without pooling hard and soft records."""

    estimand: Literal["hard", "soft"]
    passed: int
    ineligible: int
    optimization_failed: int
    numerical_failed: int
    interrupted: int
    unavailable: int

    @property
    def total(self) -> int:
        return (
            self.passed
            + self.ineligible
            + self.optimization_failed
            + self.numerical_failed
            + self.interrupted
            + self.unavailable
        )


def summarize_attempt_records(
    records: Sequence[AttemptRecord],
) -> AttemptAccounting:
    """Count all outcomes while refusing hard/soft pooling."""
    if not records:
        raise StudentSealingError("attempt accounting requires records")
    estimands = {record.estimand for record in records}
    if len(estimands) != 1:
        raise StudentSealingError("hard and soft attempt records cannot be pooled")
    estimand = next(iter(estimands))
    counts = {
        "passed": 0,
        "ineligible": 0,
        "optimization-failed": 0,
        "numerical-failed": 0,
        "interrupted": 0,
        "unavailable": 0,
    }
    for record in records:
        if record.status not in counts:
            raise StudentSealingError("unsupported attempt status")
        counts[record.status] += 1
    return AttemptAccounting(
        estimand=estimand,
        passed=counts["passed"],
        ineligible=counts["ineligible"],
        optimization_failed=counts["optimization-failed"],
        numerical_failed=counts["numerical-failed"],
        interrupted=counts["interrupted"],
        unavailable=counts["unavailable"],
    )


def seal_student_model(
    *,
    execution: StudentAttemptExecution,
    identity: StudentTrainingIdentity,
    eligibility: EligibilityRecord,
    dense_output_sha256: str,
) -> SealedStudentModelRecord:
    """Seal only passed, exact-hash-consistent P2 technical eligibility."""
    if not isinstance(eligibility, (HardEligibilityRecord, SoftEligibilityRecord)):
        raise StudentSealingError("unsupported eligibility record")
    if eligibility.status != "passed":
        raise StudentSealingError("ineligible attempt cannot be sealed")
    if execution.identity_sha256 != identity.identity_sha256:
        raise StudentSealingError("execution identity hash mismatch")
    if execution.architecture_ref != identity.architecture_ref:
        raise StudentSealingError("execution architecture mismatch")

    expected = {
        "identity_sha256": identity.identity_sha256,
        "architecture_ref": identity.architecture_ref,
        "architecture_record_sha256": identity.architecture_record_sha256,
        "task_identity_sha256": identity.task_identity_sha256,
        "teacher_record_sha256": identity.teacher_record_sha256,
        "target_cache_manifest_sha256": identity.target_cache_manifest_sha256,
    }
    for field, value in expected.items():
        if getattr(eligibility, field) != value:
            raise StudentSealingError(f"eligibility {field} mismatch")

    checkpoint_sha256 = _terminal_checkpoint_sha256(execution)
    if eligibility.checkpoint_sha256 != checkpoint_sha256:
        raise StudentSealingError("eligibility checkpoint hash mismatch")

    _require_sha256(dense_output_sha256, name="dense_output_sha256")
    if eligibility.dense_output_sha256 != dense_output_sha256:
        raise StudentSealingError("eligibility dense output hash mismatch")

    return SealedStudentModelRecord(
        schema_version=SEAL_SCHEMA_VERSION,
        estimand=eligibility.estimand,
        identity_sha256=identity.identity_sha256,
        architecture_ref=identity.architecture_ref,
        architecture_record_sha256=identity.architecture_record_sha256,
        task_identity_sha256=identity.task_identity_sha256,
        teacher_record_sha256=identity.teacher_record_sha256,
        target_cache_manifest_sha256=identity.target_cache_manifest_sha256,
        training_config_sha256=identity.training_config_sha256,
        backend_qualification_sha256=identity.backend_qualification_sha256,
        checkpoint_sha256=checkpoint_sha256,
        dense_output_sha256=dense_output_sha256,
        eligibility_record_sha256=eligibility.record_sha256,
    )


def release_student_for_discovery(
    *,
    sealed: SealedStudentModelRecord,
    eligibility: EligibilityRecord,
) -> DiscoveryReleaseRecord:
    """Create a release record only for passed, sealed, hash-consistent students."""
    if not isinstance(sealed, SealedStudentModelRecord):
        raise StudentSealingError("sealed must be SealedStudentModelRecord")
    if not isinstance(eligibility, (HardEligibilityRecord, SoftEligibilityRecord)):
        raise StudentSealingError("unsupported eligibility record")
    if eligibility.status != "passed":
        raise StudentSealingError("ineligible attempt cannot enter discovery")
    if sealed.eligibility_record_sha256 != eligibility.record_sha256:
        raise StudentSealingError("eligibility link hash mismatch")

    linked_fields = (
        "estimand",
        "identity_sha256",
        "architecture_ref",
        "architecture_record_sha256",
        "task_identity_sha256",
        "teacher_record_sha256",
        "target_cache_manifest_sha256",
        "checkpoint_sha256",
        "dense_output_sha256",
    )
    for field in linked_fields:
        if getattr(sealed, field) != getattr(eligibility, field):
            raise StudentSealingError(f"sealed {field} mismatch")

    return DiscoveryReleaseRecord(
        schema_version=RELEASE_SCHEMA_VERSION,
        estimand=sealed.estimand,
        identity_sha256=sealed.identity_sha256,
        architecture_ref=sealed.architecture_ref,
        architecture_record_sha256=sealed.architecture_record_sha256,
        task_identity_sha256=sealed.task_identity_sha256,
        teacher_record_sha256=sealed.teacher_record_sha256,
        target_cache_manifest_sha256=sealed.target_cache_manifest_sha256,
        training_config_sha256=sealed.training_config_sha256,
        backend_qualification_sha256=sealed.backend_qualification_sha256,
        checkpoint_sha256=sealed.checkpoint_sha256,
        dense_output_sha256=sealed.dense_output_sha256,
        eligibility_record_sha256=sealed.eligibility_record_sha256,
        sealed_model_record_sha256=sealed.record_sha256,
        release_status="released",
    )

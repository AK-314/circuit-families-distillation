"""Architecture-aware Stage 12-P2 technical eligibility and failure records."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from circuit_families.stage6b import decision_sha256
from circuit_families.stage6c import (
    TechnicalSoftPolicy,
    centred_soft_output_sha256,
    technical_soft_policy_sha256,
)
from circuit_families.stage6c.soft_target import (
    centre_logits_across_classes,
)

from .engine import StudentAttemptExecution
from .training import StudentTrainingIdentity

ELIGIBILITY_SCHEMA_VERSION = "stage12p2-eligibility/v1"
FAILURE_SCHEMA_VERSION = "stage12p2-attempt-failure/v1"

EligibilityStatus = Literal["passed", "ineligible"]
FailureStatus = Literal[
    "optimization-failed",
    "numerical-failed",
    "interrupted",
    "unavailable",
]


class StudentEligibilityError(ValueError):
    """Raised when P2 eligibility evidence is inconsistent."""


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
        raise StudentEligibilityError(f"{name} must be lowercase SHA-256")


def _require_nonempty(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise StudentEligibilityError(f"{name} must be a non-empty string")


def _validate_execution_identity(
    *,
    execution: StudentAttemptExecution,
    identity: StudentTrainingIdentity,
) -> None:
    if not isinstance(execution, StudentAttemptExecution):
        raise StudentEligibilityError("execution must be StudentAttemptExecution")
    if not isinstance(identity, StudentTrainingIdentity):
        raise StudentEligibilityError("identity must be StudentTrainingIdentity")
    if execution.identity_sha256 != identity.identity_sha256:
        raise StudentEligibilityError("execution identity hash disagrees with training identity")
    if execution.architecture_ref != identity.architecture_ref:
        raise StudentEligibilityError("execution architecture disagrees with training identity")


def _completed_checkpoint_sha256(
    execution: StudentAttemptExecution,
) -> str:
    if execution.status != "completed":
        raise StudentEligibilityError("eligibility requires a completed training execution")
    terminal = tuple(entry for entry in execution.checkpoints.entries if entry.role == "terminal")
    if len(terminal) != 1:
        raise StudentEligibilityError(
            "completed execution must contain exactly one terminal checkpoint"
        )
    _require_sha256(
        terminal[0].file_sha256,
        name="terminal checkpoint SHA-256",
    )
    return terminal[0].file_sha256


def _dense_tensor_sha256(tensor: torch.Tensor) -> str:
    if not isinstance(tensor, torch.Tensor):
        raise StudentEligibilityError("dense output must be a torch.Tensor")
    if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise StudentEligibilityError("dense output must be a non-empty rank-2 tensor")
    if not tensor.dtype.is_floating_point:
        raise StudentEligibilityError("dense output tensor must use a floating dtype")
    if not bool(torch.isfinite(tensor).all().item()):
        raise StudentEligibilityError("dense output tensor must contain only finite values")
    cpu = tensor.detach().to("cpu").contiguous()
    material = {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "values_sha256": hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest(),
    }
    return _sha256_json(material)


def _condition_estimand(
    identity: StudentTrainingIdentity,
) -> Literal["hard", "soft"]:
    if identity.distillation_condition == "hard_target":
        return "hard"
    if identity.distillation_condition == "soft_target":
        return "soft"
    raise StudentEligibilityError("training identity has unsupported distillation condition")


@dataclass(frozen=True)
class HardEligibilityRecord:
    schema_version: str
    estimand: Literal["hard"]
    identity_sha256: str
    architecture_ref: str
    architecture_record_sha256: str
    task_identity_sha256: str
    teacher_record_sha256: str
    target_cache_manifest_sha256: str
    checkpoint_sha256: str
    dense_output_sha256: str
    ordering_ref: str
    ordered_input_ids_sha256: str
    agreement_count: int
    total_count: int
    teacher_decisions_sha256: str
    student_decisions_sha256: str
    status: EligibilityStatus
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != ELIGIBILITY_SCHEMA_VERSION:
            raise StudentEligibilityError("invalid eligibility schema version")
        if self.estimand != "hard":
            raise StudentEligibilityError("HardEligibilityRecord estimand must be hard")
        for name in (
            "identity_sha256",
            "architecture_record_sha256",
            "task_identity_sha256",
            "teacher_record_sha256",
            "target_cache_manifest_sha256",
            "checkpoint_sha256",
            "dense_output_sha256",
            "ordered_input_ids_sha256",
            "teacher_decisions_sha256",
            "student_decisions_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        for name in ("architecture_ref", "ordering_ref"):
            _require_nonempty(getattr(self, name), name=name)
        if self.total_count <= 0:
            raise StudentEligibilityError("total_count must be positive")
        if not 0 <= self.agreement_count <= self.total_count:
            raise StudentEligibilityError("agreement_count is invalid")
        expected = "passed" if self.agreement_count == self.total_count else "ineligible"
        if self.status != expected:
            raise StudentEligibilityError("hard eligibility status disagrees with exact agreement")
        if self.scientific_data is not False:
            raise StudentEligibilityError("scientific_data must be false")
        if self.production_eligible is not False:
            raise StudentEligibilityError("production_eligible must be false")

    def to_mapping(self) -> dict[str, object]:
        return dict(self.__dict__)

    @property
    def record_sha256(self) -> str:
        return _sha256_json(self.to_mapping())


@dataclass(frozen=True)
class SoftEligibilityRecord:
    schema_version: str
    estimand: Literal["soft"]
    identity_sha256: str
    architecture_ref: str
    architecture_record_sha256: str
    task_identity_sha256: str
    teacher_record_sha256: str
    target_cache_manifest_sha256: str
    checkpoint_sha256: str
    dense_output_sha256: str
    ordering_ref: str
    ordered_input_ids_sha256: str
    policy_ref: str
    policy_sha256: str
    discrepancy_metric_ref: str
    discrepancy: float
    tolerance: float
    tolerance_passed: bool
    argmax_requirement_applied: bool
    argmax_agreement_count: int
    total_count: int
    argmax_rule_passed: bool
    teacher_soft_output_sha256: str
    student_soft_output_sha256: str
    status: EligibilityStatus
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != ELIGIBILITY_SCHEMA_VERSION:
            raise StudentEligibilityError("invalid eligibility schema version")
        if self.estimand != "soft":
            raise StudentEligibilityError("SoftEligibilityRecord estimand must be soft")
        for name in (
            "identity_sha256",
            "architecture_record_sha256",
            "task_identity_sha256",
            "teacher_record_sha256",
            "target_cache_manifest_sha256",
            "checkpoint_sha256",
            "dense_output_sha256",
            "ordered_input_ids_sha256",
            "policy_sha256",
            "teacher_soft_output_sha256",
            "student_soft_output_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        for name in (
            "architecture_ref",
            "ordering_ref",
            "policy_ref",
            "discrepancy_metric_ref",
        ):
            _require_nonempty(getattr(self, name), name=name)
        if (
            not math.isfinite(self.discrepancy)
            or not math.isfinite(self.tolerance)
            or self.tolerance < 0.0
        ):
            raise StudentEligibilityError("soft discrepancy/tolerance must be finite and valid")
        if self.total_count <= 0:
            raise StudentEligibilityError("total_count must be positive")
        if not 0 <= self.argmax_agreement_count <= self.total_count:
            raise StudentEligibilityError("argmax_agreement_count is invalid")
        eligible = self.tolerance_passed and self.argmax_rule_passed
        expected = "passed" if eligible else "ineligible"
        if self.status != expected:
            raise StudentEligibilityError("soft eligibility status disagrees with policy result")
        if self.scientific_data is not False:
            raise StudentEligibilityError("scientific_data must be false")
        if self.production_eligible is not False:
            raise StudentEligibilityError("production_eligible must be false")

    def to_mapping(self) -> dict[str, object]:
        return dict(self.__dict__)

    @property
    def record_sha256(self) -> str:
        return _sha256_json(self.to_mapping())


@dataclass(frozen=True)
class AttemptFailureRecord:
    schema_version: str
    estimand: Literal["hard", "soft"]
    identity_sha256: str
    architecture_ref: str
    architecture_record_sha256: str
    task_identity_sha256: str
    teacher_record_sha256: str
    target_cache_manifest_sha256: str
    status: FailureStatus
    reason: str
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != FAILURE_SCHEMA_VERSION:
            raise StudentEligibilityError("invalid failure schema version")
        if self.estimand not in {"hard", "soft"}:
            raise StudentEligibilityError("invalid failure estimand")
        for name in (
            "identity_sha256",
            "architecture_record_sha256",
            "task_identity_sha256",
            "teacher_record_sha256",
            "target_cache_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_nonempty(self.architecture_ref, name="architecture_ref")
        _require_nonempty(self.reason, name="reason")
        if self.status not in {
            "optimization-failed",
            "numerical-failed",
            "interrupted",
            "unavailable",
        }:
            raise StudentEligibilityError("invalid failure status")
        if self.scientific_data is not False:
            raise StudentEligibilityError("scientific_data must be false")
        if self.production_eligible is not False:
            raise StudentEligibilityError("production_eligible must be false")

    def to_mapping(self) -> dict[str, object]:
        return dict(self.__dict__)

    @property
    def record_sha256(self) -> str:
        return _sha256_json(self.to_mapping())


def evaluate_hard_student_eligibility(
    *,
    execution: StudentAttemptExecution,
    identity: StudentTrainingIdentity,
    teacher_decisions: Sequence[int] | torch.Tensor,
    student_dense_logits: torch.Tensor,
    ordering_ref: str,
    ordered_input_ids_sha256: str,
    domain_complete: bool,
) -> HardEligibilityRecord:
    """Apply Stage 6B exact argmax semantics to the supplied complete domain."""
    _validate_execution_identity(
        execution=execution,
        identity=identity,
    )
    if _condition_estimand(identity) != "hard":
        raise StudentEligibilityError("hard eligibility requires hard_target identity")
    checkpoint_sha256 = _completed_checkpoint_sha256(execution)
    if domain_complete is not True:
        raise StudentEligibilityError(
            "hard eligibility requires the supplied domain to be complete"
        )
    _require_nonempty(ordering_ref, name="ordering_ref")
    _require_sha256(
        ordered_input_ids_sha256,
        name="ordered_input_ids_sha256",
    )

    if isinstance(teacher_decisions, torch.Tensor):
        teacher = teacher_decisions.detach().to("cpu").reshape(-1)
        if teacher.dtype != torch.int64:
            raise StudentEligibilityError("teacher decisions tensor must use torch.int64")
        teacher_values = tuple(int(value) for value in teacher.tolist())
    else:
        teacher_values = tuple(teacher_decisions)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in teacher_values):
            raise StudentEligibilityError("teacher decisions must contain integers")

    dense_sha256 = _dense_tensor_sha256(student_dense_logits)
    if student_dense_logits.shape[0] != len(teacher_values):
        raise StudentEligibilityError(
            "teacher decisions and student dense outputs must have equal length"
        )
    if len(teacher_values) == 0:
        raise StudentEligibilityError("eligibility domain must be non-empty")

    student_values = tuple(
        int(value)
        for value in torch.argmax(
            student_dense_logits.detach().to("cpu"),
            dim=1,
        ).tolist()
    )
    agreement_count = sum(
        teacher_value == student_value
        for teacher_value, student_value in zip(
            teacher_values,
            student_values,
            strict=True,
        )
    )

    return HardEligibilityRecord(
        schema_version=ELIGIBILITY_SCHEMA_VERSION,
        estimand="hard",
        identity_sha256=identity.identity_sha256,
        architecture_ref=identity.architecture_ref,
        architecture_record_sha256=identity.architecture_record_sha256,
        task_identity_sha256=identity.task_identity_sha256,
        teacher_record_sha256=identity.teacher_record_sha256,
        target_cache_manifest_sha256=identity.target_cache_manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        dense_output_sha256=dense_sha256,
        ordering_ref=ordering_ref,
        ordered_input_ids_sha256=ordered_input_ids_sha256,
        agreement_count=agreement_count,
        total_count=len(teacher_values),
        teacher_decisions_sha256=decision_sha256(teacher_values),
        student_decisions_sha256=decision_sha256(student_values),
        status=("passed" if agreement_count == len(teacher_values) else "ineligible"),
    )


def evaluate_soft_student_eligibility(
    *,
    execution: StudentAttemptExecution,
    identity: StudentTrainingIdentity,
    teacher_logits: torch.Tensor,
    student_dense_logits: torch.Tensor,
    policy: TechnicalSoftPolicy,
    ordering_ref: str,
    ordered_input_ids_sha256: str,
    domain_complete: bool,
) -> SoftEligibilityRecord:
    """Apply Stage 6C gauge-invariant semantics to the supplied complete domain."""
    _validate_execution_identity(
        execution=execution,
        identity=identity,
    )
    if _condition_estimand(identity) != "soft":
        raise StudentEligibilityError("soft eligibility requires soft_target identity")
    checkpoint_sha256 = _completed_checkpoint_sha256(execution)
    if domain_complete is not True:
        raise StudentEligibilityError(
            "soft eligibility requires the supplied domain to be complete"
        )
    if not isinstance(policy, TechnicalSoftPolicy):
        raise StudentEligibilityError("policy must be TechnicalSoftPolicy")
    _require_nonempty(ordering_ref, name="ordering_ref")
    _require_sha256(
        ordered_input_ids_sha256,
        name="ordered_input_ids_sha256",
    )

    _dense_tensor_sha256(teacher_logits)
    dense_sha256 = _dense_tensor_sha256(student_dense_logits)
    if teacher_logits.shape != student_dense_logits.shape:
        raise StudentEligibilityError("teacher and student soft-output shapes must match")

    teacher_centred = centre_logits_across_classes(teacher_logits)
    student_centred = centre_logits_across_classes(student_dense_logits)
    discrepancy = float(torch.mean((student_centred - teacher_centred).square()).item())
    if not math.isfinite(discrepancy):
        raise StudentEligibilityError("soft discrepancy must be finite")

    tolerance = float(policy.tolerance.candidate_value)
    tolerance_passed = discrepancy <= tolerance
    teacher_argmax = torch.argmax(teacher_logits, dim=1)
    student_argmax = torch.argmax(student_dense_logits, dim=1)
    agreement_count = int(torch.sum(teacher_argmax == student_argmax).item())
    total_count = int(teacher_logits.shape[0])
    argmax_required = bool(
        policy.argmax_requirement is not None and policy.argmax_requirement.candidate_required
    )
    argmax_rule_passed = agreement_count == total_count if argmax_required else True
    eligible = tolerance_passed and argmax_rule_passed

    return SoftEligibilityRecord(
        schema_version=ELIGIBILITY_SCHEMA_VERSION,
        estimand="soft",
        identity_sha256=identity.identity_sha256,
        architecture_ref=identity.architecture_ref,
        architecture_record_sha256=identity.architecture_record_sha256,
        task_identity_sha256=identity.task_identity_sha256,
        teacher_record_sha256=identity.teacher_record_sha256,
        target_cache_manifest_sha256=identity.target_cache_manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        dense_output_sha256=dense_sha256,
        ordering_ref=ordering_ref,
        ordered_input_ids_sha256=ordered_input_ids_sha256,
        policy_ref=policy.policy_ref,
        policy_sha256=technical_soft_policy_sha256(policy),
        discrepancy_metric_ref=policy.tolerance.metric_ref,
        discrepancy=discrepancy,
        tolerance=tolerance,
        tolerance_passed=tolerance_passed,
        argmax_requirement_applied=argmax_required,
        argmax_agreement_count=agreement_count,
        total_count=total_count,
        argmax_rule_passed=argmax_rule_passed,
        teacher_soft_output_sha256=centred_soft_output_sha256(teacher_logits),
        student_soft_output_sha256=centred_soft_output_sha256(student_dense_logits),
        status="passed" if eligible else "ineligible",
    )


def record_noncompleted_attempt(
    *,
    execution: StudentAttemptExecution,
    identity: StudentTrainingIdentity,
) -> AttemptFailureRecord:
    """Preserve every non-completed technical attempt as an explicit record."""
    _validate_execution_identity(
        execution=execution,
        identity=identity,
    )
    mapping: dict[str, FailureStatus] = {
        "failed": "optimization-failed",
        "numerical-failure": "numerical-failed",
        "interrupted": "interrupted",
        "unavailable": "unavailable",
    }
    if execution.status not in mapping:
        raise StudentEligibilityError("completed execution is not a failure record")
    return AttemptFailureRecord(
        schema_version=FAILURE_SCHEMA_VERSION,
        estimand=_condition_estimand(identity),
        identity_sha256=identity.identity_sha256,
        architecture_ref=identity.architecture_ref,
        architecture_record_sha256=identity.architecture_record_sha256,
        task_identity_sha256=identity.task_identity_sha256,
        teacher_record_sha256=identity.teacher_record_sha256,
        target_cache_manifest_sha256=identity.target_cache_manifest_sha256,
        status=mapping[execution.status],
        reason=execution.reason,
    )

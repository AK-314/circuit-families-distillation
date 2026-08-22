"""Full-domain technical soft eligibility for Stage 6C Part D.

This module evaluates only explicitly injected technical candidates. It does
not select or authorize a production soft policy and does not resolve UD-006.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any

import torch

from circuit_families.interpretability.centred_logit_fidelity import (
    centre_logits_across_classes,
)
from circuit_families.stage4_condition_identity import (
    VERSION_REFERENCE_RE,
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    parse_condition_id,
)
from circuit_families.stage5bc.target_cache import FULL_DOMAIN_EXAMPLE_COUNT
from circuit_families.stage6c.soft_target import TechnicalSoftPolicy

SOFT_OUTPUT_HASH_VERSION = "stage6c-centred-soft-output/v1"
SOFT_ELIGIBILITY_CRITERION = "soft_policy_reference"
TECHNICAL_SOFT_DISCREPANCY_METRIC = "technical-centred-logit-mse/v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OUTPUT_ROLES = ("soft_target_teacher", "soft_target_student")


class SoftEligibilityError(ValueError):
    """Raised when full-domain soft eligibility evidence is invalid."""


def _normalise_logits(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise SoftEligibilityError("soft outputs must be torch tensors")
    if value.ndim != 2 or not value.is_floating_point():
        raise SoftEligibilityError("soft outputs must be rank-2 floating tensors")
    if value.shape[1] <= 0:
        raise SoftEligibilityError("soft outputs must contain at least one class")
    result = value.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()
    if not bool(torch.isfinite(result).all()):
        raise SoftEligibilityError("soft outputs must contain only finite values")
    return result


@dataclass(frozen=True)
class CanonicalSoftOutput:
    """One sealed soft-output matrix with canonical identity and order evidence."""

    role: str
    condition_id: str
    ordering_ref: str
    ordered_input_ids_sha256: str
    logits: torch.Tensor
    record_status: str

    def __post_init__(self) -> None:
        if self.role not in _OUTPUT_ROLES:
            raise SoftEligibilityError(f"unsupported soft-output role: {self.role!r}")
        if not isinstance(self.condition_id, str) or not self.condition_id:
            raise SoftEligibilityError("condition_id must be non-empty")
        if (
            not isinstance(self.ordering_ref, str)
            or not VERSION_REFERENCE_RE.fullmatch(self.ordering_ref)
        ):
            raise SoftEligibilityError("ordering_ref must be a version reference")
        if (
            not isinstance(self.ordered_input_ids_sha256, str)
            or not _SHA256_RE.fullmatch(self.ordered_input_ids_sha256)
        ):
            raise SoftEligibilityError(
                "ordered_input_ids_sha256 must be lowercase SHA-256 hex"
            )
        if self.record_status != "sealed":
            raise SoftEligibilityError("soft eligibility requires sealed outputs")
        object.__setattr__(self, "logits", _normalise_logits(self.logits))


def canonical_centred_soft_output_bytes(logits: torch.Tensor) -> bytes:
    """Serialize centred float64 outputs deterministically without reordering."""
    normalised = _normalise_logits(logits)
    try:
        centred = centre_logits_across_classes(normalised).contiguous()
    except (FloatingPointError, TypeError, ValueError) as exc:
        raise SoftEligibilityError(f"invalid soft outputs: {exc}") from exc
    header = json.dumps(
        {
            "class_count": centred.shape[1],
            "example_count": centred.shape[0],
            "schema_version": SOFT_OUTPUT_HASH_VERSION,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    values = b"".join(struct.pack(">d", float(value)) for value in centred.view(-1))
    return header + b"\n" + values


def centred_soft_output_sha256(logits: torch.Tensor) -> str:
    """Hash canonical class-centred outputs, invariant to per-input gauge shifts."""
    return hashlib.sha256(canonical_centred_soft_output_bytes(logits)).hexdigest()


def technical_soft_policy_sha256(policy: TechnicalSoftPolicy) -> str:
    """Hash the complete injected technical policy deterministically."""
    if not isinstance(policy, TechnicalSoftPolicy):
        raise SoftEligibilityError("policy must be TechnicalSoftPolicy")
    encoded = json.dumps(
        policy.to_mapping(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SoftEligibilityEvidence:
    """Recomputable technical evidence for one full-domain soft comparison."""

    criterion: str
    policy_ref: str
    policy_sha256: str
    teacher_condition_id: str
    student_condition_id: str
    ordering_ref: str
    ordered_input_ids_sha256: str
    total_count: int
    discrepancy_metric_ref: str
    discrepancy: float
    tolerance: float
    tolerance_comparison: str
    tolerance_passed: bool
    argmax_requirement_applied: bool
    argmax_agreement_count: int
    argmax_rule_passed: bool
    eligible: bool
    teacher_soft_output_sha256: str
    student_soft_output_sha256: str
    soft_output_hash_version: str
    scientific_data: bool = False
    production_eligible: bool = False
    resolves_ud006: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "argmax_agreement_count": self.argmax_agreement_count,
            "argmax_requirement_applied": self.argmax_requirement_applied,
            "argmax_rule_passed": self.argmax_rule_passed,
            "criterion": self.criterion,
            "discrepancy": self.discrepancy,
            "discrepancy_metric_ref": self.discrepancy_metric_ref,
            "eligible": self.eligible,
            "ordered_input_ids_sha256": self.ordered_input_ids_sha256,
            "ordering_ref": self.ordering_ref,
            "policy_ref": self.policy_ref,
            "policy_sha256": self.policy_sha256,
            "production_eligible": self.production_eligible,
            "resolves_ud006": self.resolves_ud006,
            "scientific_data": self.scientific_data,
            "soft_output_hash_version": self.soft_output_hash_version,
            "student_condition_id": self.student_condition_id,
            "student_soft_output_sha256": self.student_soft_output_sha256,
            "teacher_condition_id": self.teacher_condition_id,
            "teacher_soft_output_sha256": self.teacher_soft_output_sha256,
            "tolerance": self.tolerance,
            "tolerance_comparison": self.tolerance_comparison,
            "tolerance_passed": self.tolerance_passed,
            "total_count": self.total_count,
        }


def _parse_output_identity(
    output: CanonicalSoftOutput,
    *,
    stage3: Stage3AvailabilityIndex,
    role: str,
):
    if output.role != role:
        raise SoftEligibilityError(f"expected {role!r}, received {output.role!r}")
    try:
        identity = parse_condition_id(output.condition_id, stage3)
    except ConditionIdentityError as exc:
        raise SoftEligibilityError(f"invalid {role} identity: {exc}") from exc
    if role == "soft_target_teacher":
        if identity.depth != 3 or identity.distillation_condition != "soft_target":
            raise SoftEligibilityError(
                "teacher soft outputs require a depth-3 soft_target identity"
            )
    elif identity.depth != 4 or identity.distillation_condition != "soft_target":
        raise SoftEligibilityError(
            "student soft outputs require a depth-4 soft_target identity"
        )
    return identity


def evaluate_soft_target_eligibility(
    *,
    teacher: CanonicalSoftOutput,
    student: CanonicalSoftOutput,
    policy: TechnicalSoftPolicy,
    stage3: Stage3AvailabilityIndex,
) -> SoftEligibilityEvidence:
    """Apply one injected technical policy over exactly 12,769 canonical inputs."""
    if not isinstance(teacher, CanonicalSoftOutput):
        raise SoftEligibilityError("teacher must be CanonicalSoftOutput")
    if not isinstance(student, CanonicalSoftOutput):
        raise SoftEligibilityError("student must be CanonicalSoftOutput")
    if not isinstance(policy, TechnicalSoftPolicy):
        raise SoftEligibilityError("policy must be TechnicalSoftPolicy")
    if not isinstance(stage3, Stage3AvailabilityIndex):
        raise SoftEligibilityError("stage3 must be Stage3AvailabilityIndex")
    if policy.tolerance.metric_ref != TECHNICAL_SOFT_DISCREPANCY_METRIC:
        raise SoftEligibilityError("unsupported injected technical discrepancy metric")

    teacher_identity = _parse_output_identity(
        teacher,
        stage3=stage3,
        role="soft_target_teacher",
    )
    student_identity = _parse_output_identity(
        student,
        stage3=stage3,
        role="soft_target_student",
    )
    if teacher.condition_id != policy.representation.teacher_condition_id:
        raise SoftEligibilityError("teacher output identity does not match policy")
    if (
        teacher_identity.teacher_seed != student_identity.teacher_seed
        or teacher_identity.phase != student_identity.phase
    ):
        raise SoftEligibilityError(
            "teacher and student identities must share teacher_seed and phase"
        )
    if teacher.logits.shape != student.logits.shape:
        raise SoftEligibilityError("teacher and student soft-output shapes must match")
    if teacher.logits.shape[0] != FULL_DOMAIN_EXAMPLE_COUNT:
        raise SoftEligibilityError(
            f"soft eligibility requires exactly {FULL_DOMAIN_EXAMPLE_COUNT} inputs"
        )
    representation = policy.representation
    if (
        teacher.ordering_ref != student.ordering_ref
        or teacher.ordered_input_ids_sha256 != student.ordered_input_ids_sha256
        or teacher.ordering_ref != representation.ordering_ref
        or teacher.ordered_input_ids_sha256
        != representation.ordered_input_ids_sha256
    ):
        raise SoftEligibilityError("soft-output canonical ordering must match exactly")

    teacher_centred = centre_logits_across_classes(teacher.logits)
    student_centred = centre_logits_across_classes(student.logits)
    discrepancy = float(
        torch.mean((student_centred - teacher_centred).square()).item()
    )
    if not math.isfinite(discrepancy):
        raise SoftEligibilityError("soft discrepancy must be finite")
    tolerance = float(policy.tolerance.candidate_value)
    tolerance_passed = discrepancy <= tolerance
    teacher_argmax = torch.argmax(teacher.logits, dim=1)
    student_argmax = torch.argmax(student.logits, dim=1)
    agreement_count = int(torch.sum(teacher_argmax == student_argmax).item())
    argmax_required = bool(
        policy.argmax_requirement is not None
        and policy.argmax_requirement.candidate_required
    )
    argmax_rule_passed = (
        agreement_count == FULL_DOMAIN_EXAMPLE_COUNT if argmax_required else True
    )

    return SoftEligibilityEvidence(
        criterion=SOFT_ELIGIBILITY_CRITERION,
        policy_ref=policy.policy_ref,
        policy_sha256=technical_soft_policy_sha256(policy),
        teacher_condition_id=teacher.condition_id,
        student_condition_id=student.condition_id,
        ordering_ref=teacher.ordering_ref,
        ordered_input_ids_sha256=teacher.ordered_input_ids_sha256,
        total_count=FULL_DOMAIN_EXAMPLE_COUNT,
        discrepancy_metric_ref=policy.tolerance.metric_ref,
        discrepancy=discrepancy,
        tolerance=tolerance,
        tolerance_comparison=policy.tolerance.comparison,
        tolerance_passed=tolerance_passed,
        argmax_requirement_applied=argmax_required,
        argmax_agreement_count=agreement_count,
        argmax_rule_passed=argmax_rule_passed,
        eligible=tolerance_passed and argmax_rule_passed,
        teacher_soft_output_sha256=centred_soft_output_sha256(teacher.logits),
        student_soft_output_sha256=centred_soft_output_sha256(student.logits),
        soft_output_hash_version=SOFT_OUTPUT_HASH_VERSION,
    )

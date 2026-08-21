"""Injected hard-label loss and deterministic full-domain eligibility.

This module owns only the Stage 6B Part C core. It reuses the accepted Stage
5B trainer boundary, emits technical recomputation evidence, and does not own
training, attempt policy, checkpoint sealing, or production configuration.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from circuit_families.stage4_condition_identity import (
    VERSION_REFERENCE_RE,
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    parse_condition_id,
)
from circuit_families.stage5bc.student_trainer import PreparedTargets
from circuit_families.stage5bc.target_cache import FULL_DOMAIN_EXAMPLE_COUNT

DECISION_VECTOR_HASH_VERSION = "hard-target-decision-vector/v1"
HARD_ELIGIBILITY_CRITERION = "exact_teacher_argmax_agreement"
HARD_LABEL_LOSS_KINDS = ("cross_entropy",)
HARD_LABEL_REDUCTIONS = ("mean", "sum")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VECTOR_ROLES = ("direct_teacher", "hard_target_student")


class HardTargetLossError(ValueError):
    """Raised when injected hard-label loss inputs are inconsistent."""


class HardTargetEligibilityError(ValueError):
    """Raised when full-domain hard eligibility evidence is invalid."""


def _normalize_decisions(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise HardTargetEligibilityError("decision vector must be rank 1")
        if value.dtype != torch.int64:
            raise HardTargetEligibilityError("decision tensor must use int64")
        items = value.detach().cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
    else:
        raise HardTargetEligibilityError(
            "decisions must be an integer sequence or int64 tensor"
        )

    if any(isinstance(item, bool) or not isinstance(item, int) for item in items):
        raise HardTargetEligibilityError("decision vector must contain integers")
    if any(item < 0 for item in items):
        raise HardTargetEligibilityError("decision values must be non-negative")
    return tuple(items)


@dataclass(frozen=True)
class CanonicalDecisionVector:
    """One immutable decision vector with canonical identity and order evidence."""

    role: str
    condition_id: str
    ordering_ref: str
    ordered_input_ids_sha256: str
    decisions: tuple[int, ...] | Sequence[int] | torch.Tensor

    def __post_init__(self) -> None:
        if self.role not in _VECTOR_ROLES:
            raise HardTargetEligibilityError(
                f"unsupported decision-vector role: {self.role!r}"
            )
        if not isinstance(self.condition_id, str) or not self.condition_id:
            raise HardTargetEligibilityError("condition_id must be non-empty")
        if (
            not isinstance(self.ordering_ref, str)
            or not VERSION_REFERENCE_RE.fullmatch(self.ordering_ref)
        ):
            raise HardTargetEligibilityError(
                "ordering_ref must match the Stage 4 version-reference grammar"
            )
        if (
            not isinstance(self.ordered_input_ids_sha256, str)
            or not _SHA256_RE.fullmatch(self.ordered_input_ids_sha256)
        ):
            raise HardTargetEligibilityError(
                "ordered_input_ids_sha256 must be lowercase SHA-256 hex"
            )
        object.__setattr__(self, "decisions", _normalize_decisions(self.decisions))


def canonical_decision_bytes(decisions: Sequence[int]) -> bytes:
    """Serialize one already ordered integer vector deterministically."""
    normalized = _normalize_decisions(decisions)
    payload = {
        "decisions": list(normalized),
        "schema_version": DECISION_VECTOR_HASH_VERSION,
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def decision_sha256(decisions: Sequence[int]) -> str:
    """Hash a canonical decision vector without reordering its elements."""
    return hashlib.sha256(canonical_decision_bytes(decisions)).hexdigest()


class HardLabelLossAdapter:
    """Cross-entropy adapter for the accepted shared trainer lifecycle.

    Every setting is supplied through ``TrainerSettingsBundle.loss``. The
    adapter deliberately has no production settings or constructor defaults.
    """

    required_target_cache_kind = "teacher_argmax"

    def __call__(
        self,
        *,
        outputs: torch.Tensor,
        targets: PreparedTargets,
        settings: Mapping[str, Any],
    ) -> torch.Tensor:
        if not isinstance(targets, PreparedTargets):
            raise HardTargetLossError("targets must be PreparedTargets")
        if targets.cache_kind != self.required_target_cache_kind:
            raise HardTargetLossError(
                "hard-label loss requires teacher_argmax targets"
            )
        if not isinstance(outputs, torch.Tensor):
            raise HardTargetLossError("outputs must be a torch.Tensor")
        if outputs.ndim != 2 or not outputs.is_floating_point():
            raise HardTargetLossError(
                "hard-label outputs must be a rank-2 floating tensor"
            )
        if targets.values.ndim != 1 or targets.values.dtype != torch.int64:
            raise HardTargetLossError(
                "hard-label targets must be a rank-1 int64 tensor"
            )
        if outputs.shape[0] != targets.values.shape[0]:
            raise HardTargetLossError(
                "output and hard-label target counts must match"
            )
        if outputs.shape[1] <= 0:
            raise HardTargetLossError("outputs must contain at least one class")
        if targets.values.numel() and (
            int(targets.values.min().item()) < 0
            or int(targets.values.max().item()) >= outputs.shape[1]
        ):
            raise HardTargetLossError(
                "hard-label target is outside the injected output class range"
            )
        if not isinstance(settings, Mapping):
            raise HardTargetLossError("hard-label settings must be a mapping")
        if set(settings) != {"loss_kind", "reduction"}:
            raise HardTargetLossError(
                "hard-label settings require exactly loss_kind and reduction"
            )
        if settings["loss_kind"] not in HARD_LABEL_LOSS_KINDS:
            raise HardTargetLossError("unsupported injected hard-label loss_kind")
        if settings["reduction"] not in HARD_LABEL_REDUCTIONS:
            raise HardTargetLossError("unsupported injected hard-label reduction")

        return functional.cross_entropy(
            outputs,
            targets.values,
            reduction=settings["reduction"],
        )


@dataclass(frozen=True)
class HardEligibilityEvidence:
    """Recomputable Part C evidence; not a sealed Stage 4 record."""

    criterion: str
    teacher_condition_id: str
    student_condition_id: str
    ordering_ref: str
    ordered_input_ids_sha256: str
    agreement_count: int
    total_count: int
    eligible: bool
    teacher_decisions_sha256: str
    student_decisions_sha256: str
    decision_hash_version: str
    scientific_data: bool = False
    production_eligible: bool = False

    def to_mapping(self) -> dict[str, Any]:
        """Return deterministic metadata sufficient for later record assembly."""
        return {
            "agreement_count": self.agreement_count,
            "criterion": self.criterion,
            "decision_hash_version": self.decision_hash_version,
            "eligible": self.eligible,
            "ordered_input_ids_sha256": self.ordered_input_ids_sha256,
            "ordering_ref": self.ordering_ref,
            "production_eligible": self.production_eligible,
            "scientific_data": self.scientific_data,
            "student_condition_id": self.student_condition_id,
            "student_decisions_sha256": self.student_decisions_sha256,
            "teacher_condition_id": self.teacher_condition_id,
            "teacher_decisions_sha256": self.teacher_decisions_sha256,
            "total_count": self.total_count,
        }


def _parse_expected_identity(
    vector: CanonicalDecisionVector,
    *,
    stage3: Stage3AvailabilityIndex,
    role: str,
):
    if vector.role != role:
        raise HardTargetEligibilityError(
            f"expected {role!r} decision vector, received {vector.role!r}"
        )
    try:
        identity = parse_condition_id(vector.condition_id, stage3)
    except ConditionIdentityError as exc:
        raise HardTargetEligibilityError(
            f"invalid {role} condition identity: {exc}"
        ) from exc

    if role == "direct_teacher":
        if identity.depth != 3 or identity.distillation_condition != "direct_teacher":
            raise HardTargetEligibilityError(
                "teacher decisions require a depth-3 direct_teacher identity"
            )
    elif identity.depth != 4 or identity.distillation_condition != "hard_target":
        raise HardTargetEligibilityError(
            "student decisions require a depth-4 hard_target identity"
        )
    return identity


def evaluate_hard_target_eligibility(
    *,
    teacher: CanonicalDecisionVector,
    student: CanonicalDecisionVector,
    stage3: Stage3AvailabilityIndex,
) -> HardEligibilityEvidence:
    """Evaluate exact teacher/student agreement over the canonical universe."""
    if not isinstance(teacher, CanonicalDecisionVector):
        raise HardTargetEligibilityError("teacher must be CanonicalDecisionVector")
    if not isinstance(student, CanonicalDecisionVector):
        raise HardTargetEligibilityError("student must be CanonicalDecisionVector")
    if not isinstance(stage3, Stage3AvailabilityIndex):
        raise HardTargetEligibilityError("stage3 must be Stage3AvailabilityIndex")

    teacher_identity = _parse_expected_identity(
        teacher,
        stage3=stage3,
        role="direct_teacher",
    )
    student_identity = _parse_expected_identity(
        student,
        stage3=stage3,
        role="hard_target_student",
    )

    for name, vector in (("teacher", teacher), ("student", student)):
        if len(vector.decisions) != FULL_DOMAIN_EXAMPLE_COUNT:
            raise HardTargetEligibilityError(
                f"{name} decision vector must contain exactly "
                f"{FULL_DOMAIN_EXAMPLE_COUNT} elements"
            )

    if (
        teacher_identity.teacher_seed != student_identity.teacher_seed
        or teacher_identity.phase != student_identity.phase
    ):
        raise HardTargetEligibilityError(
            "teacher and student condition identities must share teacher_seed and phase"
        )
    if (
        teacher.ordering_ref != student.ordering_ref
        or teacher.ordered_input_ids_sha256 != student.ordered_input_ids_sha256
    ):
        raise HardTargetEligibilityError(
            "teacher and student input-order identities must match exactly"
        )

    agreement_count = sum(
        teacher_value == student_value
        for teacher_value, student_value in zip(
            teacher.decisions,
            student.decisions,
            strict=True,
        )
    )

    return HardEligibilityEvidence(
        criterion=HARD_ELIGIBILITY_CRITERION,
        teacher_condition_id=teacher.condition_id,
        student_condition_id=student.condition_id,
        ordering_ref=teacher.ordering_ref,
        ordered_input_ids_sha256=teacher.ordered_input_ids_sha256,
        agreement_count=agreement_count,
        total_count=FULL_DOMAIN_EXAMPLE_COUNT,
        eligible=agreement_count == FULL_DOMAIN_EXAMPLE_COUNT,
        teacher_decisions_sha256=decision_sha256(teacher.decisions),
        student_decisions_sha256=decision_sha256(student.decisions),
        decision_hash_version=DECISION_VECTOR_HASH_VERSION,
    )

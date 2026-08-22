"""Soft attempt records, eligible-only sealing, and the narrow release gate.

Frozen Stage 4 payloads are emitted exactly. Soft discrepancy, policy, order,
and output-hash evidence lives in typed sidecars because the frozen schemas do
not permit extension fields.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from circuit_families.stage4_condition_identity import (
    VERSION_REFERENCE_RE,
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    parse_condition_id,
)
from circuit_families.stage4_schema_records import (
    COMPONENT_ABLATION_SOURCE,
    COMPONENT_BASIS_STATUS,
    COMPONENT_COUNT,
    MASKS_SOURCE,
    STAGE8_MASKING_MANIFEST,
)
from circuit_families.stage5bc.attempt_records import (
    TECHNICAL_FAILURE_KINDS,
    TechnicalAttemptLedger,
    TechnicalAttemptRecordError,
    attempt_record_sha256,
)
from circuit_families.stage6c.eligibility import (
    SOFT_ELIGIBILITY_CRITERION,
    SoftEligibilityEvidence,
)

SOFT_FAILURE_KINDS = (
    "training_failure",
    "numerical_failure",
    "tolerance_failure",
    "argmax_rule_failure",
)

_NAMESPACE = "circuit-families-distillation"
_VOCABULARY_VERSION = "common-vocabulary/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Stage6CRecordError(ValueError):
    """Raised when Part D evidence violates a frozen lifecycle interface."""


def canonical_stage6c_record_bytes(record: Mapping[str, Any]) -> bytes:
    """Serialize one small Stage 4-compatible record deterministically."""
    if not isinstance(record, Mapping):
        raise Stage6CRecordError("record must be a mapping")
    try:
        return (
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage6CRecordError(
            f"record is not canonical-JSON serializable: {exc}"
        ) from exc


def stage6c_record_sha256(record: Mapping[str, Any]) -> str:
    """Hash deterministic Stage 6C small-record bytes."""
    return hashlib.sha256(canonical_stage6c_record_bytes(record)).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise Stage6CRecordError(f"{name} must be lowercase SHA-256 hex")
    return value


def _artifact(
    value: Mapping[str, Any],
    *,
    name: str,
    storage_class: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "storage_class",
    }:
        raise Stage6CRecordError(f"{name} keys mismatch")
    path = value["path"]
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise Stage6CRecordError(f"{name}.path must be portable and relative")
    if value["storage_class"] != storage_class:
        raise Stage6CRecordError(
            f"{name}.storage_class must be {storage_class!r}"
        )
    return {
        "path": path,
        "sha256": _require_sha256(value["sha256"], name=f"{name}.sha256"),
        "storage_class": storage_class,
    }


def _attempt_parts(
    record: Mapping[str, Any],
    stage3: Stage3AvailabilityIndex,
):
    if not isinstance(record, Mapping):
        raise Stage6CRecordError("attempt record must be a mapping")
    if record.get("record_type") != "student_attempt":
        raise Stage6CRecordError("attempt record_type must be student_attempt")
    if record.get("schema_version") != "student_attempt/v1":
        raise Stage6CRecordError("attempt schema_version mismatch")
    try:
        identity = parse_condition_id(record["condition_id"], stage3)
    except (KeyError, ConditionIdentityError) as exc:
        raise Stage6CRecordError(f"invalid attempt condition identity: {exc}") from exc
    if identity.depth != 4 or identity.distillation_condition != "soft_target":
        raise Stage6CRecordError("Part D accepts only depth-4 soft_target attempts")
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise Stage6CRecordError("attempt payload must be a mapping")
    for name in ("attempt_index", "retry_index"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Stage6CRecordError(f"attempt {name} must be non-negative")
    if payload.get("attempt_outcome") not in {"succeeded", "failed"}:
        raise Stage6CRecordError("attempt_outcome must be succeeded or failed")
    return identity, payload


def _attempt_reference(record: Mapping[str, Any]) -> dict[str, str]:
    return {
        "record_type": "student_attempt",
        "schema_version": "student_attempt/v1",
        "condition_id": record["condition_id"],
        "record_sha256": attempt_record_sha256(record),
    }


def _technical_failure_kind(payload: Mapping[str, Any]) -> str:
    reason = payload.get("failure_reason")
    if not isinstance(reason, str) or not reason:
        raise Stage6CRecordError("failed attempt requires failure_reason")
    parts = reason.split(":", 3)
    if len(parts) != 4 or parts[:2] != ["technical_failure", "v1"]:
        raise Stage6CRecordError("failed attempt reason lacks frozen taxonomy")
    low_level_kind = parts[2]
    if low_level_kind not in TECHNICAL_FAILURE_KINDS:
        raise Stage6CRecordError("failed attempt has unknown technical failure kind")
    return "numerical_failure" if low_level_kind == "numerical_failure" else "training_failure"


@dataclass(frozen=True)
class SoftEligibilityRecordEvidence:
    """Exact Stage 4 eligibility record plus soft recomputation evidence."""

    stage4_record: Mapping[str, Any]
    stage4_record_sha256: str
    evaluation: SoftEligibilityEvidence
    failure_kinds: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "failure_kinds": list(self.failure_kinds),
            "soft_policy_and_output_evidence": self.evaluation.to_mapping(),
            "stage4_record": copy.deepcopy(dict(self.stage4_record)),
            "stage4_record_sha256": self.stage4_record_sha256,
        }


@dataclass(frozen=True)
class SoftAttemptAssessment:
    """One permanent attempt with distinct zero-or-more soft failure kinds."""

    status: str
    failure_kinds: tuple[str, ...]
    attempt_record: Mapping[str, Any]
    eligibility: SoftEligibilityRecordEvidence | None


def _evaluation_failure_kinds(
    evaluation: SoftEligibilityEvidence,
) -> tuple[str, ...]:
    failures = []
    if not evaluation.tolerance_passed:
        failures.append("tolerance_failure")
    if not evaluation.argmax_rule_passed:
        failures.append("argmax_rule_failure")
    return tuple(failures)


def _emit_soft_eligibility_record(
    attempt_record: Mapping[str, Any],
    evaluation: SoftEligibilityEvidence,
) -> SoftEligibilityRecordEvidence:
    attempt_ref = _attempt_reference(attempt_record)
    payload = attempt_record["payload"]
    failure_kinds = _evaluation_failure_kinds(evaluation)
    record = {
        "namespace": _NAMESPACE,
        "vocabulary_version": _VOCABULARY_VERSION,
        "schema_version": "student_eligibility/v1",
        "record_type": "student_eligibility",
        "record_status": "sealed",
        "condition_id": attempt_record["condition_id"],
        "identity_depth": 4,
        "payload": {
            "attempt_reference": attempt_ref,
            "attempt_index": payload["attempt_index"],
            "retry_index": payload["retry_index"],
            "eligibility_status": "passed" if evaluation.eligible else "failed",
            "criterion": SOFT_ELIGIBILITY_CRITERION,
            "soft_policy_ref": evaluation.policy_ref,
        },
        "provenance": {
            "producer_lane": "lane_b",
            "creation_stage": "stage6c",
            "source_records": [attempt_ref],
        },
    }
    return SoftEligibilityRecordEvidence(
        stage4_record=record,
        stage4_record_sha256=stage6c_record_sha256(record),
        evaluation=evaluation,
        failure_kinds=failure_kinds,
    )


def assess_soft_attempt(
    *,
    attempt_record: Mapping[str, Any],
    stage3: Stage3AvailabilityIndex,
    evaluation: SoftEligibilityEvidence | None = None,
) -> SoftAttemptAssessment:
    """Classify one attempt while retaining every applicable failure kind."""
    _, payload = _attempt_parts(attempt_record, stage3)
    if payload["attempt_outcome"] == "failed":
        if evaluation is not None:
            raise Stage6CRecordError(
                "failed training attempt cannot carry eligibility evaluation"
            )
        failure = _technical_failure_kind(payload)
        return SoftAttemptAssessment(
            status="failed",
            failure_kinds=(failure,),
            attempt_record=copy.deepcopy(dict(attempt_record)),
            eligibility=None,
        )

    if not isinstance(evaluation, SoftEligibilityEvidence):
        raise Stage6CRecordError(
            "completed soft attempt requires full-domain eligibility evidence"
        )
    if evaluation.student_condition_id != attempt_record["condition_id"]:
        raise Stage6CRecordError("eligibility student identity does not match attempt")
    if evaluation.criterion != SOFT_ELIGIBILITY_CRITERION:
        raise Stage6CRecordError("soft eligibility criterion mismatch")
    if evaluation.total_count != 12_769:
        raise Stage6CRecordError("soft eligibility must evaluate exactly 12769 inputs")
    failures = _evaluation_failure_kinds(evaluation)
    if evaluation.eligible != (not failures):
        raise Stage6CRecordError("soft eligibility boolean/evidence mismatch")
    eligibility = _emit_soft_eligibility_record(attempt_record, evaluation)
    return SoftAttemptAssessment(
        status="eligible" if evaluation.eligible else "failed",
        failure_kinds=failures,
        attempt_record=copy.deepcopy(dict(attempt_record)),
        eligibility=eligibility,
    )


class SoftAttemptLedger:
    """Append-only accounting layered over the accepted Stage 5B–5C ledger."""

    def __init__(self) -> None:
        self._attempts = TechnicalAttemptLedger()
        self._assessments: dict[tuple[str, int, int], SoftAttemptAssessment] = {}

    def add(self, assessment: SoftAttemptAssessment) -> str:
        if not isinstance(assessment, SoftAttemptAssessment):
            raise Stage6CRecordError("assessment must be SoftAttemptAssessment")
        if assessment.status not in {"eligible", "failed"}:
            raise Stage6CRecordError("unknown soft-attempt status")
        if any(kind not in SOFT_FAILURE_KINDS for kind in assessment.failure_kinds):
            raise Stage6CRecordError("unknown soft-attempt failure kind")
        record = assessment.attempt_record
        payload = record["payload"]
        key = (
            record["condition_id"],
            payload["attempt_index"],
            payload["retry_index"],
        )
        try:
            digest = self._attempts.add(record)
        except TechnicalAttemptRecordError as exc:
            raise Stage6CRecordError(str(exc)) from exc
        self._assessments[key] = copy.deepcopy(assessment)
        return digest

    @property
    def attempt_count(self) -> int:
        return len(self._assessments)

    def failure_count(self, failure_kind: str) -> int:
        if failure_kind not in SOFT_FAILURE_KINDS:
            raise Stage6CRecordError("unknown soft-attempt failure kind")
        return sum(
            failure_kind in item.failure_kinds
            for item in self._assessments.values()
        )

    def assessments(self) -> tuple[SoftAttemptAssessment, ...]:
        return tuple(
            copy.deepcopy(self._assessments[key]) for key in sorted(self._assessments)
        )


def _component_basis() -> dict[str, Any]:
    return {
        "component_count": COMPONENT_COUNT,
        "status": COMPONENT_BASIS_STATUS,
        "masks_source": copy.deepcopy(MASKS_SOURCE),
        "component_ablation_source": copy.deepcopy(COMPONENT_ABLATION_SOURCE),
        "stage8_masking_manifest": copy.deepcopy(STAGE8_MASKING_MANIFEST),
    }


@dataclass(frozen=True)
class SoftSealingEvidence:
    """Eligible-only sealed-model record plus soft output/hash linkage."""

    stage4_record: Mapping[str, Any]
    stage4_record_sha256: str
    eligibility_record_sha256: str
    checkpoint: Mapping[str, str]
    checkpoint_sha256: str
    dense_output: Mapping[str, str]
    dense_output_sha256: str
    policy_sha256: str
    teacher_soft_output_sha256: str
    student_soft_output_sha256: str
    ordered_input_ids_sha256: str


def generate_soft_sealing_evidence(
    *,
    assessment: SoftAttemptAssessment,
    stage3: Stage3AvailabilityIndex,
    checkpoint: Mapping[str, Any],
    dense_output: Mapping[str, Any],
    architecture_ref: str,
) -> SoftSealingEvidence:
    """Generate sealed-model evidence only for eligible hash-consistent evidence."""
    if not isinstance(assessment, SoftAttemptAssessment):
        raise Stage6CRecordError("assessment must be SoftAttemptAssessment")
    if assessment.status != "eligible" or assessment.eligibility is None:
        raise Stage6CRecordError("only eligible soft attempts can seal")
    attempt = assessment.attempt_record
    _, payload = _attempt_parts(attempt, stage3)
    if attempt.get("record_status") != "sealed":
        raise Stage6CRecordError("draft attempts cannot seal")
    if payload["attempt_outcome"] != "succeeded":
        raise Stage6CRecordError("failed attempts cannot seal")
    eligibility = assessment.eligibility
    if stage6c_record_sha256(eligibility.stage4_record) != eligibility.stage4_record_sha256:
        raise Stage6CRecordError("eligibility record hash is inconsistent")
    if eligibility.stage4_record.get("record_status") != "sealed":
        raise Stage6CRecordError("eligibility record must be sealed")
    if eligibility.stage4_record["payload"]["eligibility_status"] != "passed":
        raise Stage6CRecordError("sealing requires passing eligibility record")
    if not isinstance(architecture_ref, str) or not VERSION_REFERENCE_RE.fullmatch(
        architecture_ref
    ):
        raise Stage6CRecordError("architecture_ref must be explicitly injected")

    checkpoint_artifact = _artifact(
        checkpoint,
        name="checkpoint",
        storage_class="external_checkpoint",
    )
    if checkpoint_artifact != payload.get("model_checkpoint"):
        raise Stage6CRecordError("checkpoint evidence does not match attempt")
    dense_artifact = _artifact(
        dense_output,
        name="dense_output",
        storage_class="external_large_object",
    )
    evaluation = eligibility.evaluation
    if dense_artifact["sha256"] != evaluation.student_soft_output_sha256:
        raise Stage6CRecordError("dense-output hash does not match evaluated student output")
    policy_hash = _require_sha256(evaluation.policy_sha256, name="policy_sha256")
    teacher_hash = _require_sha256(
        evaluation.teacher_soft_output_sha256,
        name="teacher_soft_output_sha256",
    )
    student_hash = _require_sha256(
        evaluation.student_soft_output_sha256,
        name="student_soft_output_sha256",
    )
    order_hash = _require_sha256(
        evaluation.ordered_input_ids_sha256,
        name="ordered_input_ids_sha256",
    )
    eligibility_ref = {
        "record_type": "student_eligibility",
        "schema_version": "student_eligibility/v1",
        "condition_id": attempt["condition_id"],
        "record_sha256": eligibility.stage4_record_sha256,
    }
    record = {
        "namespace": _NAMESPACE,
        "vocabulary_version": _VOCABULARY_VERSION,
        "schema_version": "sealed_dense_model/v1",
        "record_type": "sealed_dense_model",
        "record_status": "sealed",
        "condition_id": attempt["condition_id"],
        "identity_depth": 4,
        "payload": {
            "eligibility_reference": eligibility_ref,
            "eligibility_status": "passed",
            "architecture_ref": architecture_ref,
            "component_basis": _component_basis(),
            "model_checkpoint": checkpoint_artifact,
        },
        "provenance": {
            "producer_lane": "lane_b",
            "creation_stage": "stage6c",
            "source_records": [eligibility_ref],
        },
    }
    return SoftSealingEvidence(
        stage4_record=record,
        stage4_record_sha256=stage6c_record_sha256(record),
        eligibility_record_sha256=eligibility.stage4_record_sha256,
        checkpoint=checkpoint_artifact,
        checkpoint_sha256=checkpoint_artifact["sha256"],
        dense_output=dense_artifact,
        dense_output_sha256=dense_artifact["sha256"],
        policy_sha256=policy_hash,
        teacher_soft_output_sha256=teacher_hash,
        student_soft_output_sha256=student_hash,
        ordered_input_ids_sha256=order_hash,
    )


@dataclass(frozen=True)
class SoftCircuitReleaseDecision:
    """Narrow soft eligibility/sealing gate result; never a discovery job."""

    allowed: bool
    reason: str


def soft_circuit_release_gate(
    *,
    assessment: SoftAttemptAssessment,
    sealing: SoftSealingEvidence | None,
) -> SoftCircuitReleaseDecision:
    """Allow release only for complete eligible, sealed, hash-consistent evidence."""
    if assessment.status != "eligible" or assessment.eligibility is None:
        return SoftCircuitReleaseDecision(False, "soft_attempt_not_eligible")
    if sealing is None:
        return SoftCircuitReleaseDecision(False, "sealing_evidence_missing")
    if assessment.attempt_record.get("record_status") != "sealed":
        return SoftCircuitReleaseDecision(False, "attempt_not_sealed")
    eligibility = assessment.eligibility
    if eligibility.stage4_record.get("record_status") != "sealed":
        return SoftCircuitReleaseDecision(False, "eligibility_record_not_sealed")
    if eligibility.stage4_record["payload"].get("eligibility_status") != "passed":
        return SoftCircuitReleaseDecision(False, "eligibility_not_passed")
    if stage6c_record_sha256(eligibility.stage4_record) != eligibility.stage4_record_sha256:
        return SoftCircuitReleaseDecision(False, "eligibility_hash_mismatch")
    if sealing.eligibility_record_sha256 != eligibility.stage4_record_sha256:
        return SoftCircuitReleaseDecision(False, "eligibility_link_hash_mismatch")
    if stage6c_record_sha256(sealing.stage4_record) != sealing.stage4_record_sha256:
        return SoftCircuitReleaseDecision(False, "sealed_model_record_hash_mismatch")
    if sealing.stage4_record.get("record_status") != "sealed":
        return SoftCircuitReleaseDecision(False, "sealed_model_record_not_sealed")
    if sealing.stage4_record.get("condition_id") != assessment.attempt_record.get(
        "condition_id"
    ):
        return SoftCircuitReleaseDecision(False, "sealed_model_identity_mismatch")
    attempt_checkpoint = assessment.attempt_record["payload"].get("model_checkpoint")
    if sealing.checkpoint != attempt_checkpoint:
        return SoftCircuitReleaseDecision(False, "checkpoint_evidence_mismatch")
    if sealing.checkpoint.get("sha256") != sealing.checkpoint_sha256:
        return SoftCircuitReleaseDecision(False, "checkpoint_hash_mismatch")
    if sealing.dense_output.get("sha256") != sealing.dense_output_sha256:
        return SoftCircuitReleaseDecision(False, "dense_output_hash_mismatch")
    if sealing.stage4_record["payload"].get("model_checkpoint") != sealing.checkpoint:
        return SoftCircuitReleaseDecision(False, "sealed_checkpoint_link_mismatch")
    if (
        sealing.stage4_record["payload"]
        .get("eligibility_reference", {})
        .get("record_sha256")
        != eligibility.stage4_record_sha256
    ):
        return SoftCircuitReleaseDecision(False, "sealed_eligibility_link_mismatch")
    evaluation = eligibility.evaluation
    if (
        sealing.policy_sha256 != evaluation.policy_sha256
        or sealing.teacher_soft_output_sha256 != evaluation.teacher_soft_output_sha256
        or sealing.student_soft_output_sha256 != evaluation.student_soft_output_sha256
        or sealing.ordered_input_ids_sha256 != evaluation.ordered_input_ids_sha256
        or sealing.dense_output_sha256 != evaluation.student_soft_output_sha256
    ):
        return SoftCircuitReleaseDecision(False, "policy_output_or_order_hash_mismatch")
    return SoftCircuitReleaseDecision(True, "eligible_sealed_hash_consistent")

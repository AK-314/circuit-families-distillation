"""Stage 6B Part D attempt accounting, sealing evidence, and release gate.

The frozen Stage 4 eligibility and sealed-model payloads intentionally do not
have extension fields. This module therefore pairs each exact Stage 4 record
with deterministic typed evidence carrying decision/order and dense-output
hashes. It creates no new record schema and performs no training or discovery.
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
from circuit_families.stage6b.hard_target import (
    HARD_ELIGIBILITY_CRITERION,
    HardEligibilityEvidence,
)

HARD_ATTEMPT_CLASSIFICATIONS = (
    "eligible",
    "training_failure",
    "numerical_failure",
    "subperfect_agreement",
)

_NAMESPACE = "circuit-families-distillation"
_VOCABULARY_VERSION = "common-vocabulary/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Stage6BRecordError(ValueError):
    """Raised when Part D evidence violates a frozen interface."""


def canonical_stage6b_record_bytes(record: Mapping[str, Any]) -> bytes:
    """Serialize a small Stage 4-compatible record deterministically."""
    if not isinstance(record, Mapping):
        raise Stage6BRecordError("record must be a mapping")
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
        raise Stage6BRecordError(
            f"record is not canonical-JSON serializable: {exc}"
        ) from exc


def stage6b_record_sha256(record: Mapping[str, Any]) -> str:
    """Hash deterministic Stage 6B small-record bytes."""
    return hashlib.sha256(canonical_stage6b_record_bytes(record)).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise Stage6BRecordError(f"{name} must be lowercase SHA-256 hex")
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
        raise Stage6BRecordError(f"{name} keys mismatch")
    path = value["path"]
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise Stage6BRecordError(f"{name}.path must be portable and relative")
    if value["storage_class"] != storage_class:
        raise Stage6BRecordError(
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
        raise Stage6BRecordError("attempt record must be a mapping")
    if record.get("record_type") != "student_attempt":
        raise Stage6BRecordError("attempt record_type must be student_attempt")
    if record.get("schema_version") != "student_attempt/v1":
        raise Stage6BRecordError("attempt schema_version mismatch")
    try:
        identity = parse_condition_id(record["condition_id"], stage3)
    except (KeyError, ConditionIdentityError) as exc:
        raise Stage6BRecordError(f"invalid attempt condition identity: {exc}") from exc
    if identity.depth != 4 or identity.distillation_condition != "hard_target":
        raise Stage6BRecordError("Part D accepts only depth-4 hard_target attempts")
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise Stage6BRecordError("attempt payload must be a mapping")
    for name in ("attempt_index", "retry_index"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Stage6BRecordError(f"attempt {name} must be non-negative")
    if payload.get("attempt_outcome") not in {"succeeded", "failed"}:
        raise Stage6BRecordError("attempt_outcome must be succeeded or failed")
    return identity, payload


def _attempt_reference(record: Mapping[str, Any]) -> dict[str, str]:
    return {
        "record_type": "student_attempt",
        "schema_version": "student_attempt/v1",
        "condition_id": record["condition_id"],
        "record_sha256": attempt_record_sha256(record),
    }


def _failure_classification(payload: Mapping[str, Any]) -> str:
    reason = payload.get("failure_reason")
    if not isinstance(reason, str) or not reason:
        raise Stage6BRecordError("failed attempt requires failure_reason")
    parts = reason.split(":", 3)
    if len(parts) != 4 or parts[:2] != ["technical_failure", "v1"]:
        raise Stage6BRecordError("failed attempt reason lacks frozen technical taxonomy")
    low_level_kind = parts[2]
    if low_level_kind not in TECHNICAL_FAILURE_KINDS:
        raise Stage6BRecordError("failed attempt has unknown technical failure kind")
    return "numerical_failure" if low_level_kind == "numerical_failure" else "training_failure"


@dataclass(frozen=True)
class HardEligibilityRecordEvidence:
    """Exact Stage 4 eligibility record plus non-schema recomputation evidence."""

    stage4_record: Mapping[str, Any]
    stage4_record_sha256: str
    evaluation: HardEligibilityEvidence
    failure_classification: str | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "stage4_record": copy.deepcopy(dict(self.stage4_record)),
            "stage4_record_sha256": self.stage4_record_sha256,
            "decision_and_order_evidence": self.evaluation.to_mapping(),
            "failure_classification": self.failure_classification,
        }


@dataclass(frozen=True)
class HardAttemptAssessment:
    """One immutable-in-ledger hard attempt and its optional eligibility result."""

    classification: str
    attempt_record: Mapping[str, Any]
    eligibility: HardEligibilityRecordEvidence | None


def _emit_hard_eligibility_record(
    attempt_record: Mapping[str, Any],
    evaluation: HardEligibilityEvidence,
) -> HardEligibilityRecordEvidence:
    attempt_ref = _attempt_reference(attempt_record)
    payload = attempt_record["payload"]
    classification = None if evaluation.eligible else "subperfect_agreement"
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
            "criterion": HARD_ELIGIBILITY_CRITERION,
            "evaluation_example_count": evaluation.total_count,
            "teacher_argmax_agreement_count": evaluation.agreement_count,
        },
        "provenance": {
            "producer_lane": "lane_b",
            "creation_stage": "stage6b",
            "source_records": [attempt_ref],
        },
    }
    return HardEligibilityRecordEvidence(
        stage4_record=record,
        stage4_record_sha256=stage6b_record_sha256(record),
        evaluation=evaluation,
        failure_classification=classification,
    )


def assess_hard_attempt(
    *,
    attempt_record: Mapping[str, Any],
    stage3: Stage3AvailabilityIndex,
    evaluation: HardEligibilityEvidence | None = None,
) -> HardAttemptAssessment:
    """Classify one attempt without collapsing its distinct failure state."""
    _, payload = _attempt_parts(attempt_record, stage3)
    outcome = payload["attempt_outcome"]

    if outcome == "failed":
        if evaluation is not None:
            raise Stage6BRecordError(
                "failed training attempt cannot carry eligibility evaluation"
            )
        return HardAttemptAssessment(
            classification=_failure_classification(payload),
            attempt_record=copy.deepcopy(dict(attempt_record)),
            eligibility=None,
        )

    if evaluation is None:
        raise Stage6BRecordError(
            "completed hard attempt requires full-domain eligibility evidence"
        )
    if not isinstance(evaluation, HardEligibilityEvidence):
        raise Stage6BRecordError("evaluation must be HardEligibilityEvidence")
    if evaluation.student_condition_id != attempt_record["condition_id"]:
        raise Stage6BRecordError(
            "eligibility student identity does not match attempt"
        )
    if evaluation.criterion != HARD_ELIGIBILITY_CRITERION:
        raise Stage6BRecordError("hard eligibility criterion mismatch")
    if evaluation.total_count != 12_769:
        raise Stage6BRecordError("hard eligibility must evaluate exactly 12769 inputs")
    if evaluation.eligible != (evaluation.agreement_count == 12_769):
        raise Stage6BRecordError("hard eligibility boolean/count mismatch")

    eligibility = _emit_hard_eligibility_record(attempt_record, evaluation)
    return HardAttemptAssessment(
        classification=("eligible" if evaluation.eligible else "subperfect_agreement"),
        attempt_record=copy.deepcopy(dict(attempt_record)),
        eligibility=eligibility,
    )


class HardAttemptLedger:
    """Append-only Part D accounting layered over the accepted attempt ledger."""

    def __init__(self) -> None:
        self._attempts = TechnicalAttemptLedger()
        self._assessments: dict[tuple[str, int, int], HardAttemptAssessment] = {}

    def add(self, assessment: HardAttemptAssessment) -> str:
        if not isinstance(assessment, HardAttemptAssessment):
            raise Stage6BRecordError("assessment must be HardAttemptAssessment")
        if assessment.classification not in HARD_ATTEMPT_CLASSIFICATIONS:
            raise Stage6BRecordError("unknown hard-attempt classification")
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
            raise Stage6BRecordError(str(exc)) from exc
        self._assessments[key] = copy.deepcopy(assessment)
        return digest

    @property
    def attempt_count(self) -> int:
        return len(self._assessments)

    def classification_count(self, classification: str) -> int:
        if classification not in HARD_ATTEMPT_CLASSIFICATIONS:
            raise Stage6BRecordError("unknown hard-attempt classification")
        return sum(
            item.classification == classification
            for item in self._assessments.values()
        )

    def assessments(self) -> tuple[HardAttemptAssessment, ...]:
        return tuple(
            copy.deepcopy(self._assessments[key])
            for key in sorted(self._assessments)
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
class HardSealingEvidence:
    """Eligible-only sealed-model record plus checkpoint/dense-output hashes."""

    stage4_record: Mapping[str, Any]
    stage4_record_sha256: str
    eligibility_record_sha256: str
    checkpoint: Mapping[str, str]
    checkpoint_sha256: str
    dense_output: Mapping[str, str]
    dense_output_sha256: str
    teacher_decisions_sha256: str
    student_decisions_sha256: str
    ordered_input_ids_sha256: str


def generate_hard_sealing_evidence(
    *,
    assessment: HardAttemptAssessment,
    stage3: Stage3AvailabilityIndex,
    checkpoint: Mapping[str, Any],
    dense_output: Mapping[str, Any],
    architecture_ref: str,
) -> HardSealingEvidence:
    """Generate sealed-model evidence only for an eligible sealed attempt."""
    if not isinstance(assessment, HardAttemptAssessment):
        raise Stage6BRecordError("assessment must be HardAttemptAssessment")
    if assessment.classification != "eligible" or assessment.eligibility is None:
        raise Stage6BRecordError("only eligible hard attempts can seal")
    attempt = assessment.attempt_record
    _, payload = _attempt_parts(attempt, stage3)
    if attempt.get("record_status") != "sealed":
        raise Stage6BRecordError("draft attempts cannot seal")
    if payload["attempt_outcome"] != "succeeded":
        raise Stage6BRecordError("failed attempts cannot seal")
    eligibility = assessment.eligibility
    if stage6b_record_sha256(eligibility.stage4_record) != eligibility.stage4_record_sha256:
        raise Stage6BRecordError("eligibility record hash is inconsistent")
    if eligibility.stage4_record.get("record_status") != "sealed":
        raise Stage6BRecordError("eligibility record must be sealed")
    if eligibility.stage4_record["payload"]["eligibility_status"] != "passed":
        raise Stage6BRecordError("sealing requires passing eligibility record")
    if not isinstance(architecture_ref, str) or not VERSION_REFERENCE_RE.fullmatch(
        architecture_ref
    ):
        raise Stage6BRecordError("architecture_ref must be explicitly injected")

    checkpoint_artifact = _artifact(
        checkpoint,
        name="checkpoint",
        storage_class="external_checkpoint",
    )
    if checkpoint_artifact != payload.get("model_checkpoint"):
        raise Stage6BRecordError("checkpoint evidence does not match attempt")
    dense_artifact = _artifact(
        dense_output,
        name="dense_output",
        storage_class="external_large_object",
    )
    evaluation = eligibility.evaluation
    teacher_hash = _require_sha256(
        evaluation.teacher_decisions_sha256,
        name="teacher_decisions_sha256",
    )
    student_hash = _require_sha256(
        evaluation.student_decisions_sha256,
        name="student_decisions_sha256",
    )
    order_hash = _require_sha256(
        evaluation.ordered_input_ids_sha256,
        name="ordered_input_ids_sha256",
    )
    if teacher_hash != student_hash:
        raise Stage6BRecordError(
            "eligible decision hashes must match exactly"
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
            "creation_stage": "stage6b",
            "source_records": [eligibility_ref],
        },
    }
    return HardSealingEvidence(
        stage4_record=record,
        stage4_record_sha256=stage6b_record_sha256(record),
        eligibility_record_sha256=eligibility.stage4_record_sha256,
        checkpoint=checkpoint_artifact,
        checkpoint_sha256=checkpoint_artifact["sha256"],
        dense_output=dense_artifact,
        dense_output_sha256=dense_artifact["sha256"],
        teacher_decisions_sha256=teacher_hash,
        student_decisions_sha256=student_hash,
        ordered_input_ids_sha256=order_hash,
    )


@dataclass(frozen=True)
class CircuitReleaseDecision:
    """Narrow eligibility/sealing gate result; never a circuit job."""

    allowed: bool
    reason: str


def circuit_release_gate(
    *,
    assessment: HardAttemptAssessment,
    sealing: HardSealingEvidence | None,
) -> CircuitReleaseDecision:
    """Allow downstream release only for complete hash-consistent evidence."""
    if assessment.classification != "eligible" or assessment.eligibility is None:
        return CircuitReleaseDecision(False, "hard_attempt_not_eligible")
    if sealing is None:
        return CircuitReleaseDecision(False, "sealing_evidence_missing")
    if assessment.attempt_record.get("record_status") != "sealed":
        return CircuitReleaseDecision(False, "attempt_not_sealed")
    eligibility = assessment.eligibility
    if eligibility.stage4_record.get("record_status") != "sealed":
        return CircuitReleaseDecision(False, "eligibility_record_not_sealed")
    if eligibility.stage4_record["payload"].get("eligibility_status") != "passed":
        return CircuitReleaseDecision(False, "eligibility_not_passed")
    if stage6b_record_sha256(eligibility.stage4_record) != eligibility.stage4_record_sha256:
        return CircuitReleaseDecision(False, "eligibility_hash_mismatch")
    if sealing.eligibility_record_sha256 != eligibility.stage4_record_sha256:
        return CircuitReleaseDecision(False, "eligibility_link_hash_mismatch")
    if stage6b_record_sha256(sealing.stage4_record) != sealing.stage4_record_sha256:
        return CircuitReleaseDecision(False, "sealed_model_record_hash_mismatch")
    if sealing.stage4_record.get("record_status") != "sealed":
        return CircuitReleaseDecision(False, "sealed_model_record_not_sealed")
    if sealing.stage4_record.get("condition_id") != assessment.attempt_record.get(
        "condition_id"
    ):
        return CircuitReleaseDecision(False, "sealed_model_identity_mismatch")
    attempt_checkpoint = assessment.attempt_record["payload"].get("model_checkpoint")
    if sealing.checkpoint != attempt_checkpoint:
        return CircuitReleaseDecision(False, "checkpoint_evidence_mismatch")
    if sealing.checkpoint.get("sha256") != sealing.checkpoint_sha256:
        return CircuitReleaseDecision(False, "checkpoint_hash_mismatch")
    if sealing.dense_output.get("sha256") != sealing.dense_output_sha256:
        return CircuitReleaseDecision(False, "dense_output_hash_mismatch")
    if sealing.stage4_record["payload"].get("model_checkpoint") != sealing.checkpoint:
        return CircuitReleaseDecision(False, "sealed_checkpoint_link_mismatch")
    if (
        sealing.stage4_record["payload"]
        .get("eligibility_reference", {})
        .get("record_sha256")
        != eligibility.stage4_record_sha256
    ):
        return CircuitReleaseDecision(False, "sealed_eligibility_link_mismatch")
    evaluation = eligibility.evaluation
    if (
        sealing.teacher_decisions_sha256 != evaluation.teacher_decisions_sha256
        or sealing.student_decisions_sha256 != evaluation.student_decisions_sha256
        or sealing.ordered_input_ids_sha256 != evaluation.ordered_input_ids_sha256
    ):
        return CircuitReleaseDecision(False, "decision_or_order_hash_mismatch")
    return CircuitReleaseDecision(True, "eligible_sealed_hash_consistent")

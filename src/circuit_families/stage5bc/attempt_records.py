"""Stage 4-compatible synthetic student-attempt record mechanics.

This module preserves technical successes and failures as immutable-in-ledger
attempt records. It does not implement student eligibility, attempt caps,
replacement policy, or production sealed-dense-model selection.
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
from circuit_families.stage5bc.student_identity import (
    StudentAttemptIdentity,
    StudentIdentityError,
    verify_student_attempt_identity,
)
from circuit_families.stage5bc.student_trainer import (
    TechnicalTrainingResult,
)

TECHNICAL_ATTEMPT_OUTCOME_KINDS = (
    "succeeded",
    "numerical_failure",
    "interruption",
    "invalid_input",
    "configuration_rejection",
    "exhausted_technical_stop",
)

TECHNICAL_FAILURE_KINDS = TECHNICAL_ATTEMPT_OUTCOME_KINDS[1:]

TECHNICAL_FAILURE_REASON_VERSION = "technical_failure:v1"

_NAMESPACE = "circuit-families-distillation"
_VOCABULARY_VERSION = "common-vocabulary/v1"
_SCHEMA_VERSION = "student_attempt/v1"
_RECORD_TYPE = "student_attempt"
_RECORD_STATUS = "draft"
_IDENTITY_DEPTH = 4
_PRODUCER_LANE = "lane_b"
_CREATION_STAGE = "stage5b"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TechnicalAttemptRecordError(ValueError):
    """Raised when technical attempt-record mechanics are inconsistent."""


@dataclass(frozen=True, order=True)
class TechnicalAttemptKey:
    """Complete Stage 4 attempt coordinate used by the local ledger."""

    condition_id: str
    attempt_index: int
    retry_index: int


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TechnicalAttemptRecordError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def _require_version_reference(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not VERSION_REFERENCE_RE.fullmatch(value)
    ):
        raise TechnicalAttemptRecordError(
            f"{name} must match Stage 4 version-reference grammar"
        )
    return value


def _artifact(
    value: Mapping[str, Any],
    *,
    name: str,
    storage_class: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TechnicalAttemptRecordError(
            f"{name} must be a mapping"
        )

    required = {"path", "sha256", "storage_class"}
    if set(value) != required:
        raise TechnicalAttemptRecordError(
            f"{name} keys must be exactly {sorted(required)!r}"
        )

    path = value["path"]
    if not isinstance(path, str) or not path:
        raise TechnicalAttemptRecordError(
            f"{name}.path must be a non-empty relative POSIX path"
        )

    if (
        path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise TechnicalAttemptRecordError(
            f"{name}.path must be a portable relative POSIX path"
        )

    sha = _require_sha256(
        value["sha256"],
        name=f"{name}.sha256",
    )

    if value["storage_class"] != storage_class:
        raise TechnicalAttemptRecordError(
            f"{name}.storage_class must be {storage_class!r}"
        )

    return {
        "path": path,
        "sha256": sha,
        "storage_class": storage_class,
    }


def _target_cache_reference(
    value: Mapping[str, Any],
    *,
    attempt_identity: StudentAttemptIdentity,
    stage3: Stage3AvailabilityIndex,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TechnicalAttemptRecordError(
            "target_cache_reference must be a mapping"
        )

    required = {
        "record_type",
        "schema_version",
        "condition_id",
        "record_sha256",
    }
    if set(value) != required:
        raise TechnicalAttemptRecordError(
            "target_cache_reference keys mismatch"
        )

    if value["record_type"] != "teacher_output_cache":
        raise TechnicalAttemptRecordError(
            "target_cache_reference must target teacher_output_cache"
        )

    if value["schema_version"] != "teacher_output_cache/v1":
        raise TechnicalAttemptRecordError(
            "target_cache_reference schema_version mismatch"
        )

    _require_sha256(
        value["record_sha256"],
        name="target_cache_reference.record_sha256",
    )

    try:
        cache_identity = parse_condition_id(
            value["condition_id"],
            stage3,
        )
        student_identity = parse_condition_id(
            attempt_identity.condition_id,
            stage3,
        )
    except ConditionIdentityError as exc:
        raise TechnicalAttemptRecordError(
            f"invalid cache/student condition identity: {exc}"
        ) from exc

    if cache_identity.depth != 3:
        raise TechnicalAttemptRecordError(
            "target_cache_reference condition must have depth 3"
        )

    if student_identity.depth != 4:
        raise TechnicalAttemptRecordError(
            "student attempt condition must have depth 4"
        )

    if (
        cache_identity.teacher_seed != student_identity.teacher_seed
        or cache_identity.phase != student_identity.phase
        or cache_identity.distillation_condition
        != student_identity.distillation_condition
    ):
        raise TechnicalAttemptRecordError(
            "target cache does not share student distillation condition"
        )

    return copy.deepcopy(dict(value))


def _seed_mapping(
    value: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    required = {
        "seed_derivation_version",
        "seed_material",
        "digest_sha256",
        "selected_bytes_hex",
        "seed_value",
    }

    if not isinstance(value, Mapping) or set(value) != required:
        raise TechnicalAttemptRecordError(
            f"{name} seed evidence keys mismatch"
        )

    return copy.deepcopy(dict(value))


def _failure_reason(
    *,
    failure_kind: str,
    detail: str,
) -> str:
    if failure_kind not in TECHNICAL_FAILURE_KINDS:
        raise TechnicalAttemptRecordError(
            f"unsupported technical failure kind: {failure_kind!r}"
        )

    if (
        not isinstance(detail, str)
        or not detail
        or "\n" in detail
        or "\r" in detail
    ):
        raise TechnicalAttemptRecordError(
            "technical failure detail must be a non-empty single line"
        )

    return (
        f"{TECHNICAL_FAILURE_REASON_VERSION}:"
        f"{failure_kind}:{detail}"
    )


def outcome_from_training_result(
    result: TechnicalTrainingResult,
) -> tuple[str, str | None]:
    """Map the Part K terminal status into Part M record semantics."""
    if not isinstance(result, TechnicalTrainingResult):
        raise TechnicalAttemptRecordError(
            "result must be TechnicalTrainingResult"
        )

    if result.terminal_status == "stop_rule_met":
        return "succeeded", None

    if result.terminal_status == "nonfinite_failure":
        return "numerical_failure", result.terminal_reason

    if result.terminal_status == "technical_step_limit_exhausted":
        return "exhausted_technical_stop", result.terminal_reason

    raise TechnicalAttemptRecordError(
        f"unsupported terminal status: {result.terminal_status!r}"
    )


def emit_technical_attempt_record(
    *,
    stage3: Stage3AvailabilityIndex,
    attempt_identity: StudentAttemptIdentity,
    target_cache_reference: Mapping[str, Any],
    outcome_kind: str,
    student_architecture_ref: str,
    replication_policy_ref: str,
    training_config_ref: str,
    training_log: Mapping[str, Any],
    model_checkpoint: Mapping[str, Any] | None = None,
    failure_detail: str | None = None,
) -> dict[str, Any]:
    """Emit one exact Stage 4-compatible draft student_attempt record."""
    if outcome_kind not in TECHNICAL_ATTEMPT_OUTCOME_KINDS:
        raise TechnicalAttemptRecordError(
            f"unsupported technical attempt outcome: {outcome_kind!r}"
        )

    if not isinstance(attempt_identity, StudentAttemptIdentity):
        raise TechnicalAttemptRecordError(
            "attempt_identity must be StudentAttemptIdentity"
        )

    try:
        verify_student_attempt_identity(
            attempt_identity,
            stage3,
        )
    except StudentIdentityError as exc:
        raise TechnicalAttemptRecordError(
            f"attempt identity failed verification: {exc}"
        ) from exc

    cache_reference = _target_cache_reference(
        target_cache_reference,
        attempt_identity=attempt_identity,
        stage3=stage3,
    )

    architecture_ref = _require_version_reference(
        student_architecture_ref,
        name="student_architecture_ref",
    )
    replication_ref = _require_version_reference(
        replication_policy_ref,
        name="replication_policy_ref",
    )
    config_ref = _require_version_reference(
        training_config_ref,
        name="training_config_ref",
    )

    log_artifact = _artifact(
        training_log,
        name="training_log",
        storage_class="external_log",
    )

    identity_mapping = attempt_identity.to_mapping()

    payload: dict[str, Any] = {
        "target_cache": cache_reference,
        "attempt_index": attempt_identity.attempt_index,
        "retry_index": attempt_identity.retry_index,
        "attempt_outcome": (
            "succeeded"
            if outcome_kind == "succeeded"
            else "failed"
        ),
        "student_architecture_ref": architecture_ref,
        "replication_policy_ref": replication_ref,
        "training_config_ref": config_ref,
        "training_seed": _seed_mapping(
            identity_mapping["training_seed"],
            name="training_seed",
        ),
        "tie_breaking_seed": _seed_mapping(
            identity_mapping["tie_breaking_seed"],
            name="tie_breaking_seed",
        ),
        "training_log": log_artifact,
    }

    if outcome_kind == "succeeded":
        if failure_detail is not None:
            raise TechnicalAttemptRecordError(
                "succeeded attempt cannot carry failure_detail"
            )
        if model_checkpoint is None:
            raise TechnicalAttemptRecordError(
                "succeeded technical attempt requires model_checkpoint"
            )

        payload["model_checkpoint"] = _artifact(
            model_checkpoint,
            name="model_checkpoint",
            storage_class="external_checkpoint",
        )
    else:
        if model_checkpoint is not None:
            raise TechnicalAttemptRecordError(
                "failed technical attempt cannot carry model_checkpoint"
            )
        if failure_detail is None:
            raise TechnicalAttemptRecordError(
                "failed technical attempt requires failure_detail"
            )

        payload["failure_reason"] = _failure_reason(
            failure_kind=outcome_kind,
            detail=failure_detail,
        )

    return {
        "namespace": _NAMESPACE,
        "vocabulary_version": _VOCABULARY_VERSION,
        "schema_version": _SCHEMA_VERSION,
        "record_type": _RECORD_TYPE,
        "record_status": _RECORD_STATUS,
        "condition_id": attempt_identity.condition_id,
        "identity_depth": _IDENTITY_DEPTH,
        "payload": payload,
        "provenance": {
            "producer_lane": _PRODUCER_LANE,
            "creation_stage": _CREATION_STAGE,
            "source_records": [],
        },
    }


def canonical_attempt_record_bytes(
    record: Mapping[str, Any],
) -> bytes:
    """Return deterministic JSON bytes for a small synthetic attempt record."""
    if not isinstance(record, Mapping):
        raise TechnicalAttemptRecordError(
            "attempt record must be a mapping"
        )

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
        raise TechnicalAttemptRecordError(
            f"attempt record is not canonical-JSON serializable: {exc}"
        ) from exc


def attempt_record_sha256(
    record: Mapping[str, Any],
) -> str:
    """Hash the deterministic synthetic record serialization."""
    return hashlib.sha256(
        canonical_attempt_record_bytes(record)
    ).hexdigest()


def _ledger_key(
    record: Mapping[str, Any],
) -> TechnicalAttemptKey:
    if not isinstance(record, Mapping):
        raise TechnicalAttemptRecordError(
            "ledger record must be a mapping"
        )

    if record.get("record_type") != "student_attempt":
        raise TechnicalAttemptRecordError(
            "ledger accepts only student_attempt records"
        )

    condition_id = record.get("condition_id")
    payload = record.get("payload")

    if not isinstance(condition_id, str) or not condition_id:
        raise TechnicalAttemptRecordError(
            "ledger record condition_id must be non-empty"
        )

    if not isinstance(payload, Mapping):
        raise TechnicalAttemptRecordError(
            "ledger record payload must be a mapping"
        )

    attempt_index = payload.get("attempt_index")
    retry_index = payload.get("retry_index")

    for name, value in (
        ("attempt_index", attempt_index),
        ("retry_index", retry_index),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise TechnicalAttemptRecordError(
                f"ledger {name} must be a non-negative integer"
            )

    return TechnicalAttemptKey(
        condition_id=condition_id,
        attempt_index=attempt_index,
        retry_index=retry_index,
    )


def _seed_signature(
    record: Mapping[str, Any],
) -> tuple[str, str]:
    payload = record["payload"]

    try:
        training = payload["training_seed"]["digest_sha256"]
        tie = payload["tie_breaking_seed"]["digest_sha256"]
    except (KeyError, TypeError) as exc:
        raise TechnicalAttemptRecordError(
            "ledger record lacks complete seed evidence"
        ) from exc

    return (
        _require_sha256(
            training,
            name="training_seed.digest_sha256",
        ),
        _require_sha256(
            tie,
            name="tie_breaking_seed.digest_sha256",
        ),
    )


class TechnicalAttemptLedger:
    """Append-only in-memory technical attempt inventory.

    It deliberately provides no replace/remove/reindex API. Duplicate complete
    attempt coordinates are rejected, and seed evidence cannot be reassigned to
    another complete attempt identity.
    """

    def __init__(self) -> None:
        self._records: dict[
            TechnicalAttemptKey,
            dict[str, Any],
        ] = {}
        self._seed_owners: dict[
            tuple[str, str],
            TechnicalAttemptKey,
        ] = {}

    def add(
        self,
        record: Mapping[str, Any],
    ) -> str:
        """Append one record and return its canonical SHA-256."""
        key = _ledger_key(record)

        if key in self._records:
            raise TechnicalAttemptRecordError(
                "attempt identity already exists; replacement is forbidden"
            )

        seed_signature = _seed_signature(record)
        existing_owner = self._seed_owners.get(seed_signature)

        if existing_owner is not None and existing_owner != key:
            raise TechnicalAttemptRecordError(
                "seed evidence cannot be reused under another attempt identity"
            )

        stored = copy.deepcopy(dict(record))
        self._records[key] = stored
        self._seed_owners[seed_signature] = key

        return attempt_record_sha256(stored)

    @property
    def attempt_count(self) -> int:
        return len(self._records)

    @property
    def failed_attempt_count(self) -> int:
        return sum(
            record["payload"]["attempt_outcome"] == "failed"
            for record in self._records.values()
        )

    @property
    def succeeded_attempt_count(self) -> int:
        return sum(
            record["payload"]["attempt_outcome"] == "succeeded"
            for record in self._records.values()
        )

    def records(self) -> tuple[dict[str, Any], ...]:
        """Return canonical-key-ordered defensive copies."""
        return tuple(
            copy.deepcopy(self._records[key])
            for key in sorted(self._records)
        )

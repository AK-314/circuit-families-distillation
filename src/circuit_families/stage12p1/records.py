"""Compact sealed Stage 12-P1 technical records.

Records here combine only policy-neutral evidence already produced by the
task, teacher, and phase adapters. They are deliberately non-production and
contain no scientific endpoint or circuit results.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from circuit_families.stage12p1.phase import (
    phase_artifact_sha256,
    validate_phase_selection_artifact,
)
from circuit_families.stage12p1.tasks import (
    canonical_json_bytes,
    canonical_sha256,
    validate_task_record,
)
from circuit_families.stage12p1.teacher import (
    TeacherRunResult,
    TeacherTrainingRequest,
    validate_teacher_artifact,
)
from circuit_families.training.checkpoints import file_sha256

FOUNDATION_RECORD_SCHEMA_VERSION = "stage12p1-teacher-foundation-record/v1"
ATTEMPT_IDENTITY_SCHEMA_VERSION = "stage12p1-teacher-attempt-identity/v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class FoundationRecordError(ValueError):
    """Raised when compact Stage 12-P1 evidence is inconsistent."""


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FoundationRecordError(f"{name} must be lowercase SHA-256")
    return value


def _nonempty(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FoundationRecordError(f"{name} must be a non-empty string")
    if "\n" in value or "\r" in value:
        raise FoundationRecordError(f"{name} must be one line")
    return value


def _portable_relative(value: Any, *, name: str) -> str:
    text = _nonempty(value, name=name)
    if text.startswith("~") or "\\" in text:
        raise FoundationRecordError(
            f"{name} must be a portable relative POSIX path"
        )

    path = PurePosixPath(text)
    if path.is_absolute() or any(
        part in {"", ".", ".."}
        for part in path.parts
    ):
        raise FoundationRecordError(
            f"{name} must be a portable relative POSIX path"
        )

    return path.as_posix()


def teacher_attempt_identity(
    request: TeacherTrainingRequest,
) -> dict[str, Any]:
    """Return deterministic complete identity for one technical teacher attempt."""
    if not isinstance(request, TeacherTrainingRequest):
        raise FoundationRecordError(
            "request must be TeacherTrainingRequest"
        )

    material = {
        "schema_version": ATTEMPT_IDENTITY_SCHEMA_VERSION,
        "task_identity_sha256": request.task_record["hashes"][
            "task_identity_sha256"
        ],
        "resume_id": request.resume_id,
        "model_seed_id": request.model_seed_id,
        "model_seed": request.model_seed,
        "training_seed_id": request.training_seed_id,
        "training_seed": request.training_seed,
    }

    return {
        **material,
        "attempt_identity_sha256": canonical_sha256(material),
    }


def _relative_to_output_root(
    request: TeacherTrainingRequest,
    path: Path,
) -> str:
    root = request.output_root.resolve()
    candidate = path.resolve()

    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise FoundationRecordError(
            "teacher artifact escaped configured output_root"
        ) from exc

    return _portable_relative(
        relative.as_posix(),
        name="sealed_teacher.artifact_path",
    )


def _phase_result(
    *,
    teacher_status: str,
    phase_selection: Mapping[str, Any] | None,
    task_identity_sha256: str,
    teacher_artifact_sha256: str,
) -> dict[str, Any]:
    if phase_selection is None:
        if teacher_status == "completed":
            raise FoundationRecordError(
                "completed teacher requires explicit phase-selection result"
            )

        return {
            "state": "unavailable",
            "reason": f"teacher_status:{teacher_status}",
        }

    if teacher_status != "completed":
        raise FoundationRecordError(
            "non-completed teacher cannot carry selected phase evidence"
        )

    validated = validate_phase_selection_artifact(phase_selection)

    if validated["task_identity_sha256"] != task_identity_sha256:
        raise FoundationRecordError(
            "phase result task identity does not match teacher"
        )
    if validated["teacher_artifact_sha256"] != teacher_artifact_sha256:
        raise FoundationRecordError(
            "phase result teacher artifact hash does not match"
        )

    return {
        "state": "available",
        "artifact_sha256": phase_artifact_sha256(validated),
        "artifact": validated,
    }


def build_foundation_record(
    *,
    request: TeacherTrainingRequest,
    result: TeacherRunResult,
    phase_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one compact immutable task/teacher/phase technical record."""
    if not isinstance(request, TeacherTrainingRequest):
        raise FoundationRecordError(
            "request must be TeacherTrainingRequest"
        )
    if not isinstance(result, TeacherRunResult):
        raise FoundationRecordError(
            "result must be TeacherRunResult"
        )

    task = validate_task_record(request.task_record)

    if not result.artifact_path.is_file():
        raise FoundationRecordError("sealed teacher artifact is missing")

    actual_teacher_file_sha = file_sha256(result.artifact_path)
    if actual_teacher_file_sha != result.artifact_sha256:
        raise FoundationRecordError(
            "teacher artifact physical SHA-256 mismatch"
        )

    try:
        teacher = validate_teacher_artifact(
            json.loads(result.artifact_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise FoundationRecordError(
            "sealed teacher artifact is invalid"
        ) from exc

    if teacher["status"] != result.status:
        raise FoundationRecordError(
            "teacher artifact status disagrees with run result"
        )
    if teacher["resume_id"] != request.resume_id:
        raise FoundationRecordError(
            "teacher artifact resume identity disagrees with request"
        )

    task_identity_sha = task["hashes"]["task_identity_sha256"]
    if teacher["task_identity_sha256"] != task_identity_sha:
        raise FoundationRecordError(
            "teacher artifact task identity disagrees with task"
        )

    attempt = teacher_attempt_identity(request)

    checkpoint_inventory = copy.deepcopy(
        teacher["checkpoint_inventory"]
    )
    for index, item in enumerate(checkpoint_inventory):
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "sha256",
        }:
            raise FoundationRecordError(
                f"checkpoint inventory item {index} is invalid"
            )
        _portable_relative(
            item["path"],
            name=f"checkpoint_inventory[{index}].path",
        )
        _sha256(
            item["sha256"],
            name=f"checkpoint_inventory[{index}].sha256",
        )

    resume_lineage = copy.deepcopy(teacher["resume_lineage"])
    if not isinstance(resume_lineage, list):
        raise FoundationRecordError("resume_lineage must be a list")
    for index, digest in enumerate(resume_lineage):
        _sha256(
            digest,
            name=f"resume_lineage[{index}]",
        )

    phase_result = _phase_result(
        teacher_status=teacher["status"],
        phase_selection=phase_selection,
        task_identity_sha256=task_identity_sha,
        teacher_artifact_sha256=actual_teacher_file_sha,
    )

    failure_or_unavailable = teacher["status"] in {
        "failed",
        "numerical-failure",
        "unavailable",
    }

    material = {
        "schema_version": FOUNDATION_RECORD_SCHEMA_VERSION,
        "classification": "technical_fixture",
        "scientific_data": False,
        "production_eligible": False,
        "task_identity": {
            "task_id": task["task_definition"]["task_id"],
            "task_identity_sha256": task_identity_sha,
            "task_config_sha256": task["hashes"]["task_config_sha256"],
            "domain_sha256": task["hashes"]["domain_sha256"],
            "dataset_sha256": task["hashes"]["dataset_sha256"],
            "split_identity_sha256": task["hashes"][
                "split_identity_sha256"
            ],
        },
        "attempt_identity": attempt,
        "sealed_teacher": {
            "schema_version": teacher["schema_version"],
            "status": teacher["status"],
            "artifact_path": _relative_to_output_root(
                request,
                result.artifact_path,
            ),
            "file_sha256": actual_teacher_file_sha,
            "content_sha256": teacher["content_sha256"],
        },
        "checkpoint_inventory": checkpoint_inventory,
        "resume_lineage": resume_lineage,
        "phase_result": phase_result,
        "terminal_state": {
            "status": teacher["status"],
            "reason": teacher["reason"],
            "terminal": teacher["status"] != "interrupted",
            "failure_or_unavailable": failure_or_unavailable,
        },
    }

    return {
        **material,
        "content_sha256": canonical_sha256(material),
    }


def validate_foundation_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate one compact Stage 12-P1 record."""
    if not isinstance(record, Mapping):
        raise FoundationRecordError(
            "foundation record must be a mapping"
        )

    required = {
        "schema_version",
        "classification",
        "scientific_data",
        "production_eligible",
        "task_identity",
        "attempt_identity",
        "sealed_teacher",
        "checkpoint_inventory",
        "resume_lineage",
        "phase_result",
        "terminal_state",
        "content_sha256",
    }
    if set(record) != required:
        raise FoundationRecordError(
            "foundation record keys mismatch"
        )

    if record["schema_version"] != FOUNDATION_RECORD_SCHEMA_VERSION:
        raise FoundationRecordError(
            "foundation record schema_version mismatch"
        )
    if record["classification"] != "technical_fixture":
        raise FoundationRecordError(
            "foundation record classification mismatch"
        )
    if record["scientific_data"] is not False:
        raise FoundationRecordError(
            "foundation record scientific_data must be false"
        )
    if record["production_eligible"] is not False:
        raise FoundationRecordError(
            "foundation record production_eligible must be false"
        )

    task = record["task_identity"]
    if not isinstance(task, Mapping) or set(task) != {
        "task_id",
        "task_identity_sha256",
        "task_config_sha256",
        "domain_sha256",
        "dataset_sha256",
        "split_identity_sha256",
    }:
        raise FoundationRecordError(
            "task_identity structure mismatch"
        )

    _nonempty(task["task_id"], name="task_identity.task_id")
    for field in (
        "task_identity_sha256",
        "task_config_sha256",
        "domain_sha256",
        "dataset_sha256",
        "split_identity_sha256",
    ):
        _sha256(
            task[field],
            name=f"task_identity.{field}",
        )

    attempt = record["attempt_identity"]
    if not isinstance(attempt, Mapping) or set(attempt) != {
        "schema_version",
        "task_identity_sha256",
        "resume_id",
        "model_seed_id",
        "model_seed",
        "training_seed_id",
        "training_seed",
        "attempt_identity_sha256",
    }:
        raise FoundationRecordError(
            "attempt_identity structure mismatch"
        )

    if attempt["schema_version"] != ATTEMPT_IDENTITY_SCHEMA_VERSION:
        raise FoundationRecordError(
            "attempt identity schema_version mismatch"
        )
    if attempt["task_identity_sha256"] != task["task_identity_sha256"]:
        raise FoundationRecordError(
            "attempt/task identity mismatch"
        )

    for field in (
        "resume_id",
        "model_seed_id",
        "training_seed_id",
    ):
        _nonempty(
            attempt[field],
            name=f"attempt_identity.{field}",
        )

    for field in ("model_seed", "training_seed"):
        value = attempt[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 2**32 - 1
        ):
            raise FoundationRecordError(
                f"attempt_identity.{field} is invalid"
            )

    stored_attempt_hash = _sha256(
        attempt["attempt_identity_sha256"],
        name="attempt_identity.attempt_identity_sha256",
    )
    attempt_material = copy.deepcopy(dict(attempt))
    attempt_material.pop("attempt_identity_sha256")
    if canonical_sha256(attempt_material) != stored_attempt_hash:
        raise FoundationRecordError(
            "attempt identity hash mismatch"
        )

    teacher = record["sealed_teacher"]
    if not isinstance(teacher, Mapping) or set(teacher) != {
        "schema_version",
        "status",
        "artifact_path",
        "file_sha256",
        "content_sha256",
    }:
        raise FoundationRecordError(
            "sealed_teacher structure mismatch"
        )

    _nonempty(
        teacher["schema_version"],
        name="sealed_teacher.schema_version",
    )
    if teacher["status"] not in {
        "completed",
        "failed",
        "interrupted",
        "numerical-failure",
        "unavailable",
    }:
        raise FoundationRecordError(
            "sealed_teacher status is invalid"
        )
    _portable_relative(
        teacher["artifact_path"],
        name="sealed_teacher.artifact_path",
    )
    _sha256(
        teacher["file_sha256"],
        name="sealed_teacher.file_sha256",
    )
    _sha256(
        teacher["content_sha256"],
        name="sealed_teacher.content_sha256",
    )

    inventory = record["checkpoint_inventory"]
    if not isinstance(inventory, list):
        raise FoundationRecordError(
            "checkpoint_inventory must be a list"
        )

    seen_paths: set[str] = set()
    for index, item in enumerate(inventory):
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "sha256",
        }:
            raise FoundationRecordError(
                f"checkpoint_inventory[{index}] structure mismatch"
            )
        path = _portable_relative(
            item["path"],
            name=f"checkpoint_inventory[{index}].path",
        )
        if path in seen_paths:
            raise FoundationRecordError(
                "checkpoint inventory contains duplicate path"
            )
        seen_paths.add(path)
        _sha256(
            item["sha256"],
            name=f"checkpoint_inventory[{index}].sha256",
        )

    lineage = record["resume_lineage"]
    if not isinstance(lineage, list):
        raise FoundationRecordError(
            "resume_lineage must be a list"
        )
    for index, digest in enumerate(lineage):
        _sha256(
            digest,
            name=f"resume_lineage[{index}]",
        )

    phase = record["phase_result"]
    if not isinstance(phase, Mapping):
        raise FoundationRecordError(
            "phase_result must be a mapping"
        )

    if phase.get("state") == "available":
        if set(phase) != {
            "state",
            "artifact_sha256",
            "artifact",
        }:
            raise FoundationRecordError(
                "available phase_result keys mismatch"
            )

        phase_artifact = validate_phase_selection_artifact(
            phase["artifact"]
        )
        expected_phase_hash = phase_artifact_sha256(
            phase_artifact
        )
        if phase["artifact_sha256"] != expected_phase_hash:
            raise FoundationRecordError(
                "phase artifact SHA-256 mismatch"
            )
        if (
            phase_artifact["task_identity_sha256"]
            != task["task_identity_sha256"]
        ):
            raise FoundationRecordError(
                "phase/task identity mismatch"
            )
        if (
            phase_artifact["teacher_artifact_sha256"]
            != teacher["file_sha256"]
        ):
            raise FoundationRecordError(
                "phase/teacher artifact identity mismatch"
            )
        if teacher["status"] != "completed":
            raise FoundationRecordError(
                "available phase result requires completed teacher"
            )
    elif phase.get("state") == "unavailable":
        if set(phase) != {"state", "reason"}:
            raise FoundationRecordError(
                "unavailable phase_result keys mismatch"
            )
        _nonempty(
            phase["reason"],
            name="phase_result.reason",
        )
        if teacher["status"] == "completed":
            raise FoundationRecordError(
                "completed teacher may not omit phase result"
            )
    else:
        raise FoundationRecordError(
            "phase_result state is invalid"
        )

    terminal = record["terminal_state"]
    if not isinstance(terminal, Mapping) or set(terminal) != {
        "status",
        "reason",
        "terminal",
        "failure_or_unavailable",
    }:
        raise FoundationRecordError(
            "terminal_state structure mismatch"
        )

    if terminal["status"] != teacher["status"]:
        raise FoundationRecordError(
            "terminal/sealed-teacher status mismatch"
        )
    _nonempty(
        terminal["reason"],
        name="terminal_state.reason",
    )

    expected_terminal = teacher["status"] != "interrupted"
    if terminal["terminal"] is not expected_terminal:
        raise FoundationRecordError(
            "terminal_state.terminal mismatch"
        )

    expected_failure = teacher["status"] in {
        "failed",
        "numerical-failure",
        "unavailable",
    }
    if terminal["failure_or_unavailable"] is not expected_failure:
        raise FoundationRecordError(
            "terminal failure/unavailable flag mismatch"
        )

    stored_hash = _sha256(
        record["content_sha256"],
        name="content_sha256",
    )
    material = copy.deepcopy(dict(record))
    material.pop("content_sha256")

    if canonical_sha256(material) != stored_hash:
        raise FoundationRecordError(
            "foundation record content hash mismatch"
        )

    return copy.deepcopy(dict(record))


def canonical_foundation_record_bytes(
    record: Mapping[str, Any],
) -> bytes:
    return canonical_json_bytes(
        validate_foundation_record(record)
    )


def foundation_record_sha256(
    record: Mapping[str, Any],
) -> str:
    return hashlib.sha256(
        canonical_foundation_record_bytes(record)
    ).hexdigest()

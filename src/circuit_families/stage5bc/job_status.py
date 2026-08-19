"""Deterministic job status inspection and resume guards for Stage 5B/C.

Part P is read-oriented orchestration state. It inspects the declarative Part N
DAG plus isolated Part O output roots and classifies every job as exactly one
of:

    planned, blocked, running, completed, failed, stale, conflicting

The module does not execute jobs, allocate production concurrency, determine
scientific eligibility, or transfer checkpoints between attempt identities.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from circuit_families.stage5bc.job_dag import (
    TechnicalJobNode,
    TechnicalJobRegistry,
)
from circuit_families.stage5bc.job_outputs import (
    JOB_COMPLETION_FILENAME,
    JOB_COMPLETION_SCHEMA_VERSION,
    JobOutputError,
    JobOutputRoot,
    atomic_write_job_file,
    canonical_job_relative_identity,
    validate_ignored_scratch_root,
)
from circuit_families.stage5bc.student_identity import (
    StudentAttemptIdentity,
)

JOB_STATUSES = (
    "planned",
    "blocked",
    "running",
    "completed",
    "failed",
    "stale",
    "conflicting",
)

JOB_FAILURE_FILENAME = "failure.json"
JOB_FAILURE_SCHEMA_VERSION = "stage5bc-job-failure/v1"

RESUME_ACTIONS = (
    "start_new",
    "resume_existing",
    "skip_completed",
    "blocked",
    "reject_failed_attempt",
    "reject_conflict",
)


class JobStatusError(ValueError):
    """Raised when status evidence or resume binding is invalid."""


@dataclass(frozen=True)
class JobStatusReport:
    """One deterministic status observation for a declared job."""

    job_id: str
    node_type: str
    condition_id: str
    relative_identity: str
    status: str
    reason: str
    output_root_exists: bool
    completion_sha256: str | None = None
    failure_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status not in JOB_STATUSES:
            raise JobStatusError(
                f"unsupported job status: {self.status!r}"
            )

        for name in (
            "job_id",
            "node_type",
            "condition_id",
            "relative_identity",
            "reason",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise JobStatusError(
                    f"{name} must be a non-empty string"
                )

        if not isinstance(self.output_root_exists, bool):
            raise JobStatusError(
                "output_root_exists must be boolean"
            )


@dataclass(frozen=True)
class AttemptResumeDecision:
    """Mechanics-only decision for one exact student-attempt identity."""

    action: str
    reason: str
    condition_id: str
    attempt_index: int
    retry_index: int

    def __post_init__(self) -> None:
        if self.action not in RESUME_ACTIONS:
            raise JobStatusError(
                f"unsupported resume action: {self.action!r}"
            )
        if not isinstance(self.reason, str) or not self.reason:
            raise JobStatusError(
                "resume decision reason must be non-empty"
            )


def _safe_relative_artifact_path(
    value: Any,
) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise JobStatusError(
            "artifact relative_path must be a non-empty string"
        )

    if "\\" in value:
        raise JobStatusError(
            "artifact relative_path must use POSIX separators"
        )

    path = PurePosixPath(value)

    if path.is_absolute():
        raise JobStatusError(
            "artifact relative_path must not be absolute"
        )

    if any(
        part in {"", ".", ".."}
        for part in path.parts
    ):
        raise JobStatusError(
            "artifact relative_path contains escape component"
        )

    return path


def _job_root_path(
    *,
    scratch_root: Path,
    node: TechnicalJobNode,
) -> Path:
    relative = PurePosixPath(
        canonical_job_relative_identity(node)
    )
    return scratch_root.joinpath(*relative.parts)


def _has_symlink_component(
    *,
    base: Path,
    candidate: Path,
) -> bool:
    try:
        relative = candidate.relative_to(base)
    except ValueError:
        return True

    current = base

    if current.is_symlink():
        return True

    for part in relative.parts:
        current = current / part

        if os.path.lexists(current) and current.is_symlink():
            return True

    return False


def _temporary_files(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()

    return tuple(
        sorted(
            (
                path
                for path in root.rglob("*")
                if (
                    path.is_file()
                    and path.name.startswith(".")
                    and path.name.endswith(".tmp")
                )
            ),
            key=lambda path: path.as_posix(),
        )
    )


def _duplicate_completion_files(
    root: Path,
) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()

    return tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if (
                    path.is_file()
                    and path.name != JOB_COMPLETION_FILENAME
                    and (
                        path.name.startswith("completion.")
                        or path.name.startswith("completion-")
                    )
                    and (
                        path.name.endswith(".json")
                        or path.name.endswith(".record")
                    )
                )
            ),
            key=lambda path: path.name,
        )
    )


def _canonical_failure_bytes(
    *,
    output_root: JobOutputRoot,
    reason: str,
) -> bytes:
    if not isinstance(reason, str) or not reason:
        raise JobStatusError(
            "failure reason must be a non-empty string"
        )

    if "\n" in reason or "\r" in reason:
        raise JobStatusError(
            "failure reason must be one line"
        )

    record = {
        "schema_version": JOB_FAILURE_SCHEMA_VERSION,
        "scientific_data": False,
        "production_eligible": False,
        "failure_state": "failed",
        "job_id": output_root.job_id,
        "node_type": output_root.node_type,
        "condition_id": output_root.condition_id,
        "relative_identity": output_root.relative_identity,
        "reason": reason,
    }

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


def write_job_failure(
    output_root: JobOutputRoot,
    *,
    reason: str,
) -> str:
    """Atomically publish one immutable technical failure record."""
    if not isinstance(output_root, JobOutputRoot):
        raise JobStatusError(
            "output_root must be JobOutputRoot"
        )

    if not output_root.execution_allowed:
        raise JobStatusError(
            "inert placeholder jobs cannot publish failure records"
        )

    evidence = atomic_write_job_file(
        output_root,
        JOB_FAILURE_FILENAME,
        _canonical_failure_bytes(
            output_root=output_root,
            reason=reason,
        ),
    )

    return evidence.sha256


def _inspect_failure(
    *,
    node: TechnicalJobNode,
    root: Path,
) -> tuple[bool, str | None, str | None]:
    path = root / JOB_FAILURE_FILENAME

    if not path.exists():
        return False, None, None

    if not path.is_file() or path.is_symlink():
        return True, None, "failure_record_not_regular_file"

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return True, digest, "failure_record_not_valid_json"

    expected = {
        "schema_version",
        "scientific_data",
        "production_eligible",
        "failure_state",
        "job_id",
        "node_type",
        "condition_id",
        "relative_identity",
        "reason",
    }

    if not isinstance(record, Mapping) or set(record) != expected:
        return True, digest, "failure_record_keys_mismatch"

    if record["schema_version"] != JOB_FAILURE_SCHEMA_VERSION:
        return True, digest, "failure_record_schema_mismatch"

    if (
        record["scientific_data"] is not False
        or record["production_eligible"] is not False
    ):
        return True, digest, "failure_record_authority_flags_invalid"

    if record["failure_state"] != "failed":
        return True, digest, "failure_record_state_invalid"

    if record["job_id"] != node.job_id:
        return True, digest, "failure_record_job_id_mismatch"

    if record["node_type"] != node.node_type:
        return True, digest, "failure_record_node_type_mismatch"

    if record["condition_id"] != node.condition_id:
        return True, digest, "failure_record_condition_mismatch"

    if (
        record["relative_identity"]
        != canonical_job_relative_identity(node)
    ):
        return True, digest, "failure_record_relative_identity_mismatch"

    if (
        not isinstance(record["reason"], str)
        or not record["reason"]
        or "\n" in record["reason"]
        or "\r" in record["reason"]
    ):
        return True, digest, "failure_record_reason_invalid"

    return True, digest, None


def _inspect_completion(
    *,
    node: TechnicalJobNode,
    root: Path,
) -> tuple[bool, str | None, str | None]:
    path = root / JOB_COMPLETION_FILENAME

    if not path.exists():
        return False, None, None

    if not path.is_file() or path.is_symlink():
        return True, None, "completion_not_regular_file"

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return True, digest, "completion_not_valid_json"

    expected = {
        "schema_version",
        "scientific_data",
        "production_eligible",
        "completion_state",
        "job_id",
        "node_type",
        "condition_id",
        "relative_identity",
        "artifacts",
    }

    if not isinstance(record, Mapping) or set(record) != expected:
        return True, digest, "completion_keys_mismatch"

    if record["schema_version"] != JOB_COMPLETION_SCHEMA_VERSION:
        return True, digest, "completion_schema_mismatch"

    if (
        record["scientific_data"] is not False
        or record["production_eligible"] is not False
    ):
        return True, digest, "completion_authority_flags_invalid"

    if record["completion_state"] != "complete":
        return True, digest, "completion_state_invalid"

    if record["job_id"] != node.job_id:
        return True, digest, "completion_job_id_mismatch"

    if record["node_type"] != node.node_type:
        return True, digest, "completion_node_type_mismatch"

    if record["condition_id"] != node.condition_id:
        return True, digest, "completion_condition_mismatch"

    if (
        record["relative_identity"]
        != canonical_job_relative_identity(node)
    ):
        return True, digest, "completion_relative_identity_mismatch"

    artifacts = record["artifacts"]

    if not isinstance(artifacts, list):
        return True, digest, "completion_artifacts_not_list"

    seen_paths: set[str] = set()

    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            return True, digest, "completion_artifact_structure_invalid"

        try:
            portable = _safe_relative_artifact_path(
                artifact["relative_path"]
            )
        except JobStatusError:
            return True, digest, "completion_artifact_path_invalid"

        relative_path = portable.as_posix()

        if relative_path in seen_paths:
            return True, digest, "completion_duplicate_artifact_path"

        seen_paths.add(relative_path)

        target = root.joinpath(*portable.parts)

        if _has_symlink_component(
            base=root,
            candidate=target,
        ):
            return True, digest, "completion_artifact_symlink_escape"

        if not target.exists() or not target.is_file():
            return True, digest, "completion_artifact_missing"

        raw_artifact = target.read_bytes()

        expected_sha = artifact["sha256"]
        expected_size = artifact["size_bytes"]

        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sha
            )
        ):
            return True, digest, "completion_artifact_hash_invalid"

        if hashlib.sha256(raw_artifact).hexdigest() != expected_sha:
            return True, digest, "completion_artifact_hash_mismatch"

        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            return True, digest, "completion_artifact_size_invalid"

        if len(raw_artifact) != expected_size:
            return True, digest, "completion_artifact_size_mismatch"

    return True, digest, None


def _inspect_local_status(
    *,
    scratch_root: Path,
    node: TechnicalJobNode,
) -> JobStatusReport:
    relative_identity = canonical_job_relative_identity(node)
    root = _job_root_path(
        scratch_root=scratch_root,
        node=node,
    )

    if not os.path.lexists(root):
        return JobStatusReport(
            job_id=node.job_id,
            node_type=node.node_type,
            condition_id=node.condition_id,
            relative_identity=relative_identity,
            status="planned",
            reason="output_root_absent",
            output_root_exists=False,
        )

    if root.is_symlink():
        return JobStatusReport(
            job_id=node.job_id,
            node_type=node.node_type,
            condition_id=node.condition_id,
            relative_identity=relative_identity,
            status="stale",
            reason="output_root_is_symlink",
            output_root_exists=True,
        )

    if not root.is_dir():
        return JobStatusReport(
            job_id=node.job_id,
            node_type=node.node_type,
            condition_id=node.condition_id,
            relative_identity=relative_identity,
            status="stale",
            reason="output_root_not_directory",
            output_root_exists=True,
        )

    temporary = _temporary_files(root)
    duplicate_completion = _duplicate_completion_files(root)

    completion_present, completion_sha, completion_error = (
        _inspect_completion(
            node=node,
            root=root,
        )
    )
    failure_present, failure_sha, failure_error = (
        _inspect_failure(
            node=node,
            root=root,
        )
    )

    if duplicate_completion:
        return JobStatusReport(
            job_id=node.job_id,
            node_type=node.node_type,
            condition_id=node.condition_id,
            relative_identity=relative_identity,
            status="conflicting",
            reason="duplicate_completion_records_present",
            output_root_exists=True,
            completion_sha256=completion_sha,
            failure_sha256=failure_sha,
        )

    if completion_present and failure_present:
        return JobStatusReport(
            job_id=node.job_id,
            node_type=node.node_type,
            condition_id=node.condition_id,
            relative_identity=relative_identity,
            status="conflicting",
            reason="completion_and_failure_records_both_present",
            output_root_exists=True,
            completion_sha256=completion_sha,
            failure_sha256=failure_sha,
        )

    if temporary and (completion_present or failure_present):
        return JobStatusReport(
            job_id=node.job_id,
            node_type=node.node_type,
            condition_id=node.condition_id,
            relative_identity=relative_identity,
            status="conflicting",
            reason="terminal_record_with_incomplete_temporary_write",
            output_root_exists=True,
            completion_sha256=completion_sha,
            failure_sha256=failure_sha,
        )

    if completion_error is not None:
        return JobStatusReport(
            job_id=node.job_id,
            node_type=node.node_type,
            condition_id=node.condition_id,
            relative_identity=relative_identity,
            status="stale",
            reason=completion_error,
            output_root_exists=True,
            completion_sha256=completion_sha,
            failure_sha256=failure_sha,
        )

    if failure_error is not None:
        return JobStatusReport(
            job_id=node.job_id,
            node_type=node.node_type,
            condition_id=node.condition_id,
            relative_identity=relative_identity,
            status="stale",
            reason=failure_error,
            output_root_exists=True,
            completion_sha256=completion_sha,
            failure_sha256=failure_sha,
        )

    if temporary:
        return JobStatusReport(
            job_id=node.job_id,
            node_type=node.node_type,
            condition_id=node.condition_id,
            relative_identity=relative_identity,
            status="stale",
            reason="incomplete_temporary_write_present",
            output_root_exists=True,
        )

    if completion_present:
        return JobStatusReport(
            job_id=node.job_id,
            node_type=node.node_type,
            condition_id=node.condition_id,
            relative_identity=relative_identity,
            status="completed",
            reason="valid_immutable_completion_record",
            output_root_exists=True,
            completion_sha256=completion_sha,
        )

    if failure_present:
        return JobStatusReport(
            job_id=node.job_id,
            node_type=node.node_type,
            condition_id=node.condition_id,
            relative_identity=relative_identity,
            status="failed",
            reason="valid_immutable_failure_record",
            output_root_exists=True,
            failure_sha256=failure_sha,
        )

    return JobStatusReport(
        job_id=node.job_id,
        node_type=node.node_type,
        condition_id=node.condition_id,
        relative_identity=relative_identity,
        status="running",
        reason="output_root_exists_without_terminal_record",
        output_root_exists=True,
    )


def inspect_registry_statuses(
    *,
    registry: TechnicalJobRegistry,
    repository_root: str | Path,
    scratch_root: str | Path,
    protected_roots: Iterable[str | Path] = (),
) -> tuple[JobStatusReport, ...]:
    """Inspect all jobs in deterministic DAG order."""
    if not isinstance(registry, TechnicalJobRegistry):
        raise JobStatusError(
            "registry must be TechnicalJobRegistry"
        )

    try:
        scratch = validate_ignored_scratch_root(
            repository_root=repository_root,
            scratch_root=scratch_root,
            protected_roots=protected_roots,
        )
    except JobOutputError as exc:
        raise JobStatusError(
            f"scratch-root validation failed: {exc}"
        ) from exc

    status_by_id: dict[str, JobStatusReport] = {}
    reports: list[JobStatusReport] = []

    for node in registry.topological_nodes():
        local = _inspect_local_status(
            scratch_root=scratch,
            node=node,
        )

        dependency_reports = tuple(
            status_by_id[dependency_id]
            for dependency_id in node.dependencies
        )

        if not node.execution_allowed:
            if local.status == "planned":
                report = JobStatusReport(
                    job_id=node.job_id,
                    node_type=node.node_type,
                    condition_id=node.condition_id,
                    relative_identity=local.relative_identity,
                    status="blocked",
                    reason="execution_not_authorized_for_inert_placeholder",
                    output_root_exists=False,
                )
            elif (
                local.status == "running"
                and local.output_root_exists
                and not any(local_root_has_content(
                    scratch_root=scratch,
                    node=node,
                ))
            ):
                report = JobStatusReport(
                    job_id=node.job_id,
                    node_type=node.node_type,
                    condition_id=node.condition_id,
                    relative_identity=local.relative_identity,
                    status="blocked",
                    reason=(
                        "inert_placeholder_identity_root_exists_but_execution_"
                        "is_not_authorized"
                    ),
                    output_root_exists=True,
                )
            else:
                report = JobStatusReport(
                    job_id=node.job_id,
                    node_type=node.node_type,
                    condition_id=node.condition_id,
                    relative_identity=local.relative_identity,
                    status="stale",
                    reason="inert_placeholder_has_runtime_output_evidence",
                    output_root_exists=local.output_root_exists,
                    completion_sha256=local.completion_sha256,
                    failure_sha256=local.failure_sha256,
                )
        else:
            dependencies_completed = all(
                dependency.status == "completed"
                for dependency in dependency_reports
            )

            if not dependencies_completed:
                if local.status == "planned":
                    report = JobStatusReport(
                        job_id=node.job_id,
                        node_type=node.node_type,
                        condition_id=node.condition_id,
                        relative_identity=local.relative_identity,
                        status="blocked",
                        reason="dependency_not_completed",
                        output_root_exists=False,
                    )
                elif local.status in {
                    "conflicting",
                    "stale",
                    "failed",
                }:
                    report = local
                else:
                    report = JobStatusReport(
                        job_id=node.job_id,
                        node_type=node.node_type,
                        condition_id=node.condition_id,
                        relative_identity=local.relative_identity,
                        status="stale",
                        reason="runtime_output_exists_before_dependencies_completed",
                        output_root_exists=local.output_root_exists,
                        completion_sha256=local.completion_sha256,
                        failure_sha256=local.failure_sha256,
                    )
            else:
                report = local

        status_by_id[node.job_id] = report
        reports.append(report)

    return tuple(reports)


def local_root_has_content(
    *,
    scratch_root: Path,
    node: TechnicalJobNode,
) -> tuple[bool, ...]:
    """Return deterministic content-presence evidence for an allocated root."""
    root = _job_root_path(
        scratch_root=scratch_root,
        node=node,
    )

    if not root.is_dir():
        return ()

    return tuple(
        True
        for _ in sorted(
            root.iterdir(),
            key=lambda path: path.name,
        )
    )


def status_mapping(
    reports: Iterable[JobStatusReport],
) -> dict[str, str]:
    """Return deterministic job-id -> status mapping."""
    result: dict[str, str] = {}

    for report in reports:
        if not isinstance(report, JobStatusReport):
            raise JobStatusError(
                "status_mapping requires JobStatusReport values"
            )

        if report.job_id in result:
            raise JobStatusError(
                "duplicate status report for one job ID"
            )

        result[report.job_id] = report.status

    return dict(sorted(result.items()))


def _attempt_coordinates(
    identity: StudentAttemptIdentity,
) -> tuple[str, int, int]:
    if not isinstance(identity, StudentAttemptIdentity):
        raise JobStatusError(
            "attempt identity must be StudentAttemptIdentity"
        )

    return (
        identity.condition_id,
        identity.attempt_index,
        identity.retry_index,
    )


def decide_attempt_resume(
    *,
    node: TechnicalJobNode,
    status_report: JobStatusReport,
    requested_attempt_identity: StudentAttemptIdentity,
    checkpoint_attempt_identity: StudentAttemptIdentity | None = None,
) -> AttemptResumeDecision:
    """Resolve resume behavior without duplicating or transferring state."""
    if not isinstance(node, TechnicalJobNode):
        raise JobStatusError(
            "node must be TechnicalJobNode"
        )

    if node.node_type != "training":
        raise JobStatusError(
            "attempt resume decisions apply only to training jobs"
        )

    if not isinstance(status_report, JobStatusReport):
        raise JobStatusError(
            "status_report must be JobStatusReport"
        )

    if status_report.job_id != node.job_id:
        raise JobStatusError(
            "status report belongs to a different job"
        )

    requested = _attempt_coordinates(
        requested_attempt_identity
    )

    if requested[0] != node.condition_id:
        raise JobStatusError(
            "requested attempt condition does not match training job"
        )

    checkpoint: tuple[str, int, int] | None = None

    if checkpoint_attempt_identity is not None:
        checkpoint = _attempt_coordinates(
            checkpoint_attempt_identity
        )

        if checkpoint != requested:
            raise JobStatusError(
                "resume state transfer across attempt/condition identity is forbidden"
            )

    status = status_report.status

    if status == "planned":
        if checkpoint is not None:
            raise JobStatusError(
                "planned job cannot consume pre-existing resume state"
            )

        return AttemptResumeDecision(
            action="start_new",
            reason="no_existing_runtime_state",
            condition_id=requested[0],
            attempt_index=requested[1],
            retry_index=requested[2],
        )

    if status == "blocked":
        return AttemptResumeDecision(
            action="blocked",
            reason="dependency_or_execution_boundary_blocks_attempt",
            condition_id=requested[0],
            attempt_index=requested[1],
            retry_index=requested[2],
        )

    if status == "running":
        if checkpoint is None:
            raise JobStatusError(
                "running job requires exact checkpoint identity for resume"
            )

        return AttemptResumeDecision(
            action="resume_existing",
            reason="exact_attempt_checkpoint_binding_verified",
            condition_id=requested[0],
            attempt_index=requested[1],
            retry_index=requested[2],
        )

    if status == "stale":
        if checkpoint is None:
            raise JobStatusError(
                "stale job cannot resume without exact checkpoint identity"
            )

        return AttemptResumeDecision(
            action="resume_existing",
            reason="exact_attempt_checkpoint_binding_required_for_stale_recovery",
            condition_id=requested[0],
            attempt_index=requested[1],
            retry_index=requested[2],
        )

    if status == "completed":
        return AttemptResumeDecision(
            action="skip_completed",
            reason="completed_job_must_not_duplicate_attempt",
            condition_id=requested[0],
            attempt_index=requested[1],
            retry_index=requested[2],
        )

    if status == "failed":
        return AttemptResumeDecision(
            action="reject_failed_attempt",
            reason="failed_attempt_identity_must_not_be_reexecuted_as_resume",
            condition_id=requested[0],
            attempt_index=requested[1],
            retry_index=requested[2],
        )

    if status == "conflicting":
        return AttemptResumeDecision(
            action="reject_conflict",
            reason="conflicting_runtime_evidence_requires_manual_resolution",
            condition_id=requested[0],
            attempt_index=requested[1],
            retry_index=requested[2],
        )

    raise JobStatusError(
        f"unhandled job status: {status!r}"
    )

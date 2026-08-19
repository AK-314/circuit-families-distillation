"""Isolated technical job output roots and immutable atomic publication.

An output root may exist only beneath an explicitly supplied, existing,
Git-ignored scratch directory. The scratch directory must not overlap tracked
repository namespaces or explicitly protected predecessor roots.

Every DAG node has a deterministic relative identity. Only Part N executable
node types may publish files or completion records.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from circuit_families.stage5bc.job_dag import (
    TechnicalJobNode,
)

JOB_OUTPUT_LAYOUT_VERSION = "stage5bc-job-output-layout/v1"
JOB_COMPLETION_SCHEMA_VERSION = "stage5bc-job-completion/v1"
JOB_COMPLETION_FILENAME = "completion.json"


class JobOutputError(ValueError):
    """Raised when a job output path or completion violates isolation."""


@dataclass(frozen=True)
class JobOutputRoot:
    """One allocated isolated root bound to one exact DAG job."""

    repository_root: Path
    scratch_root: Path
    relative_identity: str
    path: Path
    job_id: str
    node_type: str
    condition_id: str
    execution_allowed: bool


@dataclass(frozen=True)
class JobArtifactEvidence:
    """Immutable evidence for one atomically published small artifact."""

    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class JobCompletionEvidence:
    """Evidence for one immutable completion record."""

    path: Path
    sha256: str
    size_bytes: int


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _relative_to(
    candidate: Path,
    base: Path,
    *,
    error: str,
) -> Path:
    try:
        return candidate.relative_to(base)
    except ValueError as exc:
        raise JobOutputError(error) from exc


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass

    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _ensure_no_symlink_components(
    *,
    base: Path,
    candidate: Path,
    label: str,
) -> None:
    relative = _relative_to(
        candidate,
        base,
        error=f"{label} escapes its allowed root",
    )

    current = base

    if current.is_symlink():
        raise JobOutputError(
            f"{label} base must not be a symlink"
        )

    for part in relative.parts:
        current = current / part

        if os.path.lexists(current) and current.is_symlink():
            raise JobOutputError(
                f"{label} crosses symlink component: {current}"
            )


def _git_output(
    repo_root: Path,
    *args: str,
) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise JobOutputError(
            f"Git repository inspection failed for args={args!r}"
        ) from exc


def _tracked_top_level_names(
    repo_root: Path,
) -> frozenset[str]:
    raw = _git_output(
        repo_root,
        "ls-files",
        "-z",
    )

    names = set()

    for item in raw.split(b"\0"):
        if not item:
            continue

        path = PurePosixPath(item.decode("utf-8"))
        if path.parts:
            names.add(path.parts[0])

    return frozenset(names)


def _repository_relative_path(
    *,
    repository_input: Path,
    repository_resolved: Path,
    candidate: str | Path,
    label: str,
) -> tuple[Path, Path]:
    """Preserve in-repository components across system path aliases.

    On macOS, for example, ``/var`` may resolve to ``/private/var``. The
    repository root is canonicalized, but a caller may still supply the
    scratch path through the lexical alias used to reach that repository.

    This helper removes only that repository-prefix alias. Components *inside*
    the repository remain lexical, allowing the later symlink-component audit
    to detect and reject an actual scratch/job-root symlink.
    """
    raw_candidate = Path(candidate)

    if not raw_candidate.is_absolute():
        raw_candidate = repository_input / raw_candidate

    lexical_candidate = _lexical_absolute(raw_candidate)

    relative: Path | None = None

    for base in (
        repository_input,
        repository_resolved,
    ):
        try:
            relative = lexical_candidate.relative_to(base)
        except ValueError:
            continue
        else:
            break

    if relative is None:
        raise JobOutputError(
            f"{label} must be inside repository"
        )

    canonical_candidate = repository_resolved.joinpath(
        *relative.parts
    )

    return canonical_candidate, relative


def _repository_relative_path(
    *,
    repository_input: Path,
    repository_resolved: Path,
    candidate: str | Path,
    label: str,
) -> tuple[Path, Path]:
    """Preserve in-repository components across system path aliases.

    On macOS, for example, ``/var`` may resolve to ``/private/var``. The
    repository root is canonicalized, but a caller may still supply the
    scratch path through the lexical alias used to reach that repository.

    This helper removes only that repository-prefix alias. Components inside
    the repository remain lexical, allowing the later symlink-component audit
    to detect and reject an actual scratch/job-root symlink.
    """
    raw_candidate = Path(candidate)

    if not raw_candidate.is_absolute():
        raw_candidate = repository_input / raw_candidate

    lexical_candidate = _lexical_absolute(raw_candidate)

    relative: Path | None = None

    for base in (
        repository_input,
        repository_resolved,
    ):
        try:
            relative = lexical_candidate.relative_to(base)
        except ValueError:
            continue
        else:
            break

    if relative is None:
        raise JobOutputError(
            f"{label} must be inside repository"
        )

    canonical_candidate = repository_resolved.joinpath(
        *relative.parts
    )

    return canonical_candidate, relative


def validate_ignored_scratch_root(
    *,
    repository_root: str | Path,
    scratch_root: str | Path,
    protected_roots: Iterable[str | Path] = (),
) -> Path:
    """Validate one explicitly supplied existing ignored scratch directory."""
    repository_input = _lexical_absolute(
        repository_root
    )

    try:
        repository = repository_input.resolve(strict=True)
    except OSError as exc:
        raise JobOutputError(
            "repository_root must be an existing directory"
        ) from exc

    if not repository.is_dir():
        raise JobOutputError(
            "repository_root must be an existing directory"
        )

    scratch, relative = _repository_relative_path(
        repository_input=repository_input,
        repository_resolved=repository,
        candidate=scratch_root,
        label="scratch root",
    )

    if not relative.parts:
        raise JobOutputError(
            "repository root itself cannot be the scratch root"
        )

    if ".git" in relative.parts:
        raise JobOutputError(
            "scratch root cannot be inside .git"
        )

    _ensure_no_symlink_components(
        base=repository,
        candidate=scratch,
        label="scratch root",
    )

    if not scratch.exists() or not scratch.is_dir():
        raise JobOutputError(
            "scratch root must already exist as an explicit directory"
        )

    if scratch.resolve(strict=True) != scratch:
        raise JobOutputError(
            "scratch root must not resolve through symlinks"
        )

    for protected in protected_roots:
        raw_protected = Path(protected)

        try:
            protected_path, _ = _repository_relative_path(
                repository_input=repository_input,
                repository_resolved=repository,
                candidate=raw_protected,
                label="protected root",
            )
        except JobOutputError:
            if not raw_protected.is_absolute():
                raw_protected = repository_input / raw_protected

            protected_path = _lexical_absolute(
                raw_protected
            ).resolve(strict=False)

        if _paths_overlap(scratch, protected_path):
            raise JobOutputError(
                "scratch root overlaps a protected predecessor namespace"
            )

    tracked_top_levels = _tracked_top_level_names(repository)

    if relative.parts[0] in tracked_top_levels:
        raise JobOutputError(
            "scratch root overlaps a tracked repository namespace"
        )

    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "check-ignore",
            "--quiet",
            "--",
            relative.as_posix(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    if ignored.returncode != 0:
        raise JobOutputError(
            "scratch root must be explicitly Git-ignored"
        )

    return scratch

def canonical_job_relative_identity(
    node: TechnicalJobNode,
) -> str:
    """Map one canonical Part N job ID to a portable isolated relative root."""
    if not isinstance(node, TechnicalJobNode):
        raise JobOutputError(
            "node must be TechnicalJobNode"
        )

    digest = hashlib.sha256(
        node.job_id.encode("utf-8")
    ).hexdigest()

    return (
        f"jobs/v1/{node.node_type}/"
        f"{digest}"
    )


def bind_job_output_root(
    *,
    repository_root: str | Path,
    scratch_root: str | Path,
    node: TechnicalJobNode,
    protected_roots: Iterable[str | Path] = (),
) -> JobOutputRoot:
    """Allocate a new isolated root for exactly one declared DAG job."""
    scratch = validate_ignored_scratch_root(
        repository_root=repository_root,
        scratch_root=scratch_root,
        protected_roots=protected_roots,
    )
    repository = Path(repository_root).resolve(strict=True)

    relative_identity = canonical_job_relative_identity(node)
    relative = PurePosixPath(relative_identity)

    job_path = scratch.joinpath(*relative.parts)

    _relative_to(
        job_path,
        scratch,
        error="canonical job output root escaped scratch root",
    )

    _ensure_no_symlink_components(
        base=scratch,
        candidate=job_path.parent,
        label="job output root",
    )

    if os.path.lexists(job_path):
        raise JobOutputError(
            "job output root collision: canonical root already exists"
        )

    job_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _ensure_no_symlink_components(
        base=scratch,
        candidate=job_path.parent,
        label="job output root",
    )

    job_path.mkdir(
        exist_ok=False,
    )

    _ensure_no_symlink_components(
        base=scratch,
        candidate=job_path,
        label="job output root",
    )

    return JobOutputRoot(
        repository_root=repository,
        scratch_root=scratch,
        relative_identity=relative_identity,
        path=job_path,
        job_id=node.job_id,
        node_type=node.node_type,
        condition_id=node.condition_id,
        execution_allowed=node.execution_allowed,
    )


def _validated_relative_output_path(
    relative_path: str,
) -> PurePosixPath:
    if not isinstance(relative_path, str) or not relative_path:
        raise JobOutputError(
            "job output relative path must be a non-empty string"
        )

    if "\\" in relative_path:
        raise JobOutputError(
            "job output paths must use portable POSIX separators"
        )

    path = PurePosixPath(relative_path)

    if path.is_absolute():
        raise JobOutputError(
            "absolute/private output paths are forbidden"
        )

    if any(
        part in {"", ".", ".."}
        for part in path.parts
    ):
        raise JobOutputError(
            "job output relative path contains escape component"
        )

    if not path.parts:
        raise JobOutputError(
            "job output relative path must name a file"
        )

    return path


def _resolve_job_target(
    output_root: JobOutputRoot,
    relative_path: str,
) -> Path:
    if not isinstance(output_root, JobOutputRoot):
        raise JobOutputError(
            "output_root must be JobOutputRoot"
        )

    portable = _validated_relative_output_path(
        relative_path
    )
    target = output_root.path.joinpath(*portable.parts)

    _relative_to(
        target,
        output_root.path,
        error="job output target escaped isolated root",
    )

    _ensure_no_symlink_components(
        base=output_root.path,
        candidate=target.parent,
        label="job output target",
    )

    return target


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_once(
    *,
    output_root: JobOutputRoot,
    relative_path: str,
    payload: bytes,
    allow_completion_filename: bool,
) -> JobArtifactEvidence:
    if not isinstance(payload, bytes):
        raise JobOutputError(
            "atomic job payload must be bytes"
        )

    portable = _validated_relative_output_path(
        relative_path
    )

    if (
        portable.as_posix() == JOB_COMPLETION_FILENAME
        and not allow_completion_filename
    ):
        raise JobOutputError(
            "completion.json is reserved for immutable completion publication"
        )

    target = _resolve_job_target(
        output_root,
        portable.as_posix(),
    )

    if os.path.lexists(target):
        raise JobOutputError(
            "job output collision: target already exists"
        )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _ensure_no_symlink_components(
        base=output_root.path,
        candidate=target.parent,
        label="job output target",
    )

    temporary_name: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        # Hard-link publication is atomic and refuses to overwrite an
        # independently created target, unlike POSIX os.replace().
        try:
            os.link(
                temporary_name,
                target,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise JobOutputError(
                "job output collision during atomic publication"
            ) from exc

        os.unlink(temporary_name)
        temporary_name = None
        _fsync_directory(target.parent)

    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

    digest = hashlib.sha256(payload).hexdigest()

    return JobArtifactEvidence(
        relative_path=portable.as_posix(),
        sha256=digest,
        size_bytes=len(payload),
    )


def atomic_write_job_file(
    output_root: JobOutputRoot,
    relative_path: str,
    payload: bytes,
) -> JobArtifactEvidence:
    """Atomically publish one write-once artifact inside an executable job."""
    if not output_root.execution_allowed:
        raise JobOutputError(
            "inert placeholder jobs cannot publish output artifacts"
        )

    return _atomic_publish_once(
        output_root=output_root,
        relative_path=relative_path,
        payload=payload,
        allow_completion_filename=False,
    )


def canonical_completion_bytes(
    record: dict[str, Any],
) -> bytes:
    """Return deterministic JSON bytes for a tiny completion record."""
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
        raise JobOutputError(
            f"completion record is not canonical JSON: {exc}"
        ) from exc


def write_job_completion(
    output_root: JobOutputRoot,
    *,
    artifacts: Iterable[JobArtifactEvidence],
) -> JobCompletionEvidence:
    """Publish one immutable completion record after verifying its artifacts."""
    if not isinstance(output_root, JobOutputRoot):
        raise JobOutputError(
            "output_root must be JobOutputRoot"
        )

    if not output_root.execution_allowed:
        raise JobOutputError(
            "inert placeholder jobs cannot publish completion records"
        )

    artifact_tuple = tuple(artifacts)

    if any(
        not isinstance(item, JobArtifactEvidence)
        for item in artifact_tuple
    ):
        raise JobOutputError(
            "completion artifacts must be JobArtifactEvidence records"
        )

    relative_paths = [
        item.relative_path
        for item in artifact_tuple
    ]

    if len(set(relative_paths)) != len(relative_paths):
        raise JobOutputError(
            "completion artifacts contain duplicate relative paths"
        )

    canonical_artifacts = []

    for evidence in sorted(
        artifact_tuple,
        key=lambda item: item.relative_path,
    ):
        if evidence.relative_path == JOB_COMPLETION_FILENAME:
            raise JobOutputError(
                "completion record cannot list itself as an artifact"
            )

        target = _resolve_job_target(
            output_root,
            evidence.relative_path,
        )

        if not target.is_file():
            raise JobOutputError(
                f"completion artifact is missing: {evidence.relative_path}"
            )

        raw = target.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()

        if actual_sha != evidence.sha256:
            raise JobOutputError(
                f"completion artifact hash mismatch: {evidence.relative_path}"
            )

        if len(raw) != evidence.size_bytes:
            raise JobOutputError(
                f"completion artifact size mismatch: {evidence.relative_path}"
            )

        canonical_artifacts.append(
            {
                "relative_path": evidence.relative_path,
                "sha256": evidence.sha256,
                "size_bytes": evidence.size_bytes,
            }
        )

    record = {
        "schema_version": JOB_COMPLETION_SCHEMA_VERSION,
        "scientific_data": False,
        "production_eligible": False,
        "completion_state": "complete",
        "job_id": output_root.job_id,
        "node_type": output_root.node_type,
        "condition_id": output_root.condition_id,
        "relative_identity": output_root.relative_identity,
        "artifacts": canonical_artifacts,
    }

    payload = canonical_completion_bytes(record)

    evidence = _atomic_publish_once(
        output_root=output_root,
        relative_path=JOB_COMPLETION_FILENAME,
        payload=payload,
        allow_completion_filename=True,
    )

    return JobCompletionEvidence(
        path=output_root.path / JOB_COMPLETION_FILENAME,
        sha256=evidence.sha256,
        size_bytes=evidence.size_bytes,
    )

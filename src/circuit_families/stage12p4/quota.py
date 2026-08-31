"""Injected scratch quotas, deterministic rolling retention, and audit records."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final

from circuit_families.stage12p3.records import canonical_json_bytes, safe_relative_path

from .records import QuotaProfile, RetentionProfile, Stage12P4Error

QUOTA_RECORD_VERSION: Final = "stage12p4-quota-result/v1"
RETENTION_RECORD_VERSION: Final = "stage12p4-retention-result/v1"

_CLAIMED_ROOTS: set[str] = set()
_CLAIM_GUARD = threading.Lock()


class QuotaExceededError(Stage12P4Error):
    """Explicit technical failure when safe atomic completion cannot fit."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__("scratch hard quota cannot accommodate safe finalization")
        self.record = record


class ScratchClaimError(Stage12P4Error):
    """Raised when a concurrent scratch-management claim already exists."""


@dataclass(frozen=True)
class CheckpointGeneration:
    relative_path: str
    generation: int
    artifact_class: str
    valid: bool
    protected: bool = False

    def __post_init__(self) -> None:
        safe_relative_path(self.relative_path, label="checkpoint relative_path")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise Stage12P4Error("checkpoint generation must be non-negative")
        if not isinstance(self.valid, bool) or not isinstance(self.protected, bool):
            raise Stage12P4Error("checkpoint validity/protection must be boolean")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class ScratchManager:
    """Account and retain files only beneath one explicit job scratch root."""

    def __init__(
        self,
        root: Path,
        quota: QuotaProfile,
        retention: RetentionProfile,
    ) -> None:
        self.root = root.absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise Stage12P4Error("scratch root must not be a symlink")
        self.quota = quota
        self.retention = retention
        self.lock_path = self.root / ".stage12p4-quota.lock"
        self.retention_log_path = self.root / "retention-decisions.json"

    def _resolve(self, relative_path: str) -> Path:
        safe_relative_path(relative_path, label="scratch relative_path")
        candidate = self.root.joinpath(*Path(relative_path).parts)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise Stage12P4Error("scratch path escapes root") from exc
        current = self.root
        for part in candidate.relative_to(self.root).parts:
            current = current / part
            if os.path.lexists(current) and current.is_symlink():
                raise Stage12P4Error("scratch path crosses a symlink")
        return candidate

    @contextmanager
    def claim(self) -> Iterator[None]:
        key = str(self.root)
        with _CLAIM_GUARD:
            if key in _CLAIMED_ROOTS:
                raise ScratchClaimError("scratch manager already has an active claim")
            _CLAIMED_ROOTS.add(key)
        handle: BinaryIO | None = None
        try:
            handle = self.lock_path.open("a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ScratchClaimError("scratch manager already has an active claim") from exc
            yield
        finally:
            if handle is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()
            with _CLAIM_GUARD:
                _CLAIMED_ROOTS.discard(key)

    def inventory(self) -> tuple[dict[str, Any], ...]:
        rows = []
        for path in sorted(
            self.root.rglob("*"), key=lambda item: item.relative_to(self.root).as_posix()
        ):
            if path.is_symlink():
                raise Stage12P4Error("scratch inventory contains a symlink")
            if path.is_file():
                rows.append(
                    {
                        "relative_path": path.relative_to(self.root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "partial": path.name.endswith(".partial"),
                    }
                )
        return tuple(rows)

    def used_bytes(self) -> int:
        return sum(int(row["size_bytes"]) for row in self.inventory())

    def assess_finalization(
        self,
        *,
        staging_bytes: int,
        manifest_bytes: int,
    ) -> dict[str, Any]:
        for label, value in (("staging_bytes", staging_bytes), ("manifest_bytes", manifest_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Stage12P4Error(f"{label} must be non-negative")
        used = self.used_bytes()
        projected = used + staging_bytes + manifest_bytes + self.quota.atomic_reserve_bytes
        record = {
            "schema_version": QUOTA_RECORD_VERSION,
            "quota_profile_reference": self.quota.reference,
            "used_bytes": used,
            "staging_bytes": staging_bytes,
            "manifest_bytes": manifest_bytes,
            "atomic_reserve_bytes": self.quota.atomic_reserve_bytes,
            "projected_bytes": projected,
            "warning": projected >= self.quota.warning_bytes,
            "fits": projected <= self.quota.hard_bytes,
            "failure_category": (
                None if projected <= self.quota.hard_bytes else "insufficient_finalization_reserve"
            ),
            "retryable": projected > self.quota.hard_bytes,
            "scientific_data": False,
            "production_eligible": False,
        }
        if not record["fits"]:
            raise QuotaExceededError(record)
        return record

    def apply_retention(
        self,
        generations: Sequence[CheckpointGeneration],
        *,
        stale_partials: Sequence[str] = (),
        before_delete: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        identities = [(item.generation, item.relative_path) for item in generations]
        if len(set(identities)) != len(identities):
            raise Stage12P4Error("duplicate checkpoint generation records")
        valid_unprotected = sorted(
            (
                item
                for item in generations
                if item.valid
                and not item.protected
                and item.artifact_class not in self.retention.protected_artifact_classes
            ),
            key=lambda item: (-item.generation, item.relative_path),
        )
        retained_unprotected = {
            item.relative_path
            for item in valid_unprotected[: self.retention.maximum_retained_generations]
        }
        decisions: list[dict[str, Any]] = []
        deleted: list[str] = []

        def decide(relative_path: str, reason: str, *, eligible: bool) -> None:
            path = self._resolve(relative_path)
            decision = {
                "relative_path": relative_path,
                "reason": reason,
                "eligible": eligible,
                "existed": path.is_file(),
                "deleted": False,
            }
            decisions.append(decision)
            if eligible and path.is_file():
                if before_delete is not None:
                    before_delete(relative_path)
                path.unlink()
                decision["deleted"] = True
                deleted.append(relative_path)

        try:
            for item in sorted(
                generations, key=lambda value: (value.generation, value.relative_path)
            ):
                protected = (
                    item.protected
                    or item.artifact_class in self.retention.protected_artifact_classes
                )
                if protected:
                    decide(item.relative_path, "declared_protected", eligible=False)
                elif item.relative_path in retained_unprotected:
                    decide(item.relative_path, "newest_valid_recovery_boundary", eligible=False)
                elif not item.valid:
                    decide(item.relative_path, "invalid_not_recovery_boundary", eligible=True)
                else:
                    decide(item.relative_path, "outside_bounded_valid_set", eligible=True)
            for relative_path in sorted(stale_partials):
                if not relative_path.endswith(".partial"):
                    raise Stage12P4Error("stale partial cleanup requires .partial identity")
                decide(
                    relative_path,
                    "declared_stale_partial",
                    eligible=self.retention.partial_cleanup_eligible,
                )
        finally:
            record = {
                "schema_version": RETENTION_RECORD_VERSION,
                "retention_profile_reference": self.retention.reference,
                "decisions": decisions,
                "deleted_paths": sorted(deleted),
                "retained_valid_paths": sorted(retained_unprotected),
                "valid_recovery_boundary_retained": bool(retained_unprotected),
                "scientific_data": False,
                "production_eligible": False,
            }
            _atomic_write(self.retention_log_path, canonical_json_bytes(record))
        return record

    def reconcile_retention_log(self) -> dict[str, Any]:
        try:
            value = json.loads(self.retention_log_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise Stage12P4Error("retention decision log is missing or invalid") from exc
        if not isinstance(value, dict) or value.get("schema_version") != RETENTION_RECORD_VERSION:
            raise Stage12P4Error("retention decision log schema mismatch")
        for decision in value.get("decisions", []):
            path = self._resolve(decision["relative_path"])
            if decision.get("deleted") is True and path.exists():
                raise Stage12P4Error("retention log/file state mismatch")
        return value

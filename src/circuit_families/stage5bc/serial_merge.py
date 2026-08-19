"""Deterministic serial merge of small Stage 5B/C technical records.

Part Q merges only portable status/completion evidence. It retains completed,
failed, missing and unavailable states in canonical identity order and rejects
running, stale or conflicting runtime evidence.

This is not a Stage 5D summary or scientific aggregation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from circuit_families.stage5bc.job_status import (
    JobStatusReport,
)

SYNTHETIC_REGISTRY_SCHEMA_VERSION = (
    "stage5bc-synthetic-completion-registry/v1"
)

MERGE_ENTRY_STATES = (
    "completed",
    "failed",
    "missing",
    "unavailable",
)

MERGE_SOURCE_KINDS = (
    "job_status",
    "stage3_unavailable",
)

_SHA256_HEX = frozenset("0123456789abcdef")


class SerialMergeError(ValueError):
    """Raised when portable merge evidence is ambiguous or inconsistent."""


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


@dataclass(frozen=True)
class SyntheticMergeEntry:
    """One canonical portable registry entry."""

    identity_key: str
    source_kind: str
    state: str
    observed_status: str
    reason: str
    job_id: str | None = None
    node_type: str | None = None
    condition_id: str | None = None
    record_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity_key, str) or not self.identity_key:
            raise SerialMergeError(
                "identity_key must be a non-empty string"
            )

        if self.source_kind not in MERGE_SOURCE_KINDS:
            raise SerialMergeError(
                f"unsupported merge source kind: {self.source_kind!r}"
            )

        if self.state not in MERGE_ENTRY_STATES:
            raise SerialMergeError(
                f"unsupported merge entry state: {self.state!r}"
            )

        if (
            not isinstance(self.observed_status, str)
            or not self.observed_status
        ):
            raise SerialMergeError(
                "observed_status must be a non-empty string"
            )

        if (
            not isinstance(self.reason, str)
            or not self.reason
            or "\n" in self.reason
            or "\r" in self.reason
        ):
            raise SerialMergeError(
                "reason must be a non-empty single line"
            )

        if self.source_kind == "job_status":
            for field in (
                "job_id",
                "node_type",
                "condition_id",
            ):
                value = getattr(self, field)

                if not isinstance(value, str) or not value:
                    raise SerialMergeError(
                        f"job-status entry requires non-empty {field}"
                    )

            if self.state == "unavailable":
                raise SerialMergeError(
                    "job-status entries cannot use unavailable state"
                )

            expected_identity = f"job::{self.job_id}"

            if self.identity_key != expected_identity:
                raise SerialMergeError(
                    "job-status identity_key does not match job_id"
                )

        else:
            if self.state != "unavailable":
                raise SerialMergeError(
                    "stage3_unavailable source requires unavailable state"
                )

            if self.observed_status != "unavailable":
                raise SerialMergeError(
                    "Stage 3 unavailable entry must observe unavailable"
                )

            if any(
                value is not None
                for value in (
                    self.job_id,
                    self.node_type,
                    self.condition_id,
                    self.record_sha256,
                )
            ):
                raise SerialMergeError(
                    "Stage 3 unavailable entry cannot carry downstream fields"
                )

        if self.state in {"completed", "failed"}:
            if not _valid_sha256(self.record_sha256):
                raise SerialMergeError(
                    f"{self.state} entry requires exact record SHA-256"
                )
        elif self.record_sha256 is not None:
            raise SerialMergeError(
                f"{self.state} entry must not carry terminal record SHA-256"
            )

    def to_mapping(self) -> dict[str, Any]:
        """Return the exact portable representation."""
        return {
            "identity_key": self.identity_key,
            "source_kind": self.source_kind,
            "state": self.state,
            "observed_status": self.observed_status,
            "reason": self.reason,
            "job_id": self.job_id,
            "node_type": self.node_type,
            "condition_id": self.condition_id,
            "record_sha256": self.record_sha256,
        }


def entry_from_job_status(
    report: JobStatusReport,
) -> SyntheticMergeEntry:
    """Convert one mergeable Part P report into a portable registry entry."""
    if not isinstance(report, JobStatusReport):
        raise SerialMergeError(
            "report must be JobStatusReport"
        )

    if report.status == "completed":
        state = "completed"
        record_sha = report.completion_sha256

        if not _valid_sha256(record_sha):
            raise SerialMergeError(
                "completed status lacks exact completion SHA-256"
            )

    elif report.status == "failed":
        state = "failed"
        record_sha = report.failure_sha256

        if not _valid_sha256(record_sha):
            raise SerialMergeError(
                "failed status lacks exact failure SHA-256"
            )

    elif report.status in {"planned", "blocked"}:
        state = "missing"
        record_sha = None

    elif report.status == "running":
        raise SerialMergeError(
            "running job cannot enter deterministic completion registry"
        )

    elif report.status == "stale":
        raise SerialMergeError(
            "stale job cannot enter deterministic completion registry"
        )

    elif report.status == "conflicting":
        raise SerialMergeError(
            "conflicting job cannot enter deterministic completion registry"
        )

    else:
        raise SerialMergeError(
            f"unknown Part P status: {report.status!r}"
        )

    return SyntheticMergeEntry(
        identity_key=f"job::{report.job_id}",
        source_kind="job_status",
        state=state,
        observed_status=report.status,
        reason=report.reason,
        job_id=report.job_id,
        node_type=report.node_type,
        condition_id=report.condition_id,
        record_sha256=record_sha,
    )


def stage3_unavailable_entry(
    *,
    teacher_seed: int,
    phase: str,
    reason: str,
) -> SyntheticMergeEntry:
    """Create a canonical portable marker for an unavailable Stage 3 cell."""
    if (
        isinstance(teacher_seed, bool)
        or not isinstance(teacher_seed, int)
        or teacher_seed < 0
    ):
        raise SerialMergeError(
            "teacher_seed must be a non-negative integer"
        )

    if not isinstance(phase, str) or not phase:
        raise SerialMergeError(
            "phase must be a non-empty string"
        )

    encoded_phase = quote(
        phase,
        safe="",
    )

    identity_key = (
        "stage3-unavailable/v1::"
        f"teacher_seed={teacher_seed}::"
        f"phase={encoded_phase}"
    )

    return SyntheticMergeEntry(
        identity_key=identity_key,
        source_kind="stage3_unavailable",
        state="unavailable",
        observed_status="unavailable",
        reason=reason,
    )


def merge_entries(
    entries: Iterable[SyntheticMergeEntry],
) -> dict[str, Any]:
    """Merge entries in canonical identity order with strict uniqueness."""
    materialized = tuple(entries)

    if any(
        not isinstance(entry, SyntheticMergeEntry)
        for entry in materialized
    ):
        raise SerialMergeError(
            "all merge inputs must be SyntheticMergeEntry"
        )

    by_identity: dict[
        str,
        SyntheticMergeEntry,
    ] = {}

    job_coordinates: set[
        tuple[str, str]
    ] = set()

    for entry in materialized:
        if entry.identity_key in by_identity:
            raise SerialMergeError(
                "duplicate canonical merge identity"
            )

        if entry.source_kind == "job_status":
            assert entry.node_type is not None
            assert entry.condition_id is not None

            coordinate = (
                entry.node_type,
                entry.condition_id,
            )

            if coordinate in job_coordinates:
                raise SerialMergeError(
                    "duplicate job node/condition coordinate"
                )

            job_coordinates.add(coordinate)

        by_identity[entry.identity_key] = entry

    ordered = [
        by_identity[key]
        for key in sorted(by_identity)
    ]

    return {
        "schema_version": SYNTHETIC_REGISTRY_SCHEMA_VERSION,
        "scientific_data": False,
        "production_eligible": False,
        "serialization_profile": "canonical-json-sort-keys/v1",
        "entries": [
            entry.to_mapping()
            for entry in ordered
        ],
    }


def merge_status_evidence(
    *,
    reports: Iterable[JobStatusReport],
    unavailable_entries: Iterable[SyntheticMergeEntry] = (),
) -> dict[str, Any]:
    """Merge Part P reports plus explicit unavailable Stage 3 markers."""
    job_entries = tuple(
        entry_from_job_status(report)
        for report in reports
    )
    unavailable = tuple(unavailable_entries)

    return merge_entries(
        (*job_entries, *unavailable)
    )


def canonical_registry_bytes(
    registry: Mapping[str, Any],
) -> bytes:
    """Serialize a synthetic registry deterministically."""
    if not isinstance(registry, Mapping):
        raise SerialMergeError(
            "registry must be a mapping"
        )

    try:
        return (
            json.dumps(
                copy.deepcopy(dict(registry)),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SerialMergeError(
            f"registry is not canonical-JSON serializable: {exc}"
        ) from exc


def registry_sha256(
    registry: Mapping[str, Any],
) -> str:
    """Return the exact SHA-256 of the canonical registry bytes."""
    return hashlib.sha256(
        canonical_registry_bytes(registry)
    ).hexdigest()

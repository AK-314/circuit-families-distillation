"""Serial deterministic merge with explicit duplicate and conflict evidence."""

from __future__ import annotations

import hashlib
import heapq
import os
import secrets
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from circuit_families.stage12p3.records import canonical_json_bytes, require_reference

from .compact import (
    CompactObjectEvidence,
    MetricLedgerWriter,
    MetricSchema,
    iter_ledger_rows,
    read_ledger_header,
)
from .records import CodecProfile, Stage12P4Error, sha256_file

MERGED_MANIFEST_VERSION: Final = "stage12p4-merged-ledger-manifest/v1"


class MergeConflictError(Stage12P4Error):
    """Raised when a canonical row key maps to conflicting content."""


@dataclass(frozen=True)
class ShardDescriptor:
    shard_id: str
    logical_campaign: str
    basis_identity: str
    producer_interface_version: str
    path: Path
    compact_sha256: str
    profile: CodecProfile
    lifecycle_state: str = "sealed"

    def __post_init__(self) -> None:
        for label in (
            "shard_id",
            "logical_campaign",
            "basis_identity",
            "producer_interface_version",
        ):
            require_reference(getattr(self, label), label=label)
        if len(self.compact_sha256) != 64:
            raise Stage12P4Error("shard compact SHA-256 is invalid")
        if self.lifecycle_state not in {"sealed", "failed", "unavailable", "censored"}:
            raise Stage12P4Error("unsupported shard lifecycle state")


def _atomic_manifest(path: Path, value: Mapping[str, Any]) -> bytes:
    data = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise MergeConflictError("refusing to overwrite a conflicting merged manifest")
        return data
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
    return data


def merge_ledgers(
    shards: Sequence[ShardDescriptor],
    *,
    expected_shard_ids: Sequence[str],
    schema: MetricSchema,
    output_path: Path,
    manifest_path: Path,
    output_profile: CodecProfile,
) -> tuple[CompactObjectEvidence, dict[str, Any]]:
    """Merge a closed logical shard set in canonical key order."""
    expected = tuple(sorted(expected_shard_ids))
    if len(set(expected)) != len(expected) or not expected:
        raise Stage12P4Error("expected shard identities must be non-empty and unique")
    ordered = tuple(sorted(shards, key=lambda item: item.shard_id))
    actual = tuple(item.shard_id for item in ordered)
    if actual != expected:
        raise Stage12P4Error("logical shard closure is missing or contains duplicates")
    campaigns = {item.logical_campaign for item in ordered}
    bases = {item.basis_identity for item in ordered}
    producers = {item.producer_interface_version for item in ordered}
    if len(campaigns) != 1 or len(bases) != 1 or len(producers) != 1:
        raise Stage12P4Error("shards have incompatible campaign, basis, or producer versions")

    source_inventory = []
    iterators: dict[str, Iterator[dict[str, Any]]] = {}
    for shard in ordered:
        size, digest = sha256_file(shard.path)
        if digest != shard.compact_sha256:
            raise Stage12P4Error(f"source shard hash mismatch: {shard.shard_id}")
        header = read_ledger_header(shard.path, profile=shard.profile)
        if header["schema"] != schema.to_mapping():
            raise Stage12P4Error("source shard schema drift detected")
        context = header["context"]
        expected_context = {
            "logical_campaign": shard.logical_campaign,
            "basis_identity": shard.basis_identity,
            "producer_interface_version": shard.producer_interface_version,
            "shard_id": shard.shard_id,
        }
        if context != expected_context:
            raise Stage12P4Error("source shard logical context mismatch")
        source_inventory.append(
            {
                "shard_id": shard.shard_id,
                "lifecycle_state": shard.lifecycle_state,
                "row_count": header["row_count"],
                "byte_length": size,
                "sha256": digest,
            }
        )
        iterators[shard.shard_id] = iter_ledger_rows(
            shard.path,
            profile=shard.profile,
            expected_schema=schema,
        )

    stale_partial = output_path.with_name(f".{output_path.name}.rows.partial")
    if stale_partial.exists():
        stale_partial.unlink()
    writer = MetricLedgerWriter(
        output_path,
        schema,
        context={
            "logical_campaign": next(iter(campaigns)),
            "basis_identity": next(iter(bases)),
            "producer_interface_version": next(iter(producers)),
            "merged_shard_ids": list(expected),
        },
        profile=output_profile,
    )
    heap: list[tuple[tuple[Any, ...], str, dict[str, Any]]] = []
    for shard_id, iterator in sorted(iterators.items()):
        try:
            row = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (schema.key_for(row), shard_id, row))

    duplicate_provenance = []
    unique_count = 0
    duplicate_count = 0
    try:
        while heap:
            key = heap[0][0]
            group: list[tuple[str, dict[str, Any]]] = []
            while heap and heap[0][0] == key:
                _, shard_id, row = heapq.heappop(heap)
                group.append((shard_id, row))
                try:
                    following = next(iterators[shard_id])
                except StopIteration:
                    pass
                else:
                    heapq.heappush(
                        heap,
                        (schema.key_for(following), shard_id, following),
                    )
            canonical_rows = {canonical_json_bytes(row): [] for _, row in group}
            for shard_id, row in group:
                canonical_rows[canonical_json_bytes(row)].append(shard_id)
            if len(canonical_rows) != 1:
                raise MergeConflictError(f"conflicting content for canonical key {key!r}")
            selected = min(group, key=lambda item: item[0])[1]
            writer.append(selected)
            unique_count += 1
            if len(group) > 1:
                sources = sorted(shard_id for shard_id, _ in group)
                duplicate_count += len(group) - 1
                duplicate_provenance.append(
                    {
                        "key": list(key),
                        "retained_source": sources[0],
                        "duplicate_sources": sources[1:],
                        "rule": "byte-identical-row-dedup/v1",
                    }
                )
        evidence = writer.finalize()
    except Exception:
        writer.abort()
        raise

    manifest: dict[str, Any] = {
        "schema_version": MERGED_MANIFEST_VERSION,
        "logical_campaign": next(iter(campaigns)),
        "basis_identity": next(iter(bases)),
        "producer_interface_version": next(iter(producers)),
        "schema": schema.to_mapping(),
        "canonical_key_fields": list(schema.key_fields),
        "deduplication_rule": "byte-identical-row-dedup/v1",
        "source_closure": source_inventory,
        "source_shard_count": len(source_inventory),
        "empty_shard_count": sum(item["row_count"] == 0 for item in source_inventory),
        "unique_row_count": unique_count,
        "duplicate_row_count": duplicate_count,
        "duplicate_provenance": duplicate_provenance,
        "merged_object": {
            "byte_length": evidence.compact_byte_length,
            "sha256": evidence.compact_sha256,
            "logical_row_bytes": evidence.logical_byte_length,
            "logical_row_sha256": evidence.logical_sha256,
        },
        "recomputation_evidence": {
            "all_source_hashes_verified": True,
            "canonical_order_recomputed": True,
            "semantic_conflicts": 0,
        },
        "scientific_data": False,
        "production_eligible": False,
    }
    manifest_bytes = _atomic_manifest(manifest_path, manifest)
    manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return evidence, manifest

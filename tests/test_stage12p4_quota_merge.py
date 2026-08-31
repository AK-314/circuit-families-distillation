from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from circuit_families.stage12p4 import (
    CheckpointGeneration,
    CodecProfile,
    LedgerField,
    MergeConflictError,
    MetricSchema,
    QuotaExceededError,
    QuotaProfile,
    RetentionProfile,
    ScratchClaimError,
    ScratchManager,
    ShardDescriptor,
    Stage12P4Error,
    iter_ledger_rows,
    merge_ledgers,
    write_ledger,
)


def profiles(hard: int = 1000) -> tuple[QuotaProfile, RetentionProfile]:
    return (
        QuotaProfile("quota/test/v1", hard, hard // 2, 100),
        RetentionProfile("retention/test/v1", 1, 2, ("protected",), True),
    )


def test_quota_exact_boundary_partials_and_reserve(tmp_path: Path) -> None:
    quota, retention = profiles()
    manager = ScratchManager(tmp_path, quota, retention)
    (tmp_path / "complete.bin").write_bytes(b"x" * 400)
    (tmp_path / "stage.partial").write_bytes(b"x" * 100)
    record = manager.assess_finalization(staging_bytes=300, manifest_bytes=100)
    assert record["projected_bytes"] == 1000
    assert record["fits"] is True
    with pytest.raises(QuotaExceededError) as captured:
        manager.assess_finalization(staging_bytes=301, manifest_bytes=100)
    assert captured.value.record["failure_category"] == "insufficient_finalization_reserve"


def test_retention_preserves_protected_and_older_valid_boundary(tmp_path: Path) -> None:
    quota, retention = profiles(10_000)
    manager = ScratchManager(tmp_path, quota, retention)
    generations = []
    for generation in range(5):
        relative = f"checkpoints/{generation}.bin"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([generation]))
        generations.append(
            CheckpointGeneration(
                relative,
                generation,
                "protected" if generation == 0 else "resume",
                valid=generation != 4,
            )
        )
    partial = tmp_path / "stale.partial"
    partial.write_bytes(b"partial")
    result = manager.apply_retention(generations, stale_partials=("stale.partial",))
    assert (tmp_path / "checkpoints/0.bin").exists()
    assert (tmp_path / "checkpoints/3.bin").exists()
    assert (tmp_path / "checkpoints/2.bin").exists()
    assert not (tmp_path / "checkpoints/4.bin").exists()
    assert result["valid_recovery_boundary_retained"] is True
    assert manager.reconcile_retention_log() == result


def test_interrupted_cleanup_is_audited_and_concurrent_claim_is_rejected(tmp_path: Path) -> None:
    quota, retention = profiles(10_000)
    manager = ScratchManager(tmp_path, quota, retention)
    for generation in range(3):
        path = tmp_path / f"{generation}.bin"
        path.write_bytes(b"x")
    generations = [
        CheckpointGeneration(f"{generation}.bin", generation, "resume", True)
        for generation in range(3)
    ]
    calls = 0

    def interrupt(_path: str) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("forced interruption")

    with manager.claim():
        with pytest.raises(ScratchClaimError):
            with manager.claim():
                pass
        with pytest.raises(RuntimeError, match="forced interruption"):
            manager.apply_retention(generations, before_delete=interrupt)
    assert manager.retention_log_path.exists()
    assert (tmp_path / "2.bin").exists()
    assert (tmp_path / "1.bin").exists()


def schema() -> MetricSchema:
    return MetricSchema(
        "schema/merge-test/v1",
        (
            LedgerField("key", "integer", allow_negative=False),
            LedgerField("status", "string"),
            LedgerField("value", "number", nullable=True),
        ),
        ("key",),
    )


def row(key: int, value: float | None = None, status: str = "complete") -> dict:
    return {"key": key, "status": status, "value": float(key) if value is None else value}


def shard(
    root: Path,
    shard_id: str,
    rows: list[dict],
    *,
    campaign: str = "campaign/test/v1",
    basis: str = "basis/test/v1",
) -> ShardDescriptor:
    profile = CodecProfile("codec/gzip-test/v1", "gzip", 6)
    path = root / f"{shard_id}.gz"
    evidence = write_ledger(
        path,
        rows,
        schema=schema(),
        context={
            "logical_campaign": campaign,
            "basis_identity": basis,
            "producer_interface_version": "producer/test/v1",
            "shard_id": shard_id,
        },
        profile=profile,
    )
    return ShardDescriptor(
        shard_id,
        campaign,
        basis,
        "producer/test/v1",
        path,
        evidence.compact_sha256,
        profile,
        lifecycle_state="unavailable" if not rows else "sealed",
    )


def merge(root: Path, shards: list[ShardDescriptor], name: str):
    profile = CodecProfile("codec/gzip-test/v1", "gzip", 6)
    return merge_ledgers(
        shards,
        expected_shard_ids=("a", "b", "empty"),
        schema=schema(),
        output_path=root / f"{name}.gz",
        manifest_path=root / f"{name}.json",
        output_profile=profile,
    )


def test_merge_is_arrival_order_independent_and_accounts_duplicates_empty_states(
    tmp_path: Path,
) -> None:
    shards = [
        shard(tmp_path, "a", [row(0), row(2), row(4, status="failed")]),
        shard(tmp_path, "b", [row(1), row(2), row(3, status="censored")]),
        shard(tmp_path, "empty", []),
    ]
    first, first_manifest = merge(tmp_path, shards, "first")
    second, second_manifest = merge(tmp_path, list(reversed(shards)), "second")
    assert first.compact_sha256 == second.compact_sha256
    assert first_manifest["duplicate_row_count"] == 1
    assert first_manifest["empty_shard_count"] == 1
    assert first_manifest["source_closure"] == second_manifest["source_closure"]
    rows = list(iter_ledger_rows(first.path, profile=shards[0].profile, expected_schema=schema()))
    assert [item["status"] for item in rows] == [
        "complete",
        "complete",
        "complete",
        "censored",
        "failed",
    ]


def test_merge_rejects_conflict_missing_shard_corruption_and_context_drift(tmp_path: Path) -> None:
    a = shard(tmp_path, "a", [row(0), row(2)])
    b = shard(tmp_path, "b", [row(2, 99.0)])
    empty = shard(tmp_path, "empty", [])
    with pytest.raises(MergeConflictError, match="conflicting"):
        merge(tmp_path, [a, b, empty], "conflict")
    with pytest.raises(Stage12P4Error, match="closure"):
        merge(tmp_path, [a, b], "missing")
    corrupt = ShardDescriptor(
        b.shard_id,
        b.logical_campaign,
        b.basis_identity,
        b.producer_interface_version,
        b.path,
        hashlib.sha256(b"wrong").hexdigest(),
        b.profile,
    )
    with pytest.raises(Stage12P4Error, match="hash mismatch"):
        merge(tmp_path, [a, corrupt, empty], "corrupt")
    drift = ShardDescriptor(
        b.shard_id,
        "campaign/drift/v1",
        b.basis_identity,
        b.producer_interface_version,
        b.path,
        b.compact_sha256,
        b.profile,
    )
    with pytest.raises(Stage12P4Error, match="incompatible"):
        merge(tmp_path, [a, drift, empty], "drift")


def test_merge_restart_discards_only_its_stale_partial(tmp_path: Path) -> None:
    shards = [
        shard(tmp_path, "a", [row(0)]),
        shard(tmp_path, "b", [row(1)]),
        shard(tmp_path, "empty", []),
    ]
    stale = tmp_path / ".resume.gz.rows.partial"
    stale.write_bytes(b"interrupted")
    evidence, manifest = merge(tmp_path, shards, "resume")
    assert not stale.exists()
    repeated, repeated_manifest = merge(tmp_path, list(reversed(shards)), "resume")
    assert repeated.compact_sha256 == evidence.compact_sha256
    assert repeated_manifest["manifest_sha256"] == manifest["manifest_sha256"]

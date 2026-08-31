from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from circuit_families.stage12p3 import (
    ExpectedArtifact,
    HashBoundReference,
    LogicalJobSpec,
    OutputContract,
    canonical_json_bytes,
)
from circuit_families.stage12p3.state import SEALED_OUTPUT_SCHEMA_VERSION
from circuit_families.stage12p4 import (
    CodecProfile,
    LedgerField,
    MetricLedgerWriter,
    MetricSchema,
    ProducerEvidence,
    Stage12P4Error,
    StorageObjectContract,
    iter_ledger_rows,
    producer_evidence_from_p3,
    read_mask,
    write_ledger,
    write_mask,
)


def codec(codec: str = "gzip") -> CodecProfile:
    return CodecProfile(
        f"codec/{codec}-test/v1",
        codec,
        6 if codec == "gzip" else None,
    )


def schema(*, allow_negative: bool = True) -> MetricSchema:
    return MetricSchema(
        "schema/test-ledger/v1",
        (
            LedgerField("key", "integer", allow_negative=False),
            LedgerField("status", "string"),
            LedgerField("metric", "number", nullable=True, allow_negative=allow_negative),
            LedgerField("detail", "string", nullable=True),
        ),
        ("key",),
    )


def row(key: int, *, metric: float | None = 0.0, status: str = "complete") -> dict:
    return {
        "key": key,
        "status": status,
        "metric": metric,
        "detail": None if status == "complete" else status,
    }


@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 63, 64, 65, 517])
@pytest.mark.parametrize("fill", [0, 1])
def test_mask_round_trip_across_byte_boundaries(tmp_path: Path, count: int, fill: int) -> None:
    values = (fill,) * count
    path = tmp_path / f"mask-{count}-{fill}.bin"
    write_mask(
        path,
        values,
        component_universe="universe/test/v1",
        basis_identity="basis/test/v1",
        profile=codec(),
    )
    assert (
        read_mask(
            path,
            profile=codec(),
            expected_component_universe="universe/test/v1",
            expected_basis_identity="basis/test/v1",
        )
        == values
    )


def test_mask_padding_corruption_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mask.bin"
    write_mask(
        path,
        (1,) * 9,
        component_universe="universe/test/v1",
        basis_identity="basis/test/v1",
        profile=codec("none"),
    )
    data = bytearray(path.read_bytes())
    data[-1] |= 1
    path.write_bytes(data)
    with pytest.raises(Stage12P4Error, match="hash|padding"):
        read_mask(path, profile=codec("none"))


def test_mask_universe_and_basis_mismatch_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mask.gz"
    write_mask(
        path,
        (0, 1, 0),
        component_universe="universe/a/v1",
        basis_identity="basis/a/v1",
        profile=codec(),
    )
    with pytest.raises(Stage12P4Error, match="universe"):
        read_mask(path, profile=codec(), expected_component_universe="universe/b/v1")
    with pytest.raises(Stage12P4Error, match="basis"):
        read_mask(path, profile=codec(), expected_basis_identity="basis/b/v1")


def test_mask_bytes_are_deterministic_across_directories(tmp_path: Path) -> None:
    outputs = []
    for name in ("one", "two"):
        path = tmp_path / name / "mask.gz"
        evidence = write_mask(
            path,
            tuple(int(index % 5 == 0) for index in range(101)),
            component_universe="universe/test/v1",
            basis_identity="basis/test/v1",
            profile=codec(),
        )
        outputs.append((path.read_bytes(), evidence.compact_sha256))
    assert outputs[0] == outputs[1]


def test_ledger_round_trip_preserves_all_explicit_states(tmp_path: Path) -> None:
    rows = [
        row(0, metric=-1.5),
        row(1, metric=None, status="failed"),
        row(2, metric=None, status="unavailable"),
        row(3, metric=None, status="censored"),
    ]
    path = tmp_path / "ledger.gz"
    write_ledger(path, rows, schema=schema(), context={"fixture": "states/v1"}, profile=codec())
    assert list(iter_ledger_rows(path, profile=codec(), expected_schema=schema())) == rows


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_ledger_rejects_nonfinite_numbers(tmp_path: Path, value: float) -> None:
    writer = MetricLedgerWriter(
        tmp_path / "ledger.gz",
        schema(),
        context={"fixture": "nonfinite/v1"},
        profile=codec(),
    )
    with pytest.raises(Stage12P4Error, match="non-finite"):
        writer.append(row(0, metric=value))
    writer.abort()


def test_ledger_negative_policy_and_closed_schema(tmp_path: Path) -> None:
    writer = MetricLedgerWriter(
        tmp_path / "ledger.gz",
        schema(allow_negative=False),
        context={"fixture": "negative/v1"},
        profile=codec(),
    )
    with pytest.raises(Stage12P4Error, match="negative"):
        writer.append(row(0, metric=-0.1))
    with pytest.raises(Stage12P4Error, match="closed schema"):
        writer.append({**row(0), "extra": 1})
    writer.abort()


def test_ledger_requires_strict_canonical_row_order(tmp_path: Path) -> None:
    writer = MetricLedgerWriter(
        tmp_path / "ledger.gz",
        schema(),
        context={"fixture": "ordering/v1"},
        profile=codec(),
    )
    writer.append(row(2))
    with pytest.raises(Stage12P4Error, match="canonical key order"):
        writer.append(row(1))
    writer.abort()


def test_interrupted_finalization_retains_partial_not_sealed_object(tmp_path: Path) -> None:
    path = tmp_path / "ledger.gz"
    writer = MetricLedgerWriter(
        path,
        schema(),
        context={"fixture": "interrupted/v1"},
        profile=codec(),
    )
    writer.append(row(0))
    writer.abort()
    assert not path.exists()
    assert writer.partial_path.exists()


def test_large_ledger_streams_and_reconstructs_legacy_records(tmp_path: Path) -> None:
    rows = [row(index, metric=float(index)) for index in range(10_000)]
    path = tmp_path / "large-ledger.gz"
    evidence = write_ledger(
        path,
        rows,
        schema=schema(),
        context={"legacy_record_interface": "stage6a-exact-ledger/v1"},
        profile=codec(),
    )
    assert evidence.row_count == 10_000
    iterator = iter_ledger_rows(path, profile=codec(), expected_schema=schema())
    assert next(iterator) == rows[0]
    assert sum(1 for _ in iterator) == 9_999


def _p3_job() -> LogicalJobSpec:
    reference = HashBoundReference("synthetic://input", "a" * 64, "input/v1")
    return LogicalJobSpec(
        family="synthetic-producer",
        producer_interface_version="synthetic-producer/v1",
        dependencies=(),
        expected_inputs=(reference,),
        payload_reference=reference,
        config_reference=reference,
        output_contract=OutputContract(
            "manifests/output.json",
            "synthetic-output/v1",
            (ExpectedArtifact("artifacts/data.bin", "application/octet-stream"),),
        ),
        resource_class_reference="resource/test/v1",
        priority_class_reference="priority/test/v1",
        protected_tier="tier/test/v1",
        retry_seed_namespace_reference="seed/test/v1",
    )


def test_producer_evidence_reuses_p3_job_attempt_and_sealed_manifest() -> None:
    job = _p3_job()
    manifest = {
        "schema_version": SEALED_OUTPUT_SCHEMA_VERSION,
        "declared_output_schema_version": "synthetic-output/v1",
        "campaign_id": "b" * 64,
        "job_id": job.job_id,
        "attempt_index": 0,
        "retry_index": 0,
        "artifacts": [
            {
                "relative_path": "artifacts/data.bin",
                "sha256": "c" * 64,
                "size_bytes": 1,
                "media_type": "application/octet-stream",
            }
        ],
        "scientific_data": False,
        "production_eligible": False,
    }
    raw = canonical_json_bytes(manifest)
    evidence = producer_evidence_from_p3(
        job,
        attempt_index=0,
        sealed_manifest=manifest,
        sealed_manifest_bytes=raw,
        source_relative_path="artifacts/data.bin",
    )
    assert evidence.logical_job_id == job.job_id
    assert evidence.sealed_manifest_sha256 == hashlib.sha256(raw).hexdigest()
    with pytest.raises(Stage12P4Error, match="cross-job"):
        producer_evidence_from_p3(
            job,
            attempt_index=0,
            sealed_manifest={**manifest, "job_id": "d" * 64},
            sealed_manifest_bytes=raw,
            source_relative_path="artifacts/data.bin",
        )


def test_storage_contract_rejects_unknown_fields_boundaries_and_ambiguity() -> None:
    evidence = ProducerEvidence("a" * 64, 0, "b" * 64, "c" * 64, "artifact.bin")
    contract = StorageObjectContract(
        "checkpoint",
        "producer/v1",
        evidence,
        "logical/v1",
        ("payload",),
        1,
        "d" * 64,
        "opaque-bytes/v1",
        "codec/none/v1",
        "chunking/none/v1",
        1,
        "e" * 64,
        "quota/injected/v1",
        "retention/injected/v1",
        "sealed",
    )
    assert StorageObjectContract.from_mapping(contract.to_mapping()) == contract
    with pytest.raises(Stage12P4Error, match="closed schema"):
        StorageObjectContract.from_mapping({**contract.to_mapping(), "unknown": 1})
    with pytest.raises(Stage12P4Error, match="production_eligible"):
        StorageObjectContract(**{**contract.__dict__, "production_eligible": True})
    with pytest.raises(Stage12P4Error, match="ambiguous"):
        StorageObjectContract(**{**contract.__dict__, "storage_encoding": "maybe"})

"""Portable validate-only Stage 12-P4 compact-storage lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from circuit_families.stage12p3.records import canonical_json_bytes

from .compact import (
    LedgerField,
    MetricSchema,
    iter_ledger_rows,
    read_mask,
    write_ledger,
    write_mask,
)
from .export import (
    DestinationConflictError,
    LocalFilesystemExportAdapter,
    TransferInterrupted,
    build_bundle,
    verify_destination,
)
from .merge import ShardDescriptor, merge_ledgers
from .quota import CheckpointGeneration, QuotaExceededError, ScratchManager
from .records import (
    CodecProfile,
    ProducerEvidence,
    QuotaProfile,
    RetentionProfile,
    StorageObjectContract,
)


def _write_identical(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"existing validate-only fixture conflicts: {path.name}")
    path.write_bytes(data)


def _row(record_id: int) -> dict[str, Any]:
    status = ("complete", "failed", "unavailable", "censored")[record_id % 4]
    value = None if status != "complete" else (record_id - 175) / 10.0
    detail = None if status == "complete" else f"synthetic-{status}"
    return {
        "record_id": f"row-{record_id:06d}",
        "status": status,
        "value": value,
        "detail": detail,
    }


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def run_validate_only(output_root: Path) -> dict[str, Any]:
    output_root = output_root.absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink():
        raise ValueError("validate-only output root must not be a symlink")
    profile = CodecProfile("codec/gzip-technical/v1", "gzip", 6, chunk_bytes=2048)
    quota = QuotaProfile("quota/synthetic-small/v1", 4096, 1400, 256)
    retention = RetentionProfile(
        "retention/synthetic-rolling/v1",
        checkpoint_cadence=1,
        maximum_retained_generations=2,
        protected_artifact_classes=("final-checkpoint",),
        partial_cleanup_eligible=True,
    )
    schema = MetricSchema(
        "schema/synthetic-metrics/v1",
        (
            LedgerField("record_id", "string"),
            LedgerField("status", "string"),
            LedgerField("value", "number", nullable=True, allow_negative=True),
            LedgerField("detail", "string", nullable=True),
        ),
        ("record_id",),
    )

    artifacts = output_root / "artifacts"
    mask_values = tuple(int(index % 3 == 0 or index % 11 == 0) for index in range(517))
    mask_path = artifacts / "masks" / "synthetic.mask.gz"
    mask_evidence = write_mask(
        mask_path,
        mask_values,
        component_universe="component-universe/synthetic-517/v1",
        basis_identity="basis/synthetic-canonical/v1",
        profile=profile,
    )

    shard_rows = {
        "shard-a": [_row(index) for index in range(0, 220)],
        "shard-b": [_row(index) for index in range(170, 350)],
        "shard-empty": [],
    }
    descriptors = []
    for shard_id, rows in sorted(shard_rows.items()):
        path = artifacts / "shards" / f"{shard_id}.ledger.gz"
        evidence = write_ledger(
            path,
            rows,
            schema=schema,
            context={
                "logical_campaign": "campaign/synthetic-compact/v1",
                "basis_identity": "basis/synthetic-canonical/v1",
                "producer_interface_version": "producer/synthetic-ledger/v1",
                "shard_id": shard_id,
            },
            profile=profile,
        )
        descriptors.append(
            ShardDescriptor(
                shard_id,
                "campaign/synthetic-compact/v1",
                "basis/synthetic-canonical/v1",
                "producer/synthetic-ledger/v1",
                path,
                evidence.compact_sha256,
                profile,
                lifecycle_state="unavailable" if shard_id == "shard-empty" else "sealed",
            )
        )

    merged_path = artifacts / "merged" / "metrics.ledger.gz"
    merged_manifest_path = artifacts / "merged" / "manifest.json"
    merged_evidence, merged_manifest = merge_ledgers(
        tuple(reversed(descriptors)),
        expected_shard_ids=tuple(shard_rows),
        schema=schema,
        output_path=merged_path,
        manifest_path=merged_manifest_path,
        output_profile=profile,
    )

    scratch_root = output_root / "scratch"
    manager = ScratchManager(scratch_root, quota, retention)
    checkpoint_rows = []
    for generation in range(4):
        relative = f"checkpoints/generation-{generation:03d}.bin"
        _write_identical(scratch_root / relative, bytes([generation]) * 300)
        checkpoint_rows.append(
            CheckpointGeneration(
                relative,
                generation,
                "final-checkpoint" if generation == 0 else "resume-checkpoint",
                valid=generation != 3,
                protected=generation == 0,
            )
        )
    _write_identical(scratch_root / "staging/stale.partial", b"partial" * 20)
    with manager.claim():
        warning_record = manager.assess_finalization(staging_bytes=200, manifest_bytes=100)
        try:
            manager.assess_finalization(staging_bytes=5000, manifest_bytes=100)
        except QuotaExceededError as exc:
            quota_failure_record = exc.record
        retention_record = manager.apply_retention(
            checkpoint_rows,
            stale_partials=("staging/stale.partial",),
        )
    reconciled_retention = manager.reconcile_retention_log()

    synthetic_job_id = hashlib.sha256(b"stage12p4-synthetic-job").hexdigest()
    producer = ProducerEvidence(
        synthetic_job_id,
        0,
        hashlib.sha256(b"synthetic-output-contract").hexdigest(),
        hashlib.sha256(b"synthetic-sealed-manifest").hexdigest(),
        "artifacts/synthetic.bin",
    )
    mask_contract = StorageObjectContract(
        "mask-ledger",
        "producer/synthetic-mask/v1",
        producer,
        "basis-mask/synthetic/v1",
        ("component_universe", "basis_identity", "values"),
        mask_evidence.logical_byte_length,
        mask_evidence.logical_sha256,
        "bitpack-msb0/v1",
        profile.reference,
        "chunking/profile-injected/v1",
        mask_evidence.compact_byte_length,
        mask_evidence.compact_sha256,
        quota.reference,
        retention.reference,
        "sealed",
    )
    ledger_contract = StorageObjectContract(
        "merged-ledger",
        "producer/synthetic-ledger/v1",
        producer,
        schema.reference,
        tuple(field.name for field in schema.fields),
        merged_evidence.logical_byte_length,
        merged_evidence.logical_sha256,
        "canonical-jsonl-row-array/v1",
        profile.reference,
        "chunking/profile-injected/v1",
        merged_evidence.compact_byte_length,
        merged_evidence.compact_sha256,
        quota.reference,
        retention.reference,
        "sealed",
    )
    contracts_path = artifacts / "storage-contracts.json"
    _write_identical(
        contracts_path,
        canonical_json_bytes(
            {
                "schema_version": "stage12p4-storage-contract-set/v1",
                "contracts": [mask_contract.to_mapping(), ledger_contract.to_mapping()],
                "scientific_data": False,
                "production_eligible": False,
            }
        ),
    )

    bundle_source = output_root / "bundle-source"
    for source, relative in (
        (mask_path, "masks/synthetic.mask.gz"),
        (merged_path, "merged/metrics.ledger.gz"),
        (merged_manifest_path, "merged/manifest.json"),
        (contracts_path, "storage-contracts.json"),
    ):
        _write_identical(bundle_source / relative, source.read_bytes())
    bundle_root = output_root / "bundle"
    bundle_report = build_bundle(
        bundle_source,
        (
            "masks/synthetic.mask.gz",
            "merged/metrics.ledger.gz",
            "merged/manifest.json",
            "storage-contracts.json",
        ),
        bundle_root=bundle_root,
        bundle_reference="bundle/synthetic-stage12p4/v1",
        profile=profile,
    )
    destination = output_root / "destination"
    transfer_state = output_root / "transfer-state.json"
    adapter = LocalFilesystemExportAdapter(copy_buffer_bytes=257)
    interruption_observed = False
    if not transfer_state.exists():
        try:
            adapter.export(
                bundle_root,
                destination,
                transfer_state_path=transfer_state,
                destination_reference="destination/local-synthetic/v1",
                interrupt_after_bytes=777,
            )
        except TransferInterrupted:
            interruption_observed = True
    destination_report = adapter.export(
        bundle_root,
        destination,
        transfer_state_path=transfer_state,
        destination_reference="destination/local-synthetic/v1",
    )
    repeated_report = verify_destination(
        destination,
        expected_manifest_sha256=bundle_report["manifest_sha256"],
    )

    rejection_results = {}
    first_object = bundle_report["objects"][0]["relative_path"]
    corrupt_root = output_root / "rejections" / "truncated"
    _copy_tree(destination, corrupt_root)
    corrupt_path = corrupt_root.joinpath(*Path(first_object).parts)
    corrupt_path.write_bytes(corrupt_path.read_bytes()[:-1])
    try:
        verify_destination(corrupt_root, expected_manifest_sha256=bundle_report["manifest_sha256"])
    except DestinationConflictError:
        rejection_results["truncated"] = True
    extra_root = output_root / "rejections" / "extra"
    _copy_tree(destination, extra_root)
    _write_identical(extra_root / "objects/extra.bin", b"extra")
    try:
        verify_destination(extra_root, expected_manifest_sha256=bundle_report["manifest_sha256"])
    except DestinationConflictError:
        rejection_results["extra"] = True
    conflict_root = output_root / "rejections" / "conflict"
    conflict_root.mkdir(parents=True, exist_ok=True)
    _write_identical(conflict_root.joinpath(*Path(first_object).parts), b"conflict")
    try:
        adapter.export(
            bundle_root,
            conflict_root,
            transfer_state_path=output_root / "rejections/conflict-state.json",
            destination_reference="destination/conflicting-synthetic/v1",
        )
    except DestinationConflictError:
        rejection_results["conflicting"] = True

    reconstructed_mask = read_mask(
        mask_path,
        profile=profile,
        expected_component_universe="component-universe/synthetic-517/v1",
        expected_basis_identity="basis/synthetic-canonical/v1",
    )
    reconstructed_rows = list(
        iter_ledger_rows(merged_path, profile=profile, expected_schema=schema)
    )
    verbose_mask_bytes = sum(
        len(canonical_json_bytes({"component_index": index, "retained": bool(value)}))
        for index, value in enumerate(mask_values)
    )
    verbose_row_bytes = sum(
        len(canonical_json_bytes(row)) for row in (_row(index) for index in range(350))
    )
    verbose_bytes = verbose_mask_bytes + verbose_row_bytes
    compact_bytes = mask_evidence.compact_byte_length + merged_evidence.compact_byte_length
    size_report = {
        "schema_version": "stage12p4-measured-size-report/v1",
        "fixture_identities": [
            "mask/synthetic-517/v1",
            "ledger/synthetic-350-with-explicit-states/v1",
        ],
        "logical_mask_count": 1,
        "logical_mask_component_count": len(mask_values),
        "logical_row_count": len(reconstructed_rows),
        "verbose_fixture_bytes": verbose_bytes,
        "compact_bytes": compact_bytes,
        "verbose_file_count": len(mask_values) + len(reconstructed_rows),
        "compact_file_count": 2,
        "peak_streaming_buffer_bytes": max(1024 * 1024, adapter.copy_buffer_bytes),
        "compression_ratio_verbose_over_compact": verbose_bytes / compact_bytes,
        "production_footprint_claimed": False,
        "quota_or_codec_frozen": False,
        "scientific_data": False,
        "production_eligible": False,
    }
    report: dict[str, Any] = {
        "schema_version": "stage12p4-validate-only-report/v1",
        "mask_round_trip": reconstructed_mask == mask_values,
        "ledger_round_trip": reconstructed_rows == [_row(index) for index in range(350)],
        "merged_manifest_sha256": merged_manifest["manifest_sha256"],
        "merged_object_sha256": merged_evidence.compact_sha256,
        "duplicate_row_count": merged_manifest["duplicate_row_count"],
        "empty_shard_count": merged_manifest["empty_shard_count"],
        "quota_warning": warning_record,
        "quota_failure": quota_failure_record,
        "retention": retention_record,
        "retention_restart_reconciled": reconciled_retention == retention_record,
        "interruption_observed": interruption_observed,
        "destination_verification": destination_report,
        "repeated_destination_verification_identical": repeated_report == destination_report,
        "rejections": rejection_results,
        "bundle_manifest_sha256": bundle_report["manifest_sha256"],
        "bundle_object_sha256s": [item["sha256"] for item in bundle_report["objects"]],
        "storage_contract_sha256s": [
            mask_contract.contract_sha256,
            ledger_contract.contract_sha256,
        ],
        "size_report": size_report,
        "source_artifacts_retained": all(
            path.is_file()
            for path in (mask_path, merged_path, merged_manifest_path, contracts_path)
        ),
        "rd_014_resolved": False,
        "scientific_data": False,
        "production_eligible": False,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    root = args.output_root.absolute()
    report = run_validate_only(root)
    if args.report is not None:
        report_path = args.report.absolute()
        try:
            report_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("validate-only report must be beneath output root") from exc
        _write_identical(report_path, canonical_json_bytes(report))
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

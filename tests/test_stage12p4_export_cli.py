from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from circuit_families.stage12p3.records import canonical_json_bytes
from circuit_families.stage12p4 import (
    CodecProfile,
    DestinationConflictError,
    LocalFilesystemExportAdapter,
    Stage12P4Error,
    TransferInterrupted,
    build_bundle,
    verify_destination,
)
from circuit_families.stage12p4.cli import run_validate_only


def bundle(tmp_path: Path, name: str = "bundle") -> tuple[Path, dict]:
    source = tmp_path / f"{name}-source"
    (source / "nested").mkdir(parents=True)
    deterministic_bytes = b"".join(
        hashlib.sha256(f"block-{index}".encode("ascii")).digest() for index in range(700)
    )
    (source / "a.bin").write_bytes(deterministic_bytes)
    (source / "nested/b.txt").write_text("technical fixture\n" * 100, encoding="ascii")
    root = tmp_path / name
    profile = CodecProfile("codec/gzip-export-test/v1", "gzip", 6, chunk_bytes=1000)
    report = build_bundle(
        source,
        ("nested/b.txt", "a.bin"),
        bundle_root=root,
        bundle_reference="bundle/export-test/v1",
        profile=profile,
    )
    return root, report


def test_bundle_is_deterministic_across_roots_and_has_no_host_metadata(tmp_path: Path) -> None:
    first_root, first = bundle(tmp_path, "first")
    second_root, second = bundle(tmp_path, "second")
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["objects"] == second["objects"]
    manifest = (first_root / "bundle-manifest.json").read_text(encoding="ascii")
    assert str(tmp_path) not in manifest
    assert "mtime" not in manifest


@pytest.mark.parametrize("boundary", [1, 999, 1000, 1001, 4097])
def test_export_resumes_at_multiple_boundaries_and_retains_source(
    tmp_path: Path, boundary: int
) -> None:
    root, report = bundle(tmp_path)
    destination = tmp_path / "destination"
    state = tmp_path / "state.json"
    adapter = LocalFilesystemExportAdapter(copy_buffer_bytes=137)
    with pytest.raises(TransferInterrupted):
        adapter.export(
            root,
            destination,
            transfer_state_path=state,
            destination_reference="destination/test/v1",
            interrupt_after_bytes=boundary,
        )
    result = adapter.export(
        root,
        destination,
        transfer_state_path=state,
        destination_reference="destination/test/v1",
    )
    assert result["destination_verified"] is True
    assert (
        verify_destination(destination, expected_manifest_sha256=report["manifest_sha256"])
        == result
    )
    assert (root / "bundle-manifest.json").is_file()


def test_export_rejects_corrupt_partial_stale_state_and_conflicting_final(tmp_path: Path) -> None:
    root, report = bundle(tmp_path)
    first = report["objects"][0]
    destination = tmp_path / "destination"
    partial = destination / f"{first['relative_path']}.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"wrong")
    adapter = LocalFilesystemExportAdapter()
    with pytest.raises(DestinationConflictError, match="prefix"):
        adapter.export(
            root,
            destination,
            transfer_state_path=tmp_path / "state.json",
            destination_reference="destination/test/v1",
        )
    stale_state = {
        "schema_version": "stage12p4-transfer-state/v1",
        "bundle_manifest_sha256": "0" * 64,
        "bundle_reference": "bundle/export-test/v1",
        "destination_reference": "destination/test/v1",
        "attempt_count": 0,
        "objects": {},
        "destination_verified": False,
        "scientific_data": False,
        "production_eligible": False,
    }
    stale_path = tmp_path / "stale.json"
    stale_path.write_bytes(canonical_json_bytes(stale_state))
    with pytest.raises(Stage12P4Error, match="stale transfer state"):
        adapter.export(
            root,
            tmp_path / "stale-destination",
            transfer_state_path=stale_path,
            destination_reference="destination/test/v1",
        )
    conflict_destination = tmp_path / "conflict-destination"
    conflict = conflict_destination / first["relative_path"]
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"different")
    with pytest.raises(DestinationConflictError, match="conflicting"):
        adapter.export(
            root,
            conflict_destination,
            transfer_state_path=tmp_path / "conflict-state.json",
            destination_reference="destination/test/v1",
        )


def test_export_detects_mutated_source_extra_truncated_and_read_failure(tmp_path: Path) -> None:
    root, report = bundle(tmp_path)
    first = root / report["objects"][0]["relative_path"]
    first.write_bytes(first.read_bytes() + b"mutation")
    with pytest.raises(Stage12P4Error, match="mutated"):
        LocalFilesystemExportAdapter().export(
            root,
            tmp_path / "mutation-destination",
            transfer_state_path=tmp_path / "mutation-state.json",
            destination_reference="destination/test/v1",
        )

    root, report = bundle(tmp_path, "clean")
    destination = tmp_path / "verified-destination"
    LocalFilesystemExportAdapter().export(
        root,
        destination,
        transfer_state_path=tmp_path / "verified-state.json",
        destination_reference="destination/test/v1",
    )
    (destination / "extra.bin").write_bytes(b"extra")
    with pytest.raises(DestinationConflictError, match="inventory"):
        verify_destination(destination, expected_manifest_sha256=report["manifest_sha256"])
    (destination / "extra.bin").unlink()
    object_path = destination / report["objects"][0]["relative_path"]
    object_path.write_bytes(object_path.read_bytes()[:-1])
    with pytest.raises(DestinationConflictError, match="truncated"):
        verify_destination(destination, expected_manifest_sha256=report["manifest_sha256"])

    def failing_reader(_path: Path) -> bytes:
        raise OSError("forced read failure")

    with pytest.raises(Stage12P4Error, match="read-after-write"):
        verify_destination(
            destination,
            expected_manifest_sha256=report["manifest_sha256"],
            reader=failing_reader,
        )


def test_validate_only_lifecycle_and_measured_size_report(tmp_path: Path) -> None:
    report = run_validate_only(tmp_path / "validation")
    assert report["mask_round_trip"] is True
    assert report["ledger_round_trip"] is True
    assert report["quota_warning"]["warning"] is True
    assert report["quota_failure"]["fits"] is False
    assert report["duplicate_row_count"] == 50
    assert report["empty_shard_count"] == 1
    assert report["interruption_observed"] is True
    assert report["rejections"] == {"truncated": True, "extra": True, "conflicting": True}
    assert report["size_report"]["compact_file_count"] == 2
    assert report["size_report"]["verbose_file_count"] == 867
    assert report["size_report"]["production_footprint_claimed"] is False
    assert report["scientific_data"] is False
    assert report["production_eligible"] is False


def test_cli_is_hash_seed_and_cwd_independent(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    reports = []
    for index, seed in enumerate(("1", "9173")):
        root = tmp_path / f"run-{index}"
        report_path = root / "report.json"
        cwd = repository if index == 0 else tmp_path
        environment = {
            **os.environ,
            "PYTHONHASHSEED": seed,
            "PYTHONPATH": str(repository / "src"),
        }
        subprocess.run(
            [
                sys.executable,
                "-m",
                "circuit_families.stage12p4.cli",
                "--validate-only",
                "--output-root",
                str(root),
                "--report",
                str(report_path),
            ],
            cwd=cwd,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        reports.append(json.loads(report_path.read_bytes()))
    assert reports[0]["report_sha256"] == reports[1]["report_sha256"]
    assert reports[0]["bundle_manifest_sha256"] == reports[1]["bundle_manifest_sha256"]
    assert reports[0]["bundle_object_sha256s"] == reports[1]["bundle_object_sha256s"]

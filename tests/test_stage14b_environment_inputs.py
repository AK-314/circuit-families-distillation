from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from circuit_families.stage14b.inputs import stage_input_bundle, verify_input_root
from circuit_families.stage14b.records import Stage14BError, canonical_sha256, with_boundary


def _manifest(data: bytes) -> dict[str, object]:
    record = with_boundary(
        {
            "schema_version": "stage14b-input-bundle/v1",
            "source_commit": "0" * 40,
            "chunk_rule": {"algorithm": "fixed-bytes/v1", "chunk_bytes": 16},
            "objects": [
                {
                    "relative_path": "src/example.txt",
                    "byte_length": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "role": "committed_code",
                    "provenance": f"git:{'0' * 40}",
                }
            ],
            "object_count": 1,
            "content_bytes": len(data),
            "credential_and_path_bindings": None,
            "registered_or_private_objects_present": False,
        }
    )
    record["bundle_sha256"] = canonical_sha256(record)
    return record


def test_stage_verify_and_reject_corruption_and_extra(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "src").mkdir(parents=True)
    (source / "src/example.txt").write_bytes(b"technical fixture")
    manifest = _manifest(b"technical fixture")
    report = stage_input_bundle(manifest, source, destination)
    assert report["verification"] == "PASS"
    (destination / "src/example.txt").write_bytes(b"corrupt")
    with pytest.raises(Stage14BError, match="corrupt"):
        verify_input_root(manifest, destination, allow_manifest_file=True)
    (destination / "src/example.txt").write_bytes(b"technical fixture")
    (destination / "extra.txt").write_text("extra")
    with pytest.raises(Stage14BError, match="extra"):
        verify_input_root(manifest, destination, allow_manifest_file=True)


def test_reject_symlink_and_unsafe_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    target = source / "target"
    target.write_text("technical fixture")
    (source / "src").mkdir()
    os.symlink(target, source / "src/example.txt")
    with pytest.raises(Stage14BError, match="symlink"):
        stage_input_bundle(_manifest(b"technical fixture"), source, destination)
    manifest = _manifest(b"technical fixture")
    manifest["objects"][0]["relative_path"] = "../escape"
    manifest["bundle_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    )
    with pytest.raises(Stage14BError):
        stage_input_bundle(manifest, source, destination)


def test_container_recipe_requires_injected_images() -> None:
    root = Path(__file__).resolve().parents[1]
    recipe = (root / "containers/stage14b.Containerfile").read_text()
    assert "ARG PYTHON_BASE_IMAGE" in recipe
    assert "ARG UV_BASE_IMAGE" in recipe
    assert "FROM python:" not in recipe
    assert "uv sync --frozen" in recipe

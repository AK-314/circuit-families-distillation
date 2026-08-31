"""Deterministic bundles, resumable local export, and independent verification."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tarfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from circuit_families.stage12p3.records import (
    canonical_json_bytes,
    require_reference,
    safe_relative_path,
)

from .codec import atomic_encode
from .records import CodecProfile, Stage12P4Error, sha256_file

BUNDLE_MANIFEST_VERSION: Final = "stage12p4-bundle-manifest/v1"
TRANSFER_STATE_VERSION: Final = "stage12p4-transfer-state/v1"
DESTINATION_REPORT_VERSION: Final = "stage12p4-destination-verification/v1"


class TransferInterrupted(Stage12P4Error):
    """Controlled interruption that deliberately leaves verified resumable state."""


class DestinationConflictError(Stage12P4Error):
    """Raised instead of silently overwriting different destination content."""


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise DestinationConflictError(f"conflicting object already exists: {path.name}")
        return
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


def _safe_source(root: Path, relative_path: str) -> Path:
    safe_relative_path(relative_path, label="bundle source path")
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise Stage12P4Error("bundle source escapes root") from exc
    current = root
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise Stage12P4Error("bundle source crosses a symlink")
    if not candidate.is_file():
        raise Stage12P4Error(f"bundle source is not a regular file: {relative_path}")
    return candidate


def _read_bundle_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage12P4Error("bundle manifest is missing or invalid") from exc
    required = {
        "schema_version",
        "bundle_reference",
        "codec_profile",
        "archive_encoding",
        "source_inventory",
        "source_file_count",
        "archive_logical_byte_length",
        "archive_logical_sha256",
        "objects",
        "scientific_data",
        "production_eligible",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise Stage12P4Error("bundle manifest fields mismatch")
    if value["schema_version"] != BUNDLE_MANIFEST_VERSION or canonical_json_bytes(value) != raw:
        raise Stage12P4Error("bundle manifest is not canonical")
    if value["scientific_data"] is not False or value["production_eligible"] is not False:
        raise Stage12P4Error("bundle manifest crossed the technical boundary")
    objects = value["objects"]
    if not isinstance(objects, list):
        raise Stage12P4Error("bundle object inventory must be a list")
    names = [item.get("relative_path") for item in objects if isinstance(item, Mapping)]
    if len(names) != len(objects) or len(set(names)) != len(names):
        raise Stage12P4Error("bundle object identities are duplicate or invalid")
    return value, raw


def build_bundle(
    source_root: Path,
    relative_paths: Sequence[str],
    *,
    bundle_root: Path,
    bundle_reference: str,
    profile: CodecProfile,
) -> dict[str, Any]:
    """Build a deterministic USTAR bundle and optional deterministic chunks."""
    source_root = source_root.absolute()
    bundle_root = bundle_root.absolute()
    require_reference(bundle_reference, label="bundle_reference")
    ordered_paths = tuple(sorted(relative_paths))
    if not ordered_paths or len(set(ordered_paths)) != len(ordered_paths):
        raise Stage12P4Error("bundle source paths must be non-empty and unique")
    source_inventory = []
    for relative_path in ordered_paths:
        path = _safe_source(source_root, relative_path)
        size, digest = sha256_file(path)
        source_inventory.append(
            {"relative_path": relative_path, "byte_length": size, "sha256": digest}
        )

    bundle_root.mkdir(parents=True, exist_ok=True)
    tar_path = bundle_root / ".archive.ustar.partial"
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for item in source_inventory:
            source = _safe_source(source_root, item["relative_path"])
            size, digest = sha256_file(source)
            if size != item["byte_length"] or digest != item["sha256"]:
                raise Stage12P4Error("source mutated after bundle planning")
            info = tarfile.TarInfo(name=item["relative_path"])
            info.size = size
            info.mtime = 0
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with source.open("rb") as handle:
                archive.addfile(info, handle)
    archive_size, archive_hash = sha256_file(tar_path)

    encoded_path = bundle_root / ".archive.encoded.partial"
    if encoded_path.exists():
        encoded_path.unlink()

    def tar_chunks():
        with tar_path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                yield block

    atomic_encode(encoded_path, tar_chunks(), profile)
    objects_root = bundle_root / "objects"
    objects_root.mkdir(parents=True, exist_ok=True)
    object_inventory = []
    chunk_bytes = profile.chunk_bytes
    with encoded_path.open("rb") as encoded:
        index = 0
        while True:
            if chunk_bytes is None:
                block = encoded.read()
            else:
                block = encoded.read(chunk_bytes)
            if not block:
                break
            name = "bundle.bin" if chunk_bytes is None else f"chunk-{index:06d}.bin"
            relative = f"objects/{name}"
            target = objects_root / name
            _atomic_bytes(target, block)
            object_inventory.append(
                {
                    "relative_path": relative,
                    "sequence": index,
                    "byte_length": len(block),
                    "sha256": hashlib.sha256(block).hexdigest(),
                }
            )
            index += 1
            if chunk_bytes is None:
                break
    tar_path.unlink()
    encoded_path.unlink()
    manifest = {
        "schema_version": BUNDLE_MANIFEST_VERSION,
        "bundle_reference": bundle_reference,
        "codec_profile": profile.to_mapping(),
        "archive_encoding": "ustar/v1",
        "source_inventory": source_inventory,
        "source_file_count": len(source_inventory),
        "archive_logical_byte_length": archive_size,
        "archive_logical_sha256": archive_hash,
        "objects": object_inventory,
        "scientific_data": False,
        "production_eligible": False,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    _atomic_bytes(bundle_root / "bundle-manifest.json", manifest_bytes)
    return {
        **manifest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "bundle_object_bytes": sum(item["byte_length"] for item in object_inventory),
    }


class LocalFilesystemExportAdapter:
    """Resume verified byte prefixes and publish the destination manifest last."""

    def __init__(self, *, copy_buffer_bytes: int = 64 * 1024) -> None:
        if copy_buffer_bytes <= 0:
            raise Stage12P4Error("copy buffer size must be positive")
        self.copy_buffer_bytes = copy_buffer_bytes

    def export(
        self,
        bundle_root: Path,
        destination_root: Path,
        *,
        transfer_state_path: Path,
        destination_reference: str,
        interrupt_after_bytes: int | None = None,
    ) -> dict[str, Any]:
        require_reference(destination_reference, label="destination_reference")
        manifest, manifest_bytes = _read_bundle_manifest(bundle_root / "bundle-manifest.json")
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        destination_root.mkdir(parents=True, exist_ok=True)
        state = self._load_or_initialize_state(
            transfer_state_path,
            manifest_sha=manifest_sha,
            bundle_reference=manifest["bundle_reference"],
            destination_reference=destination_reference,
        )
        copied_this_call = 0
        for item in manifest["objects"]:
            relative = safe_relative_path(item["relative_path"], label="bundle object path")
            source = bundle_root.joinpath(*PurePosixPath(relative).parts)
            source_size, source_hash = sha256_file(source)
            if source_size != item["byte_length"] or source_hash != item["sha256"]:
                raise Stage12P4Error("planned bundle object mutated before export")
            destination = destination_root.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(f"{destination.name}.partial")
            if destination.exists():
                size, digest = sha256_file(destination)
                if size != item["byte_length"] or digest != item["sha256"]:
                    raise DestinationConflictError("destination object has conflicting content")
                state["objects"][relative] = {
                    "verified_prefix_bytes": size,
                    "published": True,
                }
                self._write_state(transfer_state_path, state)
                continue
            prefix = partial.stat().st_size if partial.exists() else 0
            if prefix > item["byte_length"]:
                raise DestinationConflictError("destination partial exceeds planned object")
            if prefix:
                with source.open("rb") as source_handle, partial.open("rb") as partial_handle:
                    remaining = prefix
                    while remaining:
                        amount = min(self.copy_buffer_bytes, remaining)
                        if source_handle.read(amount) != partial_handle.read(amount):
                            raise DestinationConflictError("destination partial prefix is corrupt")
                        remaining -= amount
            state["objects"][relative] = {
                "verified_prefix_bytes": prefix,
                "published": False,
            }
            self._write_state(transfer_state_path, state)
            with source.open("rb") as source_handle:
                source_handle.seek(prefix)
                with partial.open("ab") as destination_handle:
                    while block := source_handle.read(self.copy_buffer_bytes):
                        if interrupt_after_bytes is not None:
                            remaining = interrupt_after_bytes - copied_this_call
                            if remaining <= 0:
                                destination_handle.flush()
                                os.fsync(destination_handle.fileno())
                                raise TransferInterrupted("forced transfer interruption")
                            if len(block) > remaining:
                                destination_handle.write(block[:remaining])
                                copied_this_call += remaining
                                prefix += remaining
                                destination_handle.flush()
                                os.fsync(destination_handle.fileno())
                                state["objects"][relative]["verified_prefix_bytes"] = prefix
                                self._write_state(transfer_state_path, state)
                                raise TransferInterrupted("forced transfer interruption")
                        destination_handle.write(block)
                        copied_this_call += len(block)
                        prefix += len(block)
                        state["objects"][relative]["verified_prefix_bytes"] = prefix
                        self._write_state(transfer_state_path, state)
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())
            source_size_after, source_hash_after = sha256_file(source)
            if source_size_after != item["byte_length"] or source_hash_after != item["sha256"]:
                raise Stage12P4Error("source mutated during export")
            partial_size, partial_hash = sha256_file(partial)
            if partial_size != item["byte_length"] or partial_hash != item["sha256"]:
                raise DestinationConflictError(
                    "completed destination partial failed hash verification"
                )
            os.replace(partial, destination)
            state["objects"][relative]["published"] = True
            self._write_state(transfer_state_path, state)

        destination_manifest = destination_root / "bundle-manifest.json"
        _atomic_bytes(destination_manifest, manifest_bytes)
        report = verify_destination(
            destination_root,
            expected_manifest_sha256=manifest_sha,
        )
        state["attempt_count"] += 1
        state["destination_verified"] = True
        self._write_state(transfer_state_path, state)
        return report

    @staticmethod
    def _load_or_initialize_state(
        path: Path,
        *,
        manifest_sha: str,
        bundle_reference: str,
        destination_reference: str,
    ) -> dict[str, Any]:
        if path.exists():
            try:
                value = json.loads(path.read_bytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Stage12P4Error("transfer state is invalid") from exc
            if (
                not isinstance(value, dict)
                or value.get("schema_version") != TRANSFER_STATE_VERSION
                or value.get("bundle_manifest_sha256") != manifest_sha
                or value.get("bundle_reference") != bundle_reference
                or value.get("destination_reference") != destination_reference
                or value.get("scientific_data") is not False
                or value.get("production_eligible") is not False
            ):
                raise Stage12P4Error("stale transfer state does not match export plan")
            return value
        return {
            "schema_version": TRANSFER_STATE_VERSION,
            "bundle_manifest_sha256": manifest_sha,
            "bundle_reference": bundle_reference,
            "destination_reference": destination_reference,
            "attempt_count": 0,
            "objects": {},
            "destination_verified": False,
            "scientific_data": False,
            "production_eligible": False,
        }

    @staticmethod
    def _write_state(path: Path, value: Mapping[str, Any]) -> None:
        data = canonical_json_bytes(value)
        temporary = path.with_name(f".{path.name}.partial")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def verify_destination(
    destination_root: Path,
    *,
    expected_manifest_sha256: str,
    reader: Callable[[Path], bytes] | None = None,
) -> dict[str, Any]:
    """Independently reopen every destination object and reject any extra object."""
    read = reader or (lambda path: path.read_bytes())
    manifest_path = destination_root / "bundle-manifest.json"
    try:
        manifest_bytes = read(manifest_path)
    except OSError as exc:
        raise Stage12P4Error("destination manifest read-after-write failed") from exc
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise DestinationConflictError("destination manifest hash mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage12P4Error("destination manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != BUNDLE_MANIFEST_VERSION:
        raise Stage12P4Error("destination manifest schema mismatch")
    expected_paths = {"bundle-manifest.json"}
    verified = []
    for item in manifest.get("objects", []):
        relative = safe_relative_path(item["relative_path"], label="destination object path")
        if relative in expected_paths:
            raise Stage12P4Error("duplicate destination object identity")
        expected_paths.add(relative)
        path = destination_root.joinpath(*PurePosixPath(relative).parts)
        try:
            data = read(path)
        except OSError as exc:
            raise Stage12P4Error("destination object read-after-write failed") from exc
        if len(data) != item["byte_length"]:
            raise DestinationConflictError("destination object is truncated or stale")
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["sha256"]:
            raise DestinationConflictError("destination object hash mismatch")
        verified.append({"relative_path": relative, "byte_length": len(data), "sha256": digest})
    actual_paths = {
        path.relative_to(destination_root).as_posix()
        for path in destination_root.rglob("*")
        if path.is_file()
    }
    extras = sorted(actual_paths - expected_paths)
    missing = sorted(expected_paths - actual_paths)
    if extras or missing:
        raise DestinationConflictError(
            f"destination inventory mismatch: extra={extras!r} missing={missing!r}"
        )
    return {
        "schema_version": DESTINATION_REPORT_VERSION,
        "bundle_manifest_sha256": expected_manifest_sha256,
        "verified_objects": verified,
        "verified_object_count": len(verified),
        "extra_objects": [],
        "missing_objects": [],
        "destination_verified": True,
        "scientific_data": False,
        "production_eligible": False,
    }

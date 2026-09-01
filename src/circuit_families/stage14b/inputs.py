"""Deterministic public/config/code input-bundle planning and verification."""

from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .records import (
    Stage14BError,
    atomic_write,
    canonical_json_bytes,
    canonical_sha256,
    ensure_regular_private_file,
    file_sha256,
    require_boundary,
    require_exact_fields,
    safe_relative_path,
    with_boundary,
)

DEFAULT_INCLUDED_ROOTS = (
    "src/",
    "scripts/",
    "followup/configs/stage13/",
    "followup/configs/stage14b/",
    "followup/decisions/stage13_approval_v1.json",
    "followup/manifests/stage13_",
    "followup/manifests/stage13_optional_tasks/",
    "followup/manifests/stage14b/",
    "followup/schemas/stage13/",
    "followup/fixtures/stage13/",
    "followup/reports/stage13_",
    "docs/distillation_followup/stage13_protocol_manifest_freeze.md",
    "docs/distillation_followup/stage14b_cluster_package_rehearsal.md",
    "containers/stage14b.Containerfile",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
)

FORBIDDEN_INPUT_TOKENS = (
    "checkpoint",
    "registered",
    "private",
    "credential",
    "secret",
    ".env",
    ".ssh",
)


def _git_tracked(root: Path) -> list[str]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
    return sorted(item.decode("utf-8") for item in result.stdout.split(b"\0") if item)


def _included(path: str, roots: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in roots)


def _role(path: str) -> str:
    if path == "uv.lock" or path == ".python-version" or path == "pyproject.toml":
        return "environment"
    if path.startswith("src/") or path.startswith("scripts/"):
        return "committed_code"
    if path.startswith("followup/schemas/"):
        return "schema"
    if path.startswith("followup/manifests/"):
        return "manifest"
    if path.startswith("followup/configs/") or path.startswith("followup/decisions/"):
        return "frozen_config"
    if path.startswith("followup/fixtures/") or path.startswith("followup/reports/"):
        return "excluded_synthetic_fixture"
    if path.startswith("containers/"):
        return "container_recipe"
    return "public_documentation"


def plan_input_bundle(
    repository_root: Path,
    *,
    source_commit: str,
    included_roots: Sequence[str] = DEFAULT_INCLUDED_ROOTS,
    registered_objects: Sequence[Mapping[str, Any]] = (),
    chunk_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Plan exact committed bytes; registered objects must be supplied explicitly."""
    if chunk_bytes <= 0:
        raise Stage14BError("chunk size must be positive")
    root = repository_root.absolute()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if source_commit != head:
        raise Stage14BError("input bundle source commit is stale")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if dirty:
        raise Stage14BError("input bundle requires a clean tracked checkout")
    tracked = _git_tracked(root)
    paths = [path for path in tracked if _included(path, included_roots)]
    if not paths:
        raise Stage14BError("input selection is empty")
    objects = []
    for relative in paths:
        source = ensure_regular_private_file(root, relative, label="bundle source")
        size, digest = file_sha256(source)
        objects.append(
            {
                "relative_path": relative,
                "byte_length": size,
                "sha256": digest,
                "role": _role(relative),
                "provenance": f"git:{source_commit}",
            }
        )
    for supplied in registered_objects:
        required = {"relative_path", "byte_length", "sha256", "role", "provenance"}
        if set(supplied) != required:
            raise Stage14BError("explicit registered input fields mismatch")
        relative = str(supplied["relative_path"])
        safe_relative_path(relative, label="registered input path")
        if not relative.startswith("registered-inputs/"):
            raise Stage14BError("registered inputs require the isolated registered-inputs root")
        objects.append(dict(supplied))
    objects.sort(key=lambda item: item["relative_path"])
    names = [item["relative_path"] for item in objects]
    if names != sorted(names) or len(names) != len(set(names)):
        raise Stage14BError("bundle paths must be sorted and unique")
    manifest = with_boundary(
        {
            "schema_version": "stage14b-input-bundle/v1",
            "source_commit": source_commit,
            "chunk_rule": {"algorithm": "fixed-bytes/v1", "chunk_bytes": chunk_bytes},
            "objects": objects,
            "object_count": len(objects),
            "content_bytes": sum(int(item["byte_length"]) for item in objects),
            "credential_and_path_bindings": None,
            "registered_or_private_objects_present": bool(registered_objects),
        }
    )
    manifest["bundle_sha256"] = canonical_sha256(manifest)
    return manifest


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    require_exact_fields(
        manifest,
        {
            "schema_version",
            "source_commit",
            "chunk_rule",
            "objects",
            "object_count",
            "content_bytes",
            "credential_and_path_bindings",
            "registered_or_private_objects_present",
            "scientific_data",
            "production_eligible",
            "definitive_execution_started",
            "bundle_sha256",
        },
        label="input manifest",
    )
    if manifest["schema_version"] != "stage14b-input-bundle/v1":
        raise Stage14BError("unsupported input manifest schema")
    require_boundary(manifest)
    expected = manifest.get("bundle_sha256")
    payload = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    if expected != canonical_sha256(payload):
        raise Stage14BError("input manifest integrity hash mismatch")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or manifest.get("object_count") != len(objects):
        raise Stage14BError("input manifest object count mismatch")
    names = [item.get("relative_path") for item in objects if isinstance(item, Mapping)]
    if len(names) != len(objects) or names != sorted(names) or len(names) != len(set(names)):
        raise Stage14BError("input manifest has invalid, unsorted, or duplicate paths")
    for item in objects:
        require_exact_fields(
            item,
            {"relative_path", "byte_length", "sha256", "role", "provenance"},
            label="input object",
        )
        safe_relative_path(item["relative_path"], label="input object path")
        if (
            isinstance(item["byte_length"], bool)
            or not isinstance(item["byte_length"], int)
            or item["byte_length"] < 0
        ):
            raise Stage14BError("input object byte length is invalid")
        digest = item["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise Stage14BError("input object SHA-256 is invalid")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise Stage14BError("input object SHA-256 is invalid") from exc
    if manifest["credential_and_path_bindings"] is not None:
        raise Stage14BError("credential/path bindings must remain outside the content manifest")


def verify_input_root(
    manifest: Mapping[str, Any], root: Path, *, allow_manifest_file: bool = False
) -> dict[str, Any]:
    _validate_manifest(manifest)
    root = root.absolute()
    expected = {item["relative_path"]: item for item in manifest["objects"]}
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Stage14BError("input root contains a symlink")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if allow_manifest_file and relative == "input-manifest.json":
                continue
            actual.add(relative)
    extras = sorted(actual - set(expected))
    missing = sorted(set(expected) - actual)
    if extras or missing:
        raise Stage14BError(f"input inventory mismatch; missing={missing}, extra={extras}")
    for relative, item in expected.items():
        path = ensure_regular_private_file(root, relative, label="staged input")
        size, digest = file_sha256(path)
        if size != item["byte_length"] or digest != item["sha256"]:
            raise Stage14BError(f"stale or corrupt staged input: {relative}")
    return with_boundary(
        {
            "schema_version": "stage14b-input-verification/v1",
            "bundle_sha256": manifest["bundle_sha256"],
            "object_count": len(expected),
            "content_bytes": sum(item["byte_length"] for item in expected.values()),
            "verification": "PASS",
        }
    )


def stage_input_bundle(
    manifest: Mapping[str, Any], source_root: Path, destination_root: Path
) -> dict[str, Any]:
    _validate_manifest(manifest)
    source_root = source_root.absolute()
    destination_root = destination_root.absolute()
    if destination_root.exists() and destination_root.is_symlink():
        raise Stage14BError("destination root must not be a symlink")
    destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination_root.stat().st_mode & stat.S_IWOTH:
        raise Stage14BError("destination root must not be world-writable")
    for item in manifest["objects"]:
        relative = item["relative_path"]
        if relative.startswith("registered-inputs/"):
            source = ensure_regular_private_file(source_root, relative, label="registered source")
        else:
            source = ensure_regular_private_file(source_root, relative, label="committed source")
        before = file_sha256(source)
        if before != (item["byte_length"], item["sha256"]):
            raise Stage14BError(f"source drift before staging: {relative}")
        target = destination_root.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with source.open("rb") as handle:
            atomic_write(target, handle.read())
        if file_sha256(source) != before:
            raise Stage14BError(f"source mutated during staging: {relative}")
    atomic_write(destination_root / "input-manifest.json", canonical_json_bytes(manifest))
    return verify_input_root(manifest, destination_root, allow_manifest_file=True)


def load_input_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage14BError("input manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise Stage14BError("input manifest must contain one object")
    _validate_manifest(value)
    return value


def verify_transfer_chunks(paths: Iterable[Path], expected: Sequence[Mapping[str, Any]]) -> None:
    actual = list(paths)
    if len(actual) != len(expected):
        raise Stage14BError("transfer interruption or chunk count mismatch")
    for path, item in zip(actual, expected, strict=True):
        if file_sha256(path) != (item["byte_length"], item["sha256"]):
            raise Stage14BError("transfer chunk verification failed")

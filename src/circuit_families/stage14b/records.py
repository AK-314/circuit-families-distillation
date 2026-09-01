"""Closed technical records shared by the Stage 14-B execution package."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


class Stage14BError(ValueError):
    """Raised when a Stage 14-B technical contract is violated."""


BOUNDARY_FIELDS = {
    "scientific_data": False,
    "production_eligible": False,
    "definitive_execution_started": False,
}


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Stage14BError("value is not finite canonical JSON") from exc
    return (text + "\n").encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def with_boundary(value: Mapping[str, Any]) -> dict[str, Any]:
    overlap = set(value) & set(BOUNDARY_FIELDS)
    if any(value[key] is not False for key in overlap):
        raise Stage14BError("record crossed the non-scientific boundary")
    return {**value, **BOUNDARY_FIELDS}


def require_boundary(value: Mapping[str, Any]) -> None:
    for key, expected in BOUNDARY_FIELDS.items():
        if value.get(key) is not expected:
            raise Stage14BError(f"record violates {key}=false")


def require_exact_fields(value: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    if set(value) != fields:
        raise Stage14BError(
            f"{label} fields mismatch; missing={sorted(fields - set(value))}, "
            f"unknown={sorted(set(value) - fields)}"
        )


def safe_relative_path(value: str, *, label: str = "path") -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise Stage14BError(f"{label} must be a non-empty relative POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise Stage14BError(f"{label} escapes its portable root")
    return parsed


def ensure_regular_private_file(root: Path, relative: str, *, label: str) -> Path:
    parsed = safe_relative_path(relative, label=label)
    root = root.absolute()
    candidate = root.joinpath(*parsed.parts)
    current = root
    if current.is_symlink():
        raise Stage14BError(f"{label} root must not be a symlink")
    for part in parsed.parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise Stage14BError(f"{label} crosses a symlink")
    if not candidate.is_file():
        raise Stage14BError(f"{label} is missing or not a regular file")
    mode = candidate.stat().st_mode
    if mode & stat.S_IWOTH:
        raise Stage14BError(f"{label} must not be world-writable")
    return candidate


def atomic_write(path: Path, data: bytes, *, identical_ok: bool = True) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if identical_ok and path.is_file() and path.read_bytes() == data:
            return
        raise Stage14BError(f"refusing to overwrite conflicting object: {path.name}")
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

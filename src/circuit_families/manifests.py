"""Creation of small JSON provenance manifests."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

DEFAULT_PACKAGES = ("numpy", "PyYAML", "torch")


def utc_timestamp() -> str:
    """Return the current UTC time in ISO 8601 format."""

    return datetime.now(UTC).isoformat()


def git_commit(repository_root: str | Path = ".") -> str:
    """Return the current Git commit, or 'unknown' when unavailable."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

    commit = completed.stdout.strip()
    return commit or "unknown"


def package_versions(
    packages: Sequence[str] = DEFAULT_PACKAGES,
) -> dict[str, str]:
    """Return installed versions for the requested packages."""

    versions: dict[str, str] = {}

    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"

    return versions


def device_information() -> dict[str, Any]:
    """Return basic PyTorch and hardware-device information."""

    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        selected_device = "cuda"
    elif mps_available:
        selected_device = "mps"
    else:
        selected_device = "cpu"

    cuda_devices = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]

    return {
        "selected_device": selected_device,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": cuda_devices,
        "mps_available": mps_available,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def create_manifest(
    *,
    run_id: str,
    experiment_type: str,
    repository_root: str | Path,
    config_path: str | Path,
    config_sha256: str,
    seed_name: str,
    seed: int,
    output_paths: Mapping[str, str | Path],
    hashes: Mapping[str, str] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-serialisable provenance manifest."""

    if not run_id:
        raise ValueError("run_id must be non-empty.")

    if not experiment_type:
        raise ValueError("experiment_type must be non-empty.")

    if not seed_name:
        raise ValueError("seed_name must be non-empty.")

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": utc_timestamp(),
        "run_id": run_id,
        "experiment_type": experiment_type,
        "git_commit": git_commit(repository_root),
        "config": {
            "path": str(Path(config_path)),
            "sha256": config_sha256,
        },
        "software": {
            "python": sys.version,
            "packages": package_versions(),
        },
        "device": device_information(),
        "seed": {
            "name": seed_name,
            "value": seed,
        },
        "output_paths": {
            name: str(Path(path))
            for name, path in sorted(output_paths.items())
        },
        "hashes": dict(sorted((hashes or {}).items())),
        "details": dict(details or {}),
    }

    json.dumps(manifest, allow_nan=False)
    return manifest


def write_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Write a manifest as stable, human-readable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialised = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    output_path.write_text(serialised + "\n", encoding="utf-8")

    return output_path

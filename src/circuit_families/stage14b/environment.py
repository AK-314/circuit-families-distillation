"""Locked environment and container identity capture and verification."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

from .records import (
    Stage14BError,
    canonical_sha256,
    file_sha256,
    require_boundary,
    with_boundary,
)

DETERMINISTIC_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "injected_integer",
    "TOKENIZERS_PARALLELISM": "false",
}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def resolved_package_inventory() -> list[dict[str, str]]:
    return [
        {"name": dist.metadata["Name"].lower(), "version": dist.version}
        for dist in sorted(
            importlib.metadata.distributions(),
            key=lambda item: (item.metadata["Name"] or "").lower(),
        )
        if dist.metadata["Name"]
    ]


def _torch_identity() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "cuda_runtime": None,
            "cuda_available": False,
            "mps_built": False,
            "mps_available": False,
            "build_config": None,
        }
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "build_config": torch.__config__.show(),
    }


def capture_environment(
    repository_root: Path,
    recipe_path: Path,
    *,
    image_id: str | None = None,
    immutable_digest: str | None = None,
    runtime: str | None = None,
) -> dict[str, Any]:
    root = repository_root.absolute()
    tracked_changes = _git(root, "status", "--porcelain", "--untracked-files=no")
    if tracked_changes:
        raise Stage14BError("environment evidence requires a clean tracked checkout")
    lock_size, lock_hash = file_sha256(root / "uv.lock")
    recipe_size, recipe_hash = file_sha256(recipe_path)
    packages = resolved_package_inventory()
    record = with_boundary(
        {
            "schema_version": "stage14b-environment/v1",
            "source": {
                "repository_commit": _git(root, "rev-parse", "HEAD"),
                "tracked_dirty": False,
            },
            "python": {
                "required": (root / ".python-version").read_text().strip(),
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "abi": sysconfig.get_config_var("SOABI"),
                "executable_name": Path(sys.executable).name,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "lock": {
                "path": "uv.lock",
                "byte_length": lock_size,
                "sha256": lock_hash,
                "install_mode": "uv sync --frozen",
            },
            "packages": packages,
            "packages_sha256": canonical_sha256(packages),
            "torch": _torch_identity(),
            "deterministic_environment": DETERMINISTIC_ENVIRONMENT,
            "container": {
                "recipe_path": recipe_path.relative_to(root).as_posix(),
                "recipe_byte_length": recipe_size,
                "recipe_sha256": recipe_hash,
                "build_command": (
                    "docker build --build-arg PYTHON_BASE_IMAGE=<name@sha256:digest> "
                    "--build-arg UV_BASE_IMAGE=<name@sha256:digest> "
                    "-f containers/stage14b.Containerfile ."
                ),
                "runtime": runtime,
                "image_id": image_id,
                "immutable_digest": immutable_digest,
                "build_provenance": None,
                "network_sources": ["locked Python indexes resolved by uv.lock"],
                "waiting_for": (
                    None
                    if image_id and immutable_digest and runtime
                    else "AUTHORIZED_COMPATIBLE_CONTAINER_BUILD_AND_IMMUTABLE_DIGEST"
                ),
            },
            "compatibility": {
                "cpu": "requires same lock, ABI, platform class, and passed qualification",
                "cuda": "separate driver/runtime/device-class qualification required",
                "mps": "separate macOS/device-class qualification required",
            },
            "nonportable_fields": [
                "container image digest until built",
                "accelerator driver/runtime",
                "backend numerical qualification",
            ],
        }
    )
    record["environment_sha256"] = canonical_sha256(record)
    return record


def verify_environment(
    record: dict[str, Any], repository_root: Path, recipe_path: Path
) -> dict[str, Any]:
    require_boundary(record)
    expected_hash = record.get("environment_sha256")
    payload = {key: value for key, value in record.items() if key != "environment_sha256"}
    if expected_hash != canonical_sha256(payload):
        raise Stage14BError("environment record integrity hash mismatch")
    root = repository_root.absolute()
    if record["source"]["repository_commit"] != _git(root, "rev-parse", "HEAD"):
        raise Stage14BError("environment source commit drift")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise Stage14BError("dirty tracked checkout cannot verify environment")
    if record["lock"]["sha256"] != file_sha256(root / "uv.lock")[1]:
        raise Stage14BError("lock drift")
    if record["container"]["recipe_sha256"] != file_sha256(recipe_path)[1]:
        raise Stage14BError("container recipe drift")
    if record["python"]["version"] != platform.python_version():
        raise Stage14BError("interpreter version drift")
    if record["python"]["abi"] != sysconfig.get_config_var("SOABI"):
        raise Stage14BError("interpreter ABI drift")
    if record["packages_sha256"] != canonical_sha256(resolved_package_inventory()):
        raise Stage14BError("resolved package drift")
    return with_boundary(
        {
            "schema_version": "stage14b-environment-verification/v1",
            "environment_sha256": expected_hash,
            "verification": "PASS",
        }
    )


def write_environment_record(path: Path, record: dict[str, Any]) -> None:
    from .records import atomic_write, canonical_json_bytes

    atomic_write(path, canonical_json_bytes(record))


def load_environment_record(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage14BError("environment record is unreadable") from exc
    if not isinstance(value, dict):
        raise Stage14BError("environment record must contain one object")
    return value

"""Checkpoint serialization, restoration, hashing, and validation."""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from circuit_families.config import (
    validate_model_config,
    validate_training_config,
)
from circuit_families.manifests import git_commit, package_versions
from circuit_families.training.device import device_record
from circuit_families.training.metrics import evaluate_model

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_PACKAGES = (
    "numpy",
    "PyYAML",
    "torch",
    "transformer-lens",
)


@dataclass(frozen=True)
class SavedCheckpoint:
    """Integrity identifiers for a newly written checkpoint."""

    path: Path
    file_sha256: str
    model_state_sha256: str
    optimizer_state_sha256: str


@dataclass(frozen=True)
class ResumeState:
    """State recovered from a validated checkpoint."""

    training_step: int
    model_seed: int
    metrics: dict[str, Any]
    dataset_hashes: dict[str, str]
    checkpoint_path: Path
    checkpoint_sha256: str
    model_state_sha256: str
    optimizer_state_sha256: str


def _validate_step(step: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("training step must be an integer.")

    if step < 0:
        raise ValueError("training step must be non-negative.")

    return step


def _validate_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{name} must be a lowercase SHA-256 hex digest."
        )

    return value


def checkpoint_filename(step: int) -> str:
    """Return the frozen filename for one training checkpoint."""

    return f"step_{_validate_step(step):08d}.pt"


def checkpoint_path(
    checkpoint_directory: str | Path,
    step: int,
) -> Path:
    """Return the exact path for one training checkpoint."""

    return Path(checkpoint_directory) / checkpoint_filename(step)


def file_sha256(path: str | Path) -> str:
    """Return the physical SHA-256 hash of a file."""

    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _length_prefixed(value: bytes) -> bytes:
    return len(value).to_bytes(8, byteorder="big") + value


def _canonical_key_bytes(value: Any) -> bytes:
    if value is None:
        return b"N"

    if isinstance(value, bool):
        return b"B1" if value else b"B0"

    if isinstance(value, int):
        return b"I" + str(value).encode("ascii")

    if isinstance(value, float):
        return b"F" + struct.pack(">d", value)

    if isinstance(value, str):
        return b"S" + value.encode("utf-8")

    raise TypeError(
        "Canonical mapping keys must be None, bool, int, float, or str."
    )


def _update_canonical_hash(
    digest: Any,
    value: Any,
) -> None:
    if value is None:
        digest.update(b"N")
        return

    if isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
        return

    if isinstance(value, int):
        encoded = str(value).encode("ascii")
        digest.update(b"I")
        digest.update(_length_prefixed(encoded))
        return

    if isinstance(value, float):
        digest.update(b"F")
        digest.update(struct.pack(">d", value))
        return

    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S")
        digest.update(_length_prefixed(encoded))
        return

    if isinstance(value, bytes):
        digest.update(b"Y")
        digest.update(_length_prefixed(value))
        return

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        dtype = str(tensor.dtype).encode("ascii")
        shape = ",".join(
            str(dimension)
            for dimension in tensor.shape
        ).encode("ascii")
        raw_bytes = tensor.numpy().tobytes(order="C")

        digest.update(b"R")
        digest.update(_length_prefixed(dtype))
        digest.update(_length_prefixed(shape))
        digest.update(_length_prefixed(raw_bytes))
        return

    if isinstance(value, Mapping):
        digest.update(b"M")
        digest.update(len(value).to_bytes(8, byteorder="big"))

        items = sorted(
            value.items(),
            key=lambda item: _canonical_key_bytes(item[0]),
        )

        for key, item_value in items:
            encoded_key = _canonical_key_bytes(key)
            digest.update(_length_prefixed(encoded_key))
            _update_canonical_hash(digest, item_value)
        return

    if isinstance(value, list):
        digest.update(b"L")
        digest.update(len(value).to_bytes(8, byteorder="big"))
        for item in value:
            _update_canonical_hash(digest, item)
        return

    if isinstance(value, tuple):
        digest.update(b"T")
        digest.update(len(value).to_bytes(8, byteorder="big"))
        for item in value:
            _update_canonical_hash(digest, item)
        return

    raise TypeError(
        "Unsupported value in canonical state hashing: "
        f"{type(value).__name__}."
    )


def canonical_state_hash(state: Any) -> str:
    """Hash nested model or optimizer state independently of device."""

    digest = hashlib.sha256()
    _update_canonical_hash(digest, state)
    return digest.hexdigest()


def _validated_dataset_hashes(
    dataset_hashes: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(dataset_hashes, Mapping):
        raise TypeError("dataset_hashes must be a mapping.")

    required = {"dataset_sha256", "split_sha256"}

    if not required.issubset(dataset_hashes):
        missing = sorted(required.difference(dataset_hashes))
        raise ValueError(
            "dataset_hashes is missing required values: "
            + ", ".join(missing)
        )

    return {
        name: _validate_sha256(value, f"dataset_hashes.{name}")
        for name, value in sorted(dataset_hashes.items())
    }


def save_checkpoint(
    checkpoint_directory: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    model_seed: int,
    dataset_hashes: Mapping[str, str],
    metrics: Mapping[str, Any],
    repository_root: str | Path,
    device: torch.device,
) -> SavedCheckpoint:
    """Write a complete checkpoint and return its integrity hashes."""

    step = _validate_step(step)
    validate_model_config(model_config)
    validate_training_config(training_config)

    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise TypeError("model_seed must be an integer.")

    if model_seed < 0:
        raise ValueError("model_seed must be non-negative.")

    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping.")

    validated_hashes = _validated_dataset_hashes(dataset_hashes)

    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    model_state_sha256 = canonical_state_hash(model_state)
    optimizer_state_sha256 = canonical_state_hash(optimizer_state)

    output_path = checkpoint_path(checkpoint_directory, step)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "training_step": step,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "model_config": copy.deepcopy(dict(model_config)),
        "training_config": copy.deepcopy(dict(training_config)),
        "model_seed": model_seed,
        "dataset_hashes": validated_hashes,
        "git_commit": git_commit(repository_root),
        "metrics": copy.deepcopy(dict(metrics)),
        "package_versions": package_versions(CHECKPOINT_PACKAGES),
        "device": device_record(device),
        "model_state_sha256": model_state_sha256,
        "optimizer_state_sha256": optimizer_state_sha256,
    }

    torch.save(payload, output_path)

    return SavedCheckpoint(
        path=output_path,
        file_sha256=file_sha256(output_path),
        model_state_sha256=model_state_sha256,
        optimizer_state_sha256=optimizer_state_sha256,
    )


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a checkpoint payload without restoring objects."""

    checkpoint_file = Path(path)

    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_file}"
        )

    payload = torch.load(
        checkpoint_file,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")

    required_fields = {
        "schema_version",
        "training_step",
        "model_state",
        "optimizer_state",
        "model_config",
        "training_config",
        "model_seed",
        "dataset_hashes",
        "git_commit",
        "metrics",
        "package_versions",
        "device",
        "model_state_sha256",
        "optimizer_state_sha256",
    }

    missing = sorted(required_fields.difference(payload))

    if missing:
        raise ValueError(
            "Checkpoint is missing required fields: "
            + ", ".join(missing)
        )

    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported checkpoint schema version: "
            f"{payload['schema_version']!r}."
        )

    _validate_step(payload["training_step"])
    validate_model_config(payload["model_config"])
    validate_training_config(payload["training_config"])
    _validated_dataset_hashes(payload["dataset_hashes"])

    expected_model_hash = _validate_sha256(
        payload["model_state_sha256"],
        "model_state_sha256",
    )
    expected_optimizer_hash = _validate_sha256(
        payload["optimizer_state_sha256"],
        "optimizer_state_sha256",
    )

    actual_model_hash = canonical_state_hash(payload["model_state"])
    actual_optimizer_hash = canonical_state_hash(
        payload["optimizer_state"]
    )

    if actual_model_hash != expected_model_hash:
        raise ValueError("Checkpoint model-state hash does not match.")

    if actual_optimizer_hash != expected_optimizer_hash:
        raise ValueError(
            "Checkpoint optimizer-state hash does not match."
        )

    return payload


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device=device)


def restore_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> ResumeState:
    """Restore model and optimizer state from a validated checkpoint."""

    checkpoint_file = Path(path)
    payload = load_checkpoint_payload(
        checkpoint_file,
        map_location=device,
    )

    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    _move_optimizer_state(optimizer, device)

    return ResumeState(
        training_step=payload["training_step"],
        model_seed=payload["model_seed"],
        metrics=copy.deepcopy(payload["metrics"]),
        dataset_hashes=copy.deepcopy(payload["dataset_hashes"]),
        checkpoint_path=checkpoint_file,
        checkpoint_sha256=file_sha256(checkpoint_file),
        model_state_sha256=payload["model_state_sha256"],
        optimizer_state_sha256=payload["optimizer_state_sha256"],
    )


def reload_and_reevaluate(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Restore a checkpoint and verify its recorded evaluation metrics."""

    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("Metric tolerances must be non-negative.")

    resume_state = restore_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    reevaluated = {
        "train": evaluate_model(
            model,
            train_inputs,
            train_targets,
        ),
        "test": evaluate_model(
            model,
            test_inputs,
            test_targets,
        ),
    }

    for split_name in ("train", "test"):
        recorded_split = resume_state.metrics.get(split_name)

        if not isinstance(recorded_split, Mapping):
            raise ValueError(
                f"Recorded metrics are missing the {split_name} mapping."
            )

        for metric_name in ("cross_entropy", "accuracy"):
            if metric_name not in recorded_split:
                raise ValueError(
                    "Recorded metrics are missing "
                    f"{split_name}.{metric_name}."
                )

            recorded_value = float(recorded_split[metric_name])
            reevaluated_value = reevaluated[split_name][metric_name]

            if not math.isclose(
                recorded_value,
                reevaluated_value,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ):
                raise ValueError(
                    "Reloaded checkpoint metric mismatch for "
                    f"{split_name}.{metric_name}: recorded "
                    f"{recorded_value}, reevaluated "
                    f"{reevaluated_value}."
                )

    return {
        "resume_state": resume_state,
        "train": reevaluated["train"],
        "test": reevaluated["test"],
    }

"""Atomic mechanics-only checkpoint save/restore for Stage 5B.

The checkpoint binds one exact Stage 4-backed student attempt identity to
explicit configuration and target hashes. It contains the continuation state
required by the shared Part K/L technical loop and has no production authority.
"""

from __future__ import annotations

import copy
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from circuit_families.stage4_condition_identity import Stage3AvailabilityIndex
from circuit_families.stage5bc.student_identity import (
    StudentAttemptIdentity,
    verify_student_attempt_identity,
)
from circuit_families.stage5bc.student_trainer import (
    PreparedTrainer,
    TechnicalLoopSnapshot,
    TrainerProgress,
)
from circuit_families.training.checkpoints import (
    canonical_state_hash,
    file_sha256,
)

TECHNICAL_CHECKPOINT_SCHEMA_VERSION = "stage5bc-technical-checkpoint/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "scientific_data",
        "production_eligible",
        "attempt_identity",
        "configuration_hashes",
        "target_cache_manifest_sha256",
        "snapshot",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "torch_rng_state",
        "state_hashes",
    }
)


class TechnicalCheckpointError(ValueError):
    """Raised when a technical checkpoint is corrupt, stale, or misbound."""


@dataclass(frozen=True)
class TechnicalCheckpointEvidence:
    """Integrity evidence for one atomically published checkpoint."""

    path: Path
    file_sha256: str
    model_state_sha256: str
    optimizer_state_sha256: str
    scheduler_state_sha256: str
    torch_rng_state_sha256: str
    updates_completed: int


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TechnicalCheckpointError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def _validated_hash_mapping(
    value: Mapping[str, str],
    *,
    name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise TechnicalCheckpointError(
            f"{name} must be a non-empty mapping"
        )

    result = {}

    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TechnicalCheckpointError(
                f"{name} keys must be non-empty strings"
            )
        result[key] = _require_sha256(
            item,
            name=f"{name}.{key}",
        )

    return result


def _snapshot_to_mapping(
    snapshot: TechnicalLoopSnapshot,
) -> dict[str, Any]:
    return {
        "updates_completed": snapshot.updates_completed,
        "outer_training_mode": snapshot.outer_training_mode,
        "trajectory": [
            {
                "step": item.step,
                "updates_completed": item.updates_completed,
                "metrics": copy.deepcopy(dict(item.metrics)),
            }
            for item in snapshot.trajectory
        ],
    }


def _snapshot_from_mapping(
    value: Mapping[str, Any],
) -> TechnicalLoopSnapshot:
    if not isinstance(value, Mapping):
        raise TechnicalCheckpointError(
            "checkpoint snapshot must be a mapping"
        )

    if set(value) != {
        "updates_completed",
        "outer_training_mode",
        "trajectory",
    }:
        raise TechnicalCheckpointError(
            "checkpoint snapshot keys mismatch"
        )

    raw_trajectory = value["trajectory"]
    if not isinstance(raw_trajectory, list):
        raise TechnicalCheckpointError(
            "checkpoint trajectory must be a list"
        )

    trajectory = []

    for record in raw_trajectory:
        if not isinstance(record, Mapping):
            raise TechnicalCheckpointError(
                "checkpoint trajectory entries must be mappings"
            )

        if set(record) != {
            "step",
            "updates_completed",
            "metrics",
        }:
            raise TechnicalCheckpointError(
                "checkpoint trajectory entry keys mismatch"
            )

        metrics = record["metrics"]
        if not isinstance(metrics, Mapping):
            raise TechnicalCheckpointError(
                "checkpoint trajectory metrics must be a mapping"
            )

        trajectory.append(
            TrainerProgress(
                step=record["step"],
                updates_completed=record["updates_completed"],
                metrics=copy.deepcopy(dict(metrics)),
            )
        )

    try:
        return TechnicalLoopSnapshot(
            updates_completed=value["updates_completed"],
            trajectory=tuple(trajectory),
            outer_training_mode=value["outer_training_mode"],
        )
    except ValueError as exc:
        raise TechnicalCheckpointError(
            f"invalid checkpoint snapshot: {exc}"
        ) from exc


def _scheduler_state(
    prepared: PreparedTrainer,
) -> Any:
    scheduler = prepared.optimizer_schedule.scheduler

    if scheduler is None:
        return None

    state_method = getattr(scheduler, "state_dict", None)
    if state_method is None or not callable(state_method):
        raise TechnicalCheckpointError(
            "injected scheduler must expose state_dict() for checkpointing"
        )

    return copy.deepcopy(state_method())


def _atomic_torch_save(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_name = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_name, path)
        temporary_name = None

        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            descriptor = None

        if descriptor is not None:
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def save_technical_resume_checkpoint(
    path: str | Path,
    *,
    prepared: PreparedTrainer,
    snapshot: TechnicalLoopSnapshot,
    attempt_identity: StudentAttemptIdentity,
    stage3: Stage3AvailabilityIndex,
    configuration_hashes: Mapping[str, str],
    target_cache_manifest_sha256: str,
) -> TechnicalCheckpointEvidence:
    """Atomically save exact continuation state for one technical attempt."""
    if not isinstance(prepared, PreparedTrainer):
        raise TechnicalCheckpointError(
            "prepared must be PreparedTrainer"
        )

    if not isinstance(snapshot, TechnicalLoopSnapshot):
        raise TechnicalCheckpointError(
            "snapshot must be TechnicalLoopSnapshot"
        )

    verify_student_attempt_identity(
        attempt_identity,
        stage3,
    )

    config_hashes = _validated_hash_mapping(
        configuration_hashes,
        name="configuration_hashes",
    )
    target_hash = _require_sha256(
        target_cache_manifest_sha256,
        name="target_cache_manifest_sha256",
    )

    model_state = copy.deepcopy(prepared.model.state_dict())
    optimizer_state = copy.deepcopy(
        prepared.optimizer_schedule.optimizer.state_dict()
    )
    scheduler_state = _scheduler_state(prepared)
    torch_rng_state = torch.get_rng_state().clone()

    state_hashes = {
        "model_state_sha256": canonical_state_hash(model_state),
        "optimizer_state_sha256": canonical_state_hash(optimizer_state),
        "scheduler_state_sha256": canonical_state_hash(scheduler_state),
        "torch_rng_state_sha256": canonical_state_hash(torch_rng_state),
    }

    payload = {
        "schema_version": TECHNICAL_CHECKPOINT_SCHEMA_VERSION,
        "scientific_data": False,
        "production_eligible": False,
        "attempt_identity": attempt_identity.to_mapping(),
        "configuration_hashes": config_hashes,
        "target_cache_manifest_sha256": target_hash,
        "snapshot": _snapshot_to_mapping(snapshot),
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "torch_rng_state": torch_rng_state,
        "state_hashes": state_hashes,
    }

    checkpoint_path = Path(path)

    if checkpoint_path.exists():
        raise TechnicalCheckpointError(
            f"technical checkpoint already exists: {checkpoint_path}"
        )

    _atomic_torch_save(checkpoint_path, payload)

    return TechnicalCheckpointEvidence(
        path=checkpoint_path,
        file_sha256=file_sha256(checkpoint_path),
        model_state_sha256=state_hashes["model_state_sha256"],
        optimizer_state_sha256=state_hashes["optimizer_state_sha256"],
        scheduler_state_sha256=state_hashes["scheduler_state_sha256"],
        torch_rng_state_sha256=state_hashes["torch_rng_state_sha256"],
        updates_completed=snapshot.updates_completed,
    )


def _move_optimizer_state_to_model_device(
    prepared: PreparedTrainer,
) -> None:
    devices = {
        parameter.device
        for parameter in prepared.model.parameters()
    }

    if len(devices) > 1:
        raise TechnicalCheckpointError(
            "model spans multiple devices during checkpoint restore"
        )

    device = next(iter(devices), torch.device("cpu"))

    optimizer = prepared.optimizer_schedule.optimizer

    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def load_technical_resume_checkpoint(
    path: str | Path,
    *,
    prepared: PreparedTrainer,
    expected_attempt_identity: StudentAttemptIdentity,
    stage3: Stage3AvailabilityIndex,
    expected_configuration_hashes: Mapping[str, str],
    expected_target_cache_manifest_sha256: str,
    expected_file_sha256: str | None = None,
) -> TechnicalLoopSnapshot:
    """Verify, restore and return the continuation snapshot."""
    if not isinstance(prepared, PreparedTrainer):
        raise TechnicalCheckpointError(
            "prepared must be PreparedTrainer"
        )

    verify_student_attempt_identity(
        expected_attempt_identity,
        stage3,
    )

    expected_configs = _validated_hash_mapping(
        expected_configuration_hashes,
        name="expected_configuration_hashes",
    )
    expected_target = _require_sha256(
        expected_target_cache_manifest_sha256,
        name="expected_target_cache_manifest_sha256",
    )

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise TechnicalCheckpointError(
            f"technical checkpoint does not exist: {checkpoint_path}"
        )

    actual_file_hash = file_sha256(checkpoint_path)

    if expected_file_sha256 is not None:
        expected_file = _require_sha256(
            expected_file_sha256,
            name="expected_file_sha256",
        )

        if actual_file_hash != expected_file:
            raise TechnicalCheckpointError(
                "technical checkpoint physical SHA-256 mismatch"
            )

    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise TechnicalCheckpointError(
            "technical checkpoint could not be decoded"
        ) from exc

    if not isinstance(payload, Mapping):
        raise TechnicalCheckpointError(
            "technical checkpoint payload must be a mapping"
        )

    if set(payload) != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - set(payload))
        extra = sorted(set(payload) - _REQUIRED_KEYS)
        raise TechnicalCheckpointError(
            "technical checkpoint keys mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )

    if payload["schema_version"] != TECHNICAL_CHECKPOINT_SCHEMA_VERSION:
        raise TechnicalCheckpointError(
            "technical checkpoint schema version mismatch"
        )

    if payload["scientific_data"] is not False:
        raise TechnicalCheckpointError(
            "technical checkpoint must declare scientific_data=false"
        )

    if payload["production_eligible"] is not False:
        raise TechnicalCheckpointError(
            "technical checkpoint must declare production_eligible=false"
        )

    if payload["attempt_identity"] != expected_attempt_identity.to_mapping():
        raise TechnicalCheckpointError(
            "technical checkpoint attempt identity mismatch"
        )

    if payload["configuration_hashes"] != expected_configs:
        raise TechnicalCheckpointError(
            "technical checkpoint configuration hashes are stale or mismatched"
        )

    if payload["target_cache_manifest_sha256"] != expected_target:
        raise TechnicalCheckpointError(
            "technical checkpoint target-cache hash is stale or mismatched"
        )

    state_hashes = payload["state_hashes"]

    if not isinstance(state_hashes, Mapping) or set(state_hashes) != {
        "model_state_sha256",
        "optimizer_state_sha256",
        "scheduler_state_sha256",
        "torch_rng_state_sha256",
    }:
        raise TechnicalCheckpointError(
            "technical checkpoint state_hashes structure mismatch"
        )

    actual_state_hashes = {
        "model_state_sha256": canonical_state_hash(
            payload["model_state"]
        ),
        "optimizer_state_sha256": canonical_state_hash(
            payload["optimizer_state"]
        ),
        "scheduler_state_sha256": canonical_state_hash(
            payload["scheduler_state"]
        ),
        "torch_rng_state_sha256": canonical_state_hash(
            payload["torch_rng_state"]
        ),
    }

    if dict(state_hashes) != actual_state_hashes:
        raise TechnicalCheckpointError(
            "technical checkpoint internal state hash mismatch"
        )

    snapshot = _snapshot_from_mapping(payload["snapshot"])

    scheduler = prepared.optimizer_schedule.scheduler
    saved_scheduler_state = payload["scheduler_state"]

    if scheduler is None and saved_scheduler_state is not None:
        raise TechnicalCheckpointError(
            "checkpoint contains scheduler state but prepared trainer has none"
        )

    if scheduler is not None and saved_scheduler_state is None:
        raise TechnicalCheckpointError(
            "checkpoint is missing scheduler state"
        )

    prepared.model.load_state_dict(payload["model_state"])
    prepared.optimizer_schedule.optimizer.load_state_dict(
        payload["optimizer_state"]
    )
    _move_optimizer_state_to_model_device(prepared)

    if scheduler is not None:
        load_method = getattr(scheduler, "load_state_dict", None)

        if load_method is None or not callable(load_method):
            raise TechnicalCheckpointError(
                "injected scheduler must expose load_state_dict()"
            )

        load_method(saved_scheduler_state)

    rng_state = payload["torch_rng_state"]

    if (
        not isinstance(rng_state, torch.Tensor)
        or rng_state.dtype != torch.uint8
        or rng_state.ndim != 1
    ):
        raise TechnicalCheckpointError(
            "checkpoint torch RNG state has invalid representation"
        )

    torch.set_rng_state(rng_state)

    prepared.model.train(snapshot.outer_training_mode)

    return snapshot

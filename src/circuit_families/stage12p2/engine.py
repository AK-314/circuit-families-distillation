"""Shared technical student-attempt engine for Stage 12-P2.

The engine is policy-neutral: model architecture, trainer adapters, optimizer,
schedule, stopping rule, budget, checkpoint cadence, and retention are injected.
It provides only the common technical execution/bookkeeping required by P2.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from circuit_families.stage4_condition_identity import Stage3AvailabilityIndex
from circuit_families.stage5bc.student_trainer import (
    TechnicalInterruption,
    TechnicalLoopSnapshot,
    TechnicalTrainingResult,
    TrainerLifecycle,
    TrainerSettingsBundle,
)
from circuit_families.stage5bc.target_cache import LoadedTargetCache

from .architecture import (
    ArchitectureRecord,
    validate_task_architecture_compatibility,
)
from .checkpointing import (
    load_student_resume_checkpoint,
    save_student_resume_checkpoint,
)
from .training import StudentTrainingIdentity

ATTEMPT_STATUSES = (
    "completed",
    "failed",
    "interrupted",
    "numerical-failure",
    "unavailable",
)
AttemptStatus = Literal[
    "completed",
    "failed",
    "interrupted",
    "numerical-failure",
    "unavailable",
]


class StudentAttemptEngineError(ValueError):
    """Raised when the shared P2 technical attempt contract is violated."""


class DuplicateStudentAttemptError(StudentAttemptEngineError):
    """Raised when an already-terminal attempt would be executed again."""


@dataclass(frozen=True)
class RollingCheckpointProfile:
    """Injected technical-only checkpoint cadence and retention."""

    interval_updates: int
    retention_count: int

    def __post_init__(self) -> None:
        for name in ("interval_updates", "retention_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise StudentAttemptEngineError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CheckpointInventoryEntry:
    """One retained or resume-source checkpoint."""

    role: Literal["resume_source", "rolling", "interrupted", "terminal"]
    updates_completed: int
    path: str
    file_sha256: str


@dataclass(frozen=True)
class CheckpointInventory:
    """Compact checkpoint inventory for one engine invocation."""

    entries: tuple[CheckpointInventoryEntry, ...]
    rolling_retention_count: int
    checkpoints_written: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "entries": [
                {
                    "role": entry.role,
                    "updates_completed": entry.updates_completed,
                    "path": entry.path,
                    "file_sha256": entry.file_sha256,
                }
                for entry in self.entries
            ],
            "rolling_retention_count": self.rolling_retention_count,
            "checkpoints_written": self.checkpoints_written,
        }


@dataclass(frozen=True)
class StudentAttemptExecution:
    """Compact technical execution result; not an eligibility record."""

    identity_sha256: str
    architecture_ref: str
    status: AttemptStatus
    reason: str
    updates_completed: int
    trajectory_points: int
    checkpoints: CheckpointInventory
    training_result: TechnicalTrainingResult | None
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if self.status not in ATTEMPT_STATUSES:
            raise StudentAttemptEngineError(f"unsupported attempt status: {self.status!r}")
        if self.scientific_data is not False:
            raise StudentAttemptEngineError(
                "technical attempt execution must declare scientific_data=false"
            )
        if self.production_eligible is not False:
            raise StudentAttemptEngineError(
                "technical attempt execution must declare production_eligible=false"
            )


class _RollingCheckpointManager:
    def __init__(
        self,
        *,
        root: Path,
        profile: RollingCheckpointProfile,
        prepared,
        identity: StudentTrainingIdentity,
        stage3: Stage3AvailabilityIndex,
    ) -> None:
        self.root = root
        self.profile = profile
        self.prepared = prepared
        self.identity = identity
        self.stage3 = stage3
        self.entries: list[CheckpointInventoryEntry] = []
        self.rolling_paths: list[Path] = []
        self.checkpoints_written = 0
        self.root.mkdir(parents=True, exist_ok=True)

    def add_resume_source(
        self,
        *,
        path: Path,
        updates_completed: int,
        file_sha256: str,
    ) -> None:
        self.entries.append(
            CheckpointInventoryEntry(
                role="resume_source",
                updates_completed=updates_completed,
                path=str(path),
                file_sha256=file_sha256,
            )
        )

    def _save(
        self,
        *,
        snapshot: TechnicalLoopSnapshot,
        role: Literal["rolling", "interrupted", "terminal"],
    ) -> CheckpointInventoryEntry:
        path = self.root / (f"{role}_step_{snapshot.updates_completed:08d}.pt")

        evidence = save_student_resume_checkpoint(
            path,
            prepared=self.prepared,
            snapshot=snapshot,
            identity=self.identity,
            stage3=self.stage3,
        )
        self.checkpoints_written += 1

        entry = CheckpointInventoryEntry(
            role=role,
            updates_completed=snapshot.updates_completed,
            path=str(path),
            file_sha256=evidence.file_sha256,
        )
        self.entries.append(entry)

        if role == "rolling":
            self.rolling_paths.append(path)
            while len(self.rolling_paths) > self.profile.retention_count:
                expired = self.rolling_paths.pop(0)
                expired.unlink(missing_ok=True)
                self.entries = [
                    item
                    for item in self.entries
                    if not (item.role == "rolling" and item.path == str(expired))
                ]

        return entry

    def callback(self, snapshot: TechnicalLoopSnapshot) -> None:
        if snapshot.updates_completed % self.profile.interval_updates == 0:
            self._save(snapshot=snapshot, role="rolling")

    def ensure_interrupt_checkpoint(
        self,
        snapshot: TechnicalLoopSnapshot,
    ) -> CheckpointInventoryEntry:
        for entry in reversed(self.entries):
            if (
                entry.updates_completed == snapshot.updates_completed
                and entry.role == "rolling"
                and Path(entry.path).exists()
            ):
                return entry
        return self._save(snapshot=snapshot, role="interrupted")

    def save_terminal(
        self,
        snapshot: TechnicalLoopSnapshot,
    ) -> CheckpointInventoryEntry:
        return self._save(snapshot=snapshot, role="terminal")

    def inventory(self) -> CheckpointInventory:
        retained_rolling = sum(entry.role == "rolling" for entry in self.entries)
        if retained_rolling > self.profile.retention_count:
            raise StudentAttemptEngineError("rolling checkpoint retention bound was violated")
        return CheckpointInventory(
            entries=tuple(self.entries),
            rolling_retention_count=self.profile.retention_count,
            checkpoints_written=self.checkpoints_written,
        )


def _terminal_marker_path(output_root: Path) -> Path:
    return output_root / "terminal_status.json"


def _write_terminal_marker(
    *,
    output_root: Path,
    execution: StudentAttemptExecution,
) -> None:
    path = _terminal_marker_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        {
            "identity_sha256": execution.identity_sha256,
            "architecture_ref": execution.architecture_ref,
            "status": execution.status,
            "reason": execution.reason,
            "updates_completed": execution.updates_completed,
            "scientific_data": False,
            "production_eligible": False,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _validate_execution_bindings(
    *,
    identity: StudentTrainingIdentity,
    architecture_record: ArchitectureRecord,
    task_requirements,
    cache: LoadedTargetCache,
    training_inputs: torch.Tensor,
) -> None:
    if architecture_record.architecture_ref != identity.architecture_ref:
        raise StudentAttemptEngineError(
            "architecture record disagrees with student training identity"
        )

    if architecture_record.to_mapping()["record_sha256"] != identity.architecture_record_sha256:
        raise StudentAttemptEngineError(
            "architecture record hash disagrees with student training identity"
        )

    validate_task_architecture_compatibility(
        architecture_record,
        task_requirements,
    )

    if not isinstance(cache, LoadedTargetCache):
        raise StudentAttemptEngineError("cache must be LoadedTargetCache")

    if cache.manifest.manifest_sha256() != identity.target_cache_manifest_sha256:
        raise StudentAttemptEngineError(
            "target cache manifest hash disagrees with student training identity"
        )

    manifest = cache.manifest.to_mapping()
    expected_classes = int(architecture_record.dimensions["d_vocab_out"])
    if manifest["class_count"] != expected_classes:
        raise StudentAttemptEngineError(
            "target cache class count disagrees with architecture output width"
        )

    if manifest["example_count"] != int(training_inputs.shape[0]):
        raise StudentAttemptEngineError(
            "training input count disagrees with target cache example count"
        )


def _status_from_result(
    result: TechnicalTrainingResult,
) -> tuple[AttemptStatus, str]:
    if result.terminal_status == "stop_rule_met":
        return "completed", result.terminal_reason
    if result.terminal_status == "nonfinite_failure":
        return "numerical-failure", result.terminal_reason
    if result.terminal_status == "technical_step_limit_exhausted":
        return "failed", result.terminal_reason
    raise StudentAttemptEngineError(
        f"unsupported shared-trainer terminal status: {result.terminal_status!r}"
    )


def run_student_technical_attempt(
    *,
    lifecycle: TrainerLifecycle,
    stage3: Stage3AvailabilityIndex,
    identity: StudentTrainingIdentity,
    architecture_record: ArchitectureRecord,
    task_requirements,
    cache: LoadedTargetCache,
    training_inputs: torch.Tensor,
    settings: TrainerSettingsBundle,
    device: str | torch.device,
    output_root: str | Path,
    checkpoint_profile: RollingCheckpointProfile,
    technical_safety_step_limit: int,
    resume_checkpoint: str | Path | None = None,
    resume_checkpoint_sha256: str | None = None,
    interrupt_after_updates: int | None = None,
) -> StudentAttemptExecution:
    """Run one hard or soft technical attempt through the common Stage 5 engine."""
    output_root = Path(output_root)

    if _terminal_marker_path(output_root).exists():
        raise DuplicateStudentAttemptError("attempt already has a terminal status record")

    _validate_execution_bindings(
        identity=identity,
        architecture_record=architecture_record,
        task_requirements=task_requirements,
        cache=cache,
        training_inputs=training_inputs,
    )

    prepared = lifecycle.prepare(
        cache=cache,
        model_seed=identity.model_seed,
        device=device,
        settings=settings,
    )

    manager = _RollingCheckpointManager(
        root=output_root / "checkpoints",
        profile=checkpoint_profile,
        prepared=prepared,
        identity=identity,
        stage3=stage3,
    )

    resume_snapshot = None
    if resume_checkpoint is not None:
        if resume_checkpoint_sha256 is None:
            raise StudentAttemptEngineError(
                "resume_checkpoint_sha256 is required with resume_checkpoint"
            )
        resume_path = Path(resume_checkpoint)
        resume_snapshot = load_student_resume_checkpoint(
            resume_path,
            prepared=prepared,
            expected_identity=identity,
            stage3=stage3,
            expected_file_sha256=resume_checkpoint_sha256,
        )
        manager.add_resume_source(
            path=resume_path,
            updates_completed=resume_snapshot.updates_completed,
            file_sha256=resume_checkpoint_sha256,
        )

    try:
        result = lifecycle.run_technical(
            prepared=prepared,
            training_inputs=training_inputs,
            configuration_refs=identity.configuration_refs(),
            technical_safety_step_limit=technical_safety_step_limit,
            resume_snapshot=resume_snapshot,
            snapshot_callback=manager.callback,
            interrupt_after_updates=interrupt_after_updates,
        )
    except TechnicalInterruption as interruption:
        snapshot = interruption.snapshot
        manager.ensure_interrupt_checkpoint(snapshot)
        return StudentAttemptExecution(
            identity_sha256=identity.identity_sha256,
            architecture_ref=identity.architecture_ref,
            status="interrupted",
            reason="technical_interruption",
            updates_completed=snapshot.updates_completed,
            trajectory_points=len(snapshot.trajectory),
            checkpoints=manager.inventory(),
            training_result=None,
        )

    status, reason = _status_from_result(result)
    terminal_snapshot = TechnicalLoopSnapshot(
        updates_completed=result.updates_completed,
        trajectory=result.trajectory,
        outer_training_mode=prepared.model.training,
    )
    manager.save_terminal(terminal_snapshot)

    execution = StudentAttemptExecution(
        identity_sha256=identity.identity_sha256,
        architecture_ref=identity.architecture_ref,
        status=status,
        reason=reason,
        updates_completed=result.updates_completed,
        trajectory_points=len(result.trajectory),
        checkpoints=manager.inventory(),
        training_result=result,
    )
    _write_terminal_marker(
        output_root=output_root,
        execution=execution,
    )
    return execution


def record_unavailable_student_attempt(
    *,
    identity: StudentTrainingIdentity,
    output_root: str | Path,
    reason: str,
) -> StudentAttemptExecution:
    """Record a constructed technical unavailable attempt without training."""
    if not isinstance(reason, str) or not reason:
        raise StudentAttemptEngineError("unavailable reason must be a non-empty string")

    output_root = Path(output_root)
    if _terminal_marker_path(output_root).exists():
        raise DuplicateStudentAttemptError("attempt already has a terminal status record")

    execution = StudentAttemptExecution(
        identity_sha256=identity.identity_sha256,
        architecture_ref=identity.architecture_ref,
        status="unavailable",
        reason=reason,
        updates_completed=0,
        trajectory_points=0,
        checkpoints=CheckpointInventory(
            entries=(),
            rolling_retention_count=0,
            checkpoints_written=0,
        ),
        training_result=None,
    )
    _write_terminal_marker(
        output_root=output_root,
        execution=execution,
    )
    return execution

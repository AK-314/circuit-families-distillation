"""Architecture-aware Stage 12-P2 student identity and shared-trainer adapters.

This module extends the accepted Stage 5B/5C mechanics without changing the
frozen Stage 4 student condition identity or introducing hard/soft-specific
construction loops.

All execution represented here is technical-only and non-production.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from circuit_families.stage4_condition_identity import (
    Stage3AvailabilityIndex,
    parse_condition_id,
)
from circuit_families.stage5bc.student_identity import (
    StudentAttemptIdentity,
    verify_student_attempt_identity,
)
from circuit_families.stage5bc.target_cache import TargetCacheManifest

from .architecture import (
    ArchitectureRecord,
    ArchitectureRegistry,
    canonical_architecture_sha256,
)

STUDENT_TRAINING_IDENTITY_SCHEMA_VERSION = "stage12p2-student-training-identity/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StudentTrainingContractError(ValueError):
    """Raised when a P2 training identity or architecture binding is invalid."""


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StudentTrainingContractError(f"{name} must be lowercase SHA-256 hex")
    return value


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StudentTrainingContractError(f"{name} must be a non-empty string")
    return value


def _mapping_hash(value: Mapping[str, Any], *, name: str) -> str:
    if not isinstance(value, Mapping):
        raise StudentTrainingContractError(f"{name} must be a mapping")
    try:
        return canonical_architecture_sha256(copy.deepcopy(dict(value)))
    except (TypeError, ValueError) as exc:
        raise StudentTrainingContractError(f"{name} is not canonically hashable") from exc


@dataclass(frozen=True)
class StudentTrainingIdentity:
    """One exact P2 student attempt bound to task, teacher and architecture.

    ``stage5_attempt`` remains the authoritative frozen Stage 5B/5C attempt
    identity. The P2 envelope adds the dimensions that Stage 4 deliberately did
    not contain: task, teacher/cache, architecture, initialization, training,
    and backend identity.

    The model construction seed is not a new policy. Existing Stage 5B/5C
    technical execution constructs the student with
    Stage 5 training-seed evidence remains unchanged in the embedded attempt;
    relationship explicitly.
    """

    stage5_attempt: StudentAttemptIdentity
    task_identity_sha256: str
    teacher_condition_id: str
    teacher_record_sha256: str
    target_cache_manifest_sha256: str
    phase: str
    distillation_condition: str
    student_initialization: int
    architecture_ref: str
    architecture_record_sha256: str
    initialization_ref: str
    model_seed_id: str
    model_seed: int
    training_config_ref: str
    training_config_sha256: str
    backend_ref: str
    backend_qualification_sha256: str
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        _require_nonempty_string(self.model_seed_id, name="model_seed_id")
        if (
            isinstance(self.model_seed, bool)
            or not isinstance(self.model_seed, int)
            or not 0 <= self.model_seed <= 2**32 - 1
        ):
            raise StudentTrainingContractError("model_seed must be an integer in [0, 2**32 - 1]")
        if not isinstance(self.stage5_attempt, StudentAttemptIdentity):
            raise StudentTrainingContractError("stage5_attempt must be StudentAttemptIdentity")

        for name in (
            "task_identity_sha256",
            "teacher_record_sha256",
            "target_cache_manifest_sha256",
            "architecture_record_sha256",
            "training_config_sha256",
            "backend_qualification_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)

        for name in (
            "teacher_condition_id",
            "phase",
            "distillation_condition",
            "architecture_ref",
            "initialization_ref",
            "training_config_ref",
            "backend_ref",
        ):
            _require_nonempty_string(getattr(self, name), name=name)

        if (
            isinstance(self.student_initialization, bool)
            or not isinstance(self.student_initialization, int)
            or self.student_initialization < 0
        ):
            raise StudentTrainingContractError(
                "student_initialization must be a non-negative integer"
            )

        if (
            isinstance(self.model_seed, bool)
            or not isinstance(self.model_seed, int)
            or self.model_seed < 0
        ):
            raise StudentTrainingContractError("model_seed must be a non-negative integer")

        if self.scientific_data is not False:
            raise StudentTrainingContractError(
                "Stage 12-P2 training identity must declare scientific_data=false"
            )

        if self.production_eligible is not False:
            raise StudentTrainingContractError(
                "Stage 12-P2 training identity must declare production_eligible=false"
            )

    def identity_material(self) -> dict[str, Any]:
        return {
            "schema_version": STUDENT_TRAINING_IDENTITY_SCHEMA_VERSION,
            "scientific_data": False,
            "production_eligible": False,
            "stage5_attempt": self.stage5_attempt.to_mapping(),
            "task_identity_sha256": self.task_identity_sha256,
            "teacher": {
                "condition_id": self.teacher_condition_id,
                "record_sha256": self.teacher_record_sha256,
            },
            "target_cache_manifest_sha256": self.target_cache_manifest_sha256,
            "condition": {
                "phase": self.phase,
                "distillation_condition": self.distillation_condition,
                "student_initialization": self.student_initialization,
            },
            "architecture": {
                "architecture_ref": self.architecture_ref,
                "architecture_record_sha256": self.architecture_record_sha256,
                "initialization_ref": self.initialization_ref,
                "model_seed_id": self.model_seed_id,
                "model_seed": self.model_seed,
            },
            "training": {
                "training_config_ref": self.training_config_ref,
                "training_config_sha256": self.training_config_sha256,
            },
            "backend": {
                "backend_ref": self.backend_ref,
                "backend_qualification_sha256": (self.backend_qualification_sha256),
            },
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_architecture_sha256(self.identity_material())

    def to_mapping(self) -> dict[str, Any]:
        mapping = self.identity_material()
        mapping["identity_sha256"] = self.identity_sha256
        return mapping

    def configuration_refs(self) -> dict[str, str]:
        """Return refs suitable for the accepted shared technical trainer."""
        return {
            "student_architecture": self.architecture_ref,
            "student_initialization": self.initialization_ref,
            "training_config": self.training_config_ref,
            "backend_qualification": self.backend_ref,
        }

    def checkpoint_configuration_hashes(self) -> dict[str, str]:
        """Return mandatory hashes for Stage 5B/5C checkpoint binding."""
        return {
            "student_training_identity_sha256": self.identity_sha256,
            "task_identity_sha256": self.task_identity_sha256,
            "teacher_record_sha256": self.teacher_record_sha256,
            "architecture_record_sha256": self.architecture_record_sha256,
            "training_config_sha256": self.training_config_sha256,
            "backend_qualification_sha256": self.backend_qualification_sha256,
        }


def bind_student_training_identity(
    *,
    stage3: Stage3AvailabilityIndex,
    stage5_attempt: StudentAttemptIdentity,
    task_identity_sha256: str,
    target_cache_manifest: TargetCacheManifest,
    architecture_record: ArchitectureRecord,
    model_seed_id: str,
    model_seed: int,
    training_config_ref: str,
    training_config: Mapping[str, Any],
    backend_ref: str,
    backend_qualification: Mapping[str, Any],
) -> StudentTrainingIdentity:
    """Verify and bind the accepted upstream identities without redefining them."""
    if not isinstance(stage3, Stage3AvailabilityIndex):
        raise StudentTrainingContractError("stage3 must be Stage3AvailabilityIndex")
    if not isinstance(target_cache_manifest, TargetCacheManifest):
        raise StudentTrainingContractError("target_cache_manifest must be TargetCacheManifest")
    if not isinstance(architecture_record, ArchitectureRecord):
        raise StudentTrainingContractError("architecture_record must be ArchitectureRecord")

    try:
        student_condition = verify_student_attempt_identity(
            stage5_attempt,
            stage3,
        )
    except ValueError as exc:
        raise StudentTrainingContractError(
            f"Stage 5B/5C attempt identity failed verification: {exc}"
        ) from exc

    if (
        student_condition.distillation_condition is None
        or student_condition.student_initialization is None
    ):
        raise StudentTrainingContractError(
            "verified student attempt must retain depth-4 condition coordinates"
        )

    manifest = target_cache_manifest.to_mapping()
    teacher_reference = manifest["teacher_reference"]
    teacher_condition_id = teacher_reference["condition_id"]

    try:
        teacher_condition = parse_condition_id(
            teacher_condition_id,
            stage3,
        )
    except ValueError as exc:
        raise StudentTrainingContractError(
            f"target-cache teacher identity failed verification: {exc}"
        ) from exc

    if teacher_condition.depth != 3:
        raise StudentTrainingContractError("target-cache teacher identity must have depth 3")

    if (
        teacher_condition.teacher_seed != student_condition.teacher_seed
        or teacher_condition.phase != student_condition.phase
        or teacher_condition.distillation_condition != student_condition.distillation_condition
    ):
        raise StudentTrainingContractError(
            "target-cache teacher identity disagrees with student condition"
        )

    architecture_mapping = architecture_record.to_mapping()

    return StudentTrainingIdentity(
        stage5_attempt=stage5_attempt,
        task_identity_sha256=_require_sha256(
            task_identity_sha256,
            name="task_identity_sha256",
        ),
        teacher_condition_id=teacher_condition_id,
        teacher_record_sha256=_require_sha256(
            teacher_reference["record_sha256"],
            name="teacher_reference.record_sha256",
        ),
        target_cache_manifest_sha256=target_cache_manifest.manifest_sha256(),
        phase=student_condition.phase,
        distillation_condition=student_condition.distillation_condition,
        student_initialization=student_condition.student_initialization,
        architecture_ref=architecture_record.architecture_ref,
        architecture_record_sha256=_require_sha256(
            architecture_mapping["record_sha256"],
            name="architecture_record.record_sha256",
        ),
        initialization_ref=architecture_record.initialization_ref,
        model_seed_id=model_seed_id,
        model_seed=model_seed,
        training_config_ref=_require_nonempty_string(
            training_config_ref,
            name="training_config_ref",
        ),
        training_config_sha256=_mapping_hash(
            training_config,
            name="training_config",
        ),
        backend_ref=_require_nonempty_string(
            backend_ref,
            name="backend_ref",
        ),
        backend_qualification_sha256=_mapping_hash(
            backend_qualification,
            name="backend_qualification",
        ),
    )


class FinalPositionStudentModel(torch.nn.Module):
    """Normalize an architecture's dense forward output to rank-2 logits."""

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        if not isinstance(base_model, torch.nn.Module):
            raise StudentTrainingContractError("base_model must be torch.nn.Module")
        self.base_model = base_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model(inputs)

        if not isinstance(outputs, torch.Tensor):
            raise StudentTrainingContractError("architecture forward must return a torch.Tensor")

        if outputs.ndim == 2:
            logits = outputs
        elif outputs.ndim == 3:
            if outputs.shape[1] == 0:
                raise StudentTrainingContractError(
                    "architecture sequence logits must contain at least one position"
                )
            logits = outputs[:, -1, :]
        else:
            raise StudentTrainingContractError(
                "architecture forward must return rank-2 dense logits or rank-3 sequence logits"
            )

        if not logits.is_floating_point():
            raise StudentTrainingContractError("student dense logits must use a floating dtype")

        if not bool(torch.isfinite(logits).all()):
            raise StudentTrainingContractError("student dense logits contain non-finite values")

        return logits


@dataclass(frozen=True)
class ArchitectureModelConstructor:
    """Stage5BC ModelConstructor backed only by the P2 architecture registry."""

    registry: ArchitectureRegistry
    architecture_ref: str
    architecture_record_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.registry, ArchitectureRegistry):
            raise StudentTrainingContractError("registry must be ArchitectureRegistry")

        _require_nonempty_string(
            self.architecture_ref,
            name="architecture_ref",
        )
        expected_sha = _require_sha256(
            self.architecture_record_sha256,
            name="architecture_record_sha256",
        )

        record = self.registry.architecture(self.architecture_ref)
        actual_sha = record.to_mapping()["record_sha256"]
        if actual_sha != expected_sha:
            raise StudentTrainingContractError("architecture constructor record SHA-256 mismatch")

    @classmethod
    def from_record(
        cls,
        *,
        registry: ArchitectureRegistry,
        record: ArchitectureRecord,
    ) -> ArchitectureModelConstructor:
        if not isinstance(record, ArchitectureRecord):
            raise StudentTrainingContractError("record must be ArchitectureRecord")
        return cls(
            registry=registry,
            architecture_ref=record.architecture_ref,
            architecture_record_sha256=record.to_mapping()["record_sha256"],
        )

    def __call__(
        self,
        *,
        seed: int,
        device: torch.device,
        settings: Mapping[str, Any],
    ) -> torch.nn.Module:
        if not isinstance(settings, Mapping):
            raise StudentTrainingContractError("model settings must be an explicit mapping")

        if settings.get("architecture_ref") != self.architecture_ref:
            raise StudentTrainingContractError("model settings architecture_ref mismatch")

        if settings.get("architecture_record_sha256") != self.architecture_record_sha256:
            raise StudentTrainingContractError("model settings architecture record hash mismatch")

        model = self.registry.build(
            self.architecture_ref,
            seed=seed,
            device=device,
        )

        if not isinstance(model, torch.nn.Module):
            raise StudentTrainingContractError(
                "architecture registry builder must return torch.nn.Module"
            )

        return FinalPositionStudentModel(model)

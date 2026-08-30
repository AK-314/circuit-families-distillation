"""Stage 12-P2 architecture-bound adapters over Stage 5B/5C checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from circuit_families.stage4_condition_identity import Stage3AvailabilityIndex
from circuit_families.stage5bc.student_trainer import (
    PreparedTrainer,
    TechnicalLoopSnapshot,
)
from circuit_families.stage5bc.technical_checkpoint import (
    TechnicalCheckpointEvidence,
    load_technical_resume_checkpoint,
    save_technical_resume_checkpoint,
)

from .training import StudentTrainingIdentity


class StudentCheckpointBindingError(ValueError):
    """Raised when a prepared trainer disagrees with its P2 identity."""


def _verify_prepared_architecture_binding(
    prepared: PreparedTrainer,
    identity: StudentTrainingIdentity,
) -> None:
    settings_bundle = getattr(prepared, "settings", None)
    model_settings = getattr(settings_bundle, "model", None)

    if not isinstance(model_settings, Mapping):
        raise StudentCheckpointBindingError(
            "prepared trainer must expose model settings for architecture binding"
        )

    expected = {
        "architecture_ref": identity.architecture_ref,
        "architecture_record_sha256": identity.architecture_record_sha256,
    }

    for field, value in expected.items():
        if model_settings.get(field) != value:
            raise StudentCheckpointBindingError(
                f"prepared trainer {field} disagrees with P2 training identity"
            )


def save_student_resume_checkpoint(
    path: str | Path,
    *,
    prepared: PreparedTrainer,
    snapshot: TechnicalLoopSnapshot,
    identity: StudentTrainingIdentity,
    stage3: Stage3AvailabilityIndex,
) -> TechnicalCheckpointEvidence:
    """Atomically save one Stage 5 checkpoint bound to the exact P2 identity."""
    if not isinstance(identity, StudentTrainingIdentity):
        raise StudentCheckpointBindingError("identity must be StudentTrainingIdentity")

    _verify_prepared_architecture_binding(prepared, identity)

    return save_technical_resume_checkpoint(
        path,
        prepared=prepared,
        snapshot=snapshot,
        attempt_identity=identity.stage5_attempt,
        stage3=stage3,
        configuration_hashes=identity.checkpoint_configuration_hashes(),
        target_cache_manifest_sha256=identity.target_cache_manifest_sha256,
    )


def load_student_resume_checkpoint(
    path: str | Path,
    *,
    prepared: PreparedTrainer,
    expected_identity: StudentTrainingIdentity,
    stage3: Stage3AvailabilityIndex,
    expected_file_sha256: str | None = None,
) -> TechnicalLoopSnapshot:
    """Restore only when checkpoint, cache, attempt, and architecture all agree."""
    if not isinstance(expected_identity, StudentTrainingIdentity):
        raise StudentCheckpointBindingError("expected_identity must be StudentTrainingIdentity")

    _verify_prepared_architecture_binding(prepared, expected_identity)

    return load_technical_resume_checkpoint(
        path,
        prepared=prepared,
        expected_attempt_identity=expected_identity.stage5_attempt,
        stage3=stage3,
        expected_configuration_hashes=(expected_identity.checkpoint_configuration_hashes()),
        expected_target_cache_manifest_sha256=(expected_identity.target_cache_manifest_sha256),
        expected_file_sha256=expected_file_sha256,
    )

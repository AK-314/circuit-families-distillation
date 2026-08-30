from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from circuit_families.stage12p2 import (
    CheckpointInventory,
    CheckpointInventoryEntry,
    StudentAttemptExecution,
    StudentSealingError,
    evaluate_hard_student_eligibility,
    record_noncompleted_attempt,
    release_student_for_discovery,
    seal_student_model,
    summarize_attempt_records,
)

ENGINE = runpy.run_path("tests/test_stage12p2_engine.py")
_stage3 = ENGINE["_stage3"]
_records = ENGINE["_records"]
_cache = ENGINE["_cache"]
_identity = ENGINE["_identity"]


def _case(tmp_path: Path, *, passed: bool = True):
    stage3 = _stage3()
    canonical, technical_a, _ = _records()
    cache = _cache(tmp_path, stage3=stage3, condition="hard_target")
    identity = _identity(
        stage3=stage3,
        record=technical_a,
        cache=cache,
        condition="hard_target",
    )
    execution = StudentAttemptExecution(
        identity_sha256=identity.identity_sha256,
        architecture_ref=identity.architecture_ref,
        status="completed",
        reason="stop_rule_met",
        updates_completed=1,
        trajectory_points=1,
        checkpoints=CheckpointInventory(
            entries=(
                CheckpointInventoryEntry(
                    role="terminal",
                    updates_completed=1,
                    path=str(tmp_path / "terminal.pt"),
                    file_sha256="c" * 64,
                ),
            ),
            rolling_retention_count=1,
            checkpoints_written=1,
        ),
        training_result=None,
    )
    logits = torch.zeros((2, 113), dtype=torch.float32)
    logits[0, 0] = 2.0
    logits[1, 1 if passed else 2] = 2.0
    eligibility = evaluate_hard_student_eligibility(
        execution=execution,
        identity=identity,
        teacher_decisions=(0, 1),
        student_dense_logits=logits,
        ordering_ref="technical-order/v1",
        ordered_input_ids_sha256="d" * 64,
        domain_complete=True,
    )
    return identity, execution, eligibility


def test_passed_student_seals_and_releases(tmp_path: Path) -> None:
    identity, execution, eligibility = _case(tmp_path)
    sealed = seal_student_model(
        execution=execution,
        identity=identity,
        eligibility=eligibility,
        dense_output_sha256=eligibility.dense_output_sha256,
    )
    release = release_student_for_discovery(
        sealed=sealed,
        eligibility=eligibility,
    )
    assert release.release_status == "released"
    assert release.architecture_ref == identity.architecture_ref
    assert release.training_config_sha256 == identity.training_config_sha256
    assert release.backend_qualification_sha256 == identity.backend_qualification_sha256


def test_ineligible_student_never_enters_discovery(tmp_path: Path) -> None:
    identity, execution, eligibility = _case(tmp_path, passed=False)
    assert eligibility.status == "ineligible"
    with pytest.raises(StudentSealingError, match="cannot be sealed"):
        seal_student_model(
            execution=execution,
            identity=identity,
            eligibility=eligibility,
            dense_output_sha256=eligibility.dense_output_sha256,
        )


@pytest.mark.parametrize(
    "field",
    (
        "architecture_record_sha256",
        "task_identity_sha256",
        "target_cache_manifest_sha256",
        "checkpoint_sha256",
    ),
)
def test_identity_and_checkpoint_hash_mismatch_rejects(
    tmp_path: Path,
    field: str,
) -> None:
    identity, execution, eligibility = _case(tmp_path)
    bad = replace(eligibility, **{field: "e" * 64})
    with pytest.raises(StudentSealingError, match="mismatch"):
        seal_student_model(
            execution=execution,
            identity=identity,
            eligibility=bad,
            dense_output_sha256=bad.dense_output_sha256,
        )


def test_dense_output_hash_mismatch_rejects(tmp_path: Path) -> None:
    identity, execution, eligibility = _case(tmp_path)
    with pytest.raises(StudentSealingError, match="dense output hash mismatch"):
        seal_student_model(
            execution=execution,
            identity=identity,
            eligibility=eligibility,
            dense_output_sha256="e" * 64,
        )


def test_sealed_model_cannot_change_architecture_after_eligibility(
    tmp_path: Path,
) -> None:
    identity, execution, eligibility = _case(tmp_path)
    sealed = seal_student_model(
        execution=execution,
        identity=identity,
        eligibility=eligibility,
        dense_output_sha256=eligibility.dense_output_sha256,
    )
    altered = replace(
        sealed,
        architecture_ref="technical-other/v1",
    )
    with pytest.raises(StudentSealingError, match="architecture_ref mismatch"):
        release_student_for_discovery(
            sealed=altered,
            eligibility=eligibility,
        )


def test_failed_and_ineligible_attempts_remain_counted(
    tmp_path: Path,
) -> None:
    identity, _, ineligible = _case(tmp_path, passed=False)
    failed_execution = StudentAttemptExecution(
        identity_sha256=identity.identity_sha256,
        architecture_ref=identity.architecture_ref,
        status="failed",
        reason="technical_limit",
        updates_completed=1,
        trajectory_points=1,
        checkpoints=CheckpointInventory(
            entries=(),
            rolling_retention_count=0,
            checkpoints_written=0,
        ),
        training_result=None,
    )
    failed = record_noncompleted_attempt(
        execution=failed_execution,
        identity=identity,
    )
    accounting = summarize_attempt_records((ineligible, failed))
    assert accounting.ineligible == 1
    assert accounting.optimization_failed == 1
    assert accounting.total == 2


def test_hard_and_soft_attempts_cannot_be_pooled(tmp_path: Path) -> None:
    identity, _, hard = _case(tmp_path, passed=False)
    soft_identity = replace(
        identity,
        distillation_condition="soft_target",
    )
    soft_failure = replace(
        record_noncompleted_attempt(
            execution=StudentAttemptExecution(
                identity_sha256=soft_identity.identity_sha256,
                architecture_ref=soft_identity.architecture_ref,
                status="unavailable",
                reason="technical_unavailable",
                updates_completed=0,
                trajectory_points=0,
                checkpoints=CheckpointInventory(
                    entries=(),
                    rolling_retention_count=0,
                    checkpoints_written=0,
                ),
                training_result=None,
            ),
            identity=soft_identity,
        ),
        estimand="soft",
    )
    with pytest.raises(StudentSealingError, match="cannot be pooled"):
        summarize_attempt_records((hard, soft_failure))


def test_technical_seal_cannot_assert_production_eligibility(
    tmp_path: Path,
) -> None:
    identity, execution, eligibility = _case(tmp_path)
    sealed = seal_student_model(
        execution=execution,
        identity=identity,
        eligibility=eligibility,
        dense_output_sha256=eligibility.dense_output_sha256,
    )
    with pytest.raises(StudentSealingError, match="production_eligible"):
        replace(sealed, production_eligible=True)

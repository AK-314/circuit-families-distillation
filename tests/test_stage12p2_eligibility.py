from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from circuit_families.stage6c import (
    TECHNICAL_POLICY_STATUS,
    TECHNICAL_SOFT_DISCREPANCY_METRIC,
    TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
    SoftRepresentationMetadata,
    TechnicalArgmaxRequirementMetadata,
    TechnicalSoftPolicy,
    TechnicalToleranceMetadata,
)
from circuit_families.stage6c.soft_target import CENTRING_REF
from circuit_families.stage12p2 import (
    CheckpointInventory,
    CheckpointInventoryEntry,
    StudentAttemptExecution,
    StudentEligibilityError,
    evaluate_hard_student_eligibility,
    evaluate_soft_student_eligibility,
    record_noncompleted_attempt,
)

HELPERS = runpy.run_path("tests/test_stage12p2_engine.py")
_stage3 = HELPERS["_stage3"]
_records = HELPERS["_records"]
_cache = HELPERS["_cache"]
_identity = HELPERS["_identity"]


def _completed(
    identity,
    tmp_path: Path,
) -> StudentAttemptExecution:
    checkpoint = tmp_path / "terminal.pt"
    checkpoint.write_bytes(b"technical-checkpoint")
    return StudentAttemptExecution(
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
                    path=str(checkpoint),
                    file_sha256="c" * 64,
                ),
            ),
            rolling_retention_count=1,
            checkpoints_written=1,
        ),
        training_result=None,
    )


def _soft_policy(
    *,
    required: bool,
    tolerance: float,
):
    representation = SoftRepresentationMetadata(
        representation_ref=("technical-stage12p2-centred-logits/v1"),
        cache_kind="teacher_logits",
        centering_ref=CENTRING_REF,
        teacher_condition_id=("teacher_seed=1|phase=stable post-grokking|condition=soft_target"),
        ordering_ref="technical-order/v1",
        ordered_input_ids_sha256="d" * 64,
        temperature_candidate=None,
        normalization_candidate_ref=None,
    )
    return TechnicalSoftPolicy(
        schema_version=TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
        policy_ref="technical-stage12p2-soft-policy/v1",
        status=TECHNICAL_POLICY_STATUS,
        scientific_data=False,
        production_eligible=False,
        resolves_ud006=False,
        representation=representation,
        tolerance=TechnicalToleranceMetadata(
            metric_ref=TECHNICAL_SOFT_DISCREPANCY_METRIC,
            comparison="less_than_or_equal",
            candidate_value=tolerance,
            status=TECHNICAL_POLICY_STATUS,
        ),
        argmax_requirement=TechnicalArgmaxRequirementMetadata(
            requirement_ref="technical-stage12p2-argmax/v1",
            candidate_required=required,
            status=TECHNICAL_POLICY_STATUS,
        ),
    )


def test_hard_exact_agreement_and_one_mismatch(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    canonical, _, _ = _records()
    cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="hard_target",
    )
    identity = _identity(
        stage3=stage3,
        record=canonical,
        cache=cache,
        condition="hard_target",
    )
    execution = _completed(identity, tmp_path)

    teacher = torch.tensor([0, 1], dtype=torch.int64)
    logits = torch.zeros((2, 113), dtype=torch.float32)
    logits[0, 0] = 2.0
    logits[1, 1] = 2.0

    passed = evaluate_hard_student_eligibility(
        execution=execution,
        identity=identity,
        teacher_decisions=teacher,
        student_dense_logits=logits,
        ordering_ref="technical-order/v1",
        ordered_input_ids_sha256="d" * 64,
        domain_complete=True,
    )
    assert passed.status == "passed"
    assert passed.agreement_count == 2
    assert passed.total_count == 2

    changed = logits.clone()
    changed[1, 1] = 0.0
    changed[1, 2] = 3.0
    failed = evaluate_hard_student_eligibility(
        execution=execution,
        identity=identity,
        teacher_decisions=teacher,
        student_dense_logits=changed,
        ordering_ref="technical-order/v1",
        ordered_input_ids_sha256="d" * 64,
        domain_complete=True,
    )
    assert failed.status == "ineligible"
    assert failed.agreement_count == 1


def test_hard_requires_claimed_complete_supplied_domain(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    canonical, _, _ = _records()
    cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="hard_target",
    )
    identity = _identity(
        stage3=stage3,
        record=canonical,
        cache=cache,
        condition="hard_target",
    )
    with pytest.raises(
        StudentEligibilityError,
        match="complete",
    ):
        evaluate_hard_student_eligibility(
            execution=_completed(identity, tmp_path),
            identity=identity,
            teacher_decisions=(0, 1),
            student_dense_logits=torch.zeros((2, 113)),
            ordering_ref="technical-order/v1",
            ordered_input_ids_sha256="d" * 64,
            domain_complete=False,
        )


def test_soft_is_gauge_invariant(tmp_path: Path) -> None:
    stage3 = _stage3()
    _, technical_a, _ = _records()
    cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="soft_target",
    )
    identity = _identity(
        stage3=stage3,
        record=technical_a,
        cache=cache,
        condition="soft_target",
    )
    execution = _completed(identity, tmp_path)

    teacher = torch.randn(
        (2, 113),
        generator=torch.Generator().manual_seed(1),
    )
    student = teacher + torch.tensor([[7.0], [-3.0]])

    result = evaluate_soft_student_eligibility(
        execution=execution,
        identity=identity,
        teacher_logits=teacher,
        student_dense_logits=student,
        policy=_soft_policy(
            required=False,
            tolerance=1e-10,
        ),
        ordering_ref="technical-order/v1",
        ordered_input_ids_sha256="d" * 64,
        domain_complete=True,
    )
    assert result.status == "passed"
    assert result.discrepancy == pytest.approx(
        0.0,
        abs=1e-10,
    )


def test_soft_injected_argmax_rule_can_block(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    _, technical_a, _ = _records()
    cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="soft_target",
    )
    identity = _identity(
        stage3=stage3,
        record=technical_a,
        cache=cache,
        condition="soft_target",
    )
    execution = _completed(identity, tmp_path)

    teacher = torch.zeros((2, 113), dtype=torch.float32)
    teacher[:, 0] = 1.0
    student = teacher.clone()
    student[1, 0] = 0.0
    student[1, 1] = 1.0

    no_argmax = evaluate_soft_student_eligibility(
        execution=execution,
        identity=identity,
        teacher_logits=teacher,
        student_dense_logits=student,
        policy=_soft_policy(
            required=False,
            tolerance=0.02,
        ),
        ordering_ref="technical-order/v1",
        ordered_input_ids_sha256="d" * 64,
        domain_complete=True,
    )
    required = evaluate_soft_student_eligibility(
        execution=execution,
        identity=identity,
        teacher_logits=teacher,
        student_dense_logits=student,
        policy=_soft_policy(
            required=True,
            tolerance=0.02,
        ),
        ordering_ref="technical-order/v1",
        ordered_input_ids_sha256="d" * 64,
        domain_complete=True,
    )

    assert no_argmax.status == "passed"
    assert required.status == "ineligible"
    assert required.argmax_agreement_count == 1


@pytest.mark.parametrize(
    ("engine_status", "expected"),
    (
        ("failed", "optimization-failed"),
        ("numerical-failure", "numerical-failed"),
        ("interrupted", "interrupted"),
        ("unavailable", "unavailable"),
    ),
)
def test_noncompleted_attempts_remain_explicitly_countable(
    tmp_path: Path,
    engine_status: str,
    expected: str,
) -> None:
    stage3 = _stage3()
    canonical, _, _ = _records()
    cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="hard_target",
    )
    identity = _identity(
        stage3=stage3,
        record=canonical,
        cache=cache,
        condition="hard_target",
    )
    execution = StudentAttemptExecution(
        identity_sha256=identity.identity_sha256,
        architecture_ref=identity.architecture_ref,
        status=engine_status,
        reason="technical_case",
        updates_completed=0,
        trajectory_points=0,
        checkpoints=CheckpointInventory(
            entries=(),
            rolling_retention_count=0,
            checkpoints_written=0,
        ),
        training_result=None,
    )

    record = record_noncompleted_attempt(
        execution=execution,
        identity=identity,
    )

    assert record.status == expected
    assert record.estimand == "hard"


def test_hard_and_soft_records_cannot_be_relabeled(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    canonical, _, _ = _records()
    cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="hard_target",
    )
    identity = _identity(
        stage3=stage3,
        record=canonical,
        cache=cache,
        condition="hard_target",
    )
    hard = evaluate_hard_student_eligibility(
        execution=_completed(identity, tmp_path),
        identity=identity,
        teacher_decisions=(0, 1),
        student_dense_logits=torch.nn.functional.one_hot(
            torch.tensor([0, 1]),
            num_classes=113,
        ).float(),
        ordering_ref="technical-order/v1",
        ordered_input_ids_sha256="d" * 64,
        domain_complete=True,
    )

    with pytest.raises(
        StudentEligibilityError,
        match="estimand",
    ):
        replace(hard, estimand="soft")

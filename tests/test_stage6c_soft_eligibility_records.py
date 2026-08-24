from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage4_schema_common import CommonSchemaContract
from circuit_families.stage4_schema_records import validate_part_m_record
from circuit_families.stage5bc.attempt_records import emit_technical_attempt_record
from circuit_families.stage5bc.student_identity import build_student_attempt_identity
from circuit_families.stage5bc.target_cache import FULL_DOMAIN_EXAMPLE_COUNT
from circuit_families.stage6c import (
    CENTRING_REF,
    TECHNICAL_POLICY_STATUS,
    TECHNICAL_SOFT_DISCREPANCY_METRIC,
    TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
    CanonicalSoftOutput,
    SoftAttemptLedger,
    SoftEligibilityError,
    SoftRepresentationMetadata,
    Stage6CRecordError,
    TechnicalArgmaxRequirementMetadata,
    TechnicalSoftPolicy,
    TechnicalToleranceMetadata,
    assess_soft_attempt,
    evaluate_soft_target_eligibility,
    generate_soft_sealing_evidence,
    soft_circuit_release_gate,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
VOCAB_PATH = ROOT / "followup/configs/stage4_common_vocabulary_v1.json"
IDENTITY_PATH = ROOT / "followup/configs/stage4_condition_identity_spec_v1.json"
ORDERING_REF = "technical-stage6c-full-domain-order/v1"
ORDERING_SHA256 = "a" * 64


@pytest.fixture(scope="module")
def registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry_sha() -> str:
    return hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def stage3(registry) -> Stage3AvailabilityIndex:
    return Stage3AvailabilityIndex.from_registry(registry)


@pytest.fixture(scope="module")
def contract() -> CommonSchemaContract:
    return CommonSchemaContract.from_specs(
        json.loads(VOCAB_PATH.read_text(encoding="utf-8")),
        json.loads(IDENTITY_PATH.read_text(encoding="utf-8")),
    )


def _condition_id(
    stage3: Stage3AvailabilityIndex,
    *,
    condition: str,
    initialization: int | None = None,
) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition=condition,
            student_initialization=initialization,
        ),
        stage3,
    )


def _policy(
    stage3: Stage3AvailabilityIndex,
    *,
    tolerance: float,
    argmax_required: bool | None,
) -> TechnicalSoftPolicy:
    argmax = None
    if argmax_required is not None:
        argmax = TechnicalArgmaxRequirementMetadata(
            requirement_ref="technical-stage6c-argmax-candidate/v1",
            candidate_required=argmax_required,
            status=TECHNICAL_POLICY_STATUS,
        )
    return TechnicalSoftPolicy(
        schema_version=TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
        policy_ref="technical-stage6c-soft-candidate/v1",
        status=TECHNICAL_POLICY_STATUS,
        scientific_data=False,
        production_eligible=False,
        resolves_ud006=False,
        representation=SoftRepresentationMetadata(
            representation_ref="technical-centred-teacher-logits/v1",
            cache_kind="teacher_logits",
            centering_ref=CENTRING_REF,
            teacher_condition_id=_condition_id(stage3, condition="soft_target"),
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            temperature_candidate=None,
            normalization_candidate_ref=None,
        ),
        tolerance=TechnicalToleranceMetadata(
            metric_ref=TECHNICAL_SOFT_DISCREPANCY_METRIC,
            comparison="less_than_or_equal",
            candidate_value=tolerance,
            status=TECHNICAL_POLICY_STATUS,
        ),
        argmax_requirement=argmax,
    )


def _teacher_logits() -> torch.Tensor:
    return torch.tensor([1.0, -1.0], dtype=torch.float64).repeat(
        FULL_DOMAIN_EXAMPLE_COUNT,
        1,
    )


def _outputs(
    stage3: Stage3AvailabilityIndex,
    *,
    student_logits: torch.Tensor,
) -> tuple[CanonicalSoftOutput, CanonicalSoftOutput]:
    teacher = CanonicalSoftOutput(
        role="soft_target_teacher",
        condition_id=_condition_id(stage3, condition="soft_target"),
        ordering_ref=ORDERING_REF,
        ordered_input_ids_sha256=ORDERING_SHA256,
        logits=_teacher_logits(),
        record_status="sealed",
    )
    student = CanonicalSoftOutput(
        role="soft_target_student",
        condition_id=_condition_id(
            stage3,
            condition="soft_target",
            initialization=0,
        ),
        ordering_ref=ORDERING_REF,
        ordered_input_ids_sha256=ORDERING_SHA256,
        logits=student_logits,
        record_status="sealed",
    )
    return teacher, student


def _evaluation(
    stage3: Stage3AvailabilityIndex,
    *,
    tolerance: float = 0.25,
    argmax_required: bool | None = None,
    student_logits: torch.Tensor | None = None,
):
    policy = _policy(
        stage3,
        tolerance=tolerance,
        argmax_required=argmax_required,
    )
    teacher, student = _outputs(
        stage3,
        student_logits=(
            _teacher_logits() if student_logits is None else student_logits
        ),
    )
    return evaluate_soft_target_eligibility(
        teacher=teacher,
        student=student,
        policy=policy,
        stage3=stage3,
    )


def _artifact(
    *,
    path: str,
    digest: str,
    storage_class: str,
) -> dict[str, str]:
    return {
        "path": path,
        "sha256": digest,
        "storage_class": storage_class,
    }


def _attempt(
    stage3: Stage3AvailabilityIndex,
    *,
    outcome_kind: str = "succeeded",
    attempt_index: int = 0,
    retry_index: int = 0,
    sealed: bool = False,
    condition: str = "soft_target",
):
    identity = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition=condition,
        student_initialization=0,
        attempt_index=attempt_index,
        retry_index=retry_index,
    )
    checkpoint = _artifact(
        path=f"synthetic/stage6c/student-a{attempt_index}-r{retry_index}.pt",
        digest="b" * 64,
        storage_class="external_checkpoint",
    )
    record = emit_technical_attempt_record(
        stage3=stage3,
        attempt_identity=identity,
        target_cache_reference={
            "record_type": "teacher_output_cache",
            "schema_version": "teacher_output_cache/v1",
            "condition_id": _condition_id(stage3, condition=condition),
            "record_sha256": "c" * 64,
        },
        outcome_kind=outcome_kind,
        student_architecture_ref="synthetic-student-architecture/v1",
        replication_policy_ref="synthetic-replication-policy/v1",
        training_config_ref="synthetic-training-config/v1",
        training_log=_artifact(
            path=f"synthetic/stage6c/train-a{attempt_index}-r{retry_index}.log",
            digest="d" * 64,
            storage_class="external_log",
        ),
        model_checkpoint=checkpoint if outcome_kind == "succeeded" else None,
        failure_detail=(
            None if outcome_kind == "succeeded" else "synthetic Part D failure"
        ),
    )
    if sealed:
        record = copy.deepcopy(record)
        record["record_status"] = "sealed"
    return record


def _assessment(
    stage3: Stage3AvailabilityIndex,
    *,
    evaluation=None,
    outcome_kind: str = "succeeded",
    attempt_index: int = 0,
    retry_index: int = 0,
    sealed: bool = False,
):
    attempt = _attempt(
        stage3,
        outcome_kind=outcome_kind,
        attempt_index=attempt_index,
        retry_index=retry_index,
        sealed=sealed,
    )
    return assess_soft_attempt(
        attempt_record=attempt,
        stage3=stage3,
        evaluation=evaluation if outcome_kind == "succeeded" else None,
    )


def _sealing(stage3: Stage3AvailabilityIndex, assessment):
    evaluation = assessment.eligibility.evaluation
    return generate_soft_sealing_evidence(
        assessment=assessment,
        stage3=stage3,
        checkpoint=assessment.attempt_record["payload"]["model_checkpoint"],
        dense_output=_artifact(
            path="synthetic/stage6c/centred-soft-output.bin",
            digest=evaluation.student_soft_output_sha256,
            storage_class="external_large_object",
        ),
        architecture_ref="synthetic-student-architecture/v1",
    )


def _validate_stage4(record, *, contract, stage3, registry, registry_sha) -> None:
    validate_part_m_record(
        record,
        contract=contract,
        stage3=stage3,
        stage3_registry=registry,
        stage3_registry_sha256=registry_sha,
    )


def test_full_domain_tolerance_pass_is_deterministic(stage3) -> None:
    first = _evaluation(stage3)
    second = _evaluation(stage3)

    assert first.total_count == 12_769
    assert first.discrepancy == 0.0
    assert first.tolerance_passed is True
    assert first.eligible is True
    assert first.to_mapping() == second.to_mapping()
    assert len(first.teacher_soft_output_sha256) == 64
    assert first.teacher_soft_output_sha256 == first.student_soft_output_sha256


def test_full_domain_size_and_sealed_output_boundary_are_enforced(stage3) -> None:
    policy = _policy(stage3, tolerance=0.25, argmax_required=None)
    teacher, student = _outputs(
        stage3,
        student_logits=_teacher_logits(),
    )
    short_student = replace(student, logits=student.logits[:-1])
    with pytest.raises(SoftEligibilityError, match="shapes must match"):
        evaluate_soft_target_eligibility(
            teacher=teacher,
            student=short_student,
            policy=policy,
            stage3=stage3,
        )
    with pytest.raises(SoftEligibilityError, match="requires sealed outputs"):
        replace(student, record_status="draft")


def test_full_domain_eligibility_is_gauge_invariant(stage3) -> None:
    policy = _policy(stage3, tolerance=0.0, argmax_required=True)
    teacher, student = _outputs(stage3, student_logits=_teacher_logits())
    baseline = evaluate_soft_target_eligibility(
        teacher=teacher,
        student=student,
        policy=policy,
        stage3=stage3,
    )
    shifted = evaluate_soft_target_eligibility(
        teacher=replace(teacher, logits=teacher.logits + 4.0),
        student=replace(student, logits=student.logits - 8.0),
        policy=policy,
        stage3=stage3,
    )

    assert shifted.discrepancy == baseline.discrepancy
    assert shifted.eligible == baseline.eligible
    assert shifted.teacher_soft_output_sha256 == baseline.teacher_soft_output_sha256
    assert shifted.student_soft_output_sha256 == baseline.student_soft_output_sha256


def test_tolerance_failure_remains_distinct(stage3) -> None:
    student = _teacher_logits() + torch.tensor([1.0, -1.0], dtype=torch.float64)
    evidence = _evaluation(stage3, tolerance=0.25, student_logits=student)
    assessment = _assessment(stage3, evaluation=evidence)

    assert evidence.discrepancy == 1.0
    assert evidence.tolerance_passed is False
    assert evidence.argmax_rule_passed is True
    assert assessment.failure_kinds == ("tolerance_failure",)


def test_exact_tolerance_boundary_passes_deterministically(stage3) -> None:
    student = _teacher_logits() + torch.tensor([1.0, -1.0], dtype=torch.float64)
    first = _evaluation(stage3, tolerance=1.0, student_logits=student)
    second = _evaluation(stage3, tolerance=1.0, student_logits=student.clone())

    assert first.discrepancy == first.tolerance == 1.0
    assert first.tolerance_passed is True
    assert first.eligible is True
    assert first.to_mapping() == second.to_mapping()


def test_argmax_rule_enabled_blocks_one_disagreement(stage3) -> None:
    student = _teacher_logits()
    student[0] = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    evidence = _evaluation(
        stage3,
        tolerance=1.0,
        argmax_required=True,
        student_logits=student,
    )
    assessment = _assessment(stage3, evaluation=evidence)

    assert evidence.argmax_requirement_applied is True
    assert evidence.argmax_agreement_count == 12_768
    assert evidence.tolerance_passed is True
    assert evidence.argmax_rule_passed is False
    assert assessment.failure_kinds == ("argmax_rule_failure",)


@pytest.mark.parametrize("argmax_required", [None, False])
def test_argmax_rule_disabled_does_not_gate_eligibility(
    stage3,
    argmax_required: bool | None,
) -> None:
    student = _teacher_logits()
    student[0] = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    evidence = _evaluation(
        stage3,
        tolerance=1.0,
        argmax_required=argmax_required,
        student_logits=student,
    )

    assert evidence.argmax_agreement_count == 12_768
    assert evidence.argmax_requirement_applied is False
    assert evidence.argmax_rule_passed is True
    assert evidence.eligible is True


def test_failure_taxonomy_and_permanent_attempt_accounting(stage3) -> None:
    training = _assessment(
        stage3,
        outcome_kind="exhausted_technical_stop",
    )
    numerical = _assessment(
        stage3,
        outcome_kind="numerical_failure",
        attempt_index=1,
    )
    tolerance = _assessment(
        stage3,
        evaluation=_evaluation(
            stage3,
            tolerance=0.25,
            student_logits=_teacher_logits()
            + torch.tensor([1.0, -1.0], dtype=torch.float64),
        ),
        attempt_index=2,
    )
    ledger = SoftAttemptLedger()
    for assessment in (training, numerical, tolerance):
        ledger.add(assessment)

    assert training.failure_kinds == ("training_failure",)
    assert numerical.failure_kinds == ("numerical_failure",)
    assert tolerance.failure_kinds == ("tolerance_failure",)
    assert ledger.attempt_count == 3
    assert ledger.failure_count("training_failure") == 1
    assert ledger.failure_count("numerical_failure") == 1
    assert ledger.failure_count("tolerance_failure") == 1
    with pytest.raises(Stage6CRecordError, match="replacement is forbidden"):
        ledger.add(training)
    assert ledger.attempt_count == 3


@pytest.mark.parametrize("eligible", [True, False])
def test_soft_eligibility_records_are_stage4_compatible_and_explicit(
    stage3,
    contract,
    registry,
    registry_sha,
    eligible: bool,
) -> None:
    if eligible:
        evaluation = _evaluation(stage3)
    else:
        evaluation = _evaluation(
            stage3,
            tolerance=0.25,
            student_logits=_teacher_logits()
            + torch.tensor([1.0, -1.0], dtype=torch.float64),
        )
    assessment = _assessment(stage3, evaluation=evaluation)
    record_evidence = assessment.eligibility
    assert record_evidence is not None

    _validate_stage4(
        record_evidence.stage4_record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )
    payload = record_evidence.stage4_record["payload"]
    assert payload["criterion"] == "soft_policy_reference"
    assert payload["soft_policy_ref"] == evaluation.policy_ref
    assert payload["eligibility_status"] == ("passed" if eligible else "failed")
    sidecar = record_evidence.to_mapping()["soft_policy_and_output_evidence"]
    assert sidecar["policy_sha256"] == evaluation.policy_sha256
    assert sidecar["student_soft_output_sha256"] == (
        evaluation.student_soft_output_sha256
    )


def test_soft_and_hard_attempt_identities_cannot_be_confused(stage3) -> None:
    hard_attempt = _attempt(stage3, condition="hard_target")
    with pytest.raises(Stage6CRecordError, match="only depth-4 soft_target"):
        assess_soft_attempt(
            attempt_record=hard_attempt,
            stage3=stage3,
            evaluation=_evaluation(stage3),
        )

    teacher, student = _outputs(stage3, student_logits=_teacher_logits())
    wrong_role = replace(teacher, role="soft_target_student")
    with pytest.raises(SoftEligibilityError, match="soft_target_teacher"):
        evaluate_soft_target_eligibility(
            teacher=wrong_role,
            student=student,
            policy=_policy(stage3, tolerance=0.25, argmax_required=None),
            stage3=stage3,
        )


def test_only_passed_sealed_hash_consistent_attempt_can_seal(
    stage3,
    contract,
    registry,
    registry_sha,
) -> None:
    passed = _assessment(
        stage3,
        evaluation=_evaluation(stage3),
        sealed=True,
    )
    sealing = _sealing(stage3, passed)
    _validate_stage4(
        sealing.stage4_record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )
    assert sealing.dense_output_sha256 == (
        passed.eligibility.evaluation.student_soft_output_sha256
    )

    failed = _assessment(
        stage3,
        evaluation=_evaluation(
            stage3,
            tolerance=0.0,
            student_logits=_teacher_logits()
            + torch.tensor([1.0, -1.0], dtype=torch.float64),
        ),
        sealed=True,
    )
    with pytest.raises(Stage6CRecordError, match="only eligible"):
        _sealing(stage3, failed)

    draft = _assessment(stage3, evaluation=_evaluation(stage3), sealed=False)
    with pytest.raises(Stage6CRecordError, match="draft attempts"):
        _sealing(stage3, draft)

    with pytest.raises(Stage6CRecordError, match="dense-output hash"):
        generate_soft_sealing_evidence(
            assessment=passed,
            stage3=stage3,
            checkpoint=passed.attempt_record["payload"]["model_checkpoint"],
            dense_output=_artifact(
                path="synthetic/stage6c/wrong.bin",
                digest="f" * 64,
                storage_class="external_large_object",
            ),
            architecture_ref="synthetic-student-architecture/v1",
        )


def test_narrow_soft_release_gate_blocks_failed_missing_draft_and_mismatch(
    stage3,
) -> None:
    failed = _assessment(
        stage3,
        evaluation=_evaluation(
            stage3,
            tolerance=0.0,
            student_logits=_teacher_logits()
            + torch.tensor([1.0, -1.0], dtype=torch.float64),
        ),
        sealed=True,
    )
    assert soft_circuit_release_gate(assessment=failed, sealing=None).allowed is False

    passed = _assessment(stage3, evaluation=_evaluation(stage3), sealed=True)
    missing = soft_circuit_release_gate(assessment=passed, sealing=None)
    assert missing.allowed is False
    assert missing.reason == "sealing_evidence_missing"

    valid = _sealing(stage3, passed)
    draft = _assessment(stage3, evaluation=_evaluation(stage3), sealed=False)
    assert soft_circuit_release_gate(
        assessment=draft,
        sealing=valid,
    ).reason == "attempt_not_sealed"

    checkpoint_mismatch = replace(valid, checkpoint_sha256="f" * 64)
    assert soft_circuit_release_gate(
        assessment=passed,
        sealing=checkpoint_mismatch,
    ).reason == "checkpoint_hash_mismatch"

    dense_mismatch = replace(valid, dense_output_sha256="f" * 64)
    assert soft_circuit_release_gate(
        assessment=passed,
        sealing=dense_mismatch,
    ).reason == "dense_output_hash_mismatch"


def test_valid_soft_evidence_passes_narrow_gate(stage3) -> None:
    passed = _assessment(stage3, evaluation=_evaluation(stage3), sealed=True)
    sealing = _sealing(stage3, passed)

    decision = soft_circuit_release_gate(assessment=passed, sealing=sealing)

    assert decision.allowed is True
    assert decision.reason == "eligible_sealed_hash_consistent"

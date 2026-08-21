from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

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
from circuit_families.stage6b import (
    CanonicalDecisionVector,
    HardAttemptLedger,
    Stage6BRecordError,
    assess_hard_attempt,
    circuit_release_gate,
    evaluate_hard_target_eligibility,
    generate_hard_sealing_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
VOCAB_PATH = ROOT / "followup/configs/stage4_common_vocabulary_v1.json"
IDENTITY_PATH = ROOT / "followup/configs/stage4_condition_identity_spec_v1.json"
ORDERING_REF = "modular-addition-full-domain-order/v1"
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


@pytest.fixture(scope="module")
def decisions() -> tuple[int, ...]:
    return tuple(index % 113 for index in range(FULL_DOMAIN_EXAMPLE_COUNT))


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


def _artifact(
    *,
    path: str,
    digest: str,
    storage_class: str,
) -> dict[str, str]:
    return {
        "path": path,
        "sha256": digest * 64,
        "storage_class": storage_class,
    }


def _attempt(
    stage3: Stage3AvailabilityIndex,
    *,
    outcome_kind: str,
    attempt_index: int = 0,
    retry_index: int = 0,
    sealed: bool = False,
):
    identity = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=0,
        attempt_index=attempt_index,
        retry_index=retry_index,
    )
    checkpoint = _artifact(
        path=f"synthetic/stage6b/student-a{attempt_index}-r{retry_index}.pt",
        digest="b",
        storage_class="external_checkpoint",
    )
    record = emit_technical_attempt_record(
        stage3=stage3,
        attempt_identity=identity,
        target_cache_reference={
            "record_type": "teacher_output_cache",
            "schema_version": "teacher_output_cache/v1",
            "condition_id": _condition_id(stage3, condition="hard_target"),
            "record_sha256": "c" * 64,
        },
        outcome_kind=outcome_kind,
        student_architecture_ref="synthetic-student-architecture/v1",
        replication_policy_ref="synthetic-replication-policy/v1",
        training_config_ref="synthetic-training-config/v1",
        training_log=_artifact(
            path=f"synthetic/stage6b/train-a{attempt_index}-r{retry_index}.log",
            digest="d",
            storage_class="external_log",
        ),
        model_checkpoint=(checkpoint if outcome_kind == "succeeded" else None),
        failure_detail=(
            None if outcome_kind == "succeeded" else "synthetic Part D failure"
        ),
    )
    if sealed:
        record = copy.deepcopy(record)
        record["record_status"] = "sealed"
    return record


def _evaluation(
    stage3: Stage3AvailabilityIndex,
    attempt_record,
    decisions: tuple[int, ...],
    *,
    eligible: bool,
):
    student_decisions = decisions
    if not eligible:
        changed = list(decisions)
        changed[-1] = (changed[-1] + 1) % 113
        student_decisions = tuple(changed)
    return evaluate_hard_target_eligibility(
        teacher=CanonicalDecisionVector(
            role="direct_teacher",
            condition_id=_condition_id(stage3, condition="direct_teacher"),
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            decisions=decisions,
        ),
        student=CanonicalDecisionVector(
            role="hard_target_student",
            condition_id=attempt_record["condition_id"],
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            decisions=student_decisions,
        ),
        stage3=stage3,
    )


def _assessment(
    stage3: Stage3AvailabilityIndex,
    decisions: tuple[int, ...],
    *,
    outcome_kind: str = "succeeded",
    eligible: bool = True,
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
    evaluation = (
        _evaluation(stage3, attempt, decisions, eligible=eligible)
        if outcome_kind == "succeeded"
        else None
    )
    return assess_hard_attempt(
        attempt_record=attempt,
        stage3=stage3,
        evaluation=evaluation,
    )


def _validate_stage4(record, *, contract, stage3, registry, registry_sha) -> None:
    validate_part_m_record(
        record,
        contract=contract,
        stage3=stage3,
        stage3_registry=registry,
        stage3_registry_sha256=registry_sha,
    )


def _sealing(stage3, assessment):
    checkpoint = assessment.attempt_record["payload"].get(
        "model_checkpoint",
        _artifact(
            path="synthetic/stage6b/missing-student.pt",
            digest="b",
            storage_class="external_checkpoint",
        ),
    )
    return generate_hard_sealing_evidence(
        assessment=assessment,
        stage3=stage3,
        checkpoint=checkpoint,
        dense_output=_artifact(
            path="synthetic/stage6b/dense-output.bin",
            digest="e",
            storage_class="external_large_object",
        ),
        architecture_ref="synthetic-student-architecture/v1",
    )


def test_failure_taxonomy_branches_remain_distinct(stage3, decisions) -> None:
    training = _assessment(
        stage3,
        decisions,
        outcome_kind="exhausted_technical_stop",
    )
    numerical = _assessment(
        stage3,
        decisions,
        outcome_kind="numerical_failure",
        attempt_index=1,
    )
    subperfect = _assessment(stage3, decisions, eligible=False, attempt_index=2)

    assert training.classification == "training_failure"
    assert numerical.classification == "numerical_failure"
    assert subperfect.classification == "subperfect_agreement"
    assert training.eligibility is None
    assert numerical.eligibility is None
    assert subperfect.eligibility is not None
    assert subperfect.eligibility.failure_classification == "subperfect_agreement"


@pytest.mark.parametrize("eligible", [True, False])
def test_eligibility_record_and_sidecar_are_stage4_compatible(
    stage3,
    decisions,
    contract,
    registry,
    registry_sha,
    eligible: bool,
) -> None:
    assessment = _assessment(stage3, decisions, eligible=eligible)
    eligibility = assessment.eligibility
    assert eligibility is not None

    _validate_stage4(
        eligibility.stage4_record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )
    payload = eligibility.stage4_record["payload"]
    assert payload["eligibility_status"] == ("passed" if eligible else "failed")
    assert payload["teacher_argmax_agreement_count"] == (
        FULL_DOMAIN_EXAMPLE_COUNT if eligible else FULL_DOMAIN_EXAMPLE_COUNT - 1
    )
    sidecar = eligibility.to_mapping()["decision_and_order_evidence"]
    assert len(sidecar["teacher_decisions_sha256"]) == 64
    assert len(sidecar["student_decisions_sha256"]) == 64
    assert sidecar["ordered_input_ids_sha256"] == ORDERING_SHA256


def test_failed_attempt_remains_after_later_eligible_retry(stage3, decisions) -> None:
    ledger = HardAttemptLedger()
    failed = _assessment(
        stage3,
        decisions,
        outcome_kind="exhausted_technical_stop",
    )
    eligible = _assessment(
        stage3,
        decisions,
        retry_index=1,
        sealed=True,
    )

    ledger.add(failed)
    ledger.add(eligible)

    assert ledger.attempt_count == 2
    assert ledger.classification_count("training_failure") == 1
    assert ledger.classification_count("eligible") == 1
    assert [item.classification for item in ledger.assessments()] == [
        "training_failure",
        "eligible",
    ]
    with pytest.raises(Stage6BRecordError, match="replacement is forbidden"):
        ledger.add(failed)
    assert ledger.attempt_count == 2


def test_only_passed_sealed_attempt_generates_sealing_evidence(
    stage3,
    decisions,
    contract,
    registry,
    registry_sha,
) -> None:
    passed = _assessment(stage3, decisions, sealed=True)
    sealing = _sealing(stage3, passed)

    _validate_stage4(
        sealing.stage4_record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )
    assert sealing.checkpoint_sha256 == "b" * 64
    assert sealing.dense_output_sha256 == "e" * 64
    assert sealing.stage4_record["record_status"] == "sealed"

    failed_training = _assessment(
        stage3,
        decisions,
        outcome_kind="exhausted_technical_stop",
    )
    subperfect = _assessment(stage3, decisions, eligible=False, sealed=True)
    draft_pass = _assessment(stage3, decisions, sealed=False)
    for blocked in (failed_training, subperfect):
        with pytest.raises(Stage6BRecordError, match="only eligible"):
            _sealing(stage3, blocked)
    with pytest.raises(Stage6BRecordError, match="draft attempts"):
        _sealing(stage3, draft_pass)


def test_narrow_circuit_gate_blocks_incomplete_or_failed_evidence(
    stage3,
    decisions,
) -> None:
    failed = _assessment(stage3, decisions, eligible=False, sealed=True)
    assert circuit_release_gate(assessment=failed, sealing=None).allowed is False

    passed = _assessment(stage3, decisions, sealed=True)
    missing = circuit_release_gate(assessment=passed, sealing=None)
    assert missing.allowed is False
    assert missing.reason == "sealing_evidence_missing"

    valid = _sealing(stage3, passed)
    draft = _assessment(stage3, decisions, sealed=False)
    draft_decision = circuit_release_gate(assessment=draft, sealing=valid)
    assert draft_decision.allowed is False
    assert draft_decision.reason == "attempt_not_sealed"

    mismatched = replace(valid, dense_output_sha256="f" * 64)
    mismatch = circuit_release_gate(assessment=passed, sealing=mismatched)
    assert mismatch.allowed is False
    assert mismatch.reason == "dense_output_hash_mismatch"

    checkpoint_mismatched = replace(valid, checkpoint_sha256="f" * 64)
    checkpoint_mismatch = circuit_release_gate(
        assessment=passed,
        sealing=checkpoint_mismatched,
    )
    assert checkpoint_mismatch.allowed is False
    assert checkpoint_mismatch.reason == "checkpoint_hash_mismatch"


def test_valid_sealed_hash_consistent_evidence_passes_gate(stage3, decisions) -> None:
    passed = _assessment(stage3, decisions, sealed=True)
    sealing = _sealing(stage3, passed)

    decision = circuit_release_gate(assessment=passed, sealing=sealing)

    assert decision.allowed is True
    assert decision.reason == "eligible_sealed_hash_consistent"

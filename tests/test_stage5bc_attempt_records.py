from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage4_schema_common import (
    CommonSchemaContract,
    Stage4SchemaError,
)
from circuit_families.stage4_schema_records import (
    validate_part_m_record,
)
from circuit_families.stage5bc.attempt_records import (
    TECHNICAL_FAILURE_KINDS,
    TechnicalAttemptLedger,
    TechnicalAttemptRecordError,
    attempt_record_sha256,
    canonical_attempt_record_bytes,
    emit_technical_attempt_record,
    outcome_from_training_result,
)
from circuit_families.stage5bc.student_identity import (
    build_student_attempt_identity,
)
from circuit_families.stage5bc.student_trainer import (
    TechnicalTrainingResult,
)

ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = (
    ROOT / "followup/configs/stage4_common_vocabulary_v1.json"
)
IDENTITY_SPEC_PATH = (
    ROOT / "followup/configs/stage4_condition_identity_spec_v1.json"
)
REGISTRY_PATH = (
    ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
)


@pytest.fixture(scope="module")
def registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry_sha() -> str:
    return hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def stage3(registry):
    return Stage3AvailabilityIndex.from_registry(registry)


@pytest.fixture(scope="module")
def contract():
    vocab = json.loads(
        VOCAB_PATH.read_text(encoding="utf-8")
    )
    identity_spec = json.loads(
        IDENTITY_SPEC_PATH.read_text(encoding="utf-8")
    )
    return CommonSchemaContract.from_specs(
        vocab,
        identity_spec,
    )


def _identity(
    stage3,
    *,
    condition: str = "hard_target",
    initialization: int = 0,
    attempt: int = 0,
    retry: int = 0,
):
    return build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition=condition,
        student_initialization=initialization,
        attempt_index=attempt,
        retry_index=retry,
    )


def _cache_reference(
    stage3,
    *,
    condition: str = "hard_target",
    sha_char: str = "c",
):
    condition_id = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition=condition,
        ),
        stage3,
    )

    return {
        "record_type": "teacher_output_cache",
        "schema_version": "teacher_output_cache/v1",
        "condition_id": condition_id,
        "record_sha256": sha_char * 64,
    }


def _training_log(
    *,
    suffix: str = "0",
):
    return {
        "path": f"synthetic/stage5bc/attempt-{suffix}.log",
        "sha256": suffix[-1] * 64,
        "storage_class": "external_log",
    }


def _checkpoint(
    *,
    suffix: str = "a",
):
    return {
        "path": f"synthetic/stage5bc/student-{suffix}.pt",
        "sha256": suffix[-1] * 64,
        "storage_class": "external_checkpoint",
    }


def _emit_success(stage3, *, attempt: int = 0, retry: int = 0):
    return emit_technical_attempt_record(
        stage3=stage3,
        attempt_identity=_identity(
            stage3,
            attempt=attempt,
            retry=retry,
        ),
        target_cache_reference=_cache_reference(stage3),
        outcome_kind="succeeded",
        student_architecture_ref="technical-student-architecture/v1",
        replication_policy_ref="technical-replication-policy/v1",
        training_config_ref="technical-training-config/v1",
        training_log=_training_log(suffix=str(attempt + 1)),
        model_checkpoint=_checkpoint(suffix="a"),
    )


def _emit_failure(
    stage3,
    *,
    kind: str,
    attempt: int,
    retry: int = 0,
):
    return emit_technical_attempt_record(
        stage3=stage3,
        attempt_identity=_identity(
            stage3,
            attempt=attempt,
            retry=retry,
        ),
        target_cache_reference=_cache_reference(stage3),
        outcome_kind=kind,
        student_architecture_ref="technical-student-architecture/v1",
        replication_policy_ref="technical-replication-policy/v1",
        training_config_ref="technical-training-config/v1",
        training_log=_training_log(suffix=str(attempt + 1)),
        failure_detail=f"technical {kind} fixture",
    )


def _validate(
    record,
    *,
    contract,
    stage3,
    registry,
    registry_sha,
):
    validate_part_m_record(
        record,
        contract=contract,
        stage3=stage3,
        stage3_registry=registry,
        stage3_registry_sha256=registry_sha,
    )


def test_succeeded_record_is_stage4_compatible(
    stage3,
    contract,
    registry,
    registry_sha,
) -> None:
    record = _emit_success(stage3)

    _validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )

    assert record["record_type"] == "student_attempt"
    assert record["schema_version"] == "student_attempt/v1"
    assert record["identity_depth"] == 4
    assert record["record_status"] == "draft"
    assert record["payload"]["attempt_outcome"] == "succeeded"
    assert "model_checkpoint" in record["payload"]
    assert "failure_reason" not in record["payload"]


@pytest.mark.parametrize(
    "failure_kind",
    TECHNICAL_FAILURE_KINDS,
)
def test_all_required_failure_classes_are_distinct_and_stage4_compatible(
    stage3,
    contract,
    registry,
    registry_sha,
    failure_kind: str,
) -> None:
    index = TECHNICAL_FAILURE_KINDS.index(failure_kind) + 1
    record = _emit_failure(
        stage3,
        kind=failure_kind,
        attempt=index,
    )

    _validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )

    payload = record["payload"]

    assert payload["attempt_outcome"] == "failed"
    assert "model_checkpoint" not in payload
    assert payload["failure_reason"].startswith(
        f"technical_failure:v1:{failure_kind}:"
    )


def test_part_k_terminal_status_mapping() -> None:
    common = {
        "updates_completed": 1,
        "trajectory": (),
        "configuration_refs": {"trainer": "technical/v1"},
        "target_cache_kind": "teacher_argmax",
        "model_device": "cpu",
        "model_training_mode_restored": True,
    }

    succeeded = TechnicalTrainingResult(
        terminal_status="stop_rule_met",
        terminal_reason="injected_stop_rule",
        **common,
    )
    numerical = TechnicalTrainingResult(
        terminal_status="nonfinite_failure",
        terminal_reason="nonfinite_loss",
        **common,
    )
    exhausted = TechnicalTrainingResult(
        terminal_status="technical_step_limit_exhausted",
        terminal_reason="mandatory_technical_safety_step_limit_reached",
        **common,
    )

    assert outcome_from_training_result(succeeded) == (
        "succeeded",
        None,
    )
    assert outcome_from_training_result(numerical) == (
        "numerical_failure",
        "nonfinite_loss",
    )
    assert outcome_from_training_result(exhausted) == (
        "exhausted_technical_stop",
        "mandatory_technical_safety_step_limit_reached",
    )


def test_failure_requires_reason_and_forbids_checkpoint(stage3) -> None:
    identity = _identity(stage3, attempt=1)

    common = {
        "stage3": stage3,
        "attempt_identity": identity,
        "target_cache_reference": _cache_reference(stage3),
        "outcome_kind": "numerical_failure",
        "student_architecture_ref": "technical-student-architecture/v1",
        "replication_policy_ref": "technical-replication-policy/v1",
        "training_config_ref": "technical-training-config/v1",
        "training_log": _training_log(suffix="1"),
    }

    with pytest.raises(
        TechnicalAttemptRecordError,
        match="requires failure_detail",
    ):
        emit_technical_attempt_record(**common)

    with pytest.raises(
        TechnicalAttemptRecordError,
        match="cannot carry model_checkpoint",
    ):
        emit_technical_attempt_record(
            **common,
            failure_detail="nonfinite loss",
            model_checkpoint=_checkpoint(),
        )


def test_success_requires_checkpoint_and_forbids_failure_detail(stage3) -> None:
    identity = _identity(stage3)

    common = {
        "stage3": stage3,
        "attempt_identity": identity,
        "target_cache_reference": _cache_reference(stage3),
        "outcome_kind": "succeeded",
        "student_architecture_ref": "technical-student-architecture/v1",
        "replication_policy_ref": "technical-replication-policy/v1",
        "training_config_ref": "technical-training-config/v1",
        "training_log": _training_log(suffix="1"),
    }

    with pytest.raises(
        TechnicalAttemptRecordError,
        match="requires model_checkpoint",
    ):
        emit_technical_attempt_record(**common)

    with pytest.raises(
        TechnicalAttemptRecordError,
        match="cannot carry failure_detail",
    ):
        emit_technical_attempt_record(
            **common,
            model_checkpoint=_checkpoint(),
            failure_detail="must not exist",
        )


def test_wrong_target_cache_condition_is_rejected(stage3) -> None:
    identity = _identity(
        stage3,
        condition="hard_target",
    )

    with pytest.raises(
        TechnicalAttemptRecordError,
        match="does not share student distillation condition",
    ):
        emit_technical_attempt_record(
            stage3=stage3,
            attempt_identity=identity,
            target_cache_reference=_cache_reference(
                stage3,
                condition="soft_target",
            ),
            outcome_kind="succeeded",
            student_architecture_ref="technical-student-architecture/v1",
            replication_policy_ref="technical-replication-policy/v1",
            training_config_ref="technical-training-config/v1",
            training_log=_training_log(suffix="1"),
            model_checkpoint=_checkpoint(),
        )


def test_attempt_record_bytes_and_hash_are_deterministic(stage3) -> None:
    first = _emit_success(stage3)
    second = copy.deepcopy(first)

    second = dict(reversed(list(second.items())))

    assert (
        canonical_attempt_record_bytes(first)
        == canonical_attempt_record_bytes(second)
    )
    assert attempt_record_sha256(first) == attempt_record_sha256(second)


def test_failed_attempt_remains_counted_after_later_success(stage3) -> None:
    ledger = TechnicalAttemptLedger()

    failed = _emit_failure(
        stage3,
        kind="interruption",
        attempt=0,
        retry=0,
    )
    succeeded = _emit_success(
        stage3,
        attempt=0,
        retry=1,
    )

    ledger.add(failed)
    ledger.add(succeeded)

    assert ledger.attempt_count == 2
    assert ledger.failed_attempt_count == 1
    assert ledger.succeeded_attempt_count == 1

    records = ledger.records()

    assert len(records) == 2
    assert any(
        item["payload"]["attempt_outcome"] == "failed"
        for item in records
    )


def test_duplicate_attempt_identity_cannot_replace_existing_record(
    stage3,
) -> None:
    ledger = TechnicalAttemptLedger()
    failed = _emit_failure(
        stage3,
        kind="invalid_input",
        attempt=2,
    )

    ledger.add(failed)

    with pytest.raises(
        TechnicalAttemptRecordError,
        match="replacement is forbidden",
    ):
        ledger.add(copy.deepcopy(failed))

    assert ledger.attempt_count == 1
    assert ledger.failed_attempt_count == 1


def test_seed_evidence_cannot_be_reassigned_to_another_identity(
    stage3,
) -> None:
    ledger = TechnicalAttemptLedger()

    original = _emit_failure(
        stage3,
        kind="configuration_rejection",
        attempt=3,
    )
    ledger.add(original)

    reassigned = copy.deepcopy(original)
    new_identity = _identity(
        stage3,
        initialization=1,
        attempt=3,
    )
    reassigned["condition_id"] = new_identity.condition_id

    with pytest.raises(
        TechnicalAttemptRecordError,
        match="seed evidence cannot be reused",
    ):
        ledger.add(reassigned)

    assert ledger.attempt_count == 1


def test_stage4_validator_rejects_silent_identity_reassignment(
    stage3,
    contract,
    registry,
    registry_sha,
) -> None:
    record = _emit_failure(
        stage3,
        kind="configuration_rejection",
        attempt=4,
    )

    reassigned = copy.deepcopy(record)
    reassigned["condition_id"] = _identity(
        stage3,
        initialization=1,
        attempt=4,
    ).condition_id

    with pytest.raises(Stage4SchemaError):
        _validate(
            reassigned,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )


def test_record_surface_contains_no_eligibility_or_sealed_model(stage3) -> None:
    record = _emit_success(stage3)
    rendered = json.dumps(record, sort_keys=True)

    assert "student_eligibility" not in rendered
    assert "sealed_dense_model" not in rendered

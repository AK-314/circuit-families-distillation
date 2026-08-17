from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage4_schema_common import (
    CommonSchemaContract,
    Stage4SchemaError,
)
from circuit_families.stage4_schema_records import (
    COMPONENT_ABLATION_SOURCE,
    MASKS_SOURCE,
    STAGE8_MASKING_MANIFEST,
    validate_part_m_record,
)
from circuit_families.stage4_seed_derivation import (
    SeedInputs,
    derive_seed,
)

ROOT = Path(__file__).resolve().parents[1]

VOCAB_PATH = ROOT / "followup/configs/stage4_common_vocabulary_v1.json"
IDENTITY_SPEC_PATH = (
    ROOT / "followup/configs/stage4_condition_identity_spec_v1.json"
)
REGISTRY_PATH = (
    ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
)
PRED_LINK_PATH = ROOT / "followup/manifests/predecessor_link_v1.json"


@pytest.fixture(scope="module")
def vocab():
    return json.loads(VOCAB_PATH.read_text())


@pytest.fixture(scope="module")
def identity_spec():
    return json.loads(IDENTITY_SPEC_PATH.read_text())


@pytest.fixture(scope="module")
def registry():
    return json.loads(REGISTRY_PATH.read_text())


@pytest.fixture(scope="module")
def registry_sha():
    return hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def stage3(registry):
    return Stage3AvailabilityIndex.from_registry(registry)


@pytest.fixture(scope="module")
def contract(vocab, identity_spec):
    return CommonSchemaContract.from_specs(vocab, identity_spec)


def condition_id(
    stage3,
    *,
    teacher_seed=1,
    phase="stable post-grokking",
    condition="hard_target",
    initialization=None,
):
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=teacher_seed,
            phase=phase,
            distillation_condition=condition,
            student_initialization=initialization,
        ),
        stage3,
    )


def record_ref(vocab, record_type, condition_id, char="a"):
    return {
        "record_type": record_type,
        "schema_version": vocab["schema_versions"][record_type],
        "condition_id": condition_id,
        "record_sha256": char * 64,
    }


def envelope(
    vocab,
    identity_spec,
    *,
    record_type,
    condition_id,
    payload,
    lane,
    status="draft",
):
    return {
        "namespace": vocab["namespace"],
        "vocabulary_version": vocab["vocabulary_version"],
        "schema_version": vocab["schema_versions"][record_type],
        "record_type": record_type,
        "record_status": status,
        "condition_id": condition_id,
        "identity_depth": identity_spec[
            "record_type_required_depths"
        ][record_type],
        "payload": payload,
        "provenance": {
            "producer_lane": lane,
            "creation_stage": "stage4",
            "source_records": [],
        },
    }


def teacher_record(
    *,
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    teacher_seed=1,
    phase="stable post-grokking",
):
    source = next(
        item
        for item in registry["records"]
        if (
            item["teacher_seed"] == teacher_seed
            and item["phase_label"] == phase
        )
    )

    cid = condition_id(
        stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        condition="direct_teacher",
    )

    payload = {
        "stage3_registry_path": (
            "followup/manifests/stage3_teacher_registry_v1.json"
        ),
        "stage3_registry_sha256": registry_sha,
        "stage3_registry_namespace": (
            "circuit-families-distillation/stage3-teacher-registry"
        ),
        "stage3_record_schema_version": "1",
        "canonical_run_id": source["canonical_run_id"],
        "training_step": source.get("training_step", 0),
    }

    if source["availability_status"] == "selected":
        payload["checkpoint"] = {
            "path": source["checkpoint_path"],
            "sha256": source["checkpoint_sha256"],
            "storage_class": "external_checkpoint",
        }
    else:
        payload["checkpoint"] = {
            "path": "synthetic/unavailable.pt",
            "sha256": "f" * 64,
            "storage_class": "external_checkpoint",
        }

    return envelope(
        vocab,
        identity_spec,
        record_type="teacher_reference",
        condition_id=cid,
        payload=payload,
        lane="lane_a",
    )


def cache_record(
    *,
    vocab,
    identity_spec,
    stage3,
    condition="hard_target",
    teacher_seed=1,
    phase="stable post-grokking",
):
    teacher_id = condition_id(
        stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        condition="direct_teacher",
    )
    cache_id = condition_id(
        stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        condition=condition,
    )

    kind = (
        "teacher_argmax"
        if condition == "hard_target"
        else "teacher_logits"
    )

    return envelope(
        vocab,
        identity_spec,
        record_type="teacher_output_cache",
        condition_id=cache_id,
        payload={
            "teacher_reference": record_ref(
                vocab,
                "teacher_reference",
                teacher_id,
                "1",
            ),
            "cache_kind": kind,
            "example_ordering_ref": "synthetic-ordering/v1",
            "example_count": 12769,
            "artifact": {
                "path": f"synthetic/{condition}.bin",
                "sha256": "2" * 64,
                "storage_class": "external_large_object",
            },
        },
        lane="lane_b",
    )


def seed_dict(stage3, student_id, purpose, attempt=0, retry=0):
    evidence = derive_seed(
        SeedInputs(
            condition_id=student_id,
            purpose=purpose,
            attempt_index=attempt,
            retry_index=retry,
        ),
        stage3,
    )

    return {
        "seed_derivation_version": evidence.seed_derivation_version,
        "seed_material": evidence.seed_material,
        "digest_sha256": evidence.digest_sha256,
        "selected_bytes_hex": evidence.selected_bytes_hex,
        "seed_value": evidence.seed_value,
    }


def attempt_record(
    *,
    vocab,
    identity_spec,
    stage3,
    condition="hard_target",
    initialization=0,
    attempt=0,
    retry=0,
    outcome="succeeded",
):
    cache_id = condition_id(
        stage3,
        condition=condition,
    )
    student_id = condition_id(
        stage3,
        condition=condition,
        initialization=initialization,
    )

    payload = {
        "target_cache": record_ref(
            vocab,
            "teacher_output_cache",
            cache_id,
            "3",
        ),
        "attempt_index": attempt,
        "retry_index": retry,
        "attempt_outcome": outcome,
        "student_architecture_ref": "synthetic-student-architecture/v1",
        "replication_policy_ref": "synthetic-replication-policy/v1",
        "training_config_ref": "synthetic-training/v1",
        "training_seed": seed_dict(
            stage3,
            student_id,
            "training",
            attempt,
            retry,
        ),
        "tie_breaking_seed": seed_dict(
            stage3,
            student_id,
            "tie_breaking",
            attempt,
            retry,
        ),
        "training_log": {
            "path": "synthetic/student.log",
            "sha256": "4" * 64,
            "storage_class": "external_log",
        },
    }

    if outcome == "succeeded":
        payload["model_checkpoint"] = {
            "path": "synthetic/student.pt",
            "sha256": "5" * 64,
            "storage_class": "external_checkpoint",
        }
    else:
        payload["failure_reason"] = "synthetic failure"

    return envelope(
        vocab,
        identity_spec,
        record_type="student_attempt",
        condition_id=student_id,
        payload=payload,
        lane="lane_b",
    )


def hard_eligibility_record(
    *,
    vocab,
    identity_spec,
    stage3,
    agreement=12769,
    status="passed",
):
    student_id = condition_id(
        stage3,
        condition="hard_target",
        initialization=0,
    )

    return envelope(
        vocab,
        identity_spec,
        record_type="student_eligibility",
        condition_id=student_id,
        payload={
            "attempt_reference": record_ref(
                vocab,
                "student_attempt",
                student_id,
                "6",
            ),
            "attempt_index": 0,
            "retry_index": 0,
            "eligibility_status": status,
            "criterion": "exact_teacher_argmax_agreement",
            "evaluation_example_count": 12769,
            "teacher_argmax_agreement_count": agreement,
        },
        lane="lane_b",
    )


def soft_eligibility_record(
    *,
    vocab,
    identity_spec,
    stage3,
    status="pending_policy",
):
    student_id = condition_id(
        stage3,
        condition="soft_target",
        initialization=0,
    )

    return envelope(
        vocab,
        identity_spec,
        record_type="student_eligibility",
        condition_id=student_id,
        payload={
            "attempt_reference": record_ref(
                vocab,
                "student_attempt",
                student_id,
                "7",
            ),
            "attempt_index": 0,
            "retry_index": 0,
            "eligibility_status": status,
            "criterion": "soft_policy_reference",
            "soft_policy_ref": "synthetic-soft-policy/v1",
        },
        lane="lane_b",
    )


def sealed_record(
    *,
    vocab,
    identity_spec,
    stage3,
):
    pred = json.loads(PRED_LINK_PATH.read_text())

    student_id = condition_id(
        stage3,
        condition="hard_target",
        initialization=0,
    )

    return envelope(
        vocab,
        identity_spec,
        record_type="sealed_dense_model",
        condition_id=student_id,
        payload={
            "eligibility_reference": record_ref(
                vocab,
                "student_eligibility",
                student_id,
                "8",
            ),
            "eligibility_status": "passed",
            "architecture_ref": "synthetic-student-architecture/v1",
            "component_basis": {
                "component_count": 516,
                "status": pred["component_basis"]["status"],
                "masks_source": MASKS_SOURCE,
                "component_ablation_source": COMPONENT_ABLATION_SOURCE,
                "stage8_masking_manifest": STAGE8_MASKING_MANIFEST,
            },
            "model_checkpoint": {
                "path": "synthetic/student.pt",
                "sha256": "5" * 64,
                "storage_class": "external_checkpoint",
            },
        },
        lane="lane_b",
    )


def validate(
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


def test_valid_selected_teacher_reference(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    record = teacher_record(
        vocab=vocab,
        identity_spec=identity_spec,
        registry=registry,
        registry_sha=registry_sha,
        stage3=stage3,
    )

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )


@pytest.mark.parametrize("phase", ["pre-grokking", "50%"])
def test_unavailable_stage3_cells_cannot_create_teacher_reference(
    stage3,
    phase,
):
    with pytest.raises(
        ConditionIdentityError,
        match="unavailable Stage 3 cell cannot form downstream condition identity",
    ):
        condition_id(
            stage3,
            teacher_seed=0,
            phase=phase,
            condition="direct_teacher",
        )


def test_teacher_reference_checkpoint_hash_must_match_stage3(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    record = teacher_record(
        vocab=vocab,
        identity_spec=identity_spec,
        registry=registry,
        registry_sha=registry_sha,
        stage3=stage3,
    )
    record["payload"]["checkpoint"]["sha256"] = "0" * 64

    with pytest.raises(
        Stage4SchemaError,
        match="checkpoint does not match Stage 3 record",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )


@pytest.mark.parametrize(
    "condition",
    ["hard_target", "soft_target"],
)
def test_valid_teacher_output_cache(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
    condition,
):
    record = cache_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        condition=condition,
    )

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )


def test_hard_soft_cache_ids_are_distinct(
    vocab,
    identity_spec,
    stage3,
):
    hard = cache_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        condition="hard_target",
    )
    soft = cache_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        condition="soft_target",
    )

    assert hard["condition_id"] != soft["condition_id"]


def test_cache_cannot_reference_wrong_teacher_phase(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    record = cache_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        condition="hard_target",
    )

    wrong_teacher_id = condition_id(
        stage3,
        teacher_seed=2,
        condition="direct_teacher",
    )
    record["payload"]["teacher_reference"] = record_ref(
        vocab,
        "teacher_reference",
        wrong_teacher_id,
        "9",
    )

    with pytest.raises(
        Stage4SchemaError,
        match="does not share teacher seed/phase",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )


@pytest.mark.parametrize(
    "outcome",
    ["succeeded", "failed"],
)
def test_valid_student_attempt_outcomes(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
    outcome,
):
    record = attempt_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        outcome=outcome,
    )

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )


def test_failed_attempt_is_structurally_countable_without_checkpoint(
    vocab,
    identity_spec,
    stage3,
):
    record = attempt_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        outcome="failed",
    )

    assert record["payload"]["attempt_outcome"] == "failed"
    assert "failure_reason" in record["payload"]
    assert "model_checkpoint" not in record["payload"]


def test_attempt_retry_distinct_from_student_initialization(
    vocab,
    identity_spec,
    stage3,
):
    first = attempt_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        initialization=0,
        attempt=0,
        retry=0,
    )
    retry = attempt_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        initialization=0,
        attempt=0,
        retry=1,
    )

    assert first["condition_id"] == retry["condition_id"]
    assert (
        first["payload"]["training_seed"]["seed_value"]
        != retry["payload"]["training_seed"]["seed_value"]
    )


def test_attempt_seed_must_match_attempt_coordinate(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    record = attempt_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        attempt=0,
        retry=0,
    )

    record["payload"]["attempt_index"] = 1

    with pytest.raises(
        Stage4SchemaError,
        match="seed attempt_index mismatch",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )


def test_attempt_training_and_tie_seed_purposes_cannot_swap(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    record = attempt_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
    )

    record["payload"]["training_seed"] = copy.deepcopy(
        record["payload"]["tie_breaking_seed"]
    )

    with pytest.raises(
        Stage4SchemaError,
        match="training seed purpose mismatch",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )


def test_hard_eligibility_exact_pass(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    record = hard_eligibility_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        agreement=12769,
        status="passed",
    )

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )


@pytest.mark.parametrize("agreement", [0, 1, 12768])
def test_hard_eligibility_nonperfect_agreement_cannot_pass(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
    agreement,
):
    record = hard_eligibility_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        agreement=agreement,
        status="passed",
    )

    with pytest.raises(
        Stage4SchemaError,
        match="12769/12769",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )


def test_hard_eligibility_nonperfect_agreement_can_be_failed(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    record = hard_eligibility_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
        agreement=12768,
        status="failed",
    )

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )


def test_soft_policy_reference_is_structurally_representable(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    record = soft_eligibility_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
    )

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tolerance", 0.01),
        ("temperature", 1.0),
        ("loss_threshold", 0.5),
    ],
)
def test_soft_numeric_policy_values_cannot_be_frozen_in_stage4(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
    field,
    value,
):
    record = soft_eligibility_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
    )
    record["payload"][field] = value

    with pytest.raises(
        Stage4SchemaError,
        match="keys mismatch",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )


def test_hard_and_soft_eligibility_references_cannot_mix(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    record = hard_eligibility_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
    )

    soft_student_id = condition_id(
        stage3,
        condition="soft_target",
        initialization=0,
    )
    record["payload"]["attempt_reference"] = record_ref(
        vocab,
        "student_attempt",
        soft_student_id,
        "a",
    )

    with pytest.raises(
        Stage4SchemaError,
        match="does not share distillation condition",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )


def test_valid_sealed_dense_model(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    record = sealed_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
    )

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
    )


def test_sealed_model_requires_passing_eligibility(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    record = sealed_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
    )
    record["payload"]["eligibility_status"] = "failed"

    with pytest.raises(
        Stage4SchemaError,
        match="requires passing eligibility",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("component_count", 515),
        ("component_count", 517),
        ("status", "new_basis"),
    ],
)
def test_sealed_model_component_basis_contract_rejected_when_changed(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
    field,
    replacement,
):
    record = sealed_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
    )
    record["payload"]["component_basis"][field] = replacement

    with pytest.raises(Stage4SchemaError):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )


@pytest.mark.parametrize(
    "path",
    [
        "/private/student.pt",
        "../student.pt",
        "synthetic\\student.pt",
    ],
)
def test_large_artifact_paths_must_be_portable(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
    path,
):
    record = sealed_record(
        vocab=vocab,
        identity_spec=identity_spec,
        stage3=stage3,
    )
    record["payload"]["model_checkpoint"]["path"] = path

    with pytest.raises(
        Stage4SchemaError,
        match="portable relative POSIX path",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )


def test_part_m_schemas_do_not_accept_unknown_record_type(
    vocab,
    identity_spec,
    registry,
    registry_sha,
    stage3,
    contract,
):
    teacher = teacher_record(
        vocab=vocab,
        identity_spec=identity_spec,
        registry=registry,
        registry_sha=registry_sha,
        stage3=stage3,
    )
    teacher["record_type"] = "analysis_freeze"
    teacher["schema_version"] = "analysis_freeze/v1"
    teacher["identity_depth"] = 2
    teacher["condition_id"] = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
        ),
        stage3,
    )

    with pytest.raises(Stage4SchemaError):
        validate(
            teacher,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
        )

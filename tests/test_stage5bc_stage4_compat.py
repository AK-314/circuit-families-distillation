from __future__ import annotations

import copy
import hashlib
import json
import runpy
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
    validate_part_m_record,
)
from circuit_families.stage5bc.attempt_records import (
    emit_technical_attempt_record,
)
from circuit_families.stage5bc.job_dag import (
    TechnicalJobRegistry,
    build_job_node,
)
from circuit_families.stage5bc.student_identity import (
    build_student_attempt_identity,
)

ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
)
VOCAB_PATH = (
    ROOT / "followup/configs/stage4_common_vocabulary_v1.json"
)
IDENTITY_SPEC_PATH = (
    ROOT / "followup/configs/stage4_condition_identity_spec_v1.json"
)

REGISTRY = json.loads(
    REGISTRY_PATH.read_text(encoding="utf-8")
)
REGISTRY_SHA = hashlib.sha256(
    REGISTRY_PATH.read_bytes()
).hexdigest()
VOCAB = json.loads(
    VOCAB_PATH.read_text(encoding="utf-8")
)
IDENTITY_SPEC = json.loads(
    IDENTITY_SPEC_PATH.read_text(encoding="utf-8")
)

STAGE3 = Stage3AvailabilityIndex.from_registry(REGISTRY)
CONTRACT = CommonSchemaContract.from_specs(
    VOCAB,
    IDENTITY_SPEC,
)

STAGE4_RECORD_HELPERS = runpy.run_path(
    str(ROOT / "tests/test_stage4_schema_records_part_m.py")
)
STAGE4_GRAPH_HELPERS = runpy.run_path(
    str(ROOT / "tests/test_stage4_schema_graph.py")
)


def _condition_id(
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
        STAGE3,
    )


def _cache_reference(
    *,
    condition: str,
    digest_character: str,
) -> dict[str, str]:
    return {
        "record_type": "teacher_output_cache",
        "schema_version": VOCAB["schema_versions"][
            "teacher_output_cache"
        ],
        "condition_id": _condition_id(
            condition=condition,
        ),
        "record_sha256": digest_character * 64,
    }


def _artifact(
    *,
    path: str,
    digest_character: str,
    storage_class: str,
) -> dict[str, str]:
    return {
        "path": path,
        "sha256": digest_character * 64,
        "storage_class": storage_class,
    }


def _attempt(
    *,
    condition: str,
    outcome: str,
    digest_character: str,
) -> dict:
    identity = build_student_attempt_identity(
        stage3=STAGE3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition=condition,
        student_initialization=0,
        attempt_index=0,
        retry_index=0,
    )

    return emit_technical_attempt_record(
        stage3=STAGE3,
        attempt_identity=identity,
        target_cache_reference=_cache_reference(
            condition=condition,
            digest_character=digest_character,
        ),
        outcome_kind=outcome,
        student_architecture_ref="synthetic-student-architecture/v1",
        replication_policy_ref="synthetic-replication-policy/v1",
        training_config_ref="synthetic-training/v1",
        training_log=_artifact(
            path=f"synthetic/{condition}/train.log",
            digest_character="a",
            storage_class="external_log",
        ),
        model_checkpoint=(
            _artifact(
                path=f"synthetic/{condition}/student.pt",
                digest_character="b",
                storage_class="external_checkpoint",
            )
            if outcome == "succeeded"
            else None
        ),
        failure_detail=(
            None
            if outcome == "succeeded"
            else "synthetic technical failure"
        ),
    )


def _validate_stage4_record(record: dict) -> None:
    validate_part_m_record(
        record,
        contract=CONTRACT,
        stage3=STAGE3,
        stage3_registry=REGISTRY,
        stage3_registry_sha256=REGISTRY_SHA,
    )


def test_stage3_population_remains_exactly_15_13_2() -> None:
    records = REGISTRY["records"]

    selected = [
        item
        for item in records
        if item["availability_status"] == "selected"
    ]
    unavailable = [
        item
        for item in records
        if item["availability_status"] == "unavailable"
    ]

    assert len(records) == 15
    assert len(selected) == 13
    assert len(unavailable) == 2

    assert {
        (
            item["teacher_seed"],
            item["phase_label"],
        )
        for item in unavailable
    } == {
        (0, "pre-grokking"),
        (0, "50%"),
    }


@pytest.mark.parametrize(
    "phase",
    [
        "pre-grokking",
        "50%",
    ],
)
def test_unavailable_stage3_cells_still_cannot_spawn_downstream_identity(
    phase: str,
) -> None:
    with pytest.raises(
        ConditionIdentityError,
        match="unavailable Stage 3 cell",
    ):
        build_condition_id(
            ConditionIdentity(
                teacher_seed=0,
                phase=phase,
                distillation_condition="hard_target",
            ),
            STAGE3,
        )


def test_existing_stage4_hard_and_soft_cache_records_validate() -> None:
    cache_record = STAGE4_RECORD_HELPERS["cache_record"]
    validate = STAGE4_RECORD_HELPERS["validate"]

    hard = cache_record(
        vocab=VOCAB,
        identity_spec=IDENTITY_SPEC,
        stage3=STAGE3,
        condition="hard_target",
    )
    soft = cache_record(
        vocab=VOCAB,
        identity_spec=IDENTITY_SPEC,
        stage3=STAGE3,
        condition="soft_target",
    )

    validate(
        hard,
        contract=CONTRACT,
        stage3=STAGE3,
        registry=REGISTRY,
        registry_sha=REGISTRY_SHA,
    )
    validate(
        soft,
        contract=CONTRACT,
        stage3=STAGE3,
        registry=REGISTRY,
        registry_sha=REGISTRY_SHA,
    )

    assert hard["condition_id"] != soft["condition_id"]
    assert hard["payload"]["cache_kind"] == "teacher_argmax"
    assert soft["payload"]["cache_kind"] == "teacher_logits"


@pytest.mark.parametrize(
    "condition",
    [
        "hard_target",
        "soft_target",
    ],
)
def test_new_success_attempt_emitter_is_accepted_by_stage4_validator(
    condition: str,
) -> None:
    record = _attempt(
        condition=condition,
        outcome="succeeded",
        digest_character=(
            "1" if condition == "hard_target" else "2"
        ),
    )

    _validate_stage4_record(record)

    assert record["record_type"] == "student_attempt"
    assert record["payload"]["attempt_outcome"] == "succeeded"
    assert "model_checkpoint" in record["payload"]
    assert "failure_reason" not in record["payload"]


@pytest.mark.parametrize(
    "condition",
    [
        "hard_target",
        "soft_target",
    ],
)
def test_new_failure_attempt_emitter_is_visible_to_stage4_validator(
    condition: str,
) -> None:
    record = _attempt(
        condition=condition,
        outcome="numerical_failure",
        digest_character=(
            "3" if condition == "hard_target" else "4"
        ),
    )

    _validate_stage4_record(record)

    assert record["record_type"] == "student_attempt"
    assert record["payload"]["attempt_outcome"] == "failed"
    assert record["payload"]["failure_reason"]
    assert "model_checkpoint" not in record["payload"]


def test_hard_soft_student_attempt_identity_and_seed_evidence_remain_distinct() -> None:
    hard = build_student_attempt_identity(
        stage3=STAGE3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=0,
        attempt_index=0,
        retry_index=0,
    )
    soft = build_student_attempt_identity(
        stage3=STAGE3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="soft_target",
        student_initialization=0,
        attempt_index=0,
        retry_index=0,
    )

    assert hard.condition_id != soft.condition_id
    assert hard.training_seed.seed_value != soft.training_seed.seed_value
    assert (
        hard.tie_breaking_seed.seed_value
        != soft.tie_breaking_seed.seed_value
    )


def test_adversarial_attempt_cache_condition_mismatch_is_rejected_by_stage4() -> None:
    record = _attempt(
        condition="hard_target",
        outcome="succeeded",
        digest_character="5",
    )

    record["payload"]["target_cache"] = _cache_reference(
        condition="soft_target",
        digest_character="6",
    )

    with pytest.raises(Stage4SchemaError):
        _validate_stage4_record(record)


def test_adversarial_failure_cannot_hide_as_success() -> None:
    record = _attempt(
        condition="hard_target",
        outcome="numerical_failure",
        digest_character="7",
    )

    record["payload"]["attempt_outcome"] = "succeeded"

    with pytest.raises(Stage4SchemaError):
        _validate_stage4_record(record)


def test_complete_accepted_stage4_graph_still_validates() -> None:
    records, _ = STAGE4_GRAPH_HELPERS["graph"]()
    STAGE4_GRAPH_HELPERS["validate"](records)


def test_stage4_graph_blocks_discovery_when_eligibility_is_not_sealed() -> None:
    records, hashes = STAGE4_GRAPH_HELPERS["graph"]()

    records[hashes["eligibility"]]["record_status"] = "draft"

    with pytest.raises(Stage4SchemaError):
        STAGE4_GRAPH_HELPERS["validate"](records)


def test_new_dag_preserves_future_eligibility_before_discovery_boundary() -> None:
    student_id = _condition_id(
        condition="hard_target",
        initialization=0,
    )

    identity_spec = IDENTITY_SPEC
    complete_id = identity_spec["synthetic_test_vectors"][
        "complete_a"
    ]

    cache = build_job_node(
        stage3=STAGE3,
        node_type="teacher_cache",
        condition_id=_condition_id(
            condition="hard_target",
        ),
    )
    training = build_job_node(
        stage3=STAGE3,
        node_type="training",
        condition_id=student_id,
        dependencies=(cache.job_id,),
    )
    technical_completion = build_job_node(
        stage3=STAGE3,
        node_type="technical_completion",
        condition_id=student_id,
        dependencies=(training.job_id,),
    )
    eligibility = build_job_node(
        stage3=STAGE3,
        node_type="future_eligibility",
        condition_id=student_id,
        dependencies=(technical_completion.job_id,),
    )
    discovery = build_job_node(
        stage3=STAGE3,
        node_type="discovery",
        condition_id=complete_id,
        dependencies=(eligibility.job_id,),
    )

    registry = TechnicalJobRegistry(
        stage3=STAGE3,
        nodes=(
            cache,
            training,
            technical_completion,
            eligibility,
            discovery,
        ),
    )

    ordered = registry.topological_nodes()
    positions = {
        node.job_id: index
        for index, node in enumerate(ordered)
    }

    assert discovery.dependencies == (eligibility.job_id,)
    assert positions[eligibility.job_id] < positions[discovery.job_id]
    assert eligibility.execution_allowed is False
    assert discovery.execution_allowed is False


def test_new_dag_rejects_discovery_without_future_eligibility_parent() -> None:
    student_id = _condition_id(
        condition="hard_target",
        initialization=0,
    )
    complete_id = IDENTITY_SPEC["synthetic_test_vectors"][
        "complete_a"
    ]

    cache = build_job_node(
        stage3=STAGE3,
        node_type="teacher_cache",
        condition_id=_condition_id(
            condition="hard_target",
        ),
    )
    training = build_job_node(
        stage3=STAGE3,
        node_type="training",
        condition_id=student_id,
        dependencies=(cache.job_id,),
    )
    technical_completion = build_job_node(
        stage3=STAGE3,
        node_type="technical_completion",
        condition_id=student_id,
        dependencies=(training.job_id,),
    )
    discovery = build_job_node(
        stage3=STAGE3,
        node_type="discovery",
        condition_id=complete_id,
        dependencies=(technical_completion.job_id,),
    )

    with pytest.raises(
        ValueError,
        match="discovery cannot depend on technical_completion",
    ):
        TechnicalJobRegistry(
            stage3=STAGE3,
            nodes=(
                cache,
                training,
                technical_completion,
                discovery,
            ),
        )


def test_stage4_schema_adversarial_mutation_does_not_require_schema_change() -> None:
    records, hashes = STAGE4_GRAPH_HELPERS["graph"]()
    mutated = copy.deepcopy(records)

    mutated[hashes["attempt"]]["record_status"] = "draft"

    with pytest.raises(Stage4SchemaError):
        STAGE4_GRAPH_HELPERS["validate"](mutated)

    STAGE4_GRAPH_HELPERS["validate"](records)

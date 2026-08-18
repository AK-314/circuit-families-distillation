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
from circuit_families.stage4_schema_analysis import (
    FIREWALL_PATH,
    FIREWALL_SHA256,
    validate_part_q_record,
)
from circuit_families.stage4_schema_common import (
    CommonSchemaContract,
    Stage4SchemaError,
)

ROOT = Path(__file__).resolve().parents[1]

VOCAB_PATH = ROOT / "followup/configs/stage4_common_vocabulary_v1.json"
IDENTITY_PATH = (
    ROOT / "followup/configs/stage4_condition_identity_spec_v1.json"
)
REGISTRY_PATH = (
    ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
)
DECISIONS_PATH = (
    ROOT / "followup/configs/stage2_unresolved_decisions_v1.json"
)


@pytest.fixture(scope="module")
def vocab():
    return json.loads(VOCAB_PATH.read_text())


@pytest.fixture(scope="module")
def identity_spec():
    return json.loads(IDENTITY_PATH.read_text())


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


@pytest.fixture(scope="module")
def unresolved():
    data = json.loads(DECISIONS_PATH.read_text())
    wanted = {f"UD-{n:03d}" for n in range(3, 15)}
    found = set()

    def walk(value):
        if isinstance(value, dict):
            ident = value.get("decision_id")
            if ident in wanted and value.get("status") == "unresolved":
                found.add(ident)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)

    assert found == wanted
    return found


def cid2(stage3, seed=1, phase="stable post-grokking"):
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=seed,
            phase=phase,
        ),
        stage3,
    )


def cid3(
    stage3,
    *,
    seed=1,
    phase="stable post-grokking",
    condition="hard_target",
):
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=seed,
            phase=phase,
            distillation_condition=condition,
        ),
        stage3,
    )


def cid4(
    stage3,
    *,
    seed=1,
    phase="stable post-grokking",
    condition="hard_target",
    initialization=0,
):
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=seed,
            phase=phase,
            distillation_condition=condition,
            student_initialization=initialization,
        ),
        stage3,
    )


def cid8(
    stage3,
    *,
    seed=1,
    phase="stable post-grokking",
    condition="hard_target",
    initialization=0,
):
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=seed,
            phase=phase,
            distillation_condition=condition,
            student_initialization=initialization,
            discovery_method="synthetic-method/v1",
            fidelity_setting="synthetic-fidelity/v1",
            component_cap="synthetic-cap/v1",
            overlap_setting="synthetic-overlap/v1",
        ),
        stage3,
    )


def ref(vocab, record_type, condition_id, char="a"):
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


def validate(
    record,
    *,
    contract,
    stage3,
    registry,
    registry_sha,
    unresolved,
):
    validate_part_q_record(
        record,
        contract=contract,
        stage3=stage3,
        stage3_registry=registry,
        stage3_registry_sha256=registry_sha,
        current_unresolved_decision_ids=unresolved,
    )


def summary_record(
    vocab,
    identity_spec,
    stage3,
    *,
    condition="hard_target",
):
    cell_id = cid3(stage3, condition=condition)
    student_id = cid4(stage3, condition=condition)
    endpoint_id = cid8(stage3, condition=condition)

    return envelope(
        vocab,
        identity_spec,
        record_type="student_cell_summary",
        condition_id=cell_id,
        lane="lane_d",
        payload={
            "population_unit": "teacher_seed",
            "summary_rule_ref": "synthetic-summary/v1",
            "minimum_eligible_students_ref": "synthetic-minimum/v1",
            "cell_analysis_status": "unresolved",
            "eligible_student_count": 1,
            "failed_attempt_count": 1,
            "missing_student_count": 1,
            "eligible_student_initializations": [0],
            "failed_attempts": [
                {
                    "student_initialization": 0,
                    "attempt_index": 1,
                    "retry_index": 0,
                    "attempt_reference": ref(
                        vocab,
                        "student_attempt",
                        student_id,
                        "1",
                    ),
                },
            ],
            "missing_student_initializations": [1],
            "endpoint_records": [
                ref(
                    vocab,
                    "endpoint_record",
                    endpoint_id,
                    "2",
                ),
            ],
        },
    )


def inventory_record(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    *,
    seed,
    phase,
):
    source = next(
        item
        for item in registry["records"]
        if item["teacher_seed"] == seed and item["phase_label"] == phase
    )

    inventory_id = cid2(stage3, seed=seed, phase=phase)

    if source["availability_status"] == "unavailable":
        state = "unavailable"
        reason = source["unavailable_reason"]
        summaries = []
    else:
        state = "selected"
        reason = None
        summaries = [
            ref(
                vocab,
                "student_cell_summary",
                cid3(
                    stage3,
                    seed=seed,
                    phase=phase,
                    condition="hard_target",
                ),
                "3",
            ),
            ref(
                vocab,
                "student_cell_summary",
                cid3(
                    stage3,
                    seed=seed,
                    phase=phase,
                    condition="soft_target",
                ),
                "4",
            ),
        ]

    return envelope(
        vocab,
        identity_spec,
        record_type="teacher_seed_inventory",
        condition_id=inventory_id,
        lane="lane_d",
        payload={
            "population_unit": "teacher_seed",
            "stage3_registry_path": (
                "followup/manifests/stage3_teacher_registry_v1.json"
            ),
            "stage3_registry_sha256": registry_sha,
            "stage3_availability_state": source["availability_status"],
            "cell_state": state,
            "unavailable_reason": reason,
            "student_cell_summaries": summaries,
        },
    )


def excluded_record(vocab, identity_spec, stage3):
    return envelope(
        vocab,
        identity_spec,
        record_type="excluded_development_output",
        condition_id=cid3(stage3),
        lane="lane_c",
        payload={
            "firewall_authority": {
                "path": FIREWALL_PATH,
                "sha256": FIREWALL_SHA256,
            },
            "excluded_register_path": (
                "followup/manifests/"
                "stage2_excluded_development_register_v1.json"
            ),
            "development_purpose_ref": "synthetic-pilot/v1",
            "lifecycle_state": "excluded",
            "primary_analysis_eligible": False,
            "scientific_selection_use_allowed": False,
            "regeneration_required": True,
            "regeneration_status": "pending",
            "artifact": {
                "path": "synthetic/pilot.bin",
                "sha256": "5" * 64,
                "storage_class": "external_large_object",
            },
        },
    )


def reproduction_record(vocab, identity_spec, stage3):
    endpoint_id = cid8(stage3)

    return envelope(
        vocab,
        identity_spec,
        record_type="reproduction_comparison",
        condition_id=cid3(stage3),
        lane="lane_d",
        payload={
            "comparison_rule_ref": "synthetic-reproduction/v1",
            "source_record": ref(
                vocab,
                "endpoint_record",
                endpoint_id,
                "6",
            ),
            "reproduced_record": ref(
                vocab,
                "endpoint_record",
                endpoint_id,
                "7",
            ),
            "semantic_match": True,
        },
    )


def freeze_record(
    vocab,
    identity_spec,
    stage3,
    unresolved,
    *,
    production_ready=False,
    firewall_clear=False,
):
    teacher_phase_id = cid2(stage3)
    hard_id = cid3(stage3)

    return envelope(
        vocab,
        identity_spec,
        record_type="analysis_freeze",
        condition_id=teacher_phase_id,
        lane="joint",
        status="sealed",
        payload={
            "population_unit": "teacher_seed",
            "analysis_contract_ref": "synthetic-analysis/v1",
            "firewall_authority": {
                "path": FIREWALL_PATH,
                "sha256": FIREWALL_SHA256,
            },
            "firewall_clear": firewall_clear,
            "unresolved_decision_ids": sorted(unresolved),
            "production_ready": production_ready,
            "primary_input_records": [
                ref(
                    vocab,
                    "teacher_seed_inventory",
                    teacher_phase_id,
                    "8",
                ),
                ref(
                    vocab,
                    "student_cell_summary",
                    hard_id,
                    "9",
                ),
            ],
            "excluded_development_records": [
                ref(
                    vocab,
                    "excluded_development_output",
                    hard_id,
                    "a",
                ),
            ],
        },
    )


def test_student_cell_summary_population_unit_teacher_seed(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = summary_record(vocab, identity_spec, stage3)
    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
        unresolved=unresolved,
    )
    assert record["payload"]["population_unit"] == "teacher_seed"


@pytest.mark.parametrize("condition", ["hard_target", "soft_target"])
def test_hard_soft_summary_each_valid_separately(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
    condition,
):
    record = summary_record(
        vocab,
        identity_spec,
        stage3,
        condition=condition,
    )
    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
        unresolved=unresolved,
    )


def test_hard_summary_cannot_contain_soft_endpoint(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = summary_record(vocab, identity_spec, stage3)
    record["payload"]["endpoint_records"][0] = ref(
        vocab,
        "endpoint_record",
        cid8(stage3, condition="soft_target"),
        "b",
    )

    with pytest.raises(
        Stage4SchemaError,
        match="distillation-condition prefix",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eligible_student_count", 2),
        ("failed_attempt_count", 2),
        ("missing_student_count", 2),
    ],
)
def test_summary_visibility_counts_must_match_lists(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
    field,
    value,
):
    record = summary_record(vocab, identity_spec, stage3)
    record["payload"][field] = value

    with pytest.raises(Stage4SchemaError):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_summary_eligible_and_missing_initializations_disjoint(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = summary_record(vocab, identity_spec, stage3)
    record["payload"]["missing_student_initializations"] = [0]

    with pytest.raises(
        Stage4SchemaError,
        match="must be disjoint",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_failed_attempt_coordinate_must_be_unique(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = summary_record(vocab, identity_spec, stage3)
    duplicate = copy.deepcopy(record["payload"]["failed_attempts"][0])
    record["payload"]["failed_attempts"].append(duplicate)
    record["payload"]["failed_attempt_count"] = 2

    with pytest.raises(
        Stage4SchemaError,
        match="coordinates must be unique",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_inventory_contract_validates_all_15_stage3_cells(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    records = []

    for seed in registry["canonical_seed_order"]:
        for phase in registry["canonical_phase_order"]:
            record = inventory_record(
                vocab,
                identity_spec,
                stage3,
                registry,
                registry_sha,
                seed=seed,
                phase=phase,
            )
            validate(
                record,
                contract=contract,
                stage3=stage3,
                registry=registry,
                registry_sha=registry_sha,
                unresolved=unresolved,
            )
            records.append(record)

    assert len(records) == 15
    assert sum(
        record["payload"]["cell_state"] == "unavailable"
        for record in records
    ) == 2


def test_inventory_explicitly_retains_seed0_pre_unavailable(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = inventory_record(
        vocab,
        identity_spec,
        stage3,
        registry,
        registry_sha,
        seed=0,
        phase="pre-grokking",
    )

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
        unresolved=unresolved,
    )

    assert record["payload"]["cell_state"] == "unavailable"
    assert record["payload"]["student_cell_summaries"] == []


def test_inventory_explicitly_retains_seed0_50_unavailable(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = inventory_record(
        vocab,
        identity_spec,
        stage3,
        registry,
        registry_sha,
        seed=0,
        phase="50%",
    )

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
        unresolved=unresolved,
    )

    assert record["payload"]["cell_state"] == "unavailable"
    assert record["payload"]["student_cell_summaries"] == []


@pytest.mark.parametrize("phase", ["pre-grokking", "50%"])
def test_unavailable_seed0_cannot_spawn_summary_identity(
    stage3,
    phase,
):
    with pytest.raises(
        ConditionIdentityError,
        match="unavailable Stage 3 cell cannot form downstream",
    ):
        cid3(
            stage3,
            seed=0,
            phase=phase,
            condition="hard_target",
        )


def test_selected_inventory_cannot_be_reclassified_unavailable(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = inventory_record(
        vocab,
        identity_spec,
        stage3,
        registry,
        registry_sha,
        seed=1,
        phase="stable post-grokking",
    )
    record["payload"]["cell_state"] = "unavailable"
    record["payload"]["unavailable_reason"] = "synthetic"

    with pytest.raises(
        Stage4SchemaError,
        match="cannot be reclassified unavailable",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_unavailable_inventory_reason_must_match_stage3(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = inventory_record(
        vocab,
        identity_spec,
        stage3,
        registry,
        registry_sha,
        seed=0,
        phase="pre-grokking",
    )
    record["payload"]["unavailable_reason"] = "wrong reason"

    with pytest.raises(
        Stage4SchemaError,
        match="must match Stage 3 unavailable record",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_excluded_development_output_valid_pending(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = excluded_record(vocab, identity_spec, stage3)

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
        unresolved=unresolved,
    )


def test_regenerated_development_output_remains_excluded(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = excluded_record(vocab, identity_spec, stage3)
    record["payload"]["regeneration_status"] = "regenerated"

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
        unresolved=unresolved,
    )

    assert record["payload"]["primary_analysis_eligible"] is False
    assert record["payload"]["scientific_selection_use_allowed"] is False


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("primary_analysis_eligible", True),
        ("scientific_selection_use_allowed", True),
        ("regeneration_required", False),
    ],
)
def test_excluded_development_firewall_flags_cannot_relax(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
    field,
    bad,
):
    record = excluded_record(vocab, identity_spec, stage3)
    record["payload"][field] = bad

    with pytest.raises(Stage4SchemaError):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_excluded_development_firewall_hash_is_frozen(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = excluded_record(vocab, identity_spec, stage3)
    record["payload"]["firewall_authority"]["sha256"] = "0" * 64

    with pytest.raises(
        Stage4SchemaError,
        match="firewall_authority SHA-256 mismatch",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_reproduction_comparison_requires_both_hashed_records(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = reproduction_record(vocab, identity_spec, stage3)

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
        unresolved=unresolved,
    )

    assert "record_sha256" in record["payload"]["source_record"]
    assert "record_sha256" in record["payload"]["reproduced_record"]


def test_reproduction_requires_same_record_type(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = reproduction_record(vocab, identity_spec, stage3)
    record["payload"]["reproduced_record"] = ref(
        vocab,
        "discovery_run",
        cid8(stage3),
        "7",
    )

    with pytest.raises(
        Stage4SchemaError,
        match="same record_type",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_reproduction_requires_same_condition_identity(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = reproduction_record(vocab, identity_spec, stage3)
    record["payload"]["reproduced_record"]["condition_id"] = cid8(
        stage3,
        initialization=1,
    )

    with pytest.raises(
        Stage4SchemaError,
        match="same condition_id",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_reproduction_semantic_mismatch_requires_note(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = reproduction_record(vocab, identity_spec, stage3)
    record["payload"]["semantic_match"] = False

    with pytest.raises(
        Stage4SchemaError,
        match="requires discrepancy_note",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_reproduction_semantic_mismatch_with_note_is_valid(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = reproduction_record(vocab, identity_spec, stage3)
    record["payload"]["semantic_match"] = False
    record["payload"]["discrepancy_note"] = "synthetic discrepancy"

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
        unresolved=unresolved,
    )


def test_analysis_freeze_valid_not_ready_while_decisions_unresolved(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = freeze_record(
        vocab,
        identity_spec,
        stage3,
        unresolved,
    )

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
        unresolved=unresolved,
    )

    assert record["payload"]["production_ready"] is False


def test_analysis_freeze_cannot_claim_ready_with_ud003_ud014(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = freeze_record(
        vocab,
        identity_spec,
        stage3,
        unresolved,
        production_ready=True,
        firewall_clear=True,
    )

    with pytest.raises(
        Stage4SchemaError,
        match="unresolved decisions remain",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_analysis_freeze_requires_exact_unresolved_register(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = freeze_record(
        vocab,
        identity_spec,
        stage3,
        unresolved,
    )
    record["payload"]["unresolved_decision_ids"] = ["UD-003"]

    with pytest.raises(
        Stage4SchemaError,
        match="must match the supplied decision authority",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_analysis_freeze_excluded_output_cannot_be_primary_input(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = freeze_record(
        vocab,
        identity_spec,
        stage3,
        unresolved,
    )
    record["payload"]["primary_input_records"].append(
        ref(
            vocab,
            "excluded_development_output",
            cid3(stage3),
            "b",
        )
    )

    with pytest.raises(
        Stage4SchemaError,
        match="cannot be a primary input",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_analysis_freeze_primary_inputs_must_share_teacher_phase(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    unresolved,
    contract,
):
    record = freeze_record(
        vocab,
        identity_spec,
        stage3,
        unresolved,
    )

    record["payload"]["primary_input_records"][1] = ref(
        vocab,
        "student_cell_summary",
        cid3(
            stage3,
            seed=2,
            phase="stable post-grokking",
        ),
        "9",
    )

    with pytest.raises(
        Stage4SchemaError,
        match="teacher/phase prefix",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=unresolved,
        )


def test_analysis_freeze_later_ready_state_is_structurally_representable(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    contract,
):
    resolved = set()

    record = freeze_record(
        vocab,
        identity_spec,
        stage3,
        resolved,
        production_ready=True,
        firewall_clear=True,
    )

    validate(
        record,
        contract=contract,
        stage3=stage3,
        registry=registry,
        registry_sha=registry_sha,
        unresolved=resolved,
    )

    assert record["payload"]["production_ready"] is True
    assert record["payload"]["unresolved_decision_ids"] == []


def test_analysis_freeze_not_ready_when_firewall_not_clear_even_if_resolved(
    vocab,
    identity_spec,
    stage3,
    registry,
    registry_sha,
    contract,
):
    resolved = set()

    record = freeze_record(
        vocab,
        identity_spec,
        stage3,
        resolved,
        production_ready=True,
        firewall_clear=False,
    )

    with pytest.raises(
        Stage4SchemaError,
        match="firewall_clear=false",
    ):
        validate(
            record,
            contract=contract,
            stage3=stage3,
            registry=registry,
            registry_sha=registry_sha,
            unresolved=resolved,
        )

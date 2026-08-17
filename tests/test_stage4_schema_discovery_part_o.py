from __future__ import annotations

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
from circuit_families.stage4_schema_discovery import (
    validate_part_o_record,
)
from circuit_families.stage4_seed_derivation import (
    SeedInputs,
    derive_seed,
)

ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "followup/configs/stage4_common_vocabulary_v1.json"
IDENTITY_PATH = (
    ROOT / "followup/configs/stage4_condition_identity_spec_v1.json"
)
REGISTRY_PATH = (
    ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
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
def stage3(registry):
    return Stage3AvailabilityIndex.from_registry(registry)


@pytest.fixture(scope="module")
def contract(vocab, identity_spec):
    return CommonSchemaContract.from_specs(vocab, identity_spec)


def student_id(
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


def discovery_id(
    stage3,
    *,
    seed=1,
    phase="stable post-grokking",
    condition="hard_target",
    initialization=0,
    method="synthetic-method/v1",
    fidelity="synthetic-fidelity/v1",
    cap="synthetic-cap/v1",
    overlap="synthetic-overlap/v1",
):
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=seed,
            phase=phase,
            distillation_condition=condition,
            student_initialization=initialization,
            discovery_method=method,
            fidelity_setting=fidelity,
            component_cap=cap,
            overlap_setting=overlap,
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
    *,
    record_type,
    condition_id,
    payload,
    status="draft",
):
    return {
        "namespace": vocab["namespace"],
        "vocabulary_version": vocab["vocabulary_version"],
        "schema_version": vocab["schema_versions"][record_type],
        "record_type": record_type,
        "record_status": status,
        "condition_id": condition_id,
        "identity_depth": 8,
        "payload": payload,
        "provenance": {
            "producer_lane": "lane_c",
            "creation_stage": "stage4",
            "source_records": [],
        },
    }


def seed_evidence(stage3, condition_id, attempt=0, retry=0):
    evidence = derive_seed(
        SeedInputs(
            condition_id=condition_id,
            purpose="discovery",
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


def discovery_record(vocab, stage3, *, condition_id=None):
    condition_id = condition_id or discovery_id(stage3)
    sid = student_id(stage3)

    return envelope(
        vocab,
        record_type="discovery_run",
        condition_id=condition_id,
        payload={
            "sealed_dense_model": ref(
                vocab,
                "sealed_dense_model",
                sid,
                "1",
            ),
            "attempt_index": 0,
            "retry_index": 0,
            "discovery_method_ref": "synthetic-method/v1",
            "method_budget_ref": "synthetic-budget/v1",
            "fidelity_definition_ref": "synthetic-fidelity/v1",
            "component_cap_ref": "synthetic-cap/v1",
            "overlap_setting_ref": "synthetic-overlap/v1",
            "discovery_seed": seed_evidence(
                stage3,
                condition_id,
            ),
            "search_artifact": {
                "path": "synthetic/search.bin",
                "sha256": "2" * 64,
                "storage_class": "external_large_object",
            },
        },
    )


def native_record(vocab, stage3):
    cid = discovery_id(stage3)

    return envelope(
        vocab,
        record_type="native_budget_ledger",
        condition_id=cid,
        payload={
            "discovery_run": ref(
                vocab,
                "discovery_run",
                cid,
                "3",
            ),
            "method_budget_ref": "synthetic-budget/v1",
            "native_unit_ref": "synthetic-native-unit/v1",
            "native_budget_limit": 10,
            "native_units_consumed": 7,
            "entries": [
                {
                    "sequence_index": 0,
                    "operation_ref": "synthetic-operation/v1",
                    "native_units_charged": 3,
                },
                {
                    "sequence_index": 1,
                    "operation_ref": "synthetic-operation/v1",
                    "native_units_charged": 4,
                },
            ],
        },
    )


def exact_record(vocab, stage3):
    cid = discovery_id(stage3)
    sid = student_id(stage3)

    return envelope(
        vocab,
        record_type="exact_mask_evaluation_ledger",
        condition_id=cid,
        payload={
            "sealed_dense_model": ref(
                vocab,
                "sealed_dense_model",
                sid,
                "1",
            ),
            "discovery_run": ref(
                vocab,
                "discovery_run",
                cid,
                "3",
            ),
            "fidelity_definition_ref": "synthetic-fidelity/v1",
            "exact_evaluation_allowance_ref": (
                "synthetic-exact-allowance/v1"
            ),
            "exact_evaluation_allowance": 4,
            "exact_evaluation_count": 3,
            "charged_evaluation_count": 2,
            "intact_baseline_present": True,
            "entries": [
                {
                    "evaluation_order": 0,
                    "mask_sha256": "4" * 64,
                    "mask_kind": "intact",
                    "retained_count": 516,
                    "retained_proportion": 1.0,
                    "fidelity_value": 1.0,
                    "qualifies": True,
                    "budget_charged": False,
                },
                {
                    "evaluation_order": 1,
                    "mask_sha256": "5" * 64,
                    "mask_kind": "candidate",
                    "retained_count": 258,
                    "retained_proportion": 0.5,
                    "fidelity_value": 0.95,
                    "qualifies": True,
                    "budget_charged": True,
                },
                {
                    "evaluation_order": 2,
                    "mask_sha256": "6" * 64,
                    "mask_kind": "candidate",
                    "retained_count": 206,
                    "retained_proportion": 206 / 516,
                    "fidelity_value": 0.90,
                    "qualifies": False,
                    "budget_charged": True,
                },
            ],
        },
    )


def endpoint_record(vocab, stage3, *, endpoint1=0.5, packing=1):
    cid = discovery_id(stage3)

    packed = ["5" * 64] if packing == 1 else []

    qualifying = "4" * 64 if endpoint1 == 1.0 else "5" * 64

    return envelope(
        vocab,
        record_type="endpoint_record",
        condition_id=cid,
        status="sealed",
        payload={
            "exact_ledger": ref(
                vocab,
                "exact_mask_evaluation_ledger",
                cid,
                "7",
            ),
            "fidelity_definition_ref": "synthetic-fidelity/v1",
            "component_cap_ref": "synthetic-cap/v1",
            "overlap_setting_ref": "synthetic-overlap/v1",
            "endpoint_1": {
                "smallest_recovered_component_proportion": endpoint1,
                "qualifying_mask_sha256": qualifying,
                "procedure_censored": endpoint1 == 1.0,
                "interpretation": (
                    "smallest_qualifying_proportion_in_exact_ledger"
                ),
                "global_minimum_claim": False,
            },
            "endpoint_2": {
                "packing_lower_bound": packing,
                "packed_mask_sha256s": packed,
                "packing_rule_ref": "synthetic-packing-rule/v1",
                "interpretation": (
                    "procedure_dependent_packing_lower_bound"
                ),
                "true_packing_number_claim": False,
            },
        },
    )


def validate(record, *, contract, stage3):
    validate_part_o_record(
        record,
        contract=contract,
        stage3=stage3,
    )


def test_valid_discovery_run(vocab, stage3, contract):
    validate(
        discovery_record(vocab, stage3),
        contract=contract,
        stage3=stage3,
    )


def test_discovery_uses_depth4_sealed_model_prefix(
    vocab,
    stage3,
    contract,
):
    record = discovery_record(vocab, stage3)

    assert record["payload"]["sealed_dense_model"]["condition_id"] == (
        student_id(stage3)
    )

    validate(record, contract=contract, stage3=stage3)


def test_discovery_rejects_wrong_student_initialization_prefix(
    vocab,
    stage3,
    contract,
):
    record = discovery_record(vocab, stage3)

    record["payload"]["sealed_dense_model"]["condition_id"] = student_id(
        stage3,
        initialization=1,
    )

    with pytest.raises(
        Stage4SchemaError,
        match="depth-4 prefix",
    ):
        validate(record, contract=contract, stage3=stage3)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("discovery_method_ref", "other-method/v1"),
        ("fidelity_definition_ref", "other-fidelity/v1"),
        ("component_cap_ref", "other-cap/v1"),
        ("overlap_setting_ref", "other-overlap/v1"),
    ],
)
def test_discovery_identity_setting_refs_must_match(
    vocab,
    stage3,
    contract,
    field,
    replacement,
):
    record = discovery_record(vocab, stage3)
    record["payload"][field] = replacement

    with pytest.raises(
        Stage4SchemaError,
        match="canonical condition-identity value",
    ):
        validate(record, contract=contract, stage3=stage3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method_budget_ref", "1000"),
        ("method_budget_ref", "greedy"),
    ],
)
def test_discovery_budget_is_version_reference_not_raw_value(
    vocab,
    stage3,
    contract,
    field,
    value,
):
    record = discovery_record(vocab, stage3)
    record["payload"][field] = value

    with pytest.raises(
        Stage4SchemaError,
        match="version-reference grammar",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_discovery_seed_retry_must_match(vocab, stage3, contract):
    record = discovery_record(vocab, stage3)
    record["payload"]["retry_index"] = 1

    with pytest.raises(
        Stage4SchemaError,
        match="discovery seed retry_index mismatch",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_discovery_seed_cannot_use_training_purpose(
    vocab,
    stage3,
    contract,
):
    record = discovery_record(vocab, stage3)
    cid = record["condition_id"]

    with pytest.raises(ValueError):
        derive_seed(
            SeedInputs(
                condition_id=cid,
                purpose="training",
                attempt_index=0,
                retry_index=0,
            ),
            stage3,
        )


def test_valid_native_budget_ledger(vocab, stage3, contract):
    validate(
        native_record(vocab, stage3),
        contract=contract,
        stage3=stage3,
    )


def test_native_ledger_cannot_contain_exact_count(
    vocab,
    stage3,
    contract,
):
    record = native_record(vocab, stage3)
    record["payload"]["exact_evaluation_count"] = 2

    with pytest.raises(
        Stage4SchemaError,
        match="keys mismatch",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_native_charge_sum_must_match(vocab, stage3, contract):
    record = native_record(vocab, stage3)
    record["payload"]["native_units_consumed"] = 8

    with pytest.raises(
        Stage4SchemaError,
        match="sum of ledger charges",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_native_budget_cannot_be_exceeded(vocab, stage3, contract):
    record = native_record(vocab, stage3)
    record["payload"]["native_budget_limit"] = 6

    with pytest.raises(
        Stage4SchemaError,
        match="cannot exceed native_budget_limit",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_native_sequence_must_be_contiguous(vocab, stage3, contract):
    record = native_record(vocab, stage3)
    record["payload"]["entries"][1]["sequence_index"] = 2

    with pytest.raises(
        Stage4SchemaError,
        match="contiguous from zero",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_valid_exact_ledger(vocab, stage3, contract):
    validate(
        exact_record(vocab, stage3),
        contract=contract,
        stage3=stage3,
    )


def test_exact_ledger_requires_intact_baseline_flag(
    vocab,
    stage3,
    contract,
):
    record = exact_record(vocab, stage3)
    record["payload"]["intact_baseline_present"] = False

    with pytest.raises(
        Stage4SchemaError,
        match="intact_baseline_present=true",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_exact_ledger_requires_exactly_one_intact_mask(
    vocab,
    stage3,
    contract,
):
    record = exact_record(vocab, stage3)
    record["payload"]["entries"][0]["mask_kind"] = "candidate"

    with pytest.raises(
        Stage4SchemaError,
        match="exactly one intact mask",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_intact_mask_must_retain_all_516(vocab, stage3, contract):
    record = exact_record(vocab, stage3)
    record["payload"]["entries"][0]["retained_count"] = 515
    record["payload"]["entries"][0]["retained_proportion"] = 515 / 516

    with pytest.raises(
        Stage4SchemaError,
        match="retain all 516",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_exact_masks_must_be_unique(vocab, stage3, contract):
    record = exact_record(vocab, stage3)
    record["payload"]["entries"][2]["mask_sha256"] = "5" * 64

    with pytest.raises(
        Stage4SchemaError,
        match="masks must be unique",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_exact_evaluation_count_matches_entries(
    vocab,
    stage3,
    contract,
):
    record = exact_record(vocab, stage3)
    record["payload"]["exact_evaluation_count"] = 4

    with pytest.raises(
        Stage4SchemaError,
        match="number of ledger entries",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_charged_evaluation_count_matches_entries(
    vocab,
    stage3,
    contract,
):
    record = exact_record(vocab, stage3)
    record["payload"]["charged_evaluation_count"] = 1

    with pytest.raises(
        Stage4SchemaError,
        match="charged ledger entries",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_common_exact_allowance_cannot_be_exceeded(
    vocab,
    stage3,
    contract,
):
    record = exact_record(vocab, stage3)
    record["payload"]["exact_evaluation_allowance"] = 1

    with pytest.raises(
        Stage4SchemaError,
        match="cannot exceed common allowance",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_retained_proportion_must_equal_count_over_516(
    vocab,
    stage3,
    contract,
):
    record = exact_record(vocab, stage3)
    record["payload"]["entries"][1]["retained_proportion"] = 0.6

    with pytest.raises(
        Stage4SchemaError,
        match="retained_count / 516",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_exact_ledger_cannot_contain_native_budget_fields(
    vocab,
    stage3,
    contract,
):
    record = exact_record(vocab, stage3)
    record["payload"]["native_units_consumed"] = 2

    with pytest.raises(
        Stage4SchemaError,
        match="keys mismatch",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_exact_discovery_reference_requires_exact_depth8_identity(
    vocab,
    stage3,
    contract,
):
    record = exact_record(vocab, stage3)

    other = discovery_id(
        stage3,
        fidelity="other-fidelity/v1",
    )
    record["payload"]["discovery_run"]["condition_id"] = other

    with pytest.raises(
        Stage4SchemaError,
        match="exact complete condition identity",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_valid_endpoint(vocab, stage3, contract):
    validate(
        endpoint_record(vocab, stage3),
        contract=contract,
        stage3=stage3,
    )


def test_endpoint_record_must_be_sealed(vocab, stage3, contract):
    record = endpoint_record(vocab, stage3)
    record["record_status"] = "draft"

    with pytest.raises(
        Stage4SchemaError,
        match="must be sealed",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_endpoint1_can_equal_one(vocab, stage3, contract):
    record = endpoint_record(
        vocab,
        stage3,
        endpoint1=1.0,
    )

    validate(record, contract=contract, stage3=stage3)

    assert (
        record["payload"]["endpoint_1"][
            "smallest_recovered_component_proportion"
        ]
        == 1.0
    )


def test_endpoint1_cannot_claim_global_minimum(
    vocab,
    stage3,
    contract,
):
    record = endpoint_record(vocab, stage3)
    record["payload"]["endpoint_1"]["global_minimum_claim"] = True

    with pytest.raises(
        Stage4SchemaError,
        match="cannot claim a global minimum",
    ):
        validate(record, contract=contract, stage3=stage3)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_endpoint1_must_stay_in_closed_unit_interval(
    vocab,
    stage3,
    contract,
    value,
):
    record = endpoint_record(vocab, stage3)
    record["payload"]["endpoint_1"][
        "smallest_recovered_component_proportion"
    ] = value

    with pytest.raises(
        Stage4SchemaError,
        match=r"must be in \[0, 1\]",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_endpoint2_can_equal_zero(vocab, stage3, contract):
    record = endpoint_record(
        vocab,
        stage3,
        packing=0,
    )

    validate(record, contract=contract, stage3=stage3)

    assert record["payload"]["endpoint_2"]["packing_lower_bound"] == 0
    assert record["payload"]["endpoint_2"]["packed_mask_sha256s"] == []


def test_endpoint2_cannot_claim_true_packing_number(
    vocab,
    stage3,
    contract,
):
    record = endpoint_record(vocab, stage3)
    record["payload"]["endpoint_2"]["true_packing_number_claim"] = True

    with pytest.raises(
        Stage4SchemaError,
        match="cannot claim the true packing number",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_endpoint2_count_must_match_packed_masks(
    vocab,
    stage3,
    contract,
):
    record = endpoint_record(vocab, stage3)
    record["payload"]["endpoint_2"]["packing_lower_bound"] = 2

    with pytest.raises(
        Stage4SchemaError,
        match="must equal packed_mask_sha256s length",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_endpoint2_packed_masks_must_be_unique(
    vocab,
    stage3,
    contract,
):
    record = endpoint_record(vocab, stage3)
    record["payload"]["endpoint_2"]["packing_lower_bound"] = 2
    record["payload"]["endpoint_2"]["packed_mask_sha256s"] = [
        "5" * 64,
        "5" * 64,
    ]

    with pytest.raises(
        Stage4SchemaError,
        match="must be unique",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_endpoint_setting_refs_must_match_identity(
    vocab,
    stage3,
    contract,
):
    record = endpoint_record(vocab, stage3)
    record["payload"]["component_cap_ref"] = "other-cap/v1"

    with pytest.raises(
        Stage4SchemaError,
        match="must equal condition identity",
    ):
        validate(record, contract=contract, stage3=stage3)


def test_endpoint_exact_ledger_reference_requires_exact_depth8_identity(
    vocab,
    stage3,
    contract,
):
    record = endpoint_record(vocab, stage3)

    other = discovery_id(
        stage3,
        overlap="other-overlap/v1",
    )
    record["payload"]["exact_ledger"]["condition_id"] = other

    with pytest.raises(
        Stage4SchemaError,
        match="exact complete condition identity",
    ):
        validate(record, contract=contract, stage3=stage3)

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage4_schema_analysis import (
    FIREWALL_PATH,
    FIREWALL_SHA256,
)
from circuit_families.stage4_schema_common import (
    CommonSchemaContract,
    Stage4SchemaError,
)
from circuit_families.stage4_schema_graph import (
    validate_stage4_record_graph,
)
from circuit_families.stage4_schema_records import (
    COMPONENT_ABLATION_SOURCE,
    MASKS_SOURCE,
    STAGE8_MASKING_MANIFEST,
)
from circuit_families.stage4_seed_derivation import SeedInputs, derive_seed

ROOT = Path(__file__).resolve().parents[1]
VOCAB = json.loads(
    (ROOT / "followup/configs/stage4_common_vocabulary_v1.json").read_text()
)
IDENTITY = json.loads(
    (
        ROOT
        / "followup/configs/stage4_condition_identity_spec_v1.json"
    ).read_text()
)
REGISTRY_PATH = (
    ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
)
REGISTRY = json.loads(REGISTRY_PATH.read_text())
REGISTRY_SHA = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
DECISIONS = json.loads(
    (
        ROOT / "followup/configs/stage2_unresolved_decisions_v1.json"
    ).read_text()
)

STAGE3 = Stage3AvailabilityIndex.from_registry(REGISTRY)
CONTRACT = CommonSchemaContract.from_specs(VOCAB, IDENTITY)


def unresolved_ids():
    wanted = {f"UD-{n:03d}" for n in range(3, 15)}
    result = set()

    def walk(value):
        if isinstance(value, dict):
            ident = value.get("decision_id")
            if ident in wanted and value.get("status") == "unresolved":
                result.add(ident)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(DECISIONS)
    assert result == wanted
    return result


UNRESOLVED = unresolved_ids()


def h(index):
    return f"{index:064x}"


def cid(depth, condition="hard_target", initialization=0):
    values = {
        "teacher_seed": 1,
        "phase": "stable post-grokking",
        "distillation_condition": condition,
        "student_initialization": initialization,
        "discovery_method": "synthetic-method/v1",
        "fidelity_setting": "synthetic-fidelity/v1",
        "component_cap": "synthetic-cap/v1",
        "overlap_setting": "synthetic-overlap/v1",
    }

    hierarchy = IDENTITY["canonical_hierarchy"]

    return build_condition_id(
        ConditionIdentity(
            **{
                field: values[field]
                for field in hierarchy[:depth]
            }
        ),
        STAGE3,
    )


def ref(record_type, condition_id, digest):
    return {
        "record_type": record_type,
        "schema_version": VOCAB["schema_versions"][record_type],
        "condition_id": condition_id,
        "record_sha256": digest,
    }


def envelope(
    record_type,
    condition_id,
    payload,
    *,
    lane,
    status="sealed",
    sources=(),
):
    return {
        "namespace": VOCAB["namespace"],
        "vocabulary_version": VOCAB["vocabulary_version"],
        "schema_version": VOCAB["schema_versions"][record_type],
        "record_type": record_type,
        "record_status": status,
        "condition_id": condition_id,
        "identity_depth": IDENTITY[
            "record_type_required_depths"
        ][record_type],
        "payload": payload,
        "provenance": {
            "producer_lane": lane,
            "creation_stage": "stage4",
            "source_records": list(sources),
        },
    }


def seed_dict(condition_id, purpose):
    evidence = derive_seed(
        SeedInputs(
            condition_id=condition_id,
            purpose=purpose,
            attempt_index=0,
            retry_index=0,
        ),
        STAGE3,
    )
    return {
        "seed_derivation_version": evidence.seed_derivation_version,
        "seed_material": evidence.seed_material,
        "digest_sha256": evidence.digest_sha256,
        "selected_bytes_hex": evidence.selected_bytes_hex,
        "seed_value": evidence.seed_value,
    }


def graph():
    d2 = cid(2)
    direct3 = cid(3, "direct_teacher")
    hard3 = cid(3)
    hard4 = cid(4)
    hard8 = cid(8)

    source = next(
        item
        for item in REGISTRY["records"]
        if (
            item["teacher_seed"] == 1
            and item["phase_label"] == "stable post-grokking"
        )
    )

    hashes = {
        "teacher": h(1),
        "cache": h(2),
        "attempt": h(3),
        "eligibility": h(4),
        "model": h(5),
        "discovery": h(6),
        "native": h(7),
        "exact": h(8),
        "endpoint": h(9),
        "endpoint_repro": h(10),
        "summary": h(11),
        "inventory": h(12),
        "excluded": h(13),
        "reproduction": h(14),
        "freeze": h(15),
    }

    teacher = envelope(
        "teacher_reference",
        direct3,
        {
            "stage3_registry_path": (
                "followup/manifests/stage3_teacher_registry_v1.json"
            ),
            "stage3_registry_sha256": REGISTRY_SHA,
            "stage3_registry_namespace": (
                "circuit-families-distillation/stage3-teacher-registry"
            ),
            "stage3_record_schema_version": "1",
            "canonical_run_id": source["canonical_run_id"],
            "checkpoint": {
                "path": source["checkpoint_path"],
                "sha256": source["checkpoint_sha256"],
                "storage_class": "external_checkpoint",
            },
            "training_step": source["training_step"],
        },
        lane="lane_a",
    )

    teacher_ref = ref("teacher_reference", direct3, hashes["teacher"])

    cache = envelope(
        "teacher_output_cache",
        hard3,
        {
            "teacher_reference": teacher_ref,
            "cache_kind": "teacher_argmax",
            "example_ordering_ref": "synthetic-ordering/v1",
            "example_count": 12769,
            "artifact": {
                "path": "synthetic/cache.bin",
                "sha256": "a" * 64,
                "storage_class": "external_large_object",
            },
        },
        lane="lane_b",
        sources=[teacher_ref],
    )

    cache_ref = ref("teacher_output_cache", hard3, hashes["cache"])

    attempt = envelope(
        "student_attempt",
        hard4,
        {
            "target_cache": cache_ref,
            "attempt_index": 0,
            "retry_index": 0,
            "attempt_outcome": "succeeded",
            "student_architecture_ref": "synthetic-student-architecture/v1",
            "replication_policy_ref": "synthetic-replication/v1",
            "training_config_ref": "synthetic-training/v1",
            "training_seed": seed_dict(hard4, "training"),
            "tie_breaking_seed": seed_dict(hard4, "tie_breaking"),
            "training_log": {
                "path": "synthetic/train.log",
                "sha256": "b" * 64,
                "storage_class": "external_log",
            },
            "model_checkpoint": {
                "path": "synthetic/student.pt",
                "sha256": "c" * 64,
                "storage_class": "external_checkpoint",
            },
        },
        lane="lane_b",
        sources=[cache_ref],
    )

    attempt_ref = ref("student_attempt", hard4, hashes["attempt"])

    eligibility = envelope(
        "student_eligibility",
        hard4,
        {
            "attempt_reference": attempt_ref,
            "attempt_index": 0,
            "retry_index": 0,
            "eligibility_status": "passed",
            "criterion": "exact_teacher_argmax_agreement",
            "evaluation_example_count": 12769,
            "teacher_argmax_agreement_count": 12769,
        },
        lane="lane_b",
        sources=[attempt_ref],
    )

    eligibility_ref = ref(
        "student_eligibility",
        hard4,
        hashes["eligibility"],
    )

    model = envelope(
        "sealed_dense_model",
        hard4,
        {
            "eligibility_reference": eligibility_ref,
            "eligibility_status": "passed",
            "architecture_ref": "synthetic-student-architecture/v1",
            "component_basis": {
                "component_count": 516,
                "status": "reused_predecessor_definition",
                "masks_source": MASKS_SOURCE,
                "component_ablation_source": COMPONENT_ABLATION_SOURCE,
                "stage8_masking_manifest": STAGE8_MASKING_MANIFEST,
            },
            "model_checkpoint": {
                "path": "synthetic/student.pt",
                "sha256": "c" * 64,
                "storage_class": "external_checkpoint",
            },
        },
        lane="lane_b",
        sources=[eligibility_ref],
    )

    model_ref = ref("sealed_dense_model", hard4, hashes["model"])

    discovery = envelope(
        "discovery_run",
        hard8,
        {
            "sealed_dense_model": model_ref,
            "attempt_index": 0,
            "retry_index": 0,
            "discovery_method_ref": "synthetic-method/v1",
            "method_budget_ref": "synthetic-budget/v1",
            "fidelity_definition_ref": "synthetic-fidelity/v1",
            "component_cap_ref": "synthetic-cap/v1",
            "overlap_setting_ref": "synthetic-overlap/v1",
            "discovery_seed": seed_dict(hard8, "discovery"),
            "search_artifact": {
                "path": "synthetic/search.bin",
                "sha256": "d" * 64,
                "storage_class": "external_large_object",
            },
        },
        lane="lane_c",
        sources=[model_ref],
    )

    discovery_ref = ref("discovery_run", hard8, hashes["discovery"])

    native = envelope(
        "native_budget_ledger",
        hard8,
        {
            "discovery_run": discovery_ref,
            "method_budget_ref": "synthetic-budget/v1",
            "native_unit_ref": "synthetic-native-unit/v1",
            "native_budget_limit": 4,
            "native_units_consumed": 4,
            "entries": [
                {
                    "sequence_index": 0,
                    "operation_ref": "synthetic-operation/v1",
                    "native_units_charged": 4,
                }
            ],
        },
        lane="lane_c",
        sources=[discovery_ref],
    )

    intact = "e" * 64
    candidate = "f" * 64

    exact = envelope(
        "exact_mask_evaluation_ledger",
        hard8,
        {
            "sealed_dense_model": model_ref,
            "discovery_run": discovery_ref,
            "fidelity_definition_ref": "synthetic-fidelity/v1",
            "exact_evaluation_allowance_ref": "synthetic-allowance/v1",
            "exact_evaluation_allowance": 2,
            "exact_evaluation_count": 2,
            "charged_evaluation_count": 1,
            "intact_baseline_present": True,
            "entries": [
                {
                    "evaluation_order": 0,
                    "mask_sha256": intact,
                    "mask_kind": "intact",
                    "retained_count": 516,
                    "retained_proportion": 1.0,
                    "fidelity_value": 1.0,
                    "qualifies": True,
                    "budget_charged": False,
                },
                {
                    "evaluation_order": 1,
                    "mask_sha256": candidate,
                    "mask_kind": "candidate",
                    "retained_count": 258,
                    "retained_proportion": 0.5,
                    "fidelity_value": 0.95,
                    "qualifies": True,
                    "budget_charged": True,
                },
            ],
        },
        lane="lane_c",
        sources=[model_ref, discovery_ref],
    )

    exact_ref = ref(
        "exact_mask_evaluation_ledger",
        hard8,
        hashes["exact"],
    )

    def endpoint_record():
        return envelope(
            "endpoint_record",
            hard8,
            {
                "exact_ledger": exact_ref,
                "fidelity_definition_ref": "synthetic-fidelity/v1",
                "component_cap_ref": "synthetic-cap/v1",
                "overlap_setting_ref": "synthetic-overlap/v1",
                "endpoint_1": {
                    "smallest_recovered_component_proportion": 0.5,
                    "qualifying_mask_sha256": candidate,
                    "procedure_censored": False,
                    "interpretation": (
                        "smallest_qualifying_proportion_in_exact_ledger"
                    ),
                    "global_minimum_claim": False,
                },
                "endpoint_2": {
                    "packing_lower_bound": 1,
                    "packed_mask_sha256s": [candidate],
                    "packing_rule_ref": "synthetic-packing/v1",
                    "interpretation": (
                        "procedure_dependent_packing_lower_bound"
                    ),
                    "true_packing_number_claim": False,
                },
            },
            lane="lane_c",
            sources=[exact_ref],
        )

    endpoint = endpoint_record()
    endpoint_repro = endpoint_record()

    endpoint_ref = ref("endpoint_record", hard8, hashes["endpoint"])
    endpoint_repro_ref = ref(
        "endpoint_record",
        hard8,
        hashes["endpoint_repro"],
    )

    summary = envelope(
        "student_cell_summary",
        hard3,
        {
            "population_unit": "teacher_seed",
            "summary_rule_ref": "synthetic-summary/v1",
            "minimum_eligible_students_ref": "synthetic-minimum/v1",
            "cell_analysis_status": "unresolved",
            "eligible_student_count": 1,
            "failed_attempt_count": 0,
            "missing_student_count": 0,
            "eligible_student_initializations": [0],
            "failed_attempts": [],
            "missing_student_initializations": [],
            "endpoint_records": [endpoint_ref],
        },
        lane="lane_d",
        sources=[endpoint_ref],
    )

    summary_ref = ref(
        "student_cell_summary",
        hard3,
        hashes["summary"],
    )

    inventory = envelope(
        "teacher_seed_inventory",
        d2,
        {
            "population_unit": "teacher_seed",
            "stage3_registry_path": (
                "followup/manifests/stage3_teacher_registry_v1.json"
            ),
            "stage3_registry_sha256": REGISTRY_SHA,
            "stage3_availability_state": "selected",
            "cell_state": "selected",
            "unavailable_reason": None,
            "student_cell_summaries": [summary_ref],
        },
        lane="lane_d",
        sources=[summary_ref],
    )

    excluded = envelope(
        "excluded_development_output",
        hard3,
        {
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
                "sha256": "1" * 64,
                "storage_class": "external_large_object",
            },
        },
        lane="lane_c",
    )

    reproduction = envelope(
        "reproduction_comparison",
        hard3,
        {
            "comparison_rule_ref": "synthetic-reproduction/v1",
            "source_record": endpoint_ref,
            "reproduced_record": endpoint_repro_ref,
            "semantic_match": True,
        },
        lane="lane_d",
        sources=[endpoint_ref, endpoint_repro_ref],
    )

    inventory_ref = ref(
        "teacher_seed_inventory",
        d2,
        hashes["inventory"],
    )
    reproduction_ref = ref(
        "reproduction_comparison",
        hard3,
        hashes["reproduction"],
    )
    excluded_ref = ref(
        "excluded_development_output",
        hard3,
        hashes["excluded"],
    )

    freeze = envelope(
        "analysis_freeze",
        d2,
        {
            "population_unit": "teacher_seed",
            "analysis_contract_ref": "synthetic-analysis/v1",
            "firewall_authority": {
                "path": FIREWALL_PATH,
                "sha256": FIREWALL_SHA256,
            },
            "firewall_clear": False,
            "unresolved_decision_ids": sorted(UNRESOLVED),
            "production_ready": False,
            "primary_input_records": [
                inventory_ref,
                summary_ref,
                reproduction_ref,
            ],
            "excluded_development_records": [excluded_ref],
        },
        lane="joint",
        sources=[
            inventory_ref,
            summary_ref,
            reproduction_ref,
            excluded_ref,
        ],
    )

    records = {
        hashes["teacher"]: teacher,
        hashes["cache"]: cache,
        hashes["attempt"]: attempt,
        hashes["eligibility"]: eligibility,
        hashes["model"]: model,
        hashes["discovery"]: discovery,
        hashes["native"]: native,
        hashes["exact"]: exact,
        hashes["endpoint"]: endpoint,
        hashes["endpoint_repro"]: endpoint_repro,
        hashes["summary"]: summary,
        hashes["inventory"]: inventory,
        hashes["excluded"]: excluded,
        hashes["reproduction"]: reproduction,
        hashes["freeze"]: freeze,
    }

    return records, hashes


def validate(records):
    validate_stage4_record_graph(
        records,
        contract=CONTRACT,
        stage3=STAGE3,
        stage3_registry=REGISTRY,
        stage3_registry_sha256=REGISTRY_SHA,
        current_unresolved_decision_ids=UNRESOLVED,
    )


def test_complete_graph_validates():
    records, _ = graph()
    validate(records)


def test_graph_contains_all_14_record_types():
    records, _ = graph()
    assert len({record["record_type"] for record in records.values()}) == 14


def test_dangling_reference_rejected():
    records, hashes = graph()
    records[hashes["cache"]]["payload"]["teacher_reference"][
        "record_sha256"
    ] = "0" * 64
    records[hashes["cache"]]["provenance"]["source_records"][0][
        "record_sha256"
    ] = "0" * 64

    with pytest.raises(Stage4SchemaError, match="dangling"):
        validate(records)


def test_reference_identity_mismatch_rejected():
    records, hashes = graph()
    records[hashes["cache"]]["payload"]["teacher_reference"][
        "record_type"
    ] = "teacher_output_cache"
    records[hashes["cache"]]["provenance"]["source_records"][0][
        "record_type"
    ] = "teacher_output_cache"

    with pytest.raises(Stage4SchemaError):
        validate(records)


def test_payload_dependency_must_appear_in_provenance():
    records, hashes = graph()
    records[hashes["cache"]]["provenance"]["source_records"] = []

    with pytest.raises(
        Stage4SchemaError,
        match="missing from provenance",
    ):
        validate(records)


def test_eligibility_requires_sealed_attempt():
    records, hashes = graph()
    records[hashes["attempt"]]["record_status"] = "draft"

    with pytest.raises(
        Stage4SchemaError,
        match="eligibility source student_attempt",
    ):
        validate(records)


def test_eligibility_requires_successful_attempt():
    records, hashes = graph()
    attempt = records[hashes["attempt"]]
    attempt["payload"]["attempt_outcome"] = "failed"
    del attempt["payload"]["model_checkpoint"]
    attempt["payload"]["failure_reason"] = "synthetic"

    with pytest.raises(Stage4SchemaError):
        validate(records)


def test_sealed_model_checkpoint_must_match_attempt():
    records, hashes = graph()
    records[hashes["model"]]["payload"]["model_checkpoint"][
        "sha256"
    ] = "2" * 64

    with pytest.raises(
        Stage4SchemaError,
        match="checkpoint must match student_attempt",
    ):
        validate(records)


def test_sealed_model_architecture_must_match_attempt():
    records, hashes = graph()
    records[hashes["model"]]["payload"]["architecture_ref"] = (
        "different-architecture/v1"
    )

    with pytest.raises(
        Stage4SchemaError,
        match="architecture_ref must match student_attempt",
    ):
        validate(records)


def test_discovery_requires_actually_sealed_dense_model():
    records, hashes = graph()
    records[hashes["model"]]["record_status"] = "draft"

    with pytest.raises(
        Stage4SchemaError,
        match="discovery sealed_dense_model source",
    ):
        validate(records)


def test_native_budget_ref_matches_discovery():
    records, hashes = graph()
    records[hashes["native"]]["payload"]["method_budget_ref"] = (
        "other-budget/v1"
    )

    with pytest.raises(
        Stage4SchemaError,
        match="must match discovery_run",
    ):
        validate(records)


def test_endpoint_requires_sealed_exact_ledger():
    records, hashes = graph()
    records[hashes["exact"]]["record_status"] = "draft"

    with pytest.raises(
        Stage4SchemaError,
        match="endpoint exact ledger source",
    ):
        validate(records)


def test_endpoint1_reconstructed_from_exact_ledger():
    records, hashes = graph()
    records[hashes["endpoint"]]["payload"]["endpoint_1"][
        "smallest_recovered_component_proportion"
    ] = 1.0
    records[hashes["endpoint"]]["payload"]["endpoint_1"][
        "qualifying_mask_sha256"
    ] = "e" * 64

    with pytest.raises(
        Stage4SchemaError,
        match="does not reconstruct",
    ):
        validate(records)


def test_endpoint1_mask_must_be_minimum_qualifier():
    records, hashes = graph()
    records[hashes["endpoint"]]["payload"]["endpoint_1"][
        "qualifying_mask_sha256"
    ] = "e" * 64

    with pytest.raises(
        Stage4SchemaError,
        match="not a minimum qualifying",
    ):
        validate(records)


def test_endpoint2_packed_mask_must_exist_in_exact_ledger():
    records, hashes = graph()
    records[hashes["endpoint"]]["payload"]["endpoint_2"][
        "packed_mask_sha256s"
    ] = ["3" * 64]

    with pytest.raises(
        Stage4SchemaError,
        match="absent from exact ledger",
    ):
        validate(records)


def test_endpoint2_packed_mask_must_qualify():
    records, hashes = graph()
    exact = records[hashes["exact"]]
    exact["payload"]["entries"][1]["qualifies"] = False
    endpoint = records[hashes["endpoint"]]
    endpoint["payload"]["endpoint_1"][
        "smallest_recovered_component_proportion"
    ] = 1.0
    endpoint["payload"]["endpoint_1"]["qualifying_mask_sha256"] = "e" * 64

    with pytest.raises(
        Stage4SchemaError,
        match="packed mask must be a qualifying",
    ):
        validate(records)


def test_inventory_requires_sealed_summary():
    records, hashes = graph()
    records[hashes["summary"]]["record_status"] = "draft"

    with pytest.raises(
        Stage4SchemaError,
        match="inventory student_cell_summary",
    ):
        validate(records)


def test_analysis_freeze_requires_sealed_primary_inputs():
    records, hashes = graph()
    records[hashes["reproduction"]]["record_status"] = "draft"

    with pytest.raises(
        Stage4SchemaError,
        match="analysis_freeze primary input",
    ):
        validate(records)


def test_graph_cycle_rejected():
    records, hashes = graph()

    teacher = records[hashes["teacher"]]
    freeze = records[hashes["freeze"]]

    freeze_ref = ref(
        "analysis_freeze",
        freeze["condition_id"],
        hashes["freeze"],
    )

    teacher["provenance"]["source_records"] = [freeze_ref]

    with pytest.raises(
        Stage4SchemaError,
        match="dependency cycle",
    ):
        validate(records)


def test_self_reference_rejected():
    records, hashes = graph()
    excluded = records[hashes["excluded"]]

    excluded["provenance"]["source_records"] = [
        ref(
            "excluded_development_output",
            excluded["condition_id"],
            hashes["excluded"],
        )
    ]

    with pytest.raises(
        Stage4SchemaError,
        match="cannot reference itself",
    ):
        validate(records)

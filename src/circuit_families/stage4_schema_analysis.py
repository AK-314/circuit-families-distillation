"""Stage 4 validators for analysis-side record types.

UD-011 through UD-014 remain unresolved. These validators freeze lifecycle,
population-unit, firewall, provenance, and missingness structure without
freezing the later numeric summary/orchestration choices.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from circuit_families.stage4_condition_identity import (
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    parse_condition_id,
)
from circuit_families.stage4_schema_common import (
    CommonSchemaContract,
    Stage4SchemaError,
    validate_common_envelope,
)
from circuit_families.stage4_schema_records import (
    _error,
    _exact_keys,
    _reference_identity,
    _validate_portable_artifact,
    _validate_sha256,
    _validate_uint,
    _validate_version_ref,
)

STAGE3_REGISTRY_PATH = "followup/manifests/stage3_teacher_registry_v1.json"
STAGE3_REGISTRY_SHA256 = (
    "36656fe848f7cb980cd9178f1d48cbbee74bb410135746e079cae11440b6ff0d"
)

FIREWALL_PATH = (
    "docs/distillation_followup/stage2_method_development_firewall.md"
)
FIREWALL_SHA256 = (
    "5e5ce8666f4182d7fce5f310bc42d3d45f3ff8af55bbc5882985acbc0916fadd"
)
EXCLUDED_REGISTER_PATH = (
    "followup/manifests/stage2_excluded_development_register_v1.json"
)

_PART_Q_TYPES = frozenset(
    {
        "student_cell_summary",
        "teacher_seed_inventory",
        "excluded_development_output",
        "reproduction_comparison",
        "analysis_freeze",
    }
)


def _same_depth3_prefix(left, right) -> None:
    if (
        left.teacher_seed != right.teacher_seed
        or left.phase != right.phase
        or left.distillation_condition != right.distillation_condition
    ):
        _error(
            "referenced record must share teacher/phase/"
            "distillation-condition prefix"
        )


def _same_depth2_prefix(left, right) -> None:
    if (
        left.teacher_seed != right.teacher_seed
        or left.phase != right.phase
    ):
        _error("referenced record must share teacher/phase prefix")


def _reference(
    value: Any,
    *,
    expected_record_type: str,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
):
    return _reference_identity(
        value,
        expected_record_type=expected_record_type,
        contract=contract,
        stage3=stage3,
        require_hash=True,
    )


def _generic_reference(
    value: Any,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
):
    if not isinstance(value, Mapping):
        _error("record reference must be an object")

    required = {
        "record_type",
        "schema_version",
        "condition_id",
        "record_sha256",
    }

    _exact_keys(
        value,
        required,
        label="record reference",
    )

    record_type = value["record_type"]

    if record_type not in contract.record_types:
        _error(f"unknown referenced record_type: {record_type!r}")

    expected_schema = contract.schema_versions[record_type]

    if value["schema_version"] != expected_schema:
        _error(
            "record reference schema_version does not match record_type"
        )

    _validate_sha256(
        value["record_sha256"],
        label="record reference record_sha256",
    )

    try:
        identity = parse_condition_id(
            value["condition_id"],
            stage3,
        )
    except ConditionIdentityError as exc:
        raise Stage4SchemaError(
            f"invalid referenced condition_id: {exc}"
        ) from exc

    expected_depth = contract.record_type_required_depths[record_type]

    if identity.depth != expected_depth:
        _error(
            "record reference condition depth mismatch: "
            f"actual={identity.depth} expected={expected_depth}"
        )

    return identity


def _validate_uint_list(
    values: Any,
    *,
    label: str,
) -> list[int]:
    if not isinstance(values, list):
        _error(f"{label} must be an array")

    parsed: list[int] = []

    for value in values:
        _validate_uint(value, label=label)
        parsed.append(value)

    if len(parsed) != len(set(parsed)):
        _error(f"{label} must contain unique values")

    return parsed


def _validate_firewall_authority(value: Any) -> None:
    if not isinstance(value, Mapping):
        _error("firewall_authority must be an object")

    _exact_keys(
        value,
        {"path", "sha256"},
        label="firewall_authority",
    )

    if value["path"] != FIREWALL_PATH:
        _error("firewall_authority path mismatch")

    if value["sha256"] != FIREWALL_SHA256:
        _error("firewall_authority SHA-256 mismatch")


def _validate_student_cell_summary(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    if identity.distillation_condition not in {
        "hard_target",
        "soft_target",
    }:
        _error(
            "student_cell_summary requires hard_target or soft_target"
        )

    if record["provenance"]["producer_lane"] != "lane_d":
        _error("student_cell_summary producer_lane must be lane_d")

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "population_unit",
            "summary_rule_ref",
            "minimum_eligible_students_ref",
            "cell_analysis_status",
            "eligible_student_count",
            "failed_attempt_count",
            "missing_student_count",
            "eligible_student_initializations",
            "failed_attempts",
            "missing_student_initializations",
            "endpoint_records",
        },
        label="student_cell_summary payload",
    )

    if payload["population_unit"] != "teacher_seed":
        _error("student_cell_summary population_unit must be teacher_seed")

    _validate_version_ref(
        payload["summary_rule_ref"],
        label="summary_rule_ref",
    )
    _validate_version_ref(
        payload["minimum_eligible_students_ref"],
        label="minimum_eligible_students_ref",
    )

    if payload["cell_analysis_status"] not in {
        "resolved",
        "unresolved",
    }:
        _error("cell_analysis_status is invalid")

    eligible = _validate_uint_list(
        payload["eligible_student_initializations"],
        label="eligible_student_initializations",
    )
    missing = _validate_uint_list(
        payload["missing_student_initializations"],
        label="missing_student_initializations",
    )

    if set(eligible) & set(missing):
        _error(
            "eligible and missing student initializations must be disjoint"
        )

    _validate_uint(
        payload["eligible_student_count"],
        label="eligible_student_count",
    )
    _validate_uint(
        payload["failed_attempt_count"],
        label="failed_attempt_count",
    )
    _validate_uint(
        payload["missing_student_count"],
        label="missing_student_count",
    )

    if payload["eligible_student_count"] != len(eligible):
        _error(
            "eligible_student_count must equal eligible initialization count"
        )

    if payload["missing_student_count"] != len(missing):
        _error(
            "missing_student_count must equal missing initialization count"
        )

    failed_attempts = payload["failed_attempts"]

    if not isinstance(failed_attempts, list):
        _error("failed_attempts must be an array")

    seen_failed_coordinates: set[tuple[int, int, int]] = set()

    for item in failed_attempts:
        if not isinstance(item, Mapping):
            _error("failed attempt summary item must be an object")

        _exact_keys(
            item,
            {
                "student_initialization",
                "attempt_index",
                "retry_index",
                "attempt_reference",
            },
            label="failed attempt summary item",
        )

        for field in (
            "student_initialization",
            "attempt_index",
            "retry_index",
        ):
            _validate_uint(item[field], label=field)

        attempt_identity = _reference(
            item["attempt_reference"],
            expected_record_type="student_attempt",
            contract=contract,
            stage3=stage3,
        )

        _same_depth3_prefix(identity, attempt_identity)

        if (
            attempt_identity.student_initialization
            != item["student_initialization"]
        ):
            _error(
                "failed attempt student_initialization does not match "
                "referenced student_attempt"
            )

        coordinate = (
            item["student_initialization"],
            item["attempt_index"],
            item["retry_index"],
        )

        if coordinate in seen_failed_coordinates:
            _error("failed attempt coordinates must be unique")

        seen_failed_coordinates.add(coordinate)

    if payload["failed_attempt_count"] != len(failed_attempts):
        _error(
            "failed_attempt_count must equal failed_attempts length"
        )

    endpoint_records = payload["endpoint_records"]

    if not isinstance(endpoint_records, list):
        _error("endpoint_records must be an array")

    seen_endpoint_refs: set[tuple[str, str]] = set()

    for reference in endpoint_records:
        endpoint_identity = _reference(
            reference,
            expected_record_type="endpoint_record",
            contract=contract,
            stage3=stage3,
        )

        _same_depth3_prefix(identity, endpoint_identity)

        key = (
            reference["condition_id"],
            reference["record_sha256"],
        )

        if key in seen_endpoint_refs:
            _error("endpoint_records must not contain duplicate references")

        seen_endpoint_refs.add(key)


def _find_stage3_record(
    *,
    teacher_seed: int,
    phase: str,
    registry: Mapping[str, Any],
):
    matches = [
        record
        for record in registry["records"]
        if (
            record["teacher_seed"] == teacher_seed
            and record["phase_label"] == phase
        )
    ]

    if len(matches) != 1:
        _error(
            "Stage 3 registry must contain exactly one matching "
            "teacher-phase cell"
        )

    return matches[0]


def _validate_teacher_seed_inventory(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
    stage3_registry: Mapping[str, Any],
    stage3_registry_sha256: str,
) -> None:
    if record["provenance"]["producer_lane"] != "lane_d":
        _error("teacher_seed_inventory producer_lane must be lane_d")

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "population_unit",
            "stage3_registry_path",
            "stage3_registry_sha256",
            "stage3_availability_state",
            "cell_state",
            "unavailable_reason",
            "student_cell_summaries",
        },
        label="teacher_seed_inventory payload",
    )

    if payload["population_unit"] != "teacher_seed":
        _error("teacher_seed_inventory population_unit must be teacher_seed")

    if payload["stage3_registry_path"] != STAGE3_REGISTRY_PATH:
        _error("teacher_seed_inventory Stage 3 registry path mismatch")

    if stage3_registry_sha256 != STAGE3_REGISTRY_SHA256:
        _error("provided Stage 3 registry SHA-256 is not frozen authority")

    if payload["stage3_registry_sha256"] != stage3_registry_sha256:
        _error("teacher_seed_inventory Stage 3 registry SHA-256 mismatch")

    source = _find_stage3_record(
        teacher_seed=identity.teacher_seed,
        phase=identity.phase,
        registry=stage3_registry,
    )

    expected_stage3_state = source["availability_status"]

    if payload["stage3_availability_state"] != expected_stage3_state:
        _error(
            "teacher_seed_inventory stage3_availability_state mismatch"
        )

    if payload["cell_state"] not in {
        "selected",
        "unavailable",
        "failed",
        "missing",
    }:
        _error("teacher_seed_inventory cell_state is invalid")

    summaries = payload["student_cell_summaries"]

    if not isinstance(summaries, list):
        _error("student_cell_summaries must be an array")

    seen_conditions: set[str] = set()

    for reference in summaries:
        summary_identity = _reference(
            reference,
            expected_record_type="student_cell_summary",
            contract=contract,
            stage3=stage3,
        )

        _same_depth2_prefix(identity, summary_identity)

        condition = summary_identity.distillation_condition

        if condition in seen_conditions:
            _error(
                "teacher_seed_inventory may contain at most one summary "
                "per distillation condition"
            )

        seen_conditions.add(condition)

    if expected_stage3_state == "unavailable":
        if payload["cell_state"] != "unavailable":
            _error(
                "Stage 3 unavailable cell must remain explicitly unavailable"
            )

        reason = payload["unavailable_reason"]

        if (
            not isinstance(reason, str)
            or not reason.strip()
            or reason != source.get("unavailable_reason")
        ):
            _error(
                "unavailable_reason must match Stage 3 unavailable record"
            )

        if summaries:
            _error(
                "unavailable Stage 3 cell cannot contain downstream summaries"
            )

    else:
        if payload["cell_state"] == "unavailable":
            _error(
                "selected Stage 3 cell cannot be reclassified unavailable"
            )

        if payload["unavailable_reason"] is not None:
            _error(
                "selected Stage 3 cell must use unavailable_reason=null"
            )


def _validate_excluded_development_output(
    record: Mapping[str, Any],
    identity,
) -> None:
    if record["provenance"]["producer_lane"] not in {
        "lane_b",
        "lane_c",
        "lane_d",
    }:
        _error(
            "excluded_development_output producer_lane must be "
            "lane_b, lane_c, or lane_d"
        )

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "firewall_authority",
            "excluded_register_path",
            "development_purpose_ref",
            "lifecycle_state",
            "primary_analysis_eligible",
            "scientific_selection_use_allowed",
            "regeneration_required",
            "regeneration_status",
            "artifact",
        },
        label="excluded_development_output payload",
    )

    _validate_firewall_authority(payload["firewall_authority"])

    if payload["excluded_register_path"] != EXCLUDED_REGISTER_PATH:
        _error("excluded development register path mismatch")

    _validate_version_ref(
        payload["development_purpose_ref"],
        label="development_purpose_ref",
    )

    if payload["lifecycle_state"] != "excluded":
        _error("excluded development lifecycle_state must be 'excluded'")

    if payload["primary_analysis_eligible"] is not False:
        _error(
            "excluded development output must have "
            "primary_analysis_eligible=false"
        )

    if payload["scientific_selection_use_allowed"] is not False:
        _error(
            "excluded development output cannot be used for scientific "
            "selection"
        )

    if payload["regeneration_required"] is not True:
        _error(
            "excluded development output must require regeneration"
        )

    if payload["regeneration_status"] not in {
        "pending",
        "regenerated",
    }:
        _error("regeneration_status is invalid")

    _validate_portable_artifact(
        payload["artifact"],
        label="excluded development artifact",
        required_storage_class="external_large_object",
    )


def _validate_reproduction_comparison(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    if record["provenance"]["producer_lane"] not in {
        "lane_d",
        "joint",
    }:
        _error(
            "reproduction_comparison producer_lane must be lane_d or joint"
        )

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "comparison_rule_ref",
            "source_record",
            "reproduced_record",
            "semantic_match",
        },
        {"discrepancy_note"},
        label="reproduction_comparison payload",
    )

    _validate_version_ref(
        payload["comparison_rule_ref"],
        label="comparison_rule_ref",
    )

    source_identity = _generic_reference(
        payload["source_record"],
        contract=contract,
        stage3=stage3,
    )
    reproduced_identity = _generic_reference(
        payload["reproduced_record"],
        contract=contract,
        stage3=stage3,
    )

    _same_depth3_prefix(identity, source_identity)
    _same_depth3_prefix(identity, reproduced_identity)

    source = payload["source_record"]
    reproduced = payload["reproduced_record"]

    if source["record_type"] != reproduced["record_type"]:
        _error(
            "source and reproduced records must have the same record_type"
        )

    if source["schema_version"] != reproduced["schema_version"]:
        _error(
            "source and reproduced records must have the same schema_version"
        )

    if source["condition_id"] != reproduced["condition_id"]:
        _error(
            "source and reproduced records must have the same condition_id"
        )

    if not isinstance(payload["semantic_match"], bool):
        _error("semantic_match must be boolean")

    if payload["semantic_match"]:
        if "discrepancy_note" in payload:
            note = payload["discrepancy_note"]
            if not isinstance(note, str) or not note.strip():
                _error(
                    "discrepancy_note, when present, must be non-empty"
                )
    else:
        if "discrepancy_note" not in payload:
            _error(
                "semantic mismatch requires discrepancy_note"
            )

        note = payload["discrepancy_note"]

        if not isinstance(note, str) or not note.strip():
            _error(
                "semantic mismatch discrepancy_note must be non-empty"
            )


def _validate_analysis_freeze(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
    current_unresolved_decision_ids: set[str],
) -> None:
    if record["record_status"] != "sealed":
        _error("analysis_freeze must be sealed")

    if record["provenance"]["producer_lane"] != "joint":
        _error("analysis_freeze producer_lane must be joint")

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "population_unit",
            "analysis_contract_ref",
            "firewall_authority",
            "firewall_clear",
            "unresolved_decision_ids",
            "production_ready",
            "primary_input_records",
            "excluded_development_records",
        },
        label="analysis_freeze payload",
    )

    if payload["population_unit"] != "teacher_seed":
        _error("analysis_freeze population_unit must be teacher_seed")

    _validate_version_ref(
        payload["analysis_contract_ref"],
        label="analysis_contract_ref",
    )

    _validate_firewall_authority(payload["firewall_authority"])

    if not isinstance(payload["firewall_clear"], bool):
        _error("firewall_clear must be boolean")

    unresolved = payload["unresolved_decision_ids"]

    if not isinstance(unresolved, list):
        _error("unresolved_decision_ids must be an array")

    if (
        any(not isinstance(item, str) or not item for item in unresolved)
        or len(unresolved) != len(set(unresolved))
    ):
        _error(
            "unresolved_decision_ids must contain unique non-empty strings"
        )

    if set(unresolved) != current_unresolved_decision_ids:
        _error(
            "analysis_freeze unresolved_decision_ids must match "
            "the supplied decision authority"
        )

    if not isinstance(payload["production_ready"], bool):
        _error("production_ready must be boolean")

    primary_inputs = payload["primary_input_records"]

    if not isinstance(primary_inputs, list):
        _error("primary_input_records must be an array")

    seen_primary: set[tuple[str, str, str]] = set()

    for reference in primary_inputs:
        ref_identity = _generic_reference(
            reference,
            contract=contract,
            stage3=stage3,
        )

        _same_depth2_prefix(identity, ref_identity)

        if reference["record_type"] == "excluded_development_output":
            _error(
                "excluded development output cannot be a primary input"
            )

        key = (
            reference["record_type"],
            reference["condition_id"],
            reference["record_sha256"],
        )

        if key in seen_primary:
            _error("primary_input_records must be unique")

        seen_primary.add(key)

    excluded = payload["excluded_development_records"]

    if not isinstance(excluded, list):
        _error("excluded_development_records must be an array")

    seen_excluded: set[tuple[str, str]] = set()

    for reference in excluded:
        ref_identity = _reference(
            reference,
            expected_record_type="excluded_development_output",
            contract=contract,
            stage3=stage3,
        )

        _same_depth2_prefix(identity, ref_identity)

        key = (
            reference["condition_id"],
            reference["record_sha256"],
        )

        if key in seen_excluded:
            _error("excluded_development_records must be unique")

        seen_excluded.add(key)

    if current_unresolved_decision_ids and payload["production_ready"]:
        _error(
            "analysis_freeze cannot claim production readiness while "
            "unresolved decisions remain"
        )

    if not payload["firewall_clear"] and payload["production_ready"]:
        _error(
            "analysis_freeze cannot claim production readiness while "
            "firewall_clear=false"
        )


def validate_part_q_record(
    record: Mapping[str, Any],
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
    stage3_registry: Mapping[str, Any],
    stage3_registry_sha256: str,
    current_unresolved_decision_ids: set[str],
) -> None:
    """Validate one Part Q record."""
    validate_common_envelope(
        record,
        contract=contract,
        stage3=stage3,
    )

    record_type = record["record_type"]

    if record_type not in _PART_Q_TYPES:
        _error(f"record_type is not a Part Q schema: {record_type!r}")

    try:
        identity = parse_condition_id(
            record["condition_id"],
            stage3,
        )
    except ConditionIdentityError as exc:
        raise Stage4SchemaError(
            f"invalid Part Q condition_id: {exc}"
        ) from exc

    if record_type == "student_cell_summary":
        _validate_student_cell_summary(
            record,
            identity,
            contract=contract,
            stage3=stage3,
        )
    elif record_type == "teacher_seed_inventory":
        _validate_teacher_seed_inventory(
            record,
            identity,
            contract=contract,
            stage3=stage3,
            stage3_registry=stage3_registry,
            stage3_registry_sha256=stage3_registry_sha256,
        )
    elif record_type == "excluded_development_output":
        _validate_excluded_development_output(
            record,
            identity,
        )
    elif record_type == "reproduction_comparison":
        _validate_reproduction_comparison(
            record,
            identity,
            contract=contract,
            stage3=stage3,
        )
    elif record_type == "analysis_freeze":
        _validate_analysis_freeze(
            record,
            identity,
            contract=contract,
            stage3=stage3,
            current_unresolved_decision_ids=current_unresolved_decision_ids,
        )

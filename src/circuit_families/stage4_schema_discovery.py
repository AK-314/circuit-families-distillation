"""Stage 4 validators for discovery-side record types.

UD-007 through UD-010 remain unresolved. This module validates only the
structural contract and typed/versioned references needed to resolve them
later without changing record shape.
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
    _seed_evidence,
    _validate_portable_artifact,
    _validate_sha256,
    _validate_uint,
    _validate_version_ref,
)
from circuit_families.stage4_seed_derivation import (
    SeedDerivationError,
    verify_seed_evidence,
)

COMPONENT_COUNT = 516

_PART_O_TYPES = frozenset(
    {
        "discovery_run",
        "native_budget_ledger",
        "exact_mask_evaluation_ledger",
        "endpoint_record",
    }
)


def _require_student_discovery_identity(identity) -> None:
    if identity.depth != 8:
        _error("Part O records require complete depth-8 condition identity")

    if identity.distillation_condition not in {
        "hard_target",
        "soft_target",
    }:
        _error("Part O records require hard_target or soft_target")


def _require_reference_match(
    reference: Mapping[str, Any],
    *,
    expected_record_type: str,
    record_identity,
    record_condition_id: str,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
    exact_complete_identity: bool,
):
    reference_identity = _reference_identity(
        reference,
        expected_record_type=expected_record_type,
        contract=contract,
        stage3=stage3,
        require_hash=True,
    )

    if exact_complete_identity:
        if reference["condition_id"] != record_condition_id:
            _error(
                f"{expected_record_type} reference must share exact "
                "complete condition identity"
            )
        return reference_identity

    if (
        reference_identity.teacher_seed != record_identity.teacher_seed
        or reference_identity.phase != record_identity.phase
        or (
            reference_identity.distillation_condition
            != record_identity.distillation_condition
        )
        or (
            reference_identity.student_initialization
            != record_identity.student_initialization
        )
    ):
        _error(
            f"{expected_record_type} reference must share the canonical "
            "student-condition depth-4 prefix"
        )

    return reference_identity


def _require_identity_setting_refs(identity, payload: Mapping[str, Any]) -> None:
    expected = {
        "discovery_method_ref": identity.discovery_method,
        "fidelity_definition_ref": identity.fidelity_setting,
        "component_cap_ref": identity.component_cap,
        "overlap_setting_ref": identity.overlap_setting,
    }

    for field, expected_value in expected.items():
        if field not in payload:
            _error(f"missing required identity-setting field {field!r}")

        _validate_version_ref(payload[field], label=field)

        if payload[field] != expected_value:
            _error(
                f"{field} must equal the corresponding canonical "
                "condition-identity value"
            )


def _validate_discovery_run(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    if record["provenance"]["producer_lane"] != "lane_c":
        _error("discovery_run producer_lane must be lane_c")

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "sealed_dense_model",
            "attempt_index",
            "retry_index",
            "discovery_method_ref",
            "method_budget_ref",
            "fidelity_definition_ref",
            "component_cap_ref",
            "overlap_setting_ref",
            "discovery_seed",
            "search_artifact",
        },
        label="discovery_run payload",
    )

    _require_reference_match(
        payload["sealed_dense_model"],
        expected_record_type="sealed_dense_model",
        record_identity=identity,
        record_condition_id=record["condition_id"],
        contract=contract,
        stage3=stage3,
        exact_complete_identity=False,
    )

    _validate_uint(payload["attempt_index"], label="attempt_index")
    _validate_uint(payload["retry_index"], label="retry_index")

    _require_identity_setting_refs(identity, payload)

    _validate_version_ref(
        payload["method_budget_ref"],
        label="method_budget_ref",
    )

    evidence = _seed_evidence(
        payload["discovery_seed"],
        label="discovery_seed",
    )

    try:
        seed_inputs = verify_seed_evidence(evidence, stage3)
    except SeedDerivationError as exc:
        _error(f"invalid discovery seed evidence: {exc}")

    if seed_inputs.condition_id != record["condition_id"]:
        _error("discovery seed condition_id mismatch")

    if seed_inputs.purpose != "discovery":
        _error("discovery seed purpose mismatch")

    if seed_inputs.attempt_index != payload["attempt_index"]:
        _error("discovery seed attempt_index mismatch")

    if seed_inputs.retry_index != payload["retry_index"]:
        _error("discovery seed retry_index mismatch")

    _validate_portable_artifact(
        payload["search_artifact"],
        label="discovery_run search_artifact",
        required_storage_class="external_large_object",
    )


def _validate_native_budget_ledger(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    if record["provenance"]["producer_lane"] != "lane_c":
        _error("native_budget_ledger producer_lane must be lane_c")

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "discovery_run",
            "method_budget_ref",
            "native_unit_ref",
            "native_budget_limit",
            "native_units_consumed",
            "entries",
        },
        label="native_budget_ledger payload",
    )

    _require_reference_match(
        payload["discovery_run"],
        expected_record_type="discovery_run",
        record_identity=identity,
        record_condition_id=record["condition_id"],
        contract=contract,
        stage3=stage3,
        exact_complete_identity=True,
    )

    _validate_version_ref(
        payload["method_budget_ref"],
        label="method_budget_ref",
    )
    _validate_version_ref(
        payload["native_unit_ref"],
        label="native_unit_ref",
    )
    _validate_uint(
        payload["native_budget_limit"],
        label="native_budget_limit",
    )
    _validate_uint(
        payload["native_units_consumed"],
        label="native_units_consumed",
    )

    entries = payload["entries"]

    if not isinstance(entries, list):
        _error("native budget entries must be an array")

    charged_total = 0

    for expected_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            _error("native budget entry must be an object")

        _exact_keys(
            entry,
            {
                "sequence_index",
                "operation_ref",
                "native_units_charged",
            },
            label="native budget entry",
        )

        _validate_uint(
            entry["sequence_index"],
            label="native budget sequence_index",
        )

        if entry["sequence_index"] != expected_index:
            _error(
                "native budget entry sequence_index must be contiguous "
                "from zero"
            )

        _validate_version_ref(
            entry["operation_ref"],
            label="native budget operation_ref",
        )
        _validate_uint(
            entry["native_units_charged"],
            label="native_units_charged",
        )

        charged_total += entry["native_units_charged"]

    if charged_total != payload["native_units_consumed"]:
        _error(
            "native_units_consumed must equal the sum of ledger charges"
        )

    if payload["native_units_consumed"] > payload["native_budget_limit"]:
        _error(
            "native_units_consumed cannot exceed native_budget_limit"
        )


def _validate_exact_entry(entry: Any, expected_index: int) -> None:
    if not isinstance(entry, Mapping):
        _error("exact evaluation entry must be an object")

    _exact_keys(
        entry,
        {
            "evaluation_order",
            "mask_sha256",
            "mask_kind",
            "retained_count",
            "retained_proportion",
            "fidelity_value",
            "qualifies",
            "budget_charged",
        },
        label="exact evaluation entry",
    )

    _validate_uint(
        entry["evaluation_order"],
        label="evaluation_order",
    )

    if entry["evaluation_order"] != expected_index:
        _error(
            "exact evaluation order must be contiguous from zero"
        )

    _validate_sha256(
        entry["mask_sha256"],
        label="mask_sha256",
    )

    if entry["mask_kind"] not in {"intact", "candidate"}:
        _error("mask_kind must be 'intact' or 'candidate'")

    _validate_uint(
        entry["retained_count"],
        label="retained_count",
    )

    if entry["retained_count"] > COMPONENT_COUNT:
        _error("retained_count cannot exceed 516")

    proportion = entry["retained_proportion"]

    if (
        isinstance(proportion, bool)
        or not isinstance(proportion, (int, float))
        or not 0.0 <= proportion <= 1.0
    ):
        _error("retained_proportion must be in [0, 1]")

    expected_proportion = entry["retained_count"] / COMPONENT_COUNT

    if abs(float(proportion) - expected_proportion) > 1e-12:
        _error(
            "retained_proportion must equal retained_count / 516"
        )

    fidelity = entry["fidelity_value"]

    if isinstance(fidelity, bool) or not isinstance(fidelity, (int, float)):
        _error("fidelity_value must be numeric")

    if not isinstance(entry["qualifies"], bool):
        _error("qualifies must be boolean")

    if not isinstance(entry["budget_charged"], bool):
        _error("budget_charged must be boolean")

    if entry["mask_kind"] == "intact":
        if entry["retained_count"] != COMPONENT_COUNT:
            _error("intact mask must retain all 516 components")
        if abs(float(entry["retained_proportion"]) - 1.0) > 1e-12:
            _error("intact mask retained_proportion must be 1.0")


def _validate_exact_mask_evaluation_ledger(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    if record["provenance"]["producer_lane"] != "lane_c":
        _error(
            "exact_mask_evaluation_ledger producer_lane must be lane_c"
        )

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "sealed_dense_model",
            "discovery_run",
            "fidelity_definition_ref",
            "exact_evaluation_allowance_ref",
            "exact_evaluation_allowance",
            "exact_evaluation_count",
            "charged_evaluation_count",
            "intact_baseline_present",
            "entries",
        },
        label="exact_mask_evaluation_ledger payload",
    )

    _require_reference_match(
        payload["sealed_dense_model"],
        expected_record_type="sealed_dense_model",
        record_identity=identity,
        record_condition_id=record["condition_id"],
        contract=contract,
        stage3=stage3,
        exact_complete_identity=False,
    )
    _require_reference_match(
        payload["discovery_run"],
        expected_record_type="discovery_run",
        record_identity=identity,
        record_condition_id=record["condition_id"],
        contract=contract,
        stage3=stage3,
        exact_complete_identity=True,
    )

    _validate_version_ref(
        payload["fidelity_definition_ref"],
        label="fidelity_definition_ref",
    )

    if payload["fidelity_definition_ref"] != identity.fidelity_setting:
        _error(
            "fidelity_definition_ref must equal condition identity"
        )

    _validate_version_ref(
        payload["exact_evaluation_allowance_ref"],
        label="exact_evaluation_allowance_ref",
    )

    _validate_uint(
        payload["exact_evaluation_allowance"],
        label="exact_evaluation_allowance",
    )
    _validate_uint(
        payload["exact_evaluation_count"],
        label="exact_evaluation_count",
    )
    _validate_uint(
        payload["charged_evaluation_count"],
        label="charged_evaluation_count",
    )

    if payload["intact_baseline_present"] is not True:
        _error("exact ledger requires intact_baseline_present=true")

    entries = payload["entries"]

    if not isinstance(entries, list) or not entries:
        _error("exact ledger entries must be a non-empty array")

    if payload["exact_evaluation_count"] != len(entries):
        _error(
            "exact_evaluation_count must equal number of ledger entries"
        )

    mask_hashes: set[str] = set()
    intact_count = 0
    charged_count = 0

    for expected_index, entry in enumerate(entries):
        _validate_exact_entry(entry, expected_index)

        mask_sha = entry["mask_sha256"]

        if mask_sha in mask_hashes:
            _error("exact ledger masks must be unique")

        mask_hashes.add(mask_sha)

        if entry["mask_kind"] == "intact":
            intact_count += 1

        if entry["budget_charged"]:
            charged_count += 1

    if intact_count != 1:
        _error("exact ledger must contain exactly one intact mask")

    if charged_count != payload["charged_evaluation_count"]:
        _error(
            "charged_evaluation_count must equal charged ledger entries"
        )

    if charged_count > payload["exact_evaluation_allowance"]:
        _error(
            "charged exact evaluations cannot exceed common allowance"
        )


def _validate_endpoint_record(
    record: Mapping[str, Any],
    identity,
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    if record["provenance"]["producer_lane"] != "lane_c":
        _error("endpoint_record producer_lane must be lane_c")

    if record["record_status"] != "sealed":
        _error("endpoint_record must be sealed")

    payload = record["payload"]

    _exact_keys(
        payload,
        {
            "exact_ledger",
            "fidelity_definition_ref",
            "component_cap_ref",
            "overlap_setting_ref",
            "endpoint_1",
            "endpoint_2",
        },
        label="endpoint_record payload",
    )

    _require_reference_match(
        payload["exact_ledger"],
        expected_record_type="exact_mask_evaluation_ledger",
        record_identity=identity,
        record_condition_id=record["condition_id"],
        contract=contract,
        stage3=stage3,
        exact_complete_identity=True,
    )

    for field, expected in (
        ("fidelity_definition_ref", identity.fidelity_setting),
        ("component_cap_ref", identity.component_cap),
        ("overlap_setting_ref", identity.overlap_setting),
    ):
        _validate_version_ref(payload[field], label=field)
        if payload[field] != expected:
            _error(f"{field} must equal condition identity")

    endpoint_1 = payload["endpoint_1"]

    if not isinstance(endpoint_1, Mapping):
        _error("endpoint_1 must be an object")

    _exact_keys(
        endpoint_1,
        {
            "smallest_recovered_component_proportion",
            "qualifying_mask_sha256",
            "procedure_censored",
            "interpretation",
            "global_minimum_claim",
        },
        label="endpoint_1",
    )

    proportion = endpoint_1["smallest_recovered_component_proportion"]

    if (
        isinstance(proportion, bool)
        or not isinstance(proportion, (int, float))
        or not 0.0 <= proportion <= 1.0
    ):
        _error(
            "endpoint_1 smallest_recovered_component_proportion "
            "must be in [0, 1]"
        )

    _validate_sha256(
        endpoint_1["qualifying_mask_sha256"],
        label="endpoint_1 qualifying_mask_sha256",
    )

    if not isinstance(endpoint_1["procedure_censored"], bool):
        _error("endpoint_1 procedure_censored must be boolean")

    if (
        endpoint_1["interpretation"]
        != "smallest_qualifying_proportion_in_exact_ledger"
    ):
        _error("endpoint_1 interpretation is invalid")

    if endpoint_1["global_minimum_claim"] is not False:
        _error("endpoint_1 cannot claim a global minimum")

    endpoint_2 = payload["endpoint_2"]

    if not isinstance(endpoint_2, Mapping):
        _error("endpoint_2 must be an object")

    _exact_keys(
        endpoint_2,
        {
            "packing_lower_bound",
            "packed_mask_sha256s",
            "packing_rule_ref",
            "interpretation",
            "true_packing_number_claim",
        },
        label="endpoint_2",
    )

    _validate_uint(
        endpoint_2["packing_lower_bound"],
        label="endpoint_2 packing_lower_bound",
    )

    packed = endpoint_2["packed_mask_sha256s"]

    if not isinstance(packed, list):
        _error("packed_mask_sha256s must be an array")

    if len(packed) != endpoint_2["packing_lower_bound"]:
        _error(
            "packing_lower_bound must equal packed_mask_sha256s length"
        )

    if len(set(packed)) != len(packed):
        _error("packed_mask_sha256s must be unique")

    for digest in packed:
        _validate_sha256(
            digest,
            label="endpoint_2 packed mask SHA-256",
        )

    _validate_version_ref(
        endpoint_2["packing_rule_ref"],
        label="endpoint_2 packing_rule_ref",
    )

    if (
        endpoint_2["interpretation"]
        != "procedure_dependent_packing_lower_bound"
    ):
        _error("endpoint_2 interpretation is invalid")

    if endpoint_2["true_packing_number_claim"] is not False:
        _error("endpoint_2 cannot claim the true packing number")


def validate_part_o_record(
    record: Mapping[str, Any],
    *,
    contract: CommonSchemaContract,
    stage3: Stage3AvailabilityIndex,
) -> None:
    """Validate one Part O record including cross-field semantics."""
    validate_common_envelope(
        record,
        contract=contract,
        stage3=stage3,
    )

    record_type = record["record_type"]

    if record_type not in _PART_O_TYPES:
        _error(f"record_type is not a Part O schema: {record_type!r}")

    try:
        identity = parse_condition_id(record["condition_id"], stage3)
    except ConditionIdentityError as exc:
        raise Stage4SchemaError(
            f"invalid Part O condition_id: {exc}"
        ) from exc

    _require_student_discovery_identity(identity)

    if record_type == "discovery_run":
        _validate_discovery_run(
            record,
            identity,
            contract=contract,
            stage3=stage3,
        )
    elif record_type == "native_budget_ledger":
        _validate_native_budget_ledger(
            record,
            identity,
            contract=contract,
            stage3=stage3,
        )
    elif record_type == "exact_mask_evaluation_ledger":
        _validate_exact_mask_evaluation_ledger(
            record,
            identity,
            contract=contract,
            stage3=stage3,
        )
    elif record_type == "endpoint_record":
        _validate_endpoint_record(
            record,
            identity,
            contract=contract,
            stage3=stage3,
        )

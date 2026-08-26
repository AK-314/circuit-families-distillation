"""Stage 7A teacher-seed inventory, Stage 5D bridge, and exclusion records.

The Stage 5D analysis implementation is reused directly on its accepted
synthetic fixture. Stage 7A does not create a second hierarchical reducer.
The Stage 7 inventory is linked to that accepted report bundle through an
explicit technical integration record while preserving teacher-seed population
units and separate hard/soft student tables.

Every endpoint-like Stage 7 fixture value is registered as excluded
development under the accepted Stage 2 firewall and is fail-closed against
primary scientific consumption.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from circuit_families.stage5d import (
    OUTPUT_OBJECT_IDS,
    build_stage5d_output_bundle,
    canonical_json_bytes,
    load_and_normalize_ingestion,
    load_technical_analysis_profile_set,
    validate_stage5d_output_bundle,
)

INVENTORY_SCHEMA_VERSION: Final = "stage7-teacher-seed-inventory/v1"
ANALYSIS_BRIDGE_SCHEMA_VERSION: Final = "stage7-stage5d-analysis-bridge/v1"
EXCLUSION_ENTRY_SCHEMA_VERSION: Final = "stage7-excluded-endpoint-entry/v1"
REPORT_SCHEMA_VERSION: Final = "stage7-technical-integration-report/v1"

EXPECTED_STAGE5D_PROFILE_ID: Final = "fixture_median_min2"
POPULATION_UNIT: Final = "teacher_seed"
STUDENT_MEMBER_UNIT: Final = "student_initialization"

_ALLOWED_SUBJECT_STATES: Final = {
    "teacher_direct",
    "eligible",
    "failed",
}

_ALLOWED_METHOD_STATES: Final = {
    "completed",
    "missing",
}

_ALLOWED_ENDPOINT_STATES: Final = {
    "defined",
    "missing",
}

_REQUIRED_EXCLUSION_FIELDS: Final = (
    "exclusion_id",
    "artifact_identity",
    "development_context",
    "exclusion_reason",
    "endpoint_values_emitted",
    "primary_analysis_eligible",
    "scientific_selection_eligible",
    "regeneration_required",
    "regenerate_after",
    "disposition",
    "promotion_in_place_permitted",
)


class Stage7InventoryReportingError(ValueError):
    """Raised when Part F violates hierarchy or exclusion boundaries."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()

    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage7InventoryReportingError(
            f"{label} must be a mapping"
        )

    return value


def build_teacher_seed_inventory(
    *,
    teacher_seed: int,
    phase: str,
    distillation_result: Mapping[str, Any],
    discovery_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain teacher, hard/soft, eligible/failed, method and endpoint states."""
    if (
        isinstance(teacher_seed, bool)
        or not isinstance(teacher_seed, int)
        or teacher_seed < 0
    ):
        raise Stage7InventoryReportingError(
            "teacher_seed must be a non-negative integer"
        )

    if not isinstance(phase, str) or not phase:
        raise Stage7InventoryReportingError(
            "phase must be non-empty"
        )

    distillation = _require_mapping(
        distillation_result,
        label="distillation_result",
    )
    discovery = _require_mapping(
        discovery_result,
        label="discovery_result",
    )

    if distillation.get("scientific_data") is not False:
        raise Stage7InventoryReportingError(
            "inventory accepts technical distillation only"
        )

    if discovery.get("scientific_data") is not False:
        raise Stage7InventoryReportingError(
            "inventory accepts technical discovery only"
        )

    runs = discovery.get("runs")

    if not isinstance(runs, list):
        raise Stage7InventoryReportingError(
            "discovery runs must be a list"
        )

    run_by_subject_method = {}

    for record in runs:
        run = _require_mapping(
            record,
            label="discovery run",
        )
        key = (
            run["subject_id"],
            run["discovery_method"],
        )

        if key in run_by_subject_method:
            raise Stage7InventoryReportingError(
                "duplicate subject/method discovery run"
            )

        run_by_subject_method[key] = run

    methods = tuple(
        discovery.get(
            "accepted_adapter_methods",
            (),
        )
    )

    if not methods:
        raise Stage7InventoryReportingError(
            "accepted discovery method roster is empty"
        )

    hard = _require_mapping(
        distillation["hard"],
        label="hard distillation summary",
    )
    soft = _require_mapping(
        distillation["soft"],
        label="soft distillation summary",
    )

    subject_specs = (
        {
            "subject_id": "technical-direct-teacher",
            "subject_role": "direct_teacher",
            "distillation_condition": "teacher_direct",
            "subject_state": "teacher_direct",
            "student_initialization": None,
            "source_reference_sha256": next(
                record["source_reference_sha256"]
                for record in runs
                if record["subject_id"] == "technical-direct-teacher"
            ),
        },
        {
            "subject_id": "technical-hard-eligible-student",
            "subject_role": "hard_target_student",
            "distillation_condition": "hard",
            "subject_state": "eligible",
            "student_initialization": 0,
            "source_reference_sha256": hard[
                "sealed_dense_model_sha256"
            ],
        },
        {
            "subject_id": "technical-hard-failed-student",
            "subject_role": "failed_hard_target_student",
            "distillation_condition": "hard",
            "subject_state": "failed",
            "student_initialization": 1,
            "source_reference_sha256": hard[
                "attempt_record_sha256"
            ][1],
        },
        {
            "subject_id": "technical-soft-eligible-student",
            "subject_role": "soft_target_student",
            "distillation_condition": "soft",
            "subject_state": "eligible",
            "student_initialization": 0,
            "source_reference_sha256": soft[
                "sealed_dense_model_sha256"
            ],
        },
        {
            "subject_id": "technical-soft-failed-student",
            "subject_role": "failed_soft_target_student",
            "distillation_condition": "soft",
            "subject_state": "failed",
            "student_initialization": 1,
            "source_reference_sha256": soft[
                "attempt_record_sha256"
            ][1],
        },
    )

    rows = []

    for subject in subject_specs:
        if subject["subject_state"] not in _ALLOWED_SUBJECT_STATES:
            raise Stage7InventoryReportingError(
                "invalid subject state"
            )

        for method in methods:
            run = run_by_subject_method.get(
                (
                    subject["subject_id"],
                    method,
                )
            )

            if run is None:
                method_state = "missing"
                endpoint1_state = "missing"
                endpoint2_state = "missing"
                endpoint1 = None
                endpoint2 = None
            else:
                method_state = run["stopping_status"]
                endpoint1_state = "defined"
                endpoint2_state = "defined"
                endpoint1 = dict(run["endpoint1"])
                endpoint2 = dict(run["endpoint2"])

            if method_state not in _ALLOWED_METHOD_STATES:
                raise Stage7InventoryReportingError(
                    "inventory method state is invalid"
                )

            if endpoint1_state not in _ALLOWED_ENDPOINT_STATES:
                raise Stage7InventoryReportingError(
                    "Endpoint 1 inventory state is invalid"
                )

            if endpoint2_state not in _ALLOWED_ENDPOINT_STATES:
                raise Stage7InventoryReportingError(
                    "Endpoint 2 inventory state is invalid"
                )

            if subject["subject_state"] == "failed":
                if method_state != "missing":
                    raise Stage7InventoryReportingError(
                        "failed student unexpectedly has discovery state"
                    )

                if endpoint1 is not None or endpoint2 is not None:
                    raise Stage7InventoryReportingError(
                        "failed student unexpectedly has endpoint values"
                    )

            rows.append(
                {
                    "teacher_seed": teacher_seed,
                    "phase": phase,
                    "subject_id": subject["subject_id"],
                    "subject_role": subject["subject_role"],
                    "distillation_condition": subject[
                        "distillation_condition"
                    ],
                    "subject_state": subject["subject_state"],
                    "student_initialization": subject[
                        "student_initialization"
                    ],
                    "population_unit": POPULATION_UNIT,
                    "student_member_unit": STUDENT_MEMBER_UNIT,
                    "source_reference_sha256": subject[
                        "source_reference_sha256"
                    ],
                    "discovery_method": method,
                    "method_state": method_state,
                    "endpoint1_state": endpoint1_state,
                    "endpoint2_state": endpoint2_state,
                    "endpoint1": endpoint1,
                    "endpoint2": endpoint2,
                }
            )

    rows = sorted(
        rows,
        key=canonical_json_bytes,
    )

    subject_states = {
        row["subject_state"]
        for row in rows
    }

    if subject_states != {
        "teacher_direct",
        "eligible",
        "failed",
    }:
        raise Stage7InventoryReportingError(
            "inventory does not retain all required subject states"
        )

    if not any(
        row["method_state"] == "missing"
        for row in rows
    ):
        raise Stage7InventoryReportingError(
            "inventory does not retain missing method states"
        )

    if not any(
        row["endpoint1_state"] == "missing"
        and row["endpoint2_state"] == "missing"
        for row in rows
    ):
        raise Stage7InventoryReportingError(
            "inventory does not retain missing endpoint states"
        )

    hard_rows = [
        row
        for row in rows
        if row["distillation_condition"] == "hard"
    ]

    soft_rows = [
        row
        for row in rows
        if row["distillation_condition"] == "soft"
    ]

    hard_keys = {
        (
            row["subject_id"],
            row["discovery_method"],
        )
        for row in hard_rows
    }

    soft_keys = {
        (
            row["subject_id"],
            row["discovery_method"],
        )
        for row in soft_rows
    }

    if hard_keys & soft_keys:
        raise Stage7InventoryReportingError(
            "hard and soft inventory identities collided"
        )

    record = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "classification": "synthetic_technical_only",
        "scientific_data": False,
        "production_eligible": False,
        "teacher_seed": teacher_seed,
        "phase": phase,
        "population_unit": POPULATION_UNIT,
        "student_member_unit": STUDENT_MEMBER_UNIT,
        "student_initializations_are_population_replicates": False,
        "hard_soft_pooled": False,
        "rows": rows,
    }

    record["sha256"] = _canonical_sha256(
        record
    )

    return record


def build_stage5d_analysis_bridge(
    *,
    repository_root: str | Path,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Link Stage 7 inventory to the accepted Stage 5D fixture/report machinery."""
    root = Path(
        repository_root
    ).resolve(strict=True)

    inventory_record = _require_mapping(
        inventory,
        label="inventory",
    )

    if inventory_record.get("population_unit") != POPULATION_UNIT:
        raise Stage7InventoryReportingError(
            "Stage 7 inventory population unit must be teacher_seed"
        )

    if (
        inventory_record.get(
            "student_initializations_are_population_replicates"
        )
        is not False
    ):
        raise Stage7InventoryReportingError(
            "students may not be promoted to population replicates"
        )

    ingestion_path = (
        root
        / "tests/fixtures/stage5d/"
        "synthetic_ingestion_envelope_v1.json"
    )

    profiles_path = (
        root
        / "followup/configs/stage5d/"
        "technical_analysis_profiles_v1.json"
    )

    normalized = load_and_normalize_ingestion(
        ingestion_path
    )

    profile = load_technical_analysis_profile_set(
        profiles_path
    ).require(
        EXPECTED_STAGE5D_PROFILE_ID
    )

    bundle = build_stage5d_output_bundle(
        normalized,
        profile,
    )

    validate_stage5d_output_bundle(
        bundle,
        normalized,
        profile,
    )

    objects = bundle["output_objects"]

    if set(objects) != set(OUTPUT_OBJECT_IDS):
        raise Stage7InventoryReportingError(
            "accepted Stage 5D output object roster changed"
        )

    hard_rows = objects[
        "hard_student_summaries"
    ]["rows"]

    soft_rows = objects[
        "soft_student_summaries"
    ]["rows"]

    if not hard_rows or not soft_rows:
        raise Stage7InventoryReportingError(
            "accepted Stage 5D fixture must exercise hard and soft tables"
        )

    if {
        row["key"]["distillation_condition"]
        for row in hard_rows
    } != {"hard"}:
        raise Stage7InventoryReportingError(
            "accepted Stage 5D hard table contains non-hard rows"
        )

    if {
        row["key"]["distillation_condition"]
        for row in soft_rows
    } != {"soft"}:
        raise Stage7InventoryReportingError(
            "accepted Stage 5D soft table contains non-soft rows"
        )

    hard_keys = {
        canonical_json_bytes(
            row["key"]
        )
        for row in hard_rows
    }

    soft_keys = {
        canonical_json_bytes(
            row["key"]
        )
        for row in soft_rows
    }

    if hard_keys & soft_keys:
        raise Stage7InventoryReportingError(
            "accepted Stage 5D hard/soft table identities overlap"
        )

    population_rows = (
        objects[
            "phase_population_summaries"
        ]["rows"]
        + objects[
            "teacher_student_population_summaries"
        ]["rows"]
    )

    if not population_rows:
        raise Stage7InventoryReportingError(
            "accepted Stage 5D fixture lacks population summaries"
        )

    for row in population_rows:
        if row["population_unit"] != POPULATION_UNIT:
            raise Stage7InventoryReportingError(
                "accepted Stage 5D population unit is not teacher_seed"
            )

        if "number_student_realizations" in row:
            raise Stage7InventoryReportingError(
                "accepted Stage 5D population summary pseudoreplicates students"
            )

    manifest = bundle[
        "reconstruction_manifest"
    ]

    if manifest["classification"] != "synthetic_technical_only":
        raise Stage7InventoryReportingError(
            "accepted Stage 5D bundle classification changed"
        )

    if manifest["scientific_data"] is not False:
        raise Stage7InventoryReportingError(
            "accepted Stage 5D fixture became scientific"
        )

    if manifest["production_eligible"] is not False:
        raise Stage7InventoryReportingError(
            "accepted Stage 5D fixture became production eligible"
        )

    if manifest["reducer_configuration"]["population_unit"] != POPULATION_UNIT:
        raise Stage7InventoryReportingError(
            "Stage 5D reconstruction population unit changed"
        )

    if (
        manifest["reducer_configuration"]["student_member_unit"]
        != STUDENT_MEMBER_UNIT
    ):
        raise Stage7InventoryReportingError(
            "Stage 5D student member unit changed"
        )

    return {
        "schema_version": ANALYSIS_BRIDGE_SCHEMA_VERSION,
        "classification": "synthetic_technical_only",
        "scientific_data": False,
        "production_eligible": False,
        "inventory_sha256": inventory_record["sha256"],
        "technical_analysis_profile_id": profile.profile_id,
        "stage5d_output_bundle_sha256": bundle["sha256"],
        "stage5d_output_object_sha256": {
            object_id: objects[object_id]["sha256"]
            for object_id in OUTPUT_OBJECT_IDS
        },
        "hard_soft_tables_separate": True,
        "hard_soft_pooled": False,
        "population_unit": POPULATION_UNIT,
        "student_member_unit": STUDENT_MEMBER_UNIT,
        "student_initializations_are_population_replicates": False,
        "stage5d_reducer_reimplemented": False,
        "resolved_decisions": [],
    }


def build_endpoint_exclusion_records(
    *,
    discovery_result: Mapping[str, Any],
    exclusion_register: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Register every endpoint-like Stage 7 fixture value as excluded."""
    discovery = _require_mapping(
        discovery_result,
        label="discovery_result",
    )

    register = _require_mapping(
        exclusion_register,
        label="exclusion_register",
    )

    required_fields = tuple(
        register.get(
            "required_entry_fields",
            (),
        )
    )

    if required_fields != _REQUIRED_EXCLUSION_FIELDS:
        raise Stage7InventoryReportingError(
            "Stage 2 exclusion required-entry schema changed"
        )

    firewall = _require_mapping(
        register.get("firewall"),
        label="Stage 2 firewall",
    )

    if firewall.get("primary_analysis_eligible") is not False:
        raise Stage7InventoryReportingError(
            "Stage 2 primary-analysis firewall is not active"
        )

    if firewall.get("scientific_selection_eligible") is not False:
        raise Stage7InventoryReportingError(
            "Stage 2 scientific-selection firewall is not active"
        )

    if firewall.get("post_freeze_regeneration_required") is not True:
        raise Stage7InventoryReportingError(
            "Stage 2 post-freeze regeneration firewall is not active"
        )

    allowed_dispositions = set(
        register.get(
            "allowed_dispositions",
            (),
        )
    )

    if "registered_excluded" not in allowed_dispositions:
        raise Stage7InventoryReportingError(
            "registered_excluded disposition unavailable"
        )

    runs = discovery.get("runs")

    if not isinstance(runs, list):
        raise Stage7InventoryReportingError(
            "discovery runs must be a list"
        )

    entries = []

    for run in runs:
        run_record = _require_mapping(
            run,
            label="discovery run",
        )

        for endpoint_name in (
            "endpoint1",
            "endpoint2",
        ):
            endpoint = _require_mapping(
                run_record[endpoint_name],
                label=endpoint_name,
            )

            artifact_identity = (
                f"{run_record['run_id']}:{endpoint_name}"
            )

            entry = {
                "exclusion_id": (
                    "stage7a-excluded-"
                    + _canonical_sha256(
                        artifact_identity
                    )[:20]
                ),
                "artifact_identity": artifact_identity,
                "development_context": (
                    "stage7a_portable_synthetic_technical_integration"
                ),
                "exclusion_reason": (
                    "endpoint_like_value_emitted_before_registered_"
                    "checkpoint_fixture_and_protocol_freeze"
                ),
                "endpoint_values_emitted": True,
                "primary_analysis_eligible": False,
                "scientific_selection_eligible": False,
                "regeneration_required": True,
                "regenerate_after": (
                    "registered_protocol_freeze"
                ),
                "disposition": "registered_excluded",
                "promotion_in_place_permitted": False,
            }

            if tuple(entry) != _REQUIRED_EXCLUSION_FIELDS:
                raise Stage7InventoryReportingError(
                    "exclusion entry field order/schema mismatch"
                )

            if endpoint_name == "endpoint1":
                if endpoint.get("retained_proportion") is None:
                    raise Stage7InventoryReportingError(
                        "Endpoint 1 exclusion lacks emitted value"
                    )

            if endpoint_name == "endpoint2":
                if endpoint.get("packing_lower_bound") is None:
                    raise Stage7InventoryReportingError(
                        "Endpoint 2 exclusion lacks emitted value"
                    )

            entries.append(
                entry
            )

    identities = [
        entry["artifact_identity"]
        for entry in entries
    ]

    if len(set(identities)) != len(identities):
        raise Stage7InventoryReportingError(
            "endpoint exclusion identities must be unique"
        )

    expected_count = len(runs) * 2

    if len(entries) != expected_count:
        raise Stage7InventoryReportingError(
            "not every endpoint-like fixture output received exclusion"
        )

    return tuple(
        entries
    )


def assert_rejected_as_primary_scientific_input(
    record: Mapping[str, Any],
) -> None:
    """Reject Stage 7A excluded development before primary analysis."""
    entry = _require_mapping(
        record,
        label="excluded development record",
    )

    if entry.get("primary_analysis_eligible") is not False:
        raise Stage7InventoryReportingError(
            "record does not carry primary-analysis exclusion"
        )

    if entry.get("scientific_selection_eligible") is not False:
        raise Stage7InventoryReportingError(
            "record does not carry scientific-selection exclusion"
        )

    if entry.get("regeneration_required") is not True:
        raise Stage7InventoryReportingError(
            "record does not require post-freeze regeneration"
        )

    if entry.get("promotion_in_place_permitted") is not False:
        raise Stage7InventoryReportingError(
            "record permits forbidden in-place promotion"
        )

    raise Stage7InventoryReportingError(
        "excluded Stage 7A development record rejected as primary scientific input"
    )


def build_part_f_report(
    *,
    inventory: Mapping[str, Any],
    analysis_bridge: Mapping[str, Any],
    exclusion_entries: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Create deterministic Part F technical integration reporting evidence."""
    inventory_record = _require_mapping(
        inventory,
        label="inventory",
    )

    analysis_record = _require_mapping(
        analysis_bridge,
        label="analysis_bridge",
    )

    if (
        analysis_record.get("inventory_sha256")
        != inventory_record.get("sha256")
    ):
        raise Stage7InventoryReportingError(
            "analysis bridge is not linked to inventory"
        )

    if analysis_record.get("hard_soft_pooled") is not False:
        raise Stage7InventoryReportingError(
            "hard and soft outputs may not be pooled"
        )

    if (
        analysis_record.get(
            "student_initializations_are_population_replicates"
        )
        is not False
    ):
        raise Stage7InventoryReportingError(
            "student initializations may not be population replicates"
        )

    exclusion_rows = [
        dict(entry)
        for entry in exclusion_entries
    ]

    for entry in exclusion_rows:
        try:
            assert_rejected_as_primary_scientific_input(
                entry
            )
        except Stage7InventoryReportingError as exc:
            if (
                str(exc)
                != "excluded Stage 7A development record rejected "
                "as primary scientific input"
            ):
                raise
        else:
            raise Stage7InventoryReportingError(
                "excluded endpoint record unexpectedly accepted as primary"
            )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "classification": "synthetic_technical_only",
        "scientific_data": False,
        "production_eligible": False,
        "production_default": False,
        "inventory_sha256": inventory_record["sha256"],
        "stage5d_output_bundle_sha256": analysis_record[
            "stage5d_output_bundle_sha256"
        ],
        "hard_soft_tables_separate": analysis_record[
            "hard_soft_tables_separate"
        ],
        "hard_soft_pooled": False,
        "population_unit": POPULATION_UNIT,
        "student_member_unit": STUDENT_MEMBER_UNIT,
        "student_initializations_are_population_replicates": False,
        "endpoint_like_fixture_output_count": len(exclusion_rows),
        "excluded_endpoint_like_fixture_output_count": len(exclusion_rows),
        "primary_scientific_acceptance_count": 0,
        "scientific_selection_acceptance_count": 0,
        "post_freeze_regeneration_required": True,
        "registered_fixture_execution": False,
        "real_scientific_analysis": False,
        "exclusion_entries_sha256": _canonical_sha256(
            exclusion_rows
        ),
    }

    report["sha256"] = _canonical_sha256(
        report
    )

    return report

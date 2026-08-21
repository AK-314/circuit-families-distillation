from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .models import (
    SYNTHETIC_CLASSIFICATION,
    SYNTHETIC_UNIVERSE_SCHEMA_VERSION,
    DirectTeacherEndpointRecord,
    MethodBudgetRecord,
    StudentAttemptRecord,
    StudentEligibilityRecord,
    StudentEndpointRecord,
    SyntheticCellExpectation,
    SyntheticRecordError,
    SyntheticUniverse,
    TeacherSeedInventory,
    expect_mapping,
)


def _parse_records(
    raw: Any,
    label: str,
    parser: Any,
) -> tuple[Any, ...]:
    if not isinstance(raw, list):
        raise SyntheticRecordError(f"{label} must be a list")

    return tuple(
        parser.from_mapping(
            expect_mapping(item, f"{label}[{index}]")
        )
        for index, item in enumerate(raw)
    )


def _require_unique(
    values: Iterable[str | int],
    label: str,
) -> None:
    materialized = list(values)
    if len(materialized) != len(set(materialized)):
        raise SyntheticRecordError(f"duplicate {label} detected")


def _inventory_map(
    universe: SyntheticUniverse,
) -> dict[int, TeacherSeedInventory]:
    return {
        inventory.teacher_seed: inventory
        for inventory in universe.teacher_inventories
    }


def _budget_pairs(
    universe: SyntheticUniverse,
) -> set[tuple[str, str]]:
    return {
        (record.method_id, record.budget_id)
        for record in universe.method_budgets
    }


def _validate_phase(
    inventories: Mapping[int, TeacherSeedInventory],
    teacher_seed: int,
    phase: str,
    *,
    allow_unavailable: bool,
    label: str,
) -> None:
    inventory = inventories.get(teacher_seed)
    if inventory is None:
        raise SyntheticRecordError(
            f"{label} references unknown teacher seed {teacher_seed}"
        )

    phase_state = inventory.phase_state(phase)
    if phase_state is None:
        raise SyntheticRecordError(
            f"{label} references phase {phase!r} absent from teacher inventory"
        )

    if phase_state == "unavailable" and not allow_unavailable:
        raise SyntheticRecordError(
            f"{label} references unavailable phase {phase!r}"
        )


def validate_synthetic_universe_links(
    universe: SyntheticUniverse,
) -> None:
    _require_unique(
        (record.method_id for record in universe.method_budgets),
        "method_id",
    )
    _require_unique(
        (record.budget_id for record in universe.method_budgets),
        "budget_id",
    )
    _require_unique(
        (
            inventory.teacher_seed
            for inventory in universe.teacher_inventories
        ),
        "teacher_seed inventory",
    )
    _require_unique(
        (record.attempt_id for record in universe.student_attempts),
        "attempt_id",
    )
    _require_unique(
        (
            record.eligibility_id
            for record in universe.eligibility_records
        ),
        "eligibility_id",
    )
    _require_unique(
        (
            record.record_id
            for record in universe.direct_teacher_endpoints
        ),
        "direct teacher endpoint record_id",
    )
    _require_unique(
        (record.record_id for record in universe.student_endpoints),
        "student endpoint record_id",
    )
    _require_unique(
        (record.cell_id for record in universe.cell_expectations),
        "cell_id",
    )

    inventories = _inventory_map(universe)
    budget_pairs = _budget_pairs(universe)

    attempts = {
        record.attempt_id: record
        for record in universe.student_attempts
    }

    eligibilities: dict[str, StudentEligibilityRecord] = {}
    for record in universe.eligibility_records:
        if record.attempt_id in eligibilities:
            raise SyntheticRecordError(
                f"attempt {record.attempt_id} has duplicate eligibility records"
            )
        if record.attempt_id not in attempts:
            raise SyntheticRecordError(
                f"eligibility record references unknown attempt "
                f"{record.attempt_id}"
            )
        eligibilities[record.attempt_id] = record

    if set(eligibilities) != set(attempts):
        missing = sorted(set(attempts) - set(eligibilities))
        raise SyntheticRecordError(
            f"attempts missing eligibility records: {missing}"
        )

    for attempt in universe.student_attempts:
        _validate_phase(
            inventories,
            attempt.teacher_seed,
            attempt.phase,
            allow_unavailable=False,
            label=f"attempt {attempt.attempt_id}",
        )

        eligibility = eligibilities[attempt.attempt_id]
        if attempt.outcome == "failed" and eligibility.status == "eligible":
            raise SyntheticRecordError(
                f"failed attempt {attempt.attempt_id} cannot be eligible"
            )

    for record in universe.direct_teacher_endpoints:
        identity = record.identity

        _validate_phase(
            inventories,
            identity.teacher_seed,
            identity.phase,
            allow_unavailable=record.state == "unavailable",
            label=f"direct endpoint {record.record_id}",
        )

        if (identity.method_id, identity.budget_id) not in budget_pairs:
            raise SyntheticRecordError(
                f"direct endpoint {record.record_id} has incompatible "
                "method/budget identity"
            )

        inventory = inventories[identity.teacher_seed]
        if (
            inventory.phase_state(identity.phase) == "unavailable"
            and record.state != "unavailable"
        ):
            raise SyntheticRecordError(
                f"direct endpoint {record.record_id} must preserve "
                "unavailable phase state"
            )

    for record in universe.student_endpoints:
        identity = record.identity
        attempt = attempts.get(record.attempt_id)

        if attempt is None:
            raise SyntheticRecordError(
                f"student endpoint {record.record_id} references unknown "
                f"attempt {record.attempt_id}"
            )

        expected = (
            attempt.teacher_seed,
            attempt.phase,
            attempt.distillation_condition,
            attempt.student_initialization,
        )
        observed = (
            identity.teacher_seed,
            identity.phase,
            identity.distillation_condition,
            identity.student_initialization,
        )

        if observed != expected:
            raise SyntheticRecordError(
                f"student endpoint {record.record_id} identity does not "
                "match its attempt"
            )

        _validate_phase(
            inventories,
            identity.teacher_seed,
            identity.phase,
            allow_unavailable=False,
            label=f"student endpoint {record.record_id}",
        )

        if (identity.method_id, identity.budget_id) not in budget_pairs:
            raise SyntheticRecordError(
                f"student endpoint {record.record_id} has incompatible "
                "method/budget identity"
            )

        eligibility = eligibilities[record.attempt_id]
        if record.state == "defined" and eligibility.status != "eligible":
            raise SyntheticRecordError(
                f"defined student endpoint {record.record_id} cannot come "
                f"from {eligibility.status} attempt"
            )

    for cell in universe.cell_expectations:
        identity = cell.identity

        _validate_phase(
            inventories,
            identity.teacher_seed,
            identity.phase,
            allow_unavailable=cell.state in {"unavailable", "inapplicable"},
            label=f"cell {cell.cell_id}",
        )

        if (identity.method_id, identity.budget_id) not in budget_pairs:
            raise SyntheticRecordError(
                f"cell {cell.cell_id} has incompatible method/budget identity"
            )

        inventory = inventories[identity.teacher_seed]
        phase_state = inventory.phase_state(identity.phase)

        if phase_state == "unavailable" and cell.state != "unavailable":
            raise SyntheticRecordError(
                f"cell {cell.cell_id} must preserve unavailable phase state"
            )


def validate_required_synthetic_coverage(
    universe: SyntheticUniverse,
) -> None:
    seeds = {
        inventory.teacher_seed
        for inventory in universe.teacher_inventories
    }
    if len(seeds) < 2:
        raise SyntheticRecordError(
            "fixture universe requires multiple teacher seeds"
        )

    for inventory in universe.teacher_inventories:
        available = [
            phase
            for phase, state in inventory.phase_status
            if state == "available"
        ]
        if len(available) < 2:
            raise SyntheticRecordError(
                "each synthetic teacher seed must contain repeated phases"
            )

    conditions = {
        attempt.distillation_condition
        for attempt in universe.student_attempts
    }
    if conditions != {"hard", "soft"}:
        raise SyntheticRecordError(
            "fixture universe must contain both hard and soft students"
        )

    initializations = {
        attempt.student_initialization
        for attempt in universe.student_attempts
    }
    if len(initializations) < 2:
        raise SyntheticRecordError(
            "fixture universe requires multiple student initializations"
        )

    if len(universe.method_budgets) < 2:
        raise SyntheticRecordError(
            "fixture universe requires multiple discovery methods"
        )

    native_budgets = {
        record.native_budget
        for record in universe.method_budgets
    }
    if len(native_budgets) < 2:
        raise SyntheticRecordError(
            "fixture universe must contain unequal method budgets"
        )

    if not any(
        attempt.outcome == "failed"
        for attempt in universe.student_attempts
    ):
        raise SyntheticRecordError(
            "fixture universe must contain a failed attempt"
        )

    cell_states = {
        cell.state
        for cell in universe.cell_expectations
    }
    if "unavailable" not in cell_states:
        raise SyntheticRecordError(
            "fixture universe must contain an unavailable cell"
        )
    if "unresolved" not in cell_states:
        raise SyntheticRecordError(
            "fixture universe must contain an unresolved cell"
        )

    all_endpoints = (
        *universe.direct_teacher_endpoints,
        *universe.student_endpoints,
    )

    has_endpoint_one_boundary = any(
        record.identity.endpoint_id == "endpoint_1"
        and record.state == "defined"
        and record.value == 1.0
        for record in all_endpoints
    )
    if not has_endpoint_one_boundary:
        raise SyntheticRecordError(
            "fixture universe must preserve endpoint_1 value 1.0"
        )

    has_endpoint_two_zero = any(
        record.identity.endpoint_id == "endpoint_2"
        and record.state == "defined"
        and record.value == 0
        for record in all_endpoints
    )
    if not has_endpoint_two_zero:
        raise SyntheticRecordError(
            "fixture universe must preserve endpoint_2 packing value 0"
        )


def synthetic_universe_from_mapping(
    raw: Mapping[str, Any],
) -> SyntheticUniverse:
    expected = {
        "schema_version",
        "classification",
        "method_budgets",
        "teacher_inventories",
        "student_attempts",
        "eligibility_records",
        "direct_teacher_endpoints",
        "student_endpoints",
        "cell_expectations",
    }

    actual = set(raw)
    if actual != expected:
        raise SyntheticRecordError(
            f"synthetic universe keys mismatch: "
            f"missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )

    schema_version = raw["schema_version"]
    if schema_version != SYNTHETIC_UNIVERSE_SCHEMA_VERSION:
        raise SyntheticRecordError(
            f"unsupported synthetic universe schema version: "
            f"{schema_version!r}"
        )

    classification = raw["classification"]
    if classification != SYNTHETIC_CLASSIFICATION:
        raise SyntheticRecordError(
            "synthetic universe classification must be synthetic-only"
        )

    universe = SyntheticUniverse(
        schema_version=schema_version,
        classification=classification,
        method_budgets=_parse_records(
            raw["method_budgets"],
            "method_budgets",
            MethodBudgetRecord,
        ),
        teacher_inventories=_parse_records(
            raw["teacher_inventories"],
            "teacher_inventories",
            TeacherSeedInventory,
        ),
        student_attempts=_parse_records(
            raw["student_attempts"],
            "student_attempts",
            StudentAttemptRecord,
        ),
        eligibility_records=_parse_records(
            raw["eligibility_records"],
            "eligibility_records",
            StudentEligibilityRecord,
        ),
        direct_teacher_endpoints=_parse_records(
            raw["direct_teacher_endpoints"],
            "direct_teacher_endpoints",
            DirectTeacherEndpointRecord,
        ),
        student_endpoints=_parse_records(
            raw["student_endpoints"],
            "student_endpoints",
            StudentEndpointRecord,
        ),
        cell_expectations=_parse_records(
            raw["cell_expectations"],
            "cell_expectations",
            SyntheticCellExpectation,
        ),
    )

    validate_synthetic_universe_links(universe)
    return universe


def load_synthetic_universe(path: str | Path) -> SyntheticUniverse:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    universe = synthetic_universe_from_mapping(
        expect_mapping(raw, "synthetic universe")
    )
    validate_required_synthetic_coverage(universe)
    return universe

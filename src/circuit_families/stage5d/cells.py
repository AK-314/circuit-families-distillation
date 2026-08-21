from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .profiles import TechnicalAnalysisProfile


class CellSummaryError(ValueError):
    pass


Number = int | float


@dataclass(frozen=True, slots=True, order=True)
class StudentCellKey:
    teacher_seed: int
    phase: str
    distillation_condition: str
    method_id: str
    endpoint_id: str
    protocol_id: str
    fidelity_id: str
    budget_id: str


@dataclass(frozen=True, slots=True)
class StudentCellSummary:
    key: StudentCellKey
    state: str
    reason: str | None
    reducer: str
    minimum_eligible_students: int
    eligible_initializations: tuple[int, ...]
    member_values: tuple[tuple[int, Number], ...]
    summary_value: float | None
    range_value: float | None
    mad_value: float | None
    member_unit: str = "student_initialization"
    population_unit: str = "teacher_seed"


@dataclass(frozen=True, slots=True, order=True)
class DirectTeacherKey:
    teacher_seed: int
    phase: str
    method_id: str
    endpoint_id: str
    protocol_id: str
    fidelity_id: str
    budget_id: str


@dataclass(frozen=True, slots=True)
class DirectTeacherValue:
    record_id: str
    key: DirectTeacherKey
    state: str
    value: Number | None
    population_unit: str = "teacher_seed"


def _median(values: Sequence[Number]) -> float:
    if not values:
        raise CellSummaryError("median requires at least one value")

    ordered = sorted(float(value) for value in values)
    size = len(ordered)
    midpoint = size // 2

    if size % 2:
        return ordered[midpoint]

    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def summarize_numeric_values(
    values: Sequence[Number],
    reducer: str,
) -> tuple[float, float, float]:
    if not values:
        raise CellSummaryError(
            "numeric realization summary requires at least one value"
        )

    numeric = tuple(float(value) for value in values)

    if reducer == "median":
        summary = _median(numeric)
    elif reducer == "mean":
        summary = sum(numeric) / len(numeric)
    else:
        raise CellSummaryError(
            f"unsupported injected cell reducer: {reducer}"
        )

    range_value = max(numeric) - min(numeric)

    median_value = _median(numeric)
    absolute_deviations = tuple(
        abs(value - median_value)
        for value in numeric
    )
    mad_value = _median(absolute_deviations)

    return summary, range_value, mad_value


def _student_cell_key(identity: Mapping[str, Any]) -> StudentCellKey:
    return StudentCellKey(
        teacher_seed=int(identity["teacher_seed"]),
        phase=str(identity["phase"]),
        distillation_condition=str(
            identity["distillation_condition"]
        ),
        method_id=str(identity["method_id"]),
        endpoint_id=str(identity["endpoint_id"]),
        protocol_id=str(identity["protocol_id"]),
        fidelity_id=str(identity["fidelity_id"]),
        budget_id=str(identity["budget_id"]),
    )


def _teacher_key(identity: Mapping[str, Any]) -> DirectTeacherKey:
    if identity.get("distillation_condition") is not None:
        raise CellSummaryError(
            "direct teacher identity cannot carry a distillation condition"
        )

    if identity.get("student_initialization") is not None:
        raise CellSummaryError(
            "direct teacher identity cannot carry a student initialization"
        )

    return DirectTeacherKey(
        teacher_seed=int(identity["teacher_seed"]),
        phase=str(identity["phase"]),
        method_id=str(identity["method_id"]),
        endpoint_id=str(identity["endpoint_id"]),
        protocol_id=str(identity["protocol_id"]),
        fidelity_id=str(identity["fidelity_id"]),
        budget_id=str(identity["budget_id"]),
    )


def extract_direct_teacher_values(
    normalized: Mapping[str, Any],
) -> tuple[DirectTeacherValue, ...]:
    raw_records = normalized.get("direct_teacher_endpoints")
    if not isinstance(raw_records, list):
        raise CellSummaryError(
            "normalized direct_teacher_endpoints must be a list"
        )

    results: list[DirectTeacherValue] = []
    seen: set[DirectTeacherKey] = set()

    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise CellSummaryError(
                "direct teacher endpoint must be an object"
            )

        identity = raw.get("identity")
        if not isinstance(identity, Mapping):
            raise CellSummaryError(
                "direct teacher endpoint identity must be an object"
            )

        if identity.get("subject_kind") != "teacher":
            raise CellSummaryError(
                "direct teacher collection contains non-teacher identity"
            )

        key = _teacher_key(identity)
        if key in seen:
            raise CellSummaryError(
                f"duplicate direct teacher analysis identity: {key}"
            )
        seen.add(key)

        state = str(raw["state"])
        value = raw["value"]

        if state == "defined":
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise CellSummaryError(
                    "defined direct teacher value must be numeric"
                )
            stored_value: Number | None = value
        else:
            if value is not None:
                raise CellSummaryError(
                    "non-defined direct teacher state requires null value"
                )
            stored_value = None

        results.append(
            DirectTeacherValue(
                record_id=str(raw["record_id"]),
                key=key,
                state=state,
                value=stored_value,
            )
        )

    return tuple(
        sorted(
            results,
            key=lambda record: (
                record.key,
                record.record_id,
            ),
        )
    )


def build_student_cell_summaries(
    normalized: Mapping[str, Any],
    profile: TechnicalAnalysisProfile,
) -> tuple[StudentCellSummary, ...]:
    if profile.scientific_data:
        raise CellSummaryError(
            "scientific-data profile is forbidden in Stage 5D"
        )
    if profile.production_eligible:
        raise CellSummaryError(
            "production-eligible profile is forbidden in Stage 5D"
        )
    if profile.resolves_decisions:
        raise CellSummaryError(
            "Stage 5D profile cannot resolve scientific decisions"
        )

    raw_expectations = normalized.get("cell_expectations")
    raw_endpoints = normalized.get("student_endpoints")
    raw_eligibility = normalized.get("eligibility_records")

    if not isinstance(raw_expectations, list):
        raise CellSummaryError(
            "normalized cell_expectations must be a list"
        )
    if not isinstance(raw_endpoints, list):
        raise CellSummaryError(
            "normalized student_endpoints must be a list"
        )
    if not isinstance(raw_eligibility, list):
        raise CellSummaryError(
            "normalized eligibility_records must be a list"
        )

    eligibility_by_attempt: dict[str, str] = {}

    for raw in raw_eligibility:
        if not isinstance(raw, Mapping):
            raise CellSummaryError(
                "eligibility record must be an object"
            )

        attempt_id = str(raw["attempt_id"])
        if attempt_id in eligibility_by_attempt:
            raise CellSummaryError(
                f"duplicate eligibility for attempt {attempt_id}"
            )

        eligibility_by_attempt[attempt_id] = str(raw["status"])

    expectations: dict[
        StudentCellKey,
        tuple[str, str | None],
    ] = {}

    for raw in raw_expectations:
        if not isinstance(raw, Mapping):
            raise CellSummaryError(
                "cell expectation must be an object"
            )

        identity = raw.get("identity")
        if not isinstance(identity, Mapping):
            raise CellSummaryError(
                "cell expectation identity must be an object"
            )

        key = _student_cell_key(identity)

        if key in expectations:
            raise CellSummaryError(
                f"duplicate student cell identity: {key}"
            )

        state = str(raw["state"])
        reason_raw = raw.get("reason")
        reason = None if reason_raw is None else str(reason_raw)

        expectations[key] = (state, reason)

    members: dict[
        StudentCellKey,
        dict[int, Number],
    ] = {
        key: {}
        for key in expectations
    }

    for raw in raw_endpoints:
        if not isinstance(raw, Mapping):
            raise CellSummaryError(
                "student endpoint must be an object"
            )

        identity = raw.get("identity")
        if not isinstance(identity, Mapping):
            raise CellSummaryError(
                "student endpoint identity must be an object"
            )

        if identity.get("subject_kind") != "student":
            raise CellSummaryError(
                "student endpoint collection contains non-student identity"
            )

        key = _student_cell_key(identity)

        if key not in expectations:
            raise CellSummaryError(
                f"student endpoint has no declared cell expectation: {key}"
            )

        state = str(raw["state"])
        value = raw["value"]

        if state != "defined":
            if value is not None:
                raise CellSummaryError(
                    "non-defined student endpoint requires null value"
                )
            continue

        attempt_id = str(raw["attempt_id"])
        eligibility = eligibility_by_attempt.get(attempt_id)

        if eligibility != "eligible":
            raise CellSummaryError(
                "defined student endpoint must trace to an eligible "
                f"attempt: attempt={attempt_id} status={eligibility}"
            )

        initialization_raw = identity.get("student_initialization")
        if type(initialization_raw) is not int:
            raise CellSummaryError(
                "student endpoint requires integer initialization"
            )

        initialization = initialization_raw

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise CellSummaryError(
                "defined student endpoint value must be numeric"
            )

        if initialization in members[key]:
            raise CellSummaryError(
                "multiple defined endpoint values for one student "
                f"initialization in cell {key}: initialization="
                f"{initialization}"
            )

        members[key][initialization] = value

    summaries: list[StudentCellSummary] = []

    for key in sorted(expectations):
        expected_state, expected_reason = expectations[key]

        cell_members = tuple(
            sorted(members[key].items())
        )
        initializations = tuple(
            initialization
            for initialization, _ in cell_members
        )

        if expected_state != "expected":
            summaries.append(
                StudentCellSummary(
                    key=key,
                    state=expected_state,
                    reason=expected_reason,
                    reducer=profile.settings.cell_reducer,
                    minimum_eligible_students=(
                        profile.settings.minimum_eligible_students
                    ),
                    eligible_initializations=initializations,
                    member_values=cell_members,
                    summary_value=None,
                    range_value=None,
                    mad_value=None,
                )
            )
            continue

        observed = len(cell_members)
        required = profile.settings.minimum_eligible_students

        if observed < required:
            summaries.append(
                StudentCellSummary(
                    key=key,
                    state="unresolved",
                    reason=(
                        "insufficient_eligible_students:"
                        f"observed={observed}:required={required}"
                    ),
                    reducer=profile.settings.cell_reducer,
                    minimum_eligible_students=required,
                    eligible_initializations=initializations,
                    member_values=cell_members,
                    summary_value=None,
                    range_value=None,
                    mad_value=None,
                )
            )
            continue

        summary, range_value, mad_value = summarize_numeric_values(
            tuple(value for _, value in cell_members),
            profile.settings.cell_reducer,
        )

        summaries.append(
            StudentCellSummary(
                key=key,
                state="defined",
                reason=None,
                reducer=profile.settings.cell_reducer,
                minimum_eligible_students=required,
                eligible_initializations=initializations,
                member_values=cell_members,
                summary_value=summary,
                range_value=range_value,
                mad_value=mad_value,
            )
        )

    return tuple(summaries)

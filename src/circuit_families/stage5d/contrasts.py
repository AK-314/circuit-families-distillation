from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .cells import (
    DirectTeacherKey,
    DirectTeacherValue,
    StudentCellKey,
    StudentCellSummary,
)
from .profiles import TechnicalAnalysisProfile


class ContrastError(ValueError):
    pass


CONTRAST_STATES = frozenset(
    {
        "defined",
        "absent",
        "unavailable",
        "unresolved",
        "insufficient",
        "incompatible",
        "inapplicable",
    }
)

MISSINGNESS_KINDS = frozenset(
    {
        "absent",
        "unavailable",
        "failed",
        "ineligible",
        "insufficient",
        "incompatible",
        "inapplicable",
        "unresolved",
    }
)


@dataclass(frozen=True, slots=True, order=True)
class PhaseContrastKey:
    teacher_seed: int
    distillation_condition: str
    method_id: str
    endpoint_id: str
    phase_from: str
    phase_to: str


@dataclass(frozen=True, slots=True)
class PhaseContrast:
    key: PhaseContrastKey
    state: str
    reason: str | None
    left_key: StudentCellKey | None
    right_key: StudentCellKey | None
    left_value: float | None
    right_value: float | None
    delta: float | None
    contribution_unit: str = "teacher_seed"


@dataclass(frozen=True, slots=True, order=True)
class TeacherStudentContrastKey:
    teacher_seed: int
    phase: str
    distillation_condition: str
    method_id: str
    endpoint_id: str
    protocol_id: str
    fidelity_id: str
    budget_id: str


@dataclass(frozen=True, slots=True)
class TeacherStudentContrast:
    key: TeacherStudentContrastKey
    state: str
    reason: str | None
    student_summary_value: float | None
    teacher_value: float | None
    delta: float | None
    teacher_record_id: str | None
    contribution_unit: str = "teacher_seed"


@dataclass(frozen=True, slots=True)
class MethodAccountingRow:
    method_id: str
    budget_id: str
    distillation_condition: str
    native_budget: int
    exact_eval_allowance: int
    resource_unit: str
    total_cells: int
    defined_cells: int
    unresolved_cells: int
    insufficient_cells: int
    unavailable_cells: int
    inapplicable_cells: int
    population_unit: str = "teacher_seed"


@dataclass(frozen=True, slots=True)
class CrossMethodBudgetWarning:
    left_method_id: str
    right_method_id: str
    left_budget_id: str
    right_budget_id: str
    left_native_budget: int
    right_native_budget: int
    left_resource_unit: str
    right_resource_unit: str
    exact_eval_allowance_equal: bool
    resource_imperfect: bool
    packing_counts_resource_matched: bool
    warning_code: str


@dataclass(frozen=True, slots=True, order=True)
class MissingnessRecord:
    kind: str
    scope: str
    identity: str
    reason: str


def _cell_block_state(cell: StudentCellSummary) -> str | None:
    if cell.state == "defined":
        return None

    if cell.state == "unavailable":
        return "unavailable"

    if cell.state == "inapplicable":
        return "inapplicable"

    if cell.state == "unresolved":
        if (
            cell.reason is not None
            and cell.reason.startswith(
                "insufficient_eligible_students:"
            )
        ):
            return "insufficient"
        return "unresolved"

    raise ContrastError(
        f"unsupported student-cell state for contrast: {cell.state}"
    )


def _combined_block_state(
    left: StudentCellSummary,
    right: StudentCellSummary,
) -> tuple[str | None, str | None]:
    left_state = _cell_block_state(left)
    right_state = _cell_block_state(right)

    if left_state is None and right_state is None:
        return None, None

    states = {
        state
        for state in (left_state, right_state)
        if state is not None
    }

    precedence = (
        "unavailable",
        "inapplicable",
        "insufficient",
        "unresolved",
    )

    selected = next(
        state
        for state in precedence
        if state in states
    )

    reason = (
        f"phase_contrast_blocked:"
        f"left_state={left_state or 'defined'}:"
        f"left_reason={left.reason or 'none'}:"
        f"right_state={right_state or 'defined'}:"
        f"right_reason={right.reason or 'none'}"
    )

    return selected, reason


def _technical_identity(cell: StudentCellSummary) -> tuple[str, str, str]:
    return (
        cell.key.protocol_id,
        cell.key.fidelity_id,
        cell.key.budget_id,
    )


def build_phase_contrasts(
    cells: Sequence[StudentCellSummary],
    profile: TechnicalAnalysisProfile,
) -> tuple[PhaseContrast, ...]:
    if profile.scientific_data or profile.production_eligible:
        raise ContrastError(
            "Stage 5D phase contrasts require synthetic-only profile"
        )

    if profile.resolves_decisions:
        raise ContrastError(
            "Stage 5D phase contrast profile may not resolve decisions"
        )

    grouped: dict[
        tuple[int, str, str, str],
        list[StudentCellSummary],
    ] = defaultdict(list)

    for cell in cells:
        grouped[
            (
                cell.key.teacher_seed,
                cell.key.distillation_condition,
                cell.key.method_id,
                cell.key.endpoint_id,
            )
        ].append(cell)

    outputs: list[PhaseContrast] = []

    for group_key in sorted(grouped):
        seed, condition, method_id, endpoint_id = group_key
        group = grouped[group_key]

        for phase_from, phase_to in profile.settings.phase_pairs:
            left_rows = [
                cell
                for cell in group
                if cell.key.phase == phase_from
            ]
            right_rows = [
                cell
                for cell in group
                if cell.key.phase == phase_to
            ]

            if not left_rows and not right_rows:
                continue

            key = PhaseContrastKey(
                teacher_seed=seed,
                distillation_condition=condition,
                method_id=method_id,
                endpoint_id=endpoint_id,
                phase_from=phase_from,
                phase_to=phase_to,
            )

            if not left_rows or not right_rows:
                outputs.append(
                    PhaseContrast(
                        key=key,
                        state="absent",
                        reason=(
                            "phase_contrast_missing_phase:"
                            f"left_count={len(left_rows)}:"
                            f"right_count={len(right_rows)}"
                        ),
                        left_key=(
                            left_rows[0].key
                            if len(left_rows) == 1
                            else None
                        ),
                        right_key=(
                            right_rows[0].key
                            if len(right_rows) == 1
                            else None
                        ),
                        left_value=None,
                        right_value=None,
                        delta=None,
                    )
                )
                continue

            if len(left_rows) != 1 or len(right_rows) != 1:
                outputs.append(
                    PhaseContrast(
                        key=key,
                        state="incompatible",
                        reason=(
                            "phase_contrast_ambiguous_technical_identity:"
                            f"left_count={len(left_rows)}:"
                            f"right_count={len(right_rows)}"
                        ),
                        left_key=None,
                        right_key=None,
                        left_value=None,
                        right_value=None,
                        delta=None,
                    )
                )
                continue

            left = left_rows[0]
            right = right_rows[0]

            if _technical_identity(left) != _technical_identity(right):
                outputs.append(
                    PhaseContrast(
                        key=key,
                        state="incompatible",
                        reason=(
                            "phase_contrast_identity_mismatch:"
                            f"left={_technical_identity(left)!r}:"
                            f"right={_technical_identity(right)!r}"
                        ),
                        left_key=left.key,
                        right_key=right.key,
                        left_value=None,
                        right_value=None,
                        delta=None,
                    )
                )
                continue

            blocked_state, blocked_reason = _combined_block_state(
                left,
                right,
            )

            if blocked_state is not None:
                outputs.append(
                    PhaseContrast(
                        key=key,
                        state=blocked_state,
                        reason=blocked_reason,
                        left_key=left.key,
                        right_key=right.key,
                        left_value=left.summary_value,
                        right_value=right.summary_value,
                        delta=None,
                    )
                )
                continue

            if (
                left.summary_value is None
                or right.summary_value is None
            ):
                raise ContrastError(
                    "defined phase-contrast cells require summary values"
                )

            outputs.append(
                PhaseContrast(
                    key=key,
                    state="defined",
                    reason=None,
                    left_key=left.key,
                    right_key=right.key,
                    left_value=float(left.summary_value),
                    right_value=float(right.summary_value),
                    delta=(
                        float(right.summary_value)
                        - float(left.summary_value)
                    ),
                )
            )

    return tuple(sorted(outputs, key=lambda row: row.key))


def _teacher_key_for_cell(
    cell: StudentCellSummary,
) -> DirectTeacherKey:
    return DirectTeacherKey(
        teacher_seed=cell.key.teacher_seed,
        phase=cell.key.phase,
        method_id=cell.key.method_id,
        endpoint_id=cell.key.endpoint_id,
        protocol_id=cell.key.protocol_id,
        fidelity_id=cell.key.fidelity_id,
        budget_id=cell.key.budget_id,
    )


def _teacher_near_match_key(
    key: DirectTeacherKey,
) -> tuple[int, str, str, str]:
    return (
        key.teacher_seed,
        key.phase,
        key.method_id,
        key.endpoint_id,
    )


def build_teacher_student_contrasts(
    cells: Sequence[StudentCellSummary],
    teachers: Sequence[DirectTeacherValue],
) -> tuple[TeacherStudentContrast, ...]:
    teacher_by_key: dict[DirectTeacherKey, DirectTeacherValue] = {}

    for teacher in teachers:
        if teacher.key in teacher_by_key:
            raise ContrastError(
                f"duplicate direct teacher key: {teacher.key}"
            )
        teacher_by_key[teacher.key] = teacher

    near_matches: dict[
        tuple[int, str, str, str],
        list[DirectTeacherValue],
    ] = defaultdict(list)

    for teacher in teachers:
        near_matches[
            _teacher_near_match_key(teacher.key)
        ].append(teacher)

    outputs: list[TeacherStudentContrast] = []

    for cell in sorted(cells, key=lambda row: row.key):
        key = TeacherStudentContrastKey(
            teacher_seed=cell.key.teacher_seed,
            phase=cell.key.phase,
            distillation_condition=(
                cell.key.distillation_condition
            ),
            method_id=cell.key.method_id,
            endpoint_id=cell.key.endpoint_id,
            protocol_id=cell.key.protocol_id,
            fidelity_id=cell.key.fidelity_id,
            budget_id=cell.key.budget_id,
        )

        expected_teacher_key = _teacher_key_for_cell(cell)
        teacher = teacher_by_key.get(expected_teacher_key)

        if teacher is None:
            candidates = near_matches.get(
                _teacher_near_match_key(expected_teacher_key),
                [],
            )

            if candidates:
                outputs.append(
                    TeacherStudentContrast(
                        key=key,
                        state="incompatible",
                        reason=(
                            "teacher_student_identity_mismatch:"
                            f"expected_protocol={cell.key.protocol_id}:"
                            f"expected_fidelity={cell.key.fidelity_id}:"
                            f"expected_budget={cell.key.budget_id}:"
                            f"candidate_count={len(candidates)}"
                        ),
                        student_summary_value=cell.summary_value,
                        teacher_value=None,
                        delta=None,
                        teacher_record_id=None,
                    )
                )
            else:
                outputs.append(
                    TeacherStudentContrast(
                        key=key,
                        state="absent",
                        reason="direct_teacher_record_absent",
                        student_summary_value=cell.summary_value,
                        teacher_value=None,
                        delta=None,
                        teacher_record_id=None,
                    )
                )
            continue

        cell_block = _cell_block_state(cell)

        if cell_block is not None:
            outputs.append(
                TeacherStudentContrast(
                    key=key,
                    state=cell_block,
                    reason=(
                        "student_cell_not_defined:"
                        f"state={cell_block}:"
                        f"reason={cell.reason or 'none'}"
                    ),
                    student_summary_value=None,
                    teacher_value=(
                        None
                        if teacher.value is None
                        else float(teacher.value)
                    ),
                    delta=None,
                    teacher_record_id=teacher.record_id,
                )
            )
            continue

        if teacher.state != "defined":
            if teacher.state not in {
                "unavailable",
                "unresolved",
                "inapplicable",
            }:
                raise ContrastError(
                    "unsupported direct-teacher state for contrast: "
                    f"{teacher.state}"
                )

            outputs.append(
                TeacherStudentContrast(
                    key=key,
                    state=teacher.state,
                    reason=(
                        "direct_teacher_not_defined:"
                        f"state={teacher.state}"
                    ),
                    student_summary_value=cell.summary_value,
                    teacher_value=None,
                    delta=None,
                    teacher_record_id=teacher.record_id,
                )
            )
            continue

        if cell.summary_value is None or teacher.value is None:
            raise ContrastError(
                "defined teacher-student contrast requires both values"
            )

        outputs.append(
            TeacherStudentContrast(
                key=key,
                state="defined",
                reason=None,
                student_summary_value=float(cell.summary_value),
                teacher_value=float(teacher.value),
                delta=(
                    float(cell.summary_value)
                    - float(teacher.value)
                ),
                teacher_record_id=teacher.record_id,
            )
        )

    return tuple(sorted(outputs, key=lambda row: row.key))


def _method_budget_map(
    normalized: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw = normalized.get("method_budgets")

    if not isinstance(raw, list):
        raise ContrastError(
            "normalized method_budgets must be a list"
        )

    result: dict[str, Mapping[str, Any]] = {}

    for record in raw:
        if not isinstance(record, Mapping):
            raise ContrastError(
                "method budget record must be an object"
            )

        method_id = str(record["method_id"])

        if method_id in result:
            raise ContrastError(
                f"duplicate method budget metadata: {method_id}"
            )

        result[method_id] = record

    return result


def build_method_accounting(
    normalized: Mapping[str, Any],
    cells: Sequence[StudentCellSummary],
) -> tuple[MethodAccountingRow, ...]:
    budgets = _method_budget_map(normalized)

    grouped: dict[
        tuple[str, str],
        list[StudentCellSummary],
    ] = defaultdict(list)

    for cell in cells:
        grouped[
            (
                cell.key.method_id,
                cell.key.distillation_condition,
            )
        ].append(cell)

    rows: list[MethodAccountingRow] = []

    for (method_id, condition), group in sorted(grouped.items()):
        budget = budgets.get(method_id)

        if budget is None:
            raise ContrastError(
                f"missing method budget metadata for {method_id}"
            )

        budget_id = str(budget["budget_id"])

        if any(
            cell.key.budget_id != budget_id
            for cell in group
        ):
            raise ContrastError(
                "method accounting encountered incompatible budget "
                f"identity for method {method_id}"
            )

        rows.append(
            MethodAccountingRow(
                method_id=method_id,
                budget_id=budget_id,
                distillation_condition=condition,
                native_budget=int(budget["native_budget"]),
                exact_eval_allowance=int(
                    budget["exact_eval_allowance"]
                ),
                resource_unit=str(budget["resource_unit"]),
                total_cells=len(group),
                defined_cells=sum(
                    cell.state == "defined"
                    for cell in group
                ),
                unresolved_cells=sum(
                    cell.state == "unresolved"
                    and not (
                        cell.reason is not None
                        and cell.reason.startswith(
                            "insufficient_eligible_students:"
                        )
                    )
                    for cell in group
                ),
                insufficient_cells=sum(
                    cell.state == "unresolved"
                    and cell.reason is not None
                    and cell.reason.startswith(
                        "insufficient_eligible_students:"
                    )
                    for cell in group
                ),
                unavailable_cells=sum(
                    cell.state == "unavailable"
                    for cell in group
                ),
                inapplicable_cells=sum(
                    cell.state == "inapplicable"
                    for cell in group
                ),
            )
        )

    return tuple(rows)


def build_cross_method_budget_warnings(
    normalized: Mapping[str, Any],
) -> tuple[CrossMethodBudgetWarning, ...]:
    budgets = _method_budget_map(normalized)

    rows = [
        budgets[method_id]
        for method_id in sorted(budgets)
    ]

    warnings: list[CrossMethodBudgetWarning] = []

    for left, right in combinations(rows, 2):
        warnings.append(
            CrossMethodBudgetWarning(
                left_method_id=str(left["method_id"]),
                right_method_id=str(right["method_id"]),
                left_budget_id=str(left["budget_id"]),
                right_budget_id=str(right["budget_id"]),
                left_native_budget=int(left["native_budget"]),
                right_native_budget=int(right["native_budget"]),
                left_resource_unit=str(left["resource_unit"]),
                right_resource_unit=str(right["resource_unit"]),
                exact_eval_allowance_equal=(
                    int(left["exact_eval_allowance"])
                    == int(right["exact_eval_allowance"])
                ),
                resource_imperfect=True,
                packing_counts_resource_matched=False,
                warning_code=(
                    "cross_method_raw_packing_counts_not_"
                    "perfectly_resource_matched"
                ),
            )
        )

    return tuple(warnings)


def build_missingness_records(
    normalized: Mapping[str, Any],
    cells: Sequence[StudentCellSummary],
    phase_contrasts: Sequence[PhaseContrast] = (),
    teacher_student_contrasts: Sequence[
        TeacherStudentContrast
    ] = (),
) -> tuple[MissingnessRecord, ...]:
    records: list[MissingnessRecord] = []

    raw_attempts = normalized.get("student_attempts")
    raw_eligibility = normalized.get("eligibility_records")

    if not isinstance(raw_attempts, list):
        raise ContrastError(
            "normalized student_attempts must be a list"
        )
    if not isinstance(raw_eligibility, list):
        raise ContrastError(
            "normalized eligibility_records must be a list"
        )

    for attempt in raw_attempts:
        if not isinstance(attempt, Mapping):
            raise ContrastError(
                "student attempt must be an object"
            )

        if attempt["outcome"] == "failed":
            records.append(
                MissingnessRecord(
                    kind="failed",
                    scope="student_attempt",
                    identity=str(attempt["attempt_id"]),
                    reason=str(attempt["failure_reason"]),
                )
            )

    for eligibility in raw_eligibility:
        if not isinstance(eligibility, Mapping):
            raise ContrastError(
                "eligibility record must be an object"
            )

        status = str(eligibility["status"])

        if status == "ineligible":
            records.append(
                MissingnessRecord(
                    kind="ineligible",
                    scope="student_eligibility",
                    identity=str(eligibility["eligibility_id"]),
                    reason=str(eligibility["reason"]),
                )
            )
        elif status == "inapplicable":
            records.append(
                MissingnessRecord(
                    kind="inapplicable",
                    scope="student_eligibility",
                    identity=str(eligibility["eligibility_id"]),
                    reason=str(eligibility["reason"]),
                )
            )

    for cell in cells:
        identity = repr(cell.key)

        if cell.state == "unavailable":
            records.append(
                MissingnessRecord(
                    kind="unavailable",
                    scope="student_cell",
                    identity=identity,
                    reason=cell.reason or "unavailable",
                )
            )
        elif cell.state == "inapplicable":
            records.append(
                MissingnessRecord(
                    kind="inapplicable",
                    scope="student_cell",
                    identity=identity,
                    reason=cell.reason or "inapplicable",
                )
            )
        elif cell.state == "unresolved":
            if (
                cell.reason is not None
                and cell.reason.startswith(
                    "insufficient_eligible_students:"
                )
            ):
                kind = "insufficient"
            else:
                kind = "unresolved"

            records.append(
                MissingnessRecord(
                    kind=kind,
                    scope="student_cell",
                    identity=identity,
                    reason=cell.reason or kind,
                )
            )

    for contrast in phase_contrasts:
        if contrast.state in {"absent", "incompatible"}:
            records.append(
                MissingnessRecord(
                    kind=contrast.state,
                    scope="phase_contrast",
                    identity=repr(contrast.key),
                    reason=contrast.reason or contrast.state,
                )
            )

    for contrast in teacher_student_contrasts:
        if contrast.state in {"absent", "incompatible"}:
            records.append(
                MissingnessRecord(
                    kind=contrast.state,
                    scope="teacher_student_contrast",
                    identity=repr(contrast.key),
                    reason=contrast.reason or contrast.state,
                )
            )

    for record in records:
        if record.kind not in MISSINGNESS_KINDS:
            raise ContrastError(
                f"unsupported missingness kind: {record.kind}"
            )

    return tuple(sorted(records))

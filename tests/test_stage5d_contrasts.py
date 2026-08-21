from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from circuit_families.stage5d import (
    ContrastError,
    DirectTeacherKey,
    StudentCellKey,
    build_cross_method_budget_warnings,
    build_method_accounting,
    build_missingness_records,
    build_phase_contrasts,
    build_student_cell_summaries,
    build_teacher_student_contrasts,
    extract_direct_teacher_values,
    load_and_normalize_ingestion,
    load_technical_analysis_profile_set,
)

ROOT = Path(__file__).resolve().parents[1]
ENVELOPE = (
    ROOT
    / "tests/fixtures/stage5d/"
    "synthetic_ingestion_envelope_v1.json"
)
PROFILES = (
    ROOT
    / "followup/configs/stage5d/"
    "technical_analysis_profiles_v1.json"
)


def _normalized() -> dict[str, object]:
    return load_and_normalize_ingestion(ENVELOPE)


def _profile(profile_id: str = "fixture_median_min2"):
    return load_technical_analysis_profile_set(PROFILES).require(
        profile_id
    )


def _cells(profile_id: str = "fixture_median_min2"):
    return build_student_cell_summaries(
        _normalized(),
        _profile(profile_id),
    )


def _teachers():
    return extract_direct_teacher_values(_normalized())


def test_phase_contrasts_use_injected_phase_pair_and_right_minus_left() -> None:
    cells = _cells()
    contrasts = build_phase_contrasts(cells, _profile())

    assert len(contrasts) == 16
    assert sum(row.state == "defined" for row in contrasts) == 15
    assert sum(row.state == "unresolved" for row in contrasts) == 1

    row = next(
        item
        for item in contrasts
        if item.state == "defined"
    )

    assert row.left_key is not None
    assert row.right_key is not None
    assert row.left_value is not None
    assert row.right_value is not None

    assert row.key.phase_from == "phase_early"
    assert row.key.phase_to == "phase_late"
    assert row.delta == pytest.approx(
        row.right_value - row.left_value
    )
    assert row.contribution_unit == "teacher_seed"


def test_phase_contrasts_never_pool_hard_and_soft() -> None:
    contrasts = build_phase_contrasts(_cells(), _profile())

    conditions = {
        row.key.distillation_condition
        for row in contrasts
    }

    assert conditions == {"hard", "soft"}

    keys_without_condition = [
        (
            row.key.teacher_seed,
            row.key.method_id,
            row.key.endpoint_id,
            row.key.phase_from,
            row.key.phase_to,
        )
        for row in contrasts
    ]

    assert len(keys_without_condition) > len(
        set(keys_without_condition)
    )


def test_missing_phase_is_absent_not_imputed() -> None:
    cells = list(_cells())

    target = next(
        row
        for row in cells
        if row.key.teacher_seed == 0
        and row.key.phase == "phase_late"
        and row.key.distillation_condition == "hard"
        and row.key.method_id == "method_greedy"
        and row.key.endpoint_id == "endpoint_1"
    )

    cells.remove(target)

    contrasts = build_phase_contrasts(cells, _profile())

    row = next(
        item
        for item in contrasts
        if item.key.teacher_seed == 0
        and item.key.distillation_condition == "hard"
        and item.key.method_id == "method_greedy"
        and item.key.endpoint_id == "endpoint_1"
    )

    assert row.state == "absent"
    assert row.delta is None


def test_phase_identity_mismatch_is_incompatible() -> None:
    cells = list(_cells())

    index = next(
        i
        for i, row in enumerate(cells)
        if row.key.teacher_seed == 0
        and row.key.phase == "phase_late"
        and row.key.distillation_condition == "hard"
        and row.key.method_id == "method_greedy"
        and row.key.endpoint_id == "endpoint_1"
    )

    original = cells[index]
    cells[index] = replace(
        original,
        key=replace(
            original.key,
            fidelity_id="synthetic_incompatible_fidelity",
        ),
    )

    contrasts = build_phase_contrasts(cells, _profile())

    row = next(
        item
        for item in contrasts
        if item.key.teacher_seed == 0
        and item.key.distillation_condition == "hard"
        and item.key.method_id == "method_greedy"
        and item.key.endpoint_id == "endpoint_1"
    )

    assert row.state == "incompatible"
    assert row.delta is None


def test_teacher_student_delta_is_student_summary_minus_teacher() -> None:
    contrasts = build_teacher_student_contrasts(
        _cells(),
        _teachers(),
    )

    assert len(contrasts) == 33
    assert sum(row.state == "defined" for row in contrasts) == 31
    assert sum(row.state == "unresolved" for row in contrasts) == 1
    assert sum(row.state == "unavailable" for row in contrasts) == 1

    row = next(
        item
        for item in contrasts
        if item.state == "defined"
    )

    assert row.student_summary_value is not None
    assert row.teacher_value is not None
    assert row.delta == pytest.approx(
        row.student_summary_value - row.teacher_value
    )
    assert row.teacher_record_id is not None
    assert row.contribution_unit == "teacher_seed"


def test_teacher_student_matching_requires_exact_fidelity_and_budget() -> None:
    teachers = list(_teachers())
    target_index = next(
        i
        for i, row in enumerate(teachers)
        if row.key.teacher_seed == 0
        and row.key.phase == "phase_early"
        and row.key.method_id == "method_greedy"
        and row.key.endpoint_id == "endpoint_1"
    )

    original = teachers[target_index]
    teachers[target_index] = replace(
        original,
        key=DirectTeacherKey(
            teacher_seed=original.key.teacher_seed,
            phase=original.key.phase,
            method_id=original.key.method_id,
            endpoint_id=original.key.endpoint_id,
            protocol_id=original.key.protocol_id,
            fidelity_id="wrong_fidelity",
            budget_id=original.key.budget_id,
        ),
    )

    contrasts = build_teacher_student_contrasts(
        _cells(),
        teachers,
    )

    affected = [
        row
        for row in contrasts
        if row.key.teacher_seed == 0
        and row.key.phase == "phase_early"
        and row.key.method_id == "method_greedy"
        and row.key.endpoint_id == "endpoint_1"
    ]

    assert len(affected) == 2
    assert {row.state for row in affected} == {"incompatible"}
    assert all(row.delta is None for row in affected)


def test_missing_teacher_is_absent_not_imputed() -> None:
    teachers = list(_teachers())

    teachers = [
        row
        for row in teachers
        if not (
            row.key.teacher_seed == 0
            and row.key.phase == "phase_early"
            and row.key.method_id == "method_beam"
            and row.key.endpoint_id == "endpoint_2"
        )
    ]

    contrasts = build_teacher_student_contrasts(
        _cells(),
        teachers,
    )

    affected = [
        row
        for row in contrasts
        if row.key.teacher_seed == 0
        and row.key.phase == "phase_early"
        and row.key.method_id == "method_beam"
        and row.key.endpoint_id == "endpoint_2"
    ]

    assert len(affected) == 2
    assert {row.state for row in affected} == {"absent"}
    assert all(row.delta is None for row in affected)


def test_min3_insufficient_cells_block_contrasts() -> None:
    cells = _cells("fixture_mean_min3")
    profile = _profile("fixture_mean_min3")

    phase = build_phase_contrasts(cells, profile)
    teacher_student = build_teacher_student_contrasts(
        cells,
        _teachers(),
    )

    assert any(
        row.state == "insufficient"
        for row in phase
    )
    assert any(
        row.state == "insufficient"
        for row in teacher_student
    )

    assert all(
        row.delta is None
        for row in phase
        if row.state == "insufficient"
    )


def test_method_accounting_remains_separate_by_condition() -> None:
    rows = build_method_accounting(
        _normalized(),
        _cells(),
    )

    assert len(rows) == 4

    assert {
        (row.method_id, row.distillation_condition)
        for row in rows
    } == {
        ("method_greedy", "hard"),
        ("method_greedy", "soft"),
        ("method_beam", "hard"),
        ("method_beam", "soft"),
    }

    assert all(
        row.population_unit == "teacher_seed"
        for row in rows
    )


def test_method_accounting_rejects_budget_identity_collision() -> None:
    cells = list(_cells())
    target = cells[0]

    cells[0] = replace(
        target,
        key=StudentCellKey(
            teacher_seed=target.key.teacher_seed,
            phase=target.key.phase,
            distillation_condition=(
                target.key.distillation_condition
            ),
            method_id=target.key.method_id,
            endpoint_id=target.key.endpoint_id,
            protocol_id=target.key.protocol_id,
            fidelity_id=target.key.fidelity_id,
            budget_id="wrong_budget",
        ),
    )

    with pytest.raises(
        ContrastError,
        match="incompatible budget identity",
    ):
        build_method_accounting(_normalized(), cells)


def test_cross_method_warning_forbids_resource_matched_claim() -> None:
    warnings = build_cross_method_budget_warnings(
        _normalized()
    )

    assert len(warnings) == 1

    warning = warnings[0]

    assert warning.left_native_budget != warning.right_native_budget
    assert warning.exact_eval_allowance_equal is True
    assert warning.resource_imperfect is True
    assert warning.packing_counts_resource_matched is False
    assert warning.warning_code == (
        "cross_method_raw_packing_counts_not_"
        "perfectly_resource_matched"
    )


def test_base_missingness_categories_remain_distinct() -> None:
    normalized = _normalized()
    cells = _cells()

    phase = build_phase_contrasts(cells, _profile())
    teacher_student = build_teacher_student_contrasts(
        cells,
        _teachers(),
    )

    gaps = build_missingness_records(
        normalized,
        cells,
        phase,
        teacher_student,
    )

    kinds = {row.kind for row in gaps}

    assert "failed" in kinds
    assert "ineligible" in kinds
    assert "inapplicable" in kinds
    assert "unavailable" in kinds
    assert "unresolved" in kinds


def test_insufficient_missingness_is_not_generic_unresolved() -> None:
    normalized = _normalized()
    cells = _cells("fixture_mean_min3")
    profile = _profile("fixture_mean_min3")

    phase = build_phase_contrasts(cells, profile)
    teacher_student = build_teacher_student_contrasts(
        cells,
        _teachers(),
    )

    gaps = build_missingness_records(
        normalized,
        cells,
        phase,
        teacher_student,
    )

    assert any(row.kind == "insufficient" for row in gaps)


def test_absent_and_incompatible_missingness_are_distinct() -> None:
    cells = list(_cells())
    teachers = list(_teachers())

    missing_teacher = next(
        row
        for row in teachers
        if row.key.teacher_seed == 0
        and row.key.phase == "phase_early"
        and row.key.method_id == "method_beam"
        and row.key.endpoint_id == "endpoint_2"
    )
    teachers.remove(missing_teacher)

    incompatible_index = next(
        i
        for i, row in enumerate(cells)
        if row.key.teacher_seed == 1
        and row.key.phase == "phase_late"
        and row.key.distillation_condition == "hard"
        and row.key.method_id == "method_greedy"
        and row.key.endpoint_id == "endpoint_1"
    )

    original = cells[incompatible_index]
    cells[incompatible_index] = replace(
        original,
        key=replace(
            original.key,
            fidelity_id="incompatible_fixture_fidelity",
        ),
    )

    phase = build_phase_contrasts(cells, _profile())
    teacher_student = build_teacher_student_contrasts(
        cells,
        teachers,
    )

    gaps = build_missingness_records(
        _normalized(),
        cells,
        phase,
        teacher_student,
    )

    kinds = {row.kind for row in gaps}

    assert "absent" in kinds
    assert "incompatible" in kinds

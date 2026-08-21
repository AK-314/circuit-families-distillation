from __future__ import annotations

from pathlib import Path

import pytest

from circuit_families.stage5d import (
    PhaseContrast,
    PhaseContrastKey,
    PopulationAggregationError,
    StudentCellKey,
    TeacherStudentContrast,
    TeacherStudentContrastKey,
    build_phase_contrasts,
    build_phase_population_summaries,
    build_student_cell_summaries,
    build_teacher_student_contrasts,
    build_teacher_student_population_summaries,
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


def _profile(profile_id: str = "fixture_median_min2"):
    return load_technical_analysis_profile_set(PROFILES).require(
        profile_id
    )


def _student_key(seed: int, phase: str) -> StudentCellKey:
    return StudentCellKey(
        teacher_seed=seed,
        phase=phase,
        distillation_condition="hard",
        method_id="method_greedy",
        endpoint_id="endpoint_1",
        protocol_id="synthetic_protocol_v1",
        fidelity_id="synthetic_fidelity_v1",
        budget_id="budget_greedy_v1",
    )


def _phase_contrast(
    seed: int,
    *,
    left: float | None = None,
    right: float | None = None,
    delta: float | None = None,
    state: str = "defined",
) -> PhaseContrast:
    return PhaseContrast(
        key=PhaseContrastKey(
            teacher_seed=seed,
            distillation_condition="hard",
            method_id="method_greedy",
            endpoint_id="endpoint_1",
            phase_from="phase_early",
            phase_to="phase_late",
        ),
        state=state,
        reason=None if state == "defined" else f"fixture_{state}",
        left_key=(
            _student_key(seed, "phase_early")
            if state == "defined"
            else None
        ),
        right_key=(
            _student_key(seed, "phase_late")
            if state == "defined"
            else None
        ),
        left_value=left,
        right_value=right,
        delta=delta,
    )


def _teacher_student_contrast(
    seed: int,
    *,
    student: float | None = None,
    teacher: float | None = None,
    delta: float | None = None,
    state: str = "defined",
) -> TeacherStudentContrast:
    return TeacherStudentContrast(
        key=TeacherStudentContrastKey(
            teacher_seed=seed,
            phase="phase_early",
            distillation_condition="hard",
            method_id="method_greedy",
            endpoint_id="endpoint_1",
            protocol_id="synthetic_protocol_v1",
            fidelity_id="synthetic_fidelity_v1",
            budget_id="budget_greedy_v1",
        ),
        state=state,
        reason=None if state == "defined" else f"fixture_{state}",
        student_summary_value=student,
        teacher_value=teacher,
        delta=delta,
        teacher_record_id=(
            f"teacher_{seed}"
            if state == "defined"
            else None
        ),
    )


def test_population_n_is_teacher_seeds_not_student_initializations() -> None:
    normalized = load_and_normalize_ingestion(ENVELOPE)
    profile = _profile()
    cells = build_student_cell_summaries(normalized, profile)
    phase = build_phase_contrasts(cells, profile)
    summaries = build_phase_population_summaries(phase, profile)

    teacher_seeds = {
        row.key.teacher_seed
        for row in phase
    }
    student_initialization_count = sum(
        len(row.eligible_initializations)
        for row in cells
    )

    assert student_initialization_count > len(teacher_seeds)
    assert summaries
    assert all(row.population_unit == "teacher_seed" for row in summaries)
    assert all(
        row.number_defined_teacher_seeds
        == len(row.contributing_teacher_seeds)
        for row in summaries
    )
    assert all(
        row.number_defined_teacher_seeds <= len(teacher_seeds)
        for row in summaries
    )


def test_population_reducers_are_deterministic_and_configured() -> None:
    rows = (
        _phase_contrast(2, left=0.0, right=100.0, delta=100.0),
        _phase_contrast(0, left=1.0, right=1.0, delta=0.0),
        _phase_contrast(1, left=2.0, right=4.0, delta=2.0),
    )

    median_summary = build_phase_population_summaries(
        rows,
        _profile("fixture_median_min2"),
    )[0]
    reversed_median = build_phase_population_summaries(
        tuple(reversed(rows)),
        _profile("fixture_median_min2"),
    )[0]
    mean_summary = build_phase_population_summaries(
        rows,
        _profile("fixture_mean_min3"),
    )[0]

    assert median_summary.reducer == "median"
    assert median_summary.aggregated_value == pytest.approx(2.0)
    assert reversed_median == median_summary
    assert mean_summary.reducer == "mean"
    assert mean_summary.aggregated_value == pytest.approx(34.0)
    assert median_summary.contributing_teacher_seeds == (0, 1, 2)


def test_population_missingness_states_remain_distinct_without_imputation() -> None:
    rows = (
        _teacher_student_contrast(
            0,
            student=3.0,
            teacher=1.0,
            delta=2.0,
        ),
        _teacher_student_contrast(1, state="failed"),
        _teacher_student_contrast(2, state="ineligible"),
        _teacher_student_contrast(3, state="unavailable"),
        _teacher_student_contrast(4, state="unresolved"),
        _teacher_student_contrast(5, state="insufficient"),
        _teacher_student_contrast(6, state="inapplicable"),
    )

    summary = build_teacher_student_population_summaries(
        rows,
        _profile(),
    )[0]

    assert summary.aggregated_value == pytest.approx(2.0)
    assert summary.contributing_teacher_seeds == (0,)
    assert summary.number_defined_teacher_seeds == 1
    assert summary.failed_seeds == (1,)
    assert summary.ineligible_seeds == (2,)
    assert summary.unavailable_seeds == (3,)
    assert summary.unresolved_seeds == (4,)
    assert summary.insufficient_seeds == (5,)
    assert summary.inapplicable_seeds == (6,)
    assert summary.low_population_warning is not None

    no_defined = build_teacher_student_population_summaries(
        (_teacher_student_contrast(0, state="unavailable"),),
        _profile(),
    )[0]
    assert no_defined.aggregated_value is None
    assert no_defined.contributing_teacher_seeds == ()


def test_duplicate_teacher_seed_contributions_are_rejected() -> None:
    row = _phase_contrast(0, left=1.0, right=2.0, delta=1.0)

    with pytest.raises(
        PopulationAggregationError,
        match="duplicate population contribution",
    ):
        build_phase_population_summaries((row, row), _profile())


def test_population_input_must_be_part_f_contrasts() -> None:
    normalized = load_and_normalize_ingestion(ENVELOPE)
    cell = build_student_cell_summaries(normalized, _profile())[0]

    with pytest.raises(
        PopulationAggregationError,
        match="PhaseContrast objects",
    ):
        build_phase_population_summaries((cell,), _profile())  # type: ignore[arg-type]


def test_phase_direction_remains_right_minus_left() -> None:
    row = _phase_contrast(0, left=1.5, right=4.0, delta=2.5)
    summary = build_phase_population_summaries((row,), _profile())[0]

    assert summary.aggregated_value == pytest.approx(4.0 - 1.5)

    reversed_row = _phase_contrast(
        0,
        left=1.5,
        right=4.0,
        delta=-2.5,
    )
    with pytest.raises(
        PopulationAggregationError,
        match="right phase minus left phase",
    ):
        build_phase_population_summaries((reversed_row,), _profile())


def test_teacher_student_direction_remains_student_minus_teacher() -> None:
    row = _teacher_student_contrast(
        0,
        student=5.0,
        teacher=1.25,
        delta=3.75,
    )
    summary = build_teacher_student_population_summaries(
        (row,),
        _profile(),
    )[0]

    assert summary.aggregated_value == pytest.approx(5.0 - 1.25)

    reversed_row = _teacher_student_contrast(
        0,
        student=5.0,
        teacher=1.25,
        delta=-3.75,
    )
    with pytest.raises(
        PopulationAggregationError,
        match="student summary minus direct teacher",
    ):
        build_teacher_student_population_summaries(
            (reversed_row,),
            _profile(),
        )


def test_fixture_teacher_student_rollups_use_seed_level_part_f_values() -> None:
    normalized = load_and_normalize_ingestion(ENVELOPE)
    profile = _profile()
    cells = build_student_cell_summaries(normalized, profile)
    contrasts = build_teacher_student_contrasts(
        cells,
        extract_direct_teacher_values(normalized),
    )
    summaries = build_teacher_student_population_summaries(
        contrasts,
        profile,
    )

    assert summaries
    assert all(row.population_unit == "teacher_seed" for row in summaries)
    assert all(
        row.metric_kind == "teacher_student_contrast"
        for row in summaries
    )
    assert all(row.low_population_warning is not None for row in summaries)
    assert profile.resolves_decisions == ()
    assert set(profile.decision_dependencies) == {
        "UD-004",
        "UD-011",
        "UD-012",
    }

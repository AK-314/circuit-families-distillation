from __future__ import annotations

import copy
from pathlib import Path

import pytest

from circuit_families.stage5d import (
    CellSummaryError,
    build_student_cell_summaries,
    extract_direct_teacher_values,
    load_and_normalize_ingestion,
    load_technical_analysis_profile_set,
    summarize_numeric_values,
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


def _profile(profile_id: str):
    return load_technical_analysis_profile_set(PROFILES).require(
        profile_id
    )


def test_even_sample_median_range_and_mad_are_exact() -> None:
    summary, range_value, mad_value = summarize_numeric_values(
        (1, 3, 5, 7),
        "median",
    )

    assert summary == 4.0
    assert range_value == 6.0
    assert mad_value == 2.0

    summary_two, range_two, mad_two = summarize_numeric_values(
        (1, 2),
        "median",
    )

    assert summary_two == 1.5
    assert range_two == 1.0
    assert mad_two == 0.5


def test_injected_mean_changes_only_cell_reducer() -> None:
    summary, range_value, mad_value = summarize_numeric_values(
        (1, 2, 9),
        "mean",
    )

    assert summary == 4.0
    assert range_value == 8.0

    # MAD is conventionally centred on the sample median, not
    # on the injected cell-summary reducer.
    assert mad_value == 1.0


def test_min2_fixture_builds_exact_declared_cells() -> None:
    summaries = build_student_cell_summaries(
        _normalized(),
        _profile("fixture_median_min2"),
    )

    assert len(summaries) == 33

    assert sum(row.state == "defined" for row in summaries) == 31
    assert sum(row.state == "unresolved" for row in summaries) == 1
    assert sum(row.state == "unavailable" for row in summaries) == 1

    assert all(
        row.member_unit == "student_initialization"
        for row in summaries
    )
    assert all(
        row.population_unit == "teacher_seed"
        for row in summaries
    )


def test_hard_and_soft_are_distinct_cell_identities() -> None:
    summaries = build_student_cell_summaries(
        _normalized(),
        _profile("fixture_median_min2"),
    )

    hard = {
        (
            row.key.teacher_seed,
            row.key.phase,
            row.key.method_id,
            row.key.endpoint_id,
            row.key.protocol_id,
            row.key.fidelity_id,
            row.key.budget_id,
        )
        for row in summaries
        if row.key.distillation_condition == "hard"
    }

    soft = {
        (
            row.key.teacher_seed,
            row.key.phase,
            row.key.method_id,
            row.key.endpoint_id,
            row.key.protocol_id,
            row.key.fidelity_id,
            row.key.budget_id,
        )
        for row in summaries
        if row.key.distillation_condition == "soft"
    }

    assert hard
    assert soft

    # Matching lower-dimensional coordinates exist in both conditions,
    # but the full cell key retains condition and therefore never pools them.
    assert hard & soft


def test_min3_marks_insufficient_cells_unresolved_without_imputation() -> None:
    summaries = build_student_cell_summaries(
        _normalized(),
        _profile("fixture_mean_min3"),
    )

    unresolved = [
        row
        for row in summaries
        if row.state == "unresolved"
    ]

    assert len(unresolved) == 12

    for row in unresolved:
        assert row.summary_value is None
        assert row.range_value is None
        assert row.mad_value is None

    insufficient = [
        row
        for row in unresolved
        if row.reason is not None
        and row.reason.startswith(
            "insufficient_eligible_students:"
        )
    ]

    assert len(insufficient) == 11

    assert all(
        len(row.member_values) < 3
        for row in insufficient
    )


def test_explicit_unavailable_and_unresolved_states_are_preserved() -> None:
    summaries = build_student_cell_summaries(
        _normalized(),
        _profile("fixture_median_min2"),
    )

    unavailable = [
        row for row in summaries
        if row.state == "unavailable"
    ]
    unresolved = [
        row for row in summaries
        if row.state == "unresolved"
    ]

    assert len(unavailable) == 1
    assert unavailable[0].reason == (
        "synthetic_teacher_phase_unavailable"
    )

    assert len(unresolved) == 1
    assert unresolved[0].reason == (
        "synthetic_missing_cell_fixture"
    )

    assert unavailable[0].summary_value is None
    assert unresolved[0].summary_value is None


def test_boundary_values_remain_ordinary_numeric_members() -> None:
    summaries = build_student_cell_summaries(
        _normalized(),
        _profile("fixture_median_min2"),
    )

    members = [
        value
        for row in summaries
        for _, value in row.member_values
    ]

    assert 1.0 in members
    assert 0 in members


def test_direct_teachers_are_kept_separate_from_student_conditions() -> None:
    values = extract_direct_teacher_values(_normalized())

    assert len(values) == 17
    assert sum(row.state == "defined" for row in values) == 16
    assert sum(row.state == "unavailable" for row in values) == 1
    assert all(row.population_unit == "teacher_seed" for row in values)

    assert any(
        row.key.endpoint_id == "endpoint_2"
        and row.state == "defined"
        for row in values
    )


def test_duplicate_initialization_inside_exact_cell_is_rejected() -> None:
    normalized = _normalized()
    endpoints = normalized["student_endpoints"]
    assert isinstance(endpoints, list)

    duplicate = copy.deepcopy(endpoints[0])
    duplicate["record_id"] = "synthetic_duplicate_semantic_member"
    endpoints.append(duplicate)

    with pytest.raises(
        CellSummaryError,
        match="multiple defined endpoint values",
    ):
        build_student_cell_summaries(
            normalized,
            _profile("fixture_median_min2"),
        )


def test_student_endpoint_without_declared_cell_is_rejected() -> None:
    normalized = _normalized()
    endpoints = normalized["student_endpoints"]
    assert isinstance(endpoints, list)

    endpoint = endpoints[0]
    identity = endpoint["identity"]
    assert isinstance(identity, dict)

    identity["protocol_id"] = "undeclared_protocol"

    with pytest.raises(
        CellSummaryError,
        match="no declared cell expectation",
    ):
        build_student_cell_summaries(
            normalized,
            _profile("fixture_median_min2"),
        )


def test_defined_endpoint_from_noneligible_attempt_is_rejected_again() -> None:
    normalized = _normalized()

    endpoints = normalized["student_endpoints"]
    eligibility = normalized["eligibility_records"]

    assert isinstance(endpoints, list)
    assert isinstance(eligibility, list)

    attempt_id = endpoints[0]["attempt_id"]

    record = next(
        row
        for row in eligibility
        if row["attempt_id"] == attempt_id
    )
    record["status"] = "ineligible"
    record["reason"] = "synthetic_part_e_mutation"

    with pytest.raises(
        CellSummaryError,
        match="must trace to an eligible attempt",
    ):
        build_student_cell_summaries(
            normalized,
            _profile("fixture_median_min2"),
        )

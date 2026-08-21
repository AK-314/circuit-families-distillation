from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from circuit_families.stage5d import (
    AnalysisIdentity,
    SyntheticRecordError,
    TechnicalAnalysisProfileError,
    load_synthetic_universe,
    load_technical_analysis_profile_set,
    synthetic_universe_from_mapping,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    ROOT
    / "followup/configs/stage5d/technical_analysis_profiles_v1.json"
)
FIXTURE_PATH = ROOT / "tests/fixtures/stage5d/synthetic_universe_v1.json"


def _fixture_mapping() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_profiles_require_explicit_injection_and_resolve_nothing() -> None:
    profile_set = load_technical_analysis_profile_set(PROFILE_PATH)

    assert len(profile_set.profiles) == 2
    assert profile_set.require("fixture_median_min2").profile_id == (
        "fixture_median_min2"
    )
    assert profile_set.require("fixture_mean_min3").profile_id == (
        "fixture_mean_min3"
    )

    for profile in profile_set.profiles:
        assert profile.synthetic_only is True
        assert profile.scientific_data is False
        assert profile.production_eligible is False
        assert profile.resolves_decisions == ()
        assert set(profile.decision_dependencies) == {
            "UD-004",
            "UD-011",
            "UD-012",
        }


def test_profile_rejects_scientific_data() -> None:
    raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    raw["profiles"][0]["scientific_data"] = True

    from circuit_families.stage5d.profiles import (
        TechnicalAnalysisProfileSet,
    )

    with pytest.raises(
        TechnicalAnalysisProfileError,
        match="cannot permit scientific data",
    ):
        TechnicalAnalysisProfileSet.from_mapping(raw)


def test_profile_rejects_resolution_of_unresolved_decision() -> None:
    raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    raw["profiles"][0]["resolves_decisions"] = ["UD-011"]

    from circuit_families.stage5d.profiles import (
        TechnicalAnalysisProfileSet,
    )

    with pytest.raises(
        TechnicalAnalysisProfileError,
        match="may not resolve",
    ):
        TechnicalAnalysisProfileSet.from_mapping(raw)


def test_profile_rejects_unknown_field() -> None:
    raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    raw["profiles"][0]["scientific_default"] = "median"

    from circuit_families.stage5d.profiles import (
        TechnicalAnalysisProfileSet,
    )

    with pytest.raises(
        TechnicalAnalysisProfileError,
        match="keys mismatch",
    ):
        TechnicalAnalysisProfileSet.from_mapping(raw)


def test_teacher_identity_rejects_student_fields() -> None:
    raw = {
        "subject_kind": "teacher",
        "teacher_seed": 0,
        "phase": "phase_early",
        "distillation_condition": "hard",
        "student_initialization": None,
        "method_id": "method_greedy",
        "endpoint_id": "endpoint_1",
        "protocol_id": "synthetic_protocol_v1",
        "fidelity_id": "synthetic_fidelity_v1",
        "budget_id": "budget_greedy_v1",
    }

    with pytest.raises(
        SyntheticRecordError,
        match="teacher identities cannot",
    ):
        AnalysisIdentity.from_mapping(raw)


def test_student_identity_requires_condition_and_initialization() -> None:
    raw = {
        "subject_kind": "student",
        "teacher_seed": 0,
        "phase": "phase_early",
        "distillation_condition": None,
        "student_initialization": None,
        "method_id": "method_greedy",
        "endpoint_id": "endpoint_1",
        "protocol_id": "synthetic_protocol_v1",
        "fidelity_id": "synthetic_fidelity_v1",
        "budget_id": "budget_greedy_v1",
    }

    with pytest.raises(
        SyntheticRecordError,
        match="require hard or soft",
    ):
        AnalysisIdentity.from_mapping(raw)


def test_synthetic_universe_has_required_coverage() -> None:
    universe = load_synthetic_universe(FIXTURE_PATH)

    assert len(universe.teacher_inventories) == 2
    assert {record.method_id for record in universe.method_budgets} == {
        "method_greedy",
        "method_beam",
    }
    assert {
        record.native_budget for record in universe.method_budgets
    } == {100, 180}

    assert any(
        record.outcome == "failed"
        for record in universe.student_attempts
    )
    assert any(
        record.state == "unavailable"
        for record in universe.cell_expectations
    )
    assert any(
        record.state == "unresolved"
        for record in universe.cell_expectations
    )


def test_endpoint_boundary_values_are_ordinary_defined_results() -> None:
    universe = load_synthetic_universe(FIXTURE_PATH)

    records = (
        *universe.direct_teacher_endpoints,
        *universe.student_endpoints,
    )

    endpoint_one = [
        record
        for record in records
        if record.identity.endpoint_id == "endpoint_1"
        and record.state == "defined"
        and record.value == 1.0
    ]
    endpoint_two = [
        record
        for record in records
        if record.identity.endpoint_id == "endpoint_2"
        and record.state == "defined"
        and record.value == 0
    ]

    assert endpoint_one
    assert endpoint_two


def test_defined_student_endpoint_requires_eligible_attempt() -> None:
    raw = _fixture_mapping()
    student_endpoints = raw["student_endpoints"]
    eligibility_records = raw["eligibility_records"]

    assert isinstance(student_endpoints, list)
    assert isinstance(eligibility_records, list)

    endpoint = student_endpoints[0]
    assert isinstance(endpoint, dict)
    attempt_id = endpoint["attempt_id"]

    target = next(
        record
        for record in eligibility_records
        if isinstance(record, dict)
        and record["attempt_id"] == attempt_id
    )
    target["status"] = "ineligible"
    target["reason"] = "synthetic_mutation"

    with pytest.raises(
        SyntheticRecordError,
        match="cannot come from ineligible",
    ):
        synthetic_universe_from_mapping(raw)


def test_student_endpoint_identity_must_match_attempt() -> None:
    raw = _fixture_mapping()
    mutated = copy.deepcopy(raw)

    student_endpoints = mutated["student_endpoints"]
    assert isinstance(student_endpoints, list)
    endpoint = student_endpoints[0]
    assert isinstance(endpoint, dict)

    identity = endpoint["identity"]
    assert isinstance(identity, dict)
    identity["teacher_seed"] = 1

    with pytest.raises(
        SyntheticRecordError,
        match="identity does not match",
    ):
        synthetic_universe_from_mapping(mutated)


def test_method_budget_identity_must_match_registered_method() -> None:
    raw = _fixture_mapping()

    student_endpoints = raw["student_endpoints"]
    assert isinstance(student_endpoints, list)
    endpoint = student_endpoints[0]
    assert isinstance(endpoint, dict)

    identity = endpoint["identity"]
    assert isinstance(identity, dict)
    identity["budget_id"] = "budget_beam_v1"

    with pytest.raises(
        SyntheticRecordError,
        match="incompatible method/budget",
    ):
        synthetic_universe_from_mapping(raw)

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from circuit_families.stage5bc.technical_profiles import (
    ALLOWED_DECISION_DEPENDENCIES,
    PROFILE_KINDS,
    PROFILE_SCHEMA_VERSION,
    TechnicalProfileError,
    TechnicalProfileSet,
    load_technical_profile_set,
)

FIXTURE = Path("tests/fixtures/stage5bc/technical_profile_set_v1.json")
SCHEMA = Path("followup/schemas/stage5bc/technical_profile_v1.schema.json")


def _raw_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_loads_with_exact_five_technical_profile_kinds() -> None:
    profile_set = load_technical_profile_set(FIXTURE)

    assert tuple(
        profile["profile_kind"]
        for profile in profile_set.to_mapping()["profiles"]
    ) == PROFILE_KINDS

    for profile in profile_set.profiles:
        assert profile.schema_version == PROFILE_SCHEMA_VERSION
        assert profile.classification == "technical_fixture"
        assert profile.scientific_data is False
        assert profile.production_eligible is False
        assert profile.settings_status == "technical_candidate_only"
        assert profile.resolves_decisions == ()
        assert profile.decision_dependencies
        assert set(profile.decision_dependencies) <= ALLOWED_DECISION_DEPENDENCIES


@pytest.mark.parametrize(
    "field",
    [
        "scientific_data",
        "production_eligible",
        "decision_dependencies",
        "resolves_decisions",
        "settings",
    ],
)
def test_required_boundary_fields_have_no_implicit_defaults(field: str) -> None:
    raw = _raw_fixture()
    del raw["profiles"][0][field]

    with pytest.raises(TechnicalProfileError, match="keys mismatch"):
        TechnicalProfileSet.from_mapping(raw)


def test_scientific_data_true_is_rejected() -> None:
    raw = _raw_fixture()
    raw["profiles"][0]["scientific_data"] = True

    with pytest.raises(TechnicalProfileError, match="scientific_data=false"):
        TechnicalProfileSet.from_mapping(raw)


def test_production_eligible_true_is_rejected() -> None:
    raw = _raw_fixture()
    raw["profiles"][0]["production_eligible"] = True

    with pytest.raises(TechnicalProfileError, match="production_eligible=false"):
        TechnicalProfileSet.from_mapping(raw)


def test_resolving_an_unresolved_decision_is_rejected() -> None:
    raw = _raw_fixture()
    raw["profiles"][0]["resolves_decisions"] = ["UD-003"]

    with pytest.raises(TechnicalProfileError, match="resolve no scientific"):
        TechnicalProfileSet.from_mapping(raw)


def test_unknown_decision_dependency_is_rejected() -> None:
    raw = _raw_fixture()
    raw["profiles"][0]["decision_dependencies"] = ["UD-999"]

    with pytest.raises(TechnicalProfileError, match="unknown unresolved"):
        TechnicalProfileSet.from_mapping(raw)


def test_empty_decision_dependency_list_is_rejected() -> None:
    raw = _raw_fixture()
    raw["profiles"][0]["decision_dependencies"] = []

    with pytest.raises(TechnicalProfileError, match="non-empty explicit list"):
        TechnicalProfileSet.from_mapping(raw)


def test_missing_profile_kind_is_rejected() -> None:
    raw = _raw_fixture()
    raw["profiles"] = raw["profiles"][:-1]

    with pytest.raises(TechnicalProfileError, match="exactly these kinds"):
        TechnicalProfileSet.from_mapping(raw)


def test_duplicate_profile_kind_is_rejected() -> None:
    raw = _raw_fixture()
    duplicate = copy.deepcopy(raw["profiles"][0])
    duplicate["profile_id"] = "technical-duplicate-architecture/v1"
    raw["profiles"][-1] = duplicate

    with pytest.raises(TechnicalProfileError, match="profile_kind once"):
        TechnicalProfileSet.from_mapping(raw)


def test_round_trip_is_canonical_by_profile_kind() -> None:
    raw = _raw_fixture()
    raw["profiles"].reverse()

    profile_set = TechnicalProfileSet.from_mapping(raw)
    rendered = profile_set.to_mapping()

    assert [item["profile_kind"] for item in rendered["profiles"]] == list(
        PROFILE_KINDS
    )
    assert all(item["resolves_decisions"] == [] for item in rendered["profiles"])


def test_json_schema_encodes_the_same_scientific_firewall() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    properties = schema["properties"]

    assert schema["additionalProperties"] is False
    assert properties["scientific_data"]["const"] is False
    assert properties["production_eligible"]["const"] is False
    assert properties["resolves_decisions"]["maxItems"] == 0
    assert set(properties["decision_dependencies"]["items"]["enum"]) == (
        ALLOWED_DECISION_DEPENDENCIES
    )
    assert tuple(properties["profile_kind"]["enum"]) == PROFILE_KINDS

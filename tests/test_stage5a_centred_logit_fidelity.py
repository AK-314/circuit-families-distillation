"""Focused Stage 5A centred-logit fidelity contract tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from circuit_families.interpretability.centred_logit_fidelity import (
    FIDELITY_FORMULA_REF,
    TECHNICAL_PROFILE_SET_VERSION,
    TechnicalNumericalProfile,
    load_technical_profile_set,
    technical_profile_from_record,
    validate_technical_profile_set,
)

REPOSITORY = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    REPOSITORY
    / "followup"
    / "configs"
    / "stage5a_technical_fidelity_profiles_v1.json"
)


def _raw_profile_set() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _raw_profile() -> dict[str, object]:
    profile_set = _raw_profile_set()
    profiles = profile_set["profiles"]
    assert isinstance(profiles, list)
    profile = profiles[0]
    assert isinstance(profile, dict)
    return copy.deepcopy(profile)


def test_committed_technical_profile_loads_and_is_explicitly_nonproduction() -> None:
    profiles = load_technical_profile_set(PROFILE_PATH)

    assert len(profiles) == 1
    profile = profiles[0]

    assert isinstance(profile, TechnicalNumericalProfile)
    assert profile.profile_ref == "centred-logit-technical-f64-sequential/v1"
    assert profile.formula_ref == FIDELITY_FORMULA_REF
    assert profile.profile_status == "technical_candidate"
    assert profile.scientific_data is False
    assert profile.production_eligible is False
    assert profile.resolves_ud007 is False
    assert profile.accumulation_dtype == "float64"
    assert profile.centering_dtype == "preserve_input_float_dtype"
    assert (
        profile.accumulation_order
        == "canonical_example_order_then_class_sum"
    )
    assert (
        profile.canonical_order_policy
        == "explicit_index_strict_contiguous_ascending"
    )
    assert (
        profile.batch_semantics
        == "batch_boundaries_do_not_change_logical_order"
    )
    assert profile.nonfinite_policy == "reject"
    assert (
        profile.denominator_guard_candidate
        == "classify_exact_zero_else_positive"
    )
    assert profile.near_zero_guard_defined is False


def test_profile_set_is_explicitly_stage5a_technical_only() -> None:
    raw = _raw_profile_set()

    assert raw["profile_set_version"] == TECHNICAL_PROFILE_SET_VERSION
    assert raw["stage"] == "5A"
    assert raw["purpose"] == "synthetic_and_technical_validation_only"
    assert raw["scientific_data"] is False
    assert raw["production_eligible"] is False
    assert raw["resolves_ud007"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("production_eligible", True, "production_eligible=false"),
        ("resolves_ud007", True, "resolves_ud007=false"),
        ("scientific_data", True, "scientific_data=false"),
        (
            "profile_status",
            "production_frozen",
            "profile_status='technical_candidate'",
        ),
        (
            "near_zero_guard_defined",
            True,
            "near-zero production guard unresolved",
        ),
    ],
)
def test_profile_rejects_authoritative_or_ud007_resolving_claims(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = _raw_profile()
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        technical_profile_from_record(raw)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "fidelity_threshold",
        "primary_fidelity_threshold",
        "threshold",
        "sensitivity_grid",
        "fidelity_sensitivity_grid",
        "production_threshold",
        "production_precision",
        "final_precision",
        "final_denominator_guard",
        "final_reduction_order",
    ],
)
def test_profile_rejects_premature_scientific_choice_fields(
    forbidden_field: str,
) -> None:
    raw = _raw_profile()
    raw[forbidden_field] = 0.99

    with pytest.raises(
        ValueError,
        match="premature scientific choice field",
    ):
        technical_profile_from_record(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accumulation_dtype", "float32"),
        ("centering_dtype", "force_float64"),
        ("accumulation_order", "parallel_unordered_sum"),
        ("canonical_order_policy", "arbitrary"),
        ("batch_semantics", "batch_dependent"),
        ("nonfinite_policy", "ignore"),
        ("denominator_guard_candidate", "epsilon_clip"),
    ],
)
def test_unregistered_technical_behaviour_is_rejected(
    field: str,
    value: str,
) -> None:
    raw = _raw_profile()
    raw[field] = value

    with pytest.raises(ValueError, match="Unsupported Stage 5A technical"):
        technical_profile_from_record(raw)


def test_profile_set_rejects_production_or_ud007_resolution() -> None:
    for field in ("scientific_data", "production_eligible", "resolves_ud007"):
        raw = _raw_profile_set()
        raw[field] = True
        with pytest.raises(ValueError):
            validate_technical_profile_set(raw)


def test_profile_set_rejects_duplicate_profile_references() -> None:
    raw = _raw_profile_set()
    profiles = raw["profiles"]
    assert isinstance(profiles, list)
    profiles.append(copy.deepcopy(profiles[0]))

    with pytest.raises(ValueError, match="profile_ref values must be unique"):
        validate_technical_profile_set(raw)


def test_profile_record_is_json_safe_and_round_trips() -> None:
    profile = load_technical_profile_set(PROFILE_PATH)[0]

    record = profile.to_record()
    encoded = json.dumps(
        record,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    decoded = json.loads(encoded)

    assert technical_profile_from_record(decoded) == profile


def test_profile_contains_no_threshold_or_sensitivity_grid() -> None:
    raw_text = PROFILE_PATH.read_text(encoding="utf-8").lower()

    forbidden_json_keys = (
        '"fidelity_threshold"',
        '"primary_fidelity_threshold"',
        '"threshold"',
        '"sensitivity_grid"',
        '"fidelity_sensitivity_grid"',
    )
    assert all(key not in raw_text for key in forbidden_json_keys)


def test_metric_record_json_roundtrip() -> None:
    import json

    from circuit_families.interpretability.centred_logit_fidelity import (
        CentredLogitPredictiveMetricRecord,
        centred_logit_predictive_metric_record_from_record,
    )

    record = CentredLogitPredictiveMetricRecord(
        formula_ref="centred-logit-predictive-fidelity/v1",
        profile_ref="centred-logit-technical-f64-sequential/v1",
        record_status="technical_candidate",
        evaluated_example_count=2,
        class_count=3,
        numerator=10.0,
        denominator=5.0,
        predictive_fidelity=-1.0,
        denominator_status="valid",
        canonical_order_policy="canonical",
        accumulation_order="sequential",
        nonfinite_rejected=True,
        notes="technical-only",
    )

    decoded = json.loads(
        json.dumps(record.to_record(), allow_nan=False)
    )

    restored = centred_logit_predictive_metric_record_from_record(decoded)

    assert restored == record


def test_metric_record_roundtrip_preserves_negative_fidelity() -> None:
    from circuit_families.interpretability.centred_logit_fidelity import (
        CentredLogitPredictiveMetricRecord,
        centred_logit_predictive_metric_record_from_record,
    )

    record = CentredLogitPredictiveMetricRecord(
        formula_ref="centred-logit-predictive-fidelity/v1",
        profile_ref="centred-logit-technical-f64-sequential/v1",
        record_status="technical_candidate",
        evaluated_example_count=1,
        class_count=2,
        numerator=3.0,
        denominator=1.0,
        predictive_fidelity=-2.0,
        denominator_status="valid",
        canonical_order_policy="canonical",
        accumulation_order="sequential",
        nonfinite_rejected=True,
        notes="technical-only",
    )

    restored = centred_logit_predictive_metric_record_from_record(
        record.to_record()
    )

    assert restored.predictive_fidelity == -2.0


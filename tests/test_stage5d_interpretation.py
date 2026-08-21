from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from circuit_families.stage5d import (
    Stage5DInterpretationError,
    build_neutral_interpretation_audit,
    build_stage5d_output_bundle,
    frozen_outcome_labels,
    load_and_normalize_ingestion,
    load_technical_analysis_profile_set,
    stage5d_boundary_audit,
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
BOUNDARY_DOCUMENT = (
    ROOT
    / "docs/distillation_followup/"
    "stage5d_barrier1_boundary.md"
)
UNRESOLVED_REGISTER = (
    ROOT / "followup/configs/stage2_unresolved_decisions_v1.json"
)


@pytest.fixture
def output_bundle():
    normalized = load_and_normalize_ingestion(ENVELOPE)
    profile = load_technical_analysis_profile_set(PROFILES).require(
        "fixture_median_min2"
    )
    return build_stage5d_output_bundle(normalized, profile)


def test_six_frozen_interpretations_use_neutral_labels_only() -> None:
    labels = frozen_outcome_labels()

    assert len(labels) == 6
    assert tuple(label.frozen_order for label in labels) == tuple(range(1, 7))
    assert tuple(label.interpretation_id for label in labels) == (
        "frozen_outcome_01",
        "frozen_outcome_02",
        "frozen_outcome_03",
        "frozen_outcome_04",
        "frozen_outcome_05",
        "frozen_outcome_06",
    )
    assert tuple(label.neutral_label for label in labels) == (
        "teacher_phase_across_student_conditions",
        "within_teacher_student_realization_variation",
        "cross_student_compressibility",
        "method_fidelity_sensitivity",
        "function_realization_relationship",
        "predictive_fidelity_transition_comparison",
    )
    assert {label.status for label in labels} == {
        "label_only_not_assessed"
    }

    forbidden_directional_or_causal_terms = {
        "increase",
        "decrease",
        "higher",
        "lower",
        "persists",
        "dominates",
        "contributes",
        "causes",
        "drives",
        "enables",
        "controls",
        "disappears",
    }
    words = {
        word
        for label in labels
        for word in label.neutral_label.split("_")
    }
    assert words.isdisjoint(forbidden_directional_or_causal_terms)


def test_synthetic_values_cannot_create_conclusions(output_bundle) -> None:
    original = build_neutral_interpretation_audit(output_bundle)
    changed_values = copy.deepcopy(output_bundle)
    first_row = changed_values["output_objects"][
        "direct_teacher_summaries"
    ]["rows"][0]
    first_row["value"] = -999999.0
    altered = build_neutral_interpretation_audit(changed_values)

    assert altered.labels == original.labels
    assert original.directional_predictions == ()
    assert original.automatic_conclusions == ()
    assert original.causal_conclusions == ()
    assert original.scientific_claims == ()
    assert altered.directional_predictions == ()
    assert altered.automatic_conclusions == ()
    assert altered.causal_conclusions == ()
    assert altered.scientific_claims == ()


def test_stage5d_boundary_keeps_required_decisions_unresolved() -> None:
    boundary = stage5d_boundary_audit()
    decisions = {
        decision.decision_id: decision
        for decision in boundary.unresolved_decisions
    }

    assert set(decisions) == {"UD-004", "UD-011", "UD-012", "UD-014"}
    assert all(decision.status == "unresolved" for decision in decisions.values())
    assert (
        decisions["UD-004"].owner,
        decisions["UD-004"].lane,
        decisions["UD-004"].resolution_stage,
    ) == ("Austin", "Lane B", "Stage 11")
    assert (
        decisions["UD-011"].owner,
        decisions["UD-011"].lane,
        decisions["UD-011"].resolution_stage,
    ) == ("Alex", "Lane D", "Stage 13")
    assert (
        decisions["UD-012"].owner,
        decisions["UD-012"].lane,
        decisions["UD-012"].resolution_stage,
    ) == ("Alex", "Lane D", "Stage 13")
    assert (
        decisions["UD-014"].owner,
        decisions["UD-014"].lane,
        decisions["UD-014"].resolution_stage,
    ) == ("Joint", "Joint", "Stage 14 / Barrier 3")

    canonical = json.loads(
        UNRESOLVED_REGISTER.read_text(encoding="utf-8")
    )
    canonical_by_id = {
        record["decision_id"]: record
        for record in canonical["decisions"]
    }
    for decision_id, boundary_record in decisions.items():
        canonical_record = canonical_by_id[decision_id]
        assert boundary_record.decision_family == (
            canonical_record["decision_family"]
        )
        assert boundary_record.owner == canonical_record["owner"]
        assert boundary_record.lane == canonical_record["lane"]
        assert boundary_record.resolution_stage == (
            canonical_record["resolution_stage"]
        )
        assert canonical_record["status"] == "unresolved"


def test_stage5d_boundary_has_no_scientific_execution_paths() -> None:
    boundary = stage5d_boundary_audit()

    assert boundary.barrier_id == "Barrier 1"
    assert boundary.purpose == (
        "deterministic hierarchy exercise on synthetic fixtures only"
    )
    assert boundary.synthetic_only is True
    assert boundary.real_result_ingestion is False
    assert boundary.stage6_execution is False
    assert boundary.production_training is False
    assert boundary.scientific_analysis is False
    assert boundary.final_scientific_freeze is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "classification",
            "scientific_result",
            "real or production-result ingestion",
        ),
        (
            "scientific_data",
            True,
            "scientific data cannot enter",
        ),
        (
            "production_eligible",
            True,
            "production results cannot enter",
        ),
    ),
)
def test_non_synthetic_or_production_results_cannot_be_ingested(
    output_bundle,
    field,
    value,
    message,
) -> None:
    forbidden = copy.deepcopy(output_bundle)
    forbidden[field] = value

    with pytest.raises(Stage5DInterpretationError, match=message):
        build_neutral_interpretation_audit(forbidden)


def test_boundary_document_records_required_absences_and_ud_status() -> None:
    text = BOUNDARY_DOCUMENT.read_text(encoding="utf-8")

    assert "deterministic hierarchy exercise on synthetic fixtures only" in text
    for decision_id in ("UD-004", "UD-011", "UD-012", "UD-014"):
        assert decision_id in text
    for absent_capability in (
        "real-result ingestion",
        "Stage 6 execution",
        "production training",
        "scientific analysis",
        "final scientific freeze",
    ):
        assert absent_capability in text
    assert "Nothing in Stage 5D resolves" in text

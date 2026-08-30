"""Focused/adversarial Stage 12-P1 phase-adapter tests."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import pytest

from circuit_families.analysis.phase_detection import (
    find_pre_grokking_checkpoint,
    find_stable_post_sequence,
    select_transition_landmarks,
)
from circuit_families.stage12p1.phase import (
    HISTORICAL_RULE_ID,
    PHASE_ROLES,
    PhaseAdapterError,
    PhaseDecision,
    build_teacher_trajectory,
    phase_artifact_sha256,
    select_teacher_phases,
    validate_phase_selection_artifact,
    validate_teacher_trajectory,
)

TASK_HASH = "1" * 64
TEACHER_ARTIFACT_HASH = "2" * 64


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row(
    step: int,
    *,
    train: float,
    test: float,
    train_loss: float = 0.1,
    test_loss: float = 1.0,
) -> dict:
    return {
        "training_step": step,
        "train_accuracy": train,
        "test_accuracy": test,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "checkpoint_path": f"checkpoints/technical/step_{step:08d}.pt",
        "checkpoint_sha256": _sha(f"technical-checkpoint-{step}"),
    }


def _historical_rows() -> list[dict]:
    return [
        _row(0, train=0.50, test=0.00),
        _row(50, train=1.00, test=0.02),
        _row(100, train=1.00, test=0.15),
        _row(150, train=1.00, test=0.50),
        _row(200, train=1.00, test=0.90),
        _row(250, train=1.00, test=0.990),
        _row(300, train=1.00, test=0.995),
        _row(350, train=1.00, test=0.997),
        _row(400, train=1.00, test=0.999),
        _row(450, train=1.00, test=1.000),
    ]


def _trajectory(rows=None, *, seed=17):
    return build_teacher_trajectory(
        teacher_seed_id="technical-teacher-seed/v1",
        teacher_seed=seed,
        task_identity_sha256=TASK_HASH,
        teacher_artifact_sha256=TEACHER_ARTIFACT_HASH,
        records=_historical_rows() if rows is None else rows,
    )


def test_historical_adapter_matches_frozen_stage3_selectors() -> None:
    rows = _historical_rows()
    trajectory = _trajectory(rows)
    artifact = select_teacher_phases(trajectory)

    pre = find_pre_grokking_checkpoint(rows)
    stable_sequence, stable = find_stable_post_sequence(rows)
    landmarks = select_transition_landmarks(
        rows,
        pre_step=int(pre["training_step"]),
        stable_post_step=int(stable["training_step"]),
    )

    by_role = {
        record["role"]: record
        for record in artifact["phase_records"]
    }

    assert artifact["selection_rule"] == {
        "rule_id": "historical-stage3-phase-selection",
        "rule_version": "stage3-frozen/v1",
    }
    assert by_role["pre"]["training_step"] == pre["training_step"]
    assert by_role["transition"]["training_step"] == (
        landmarks["50%"]["training_step"]
    )
    assert by_role["stable_post"]["training_step"] == stable["training_step"]
    assert by_role["stable_post"]["stable_supporting_steps"] == [
        item["training_step"]
        for item in stable_sequence
    ]


def test_sealed_trajectory_round_trip_and_content_hash() -> None:
    trajectory = _trajectory()
    sealed = trajectory.sealed_mapping()
    rebuilt = validate_teacher_trajectory(sealed)

    assert rebuilt == trajectory
    assert rebuilt.content_sha256() == sealed["content_sha256"]


def test_trajectory_rejects_non_metric_scientific_payload_fields() -> None:
    row = _historical_rows()[0]
    row["circuit_effect"] = 0.91

    with pytest.raises(PhaseAdapterError, match="keys mismatch"):
        _trajectory([row])


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "distillation_condition",
        "endpoint_value",
        "discovery_method",
        "causal_effect",
        "component_set",
    ),
)
def test_every_forbidden_scientific_field_is_structurally_excluded(
    forbidden_key: str,
) -> None:
    row = _historical_rows()[0]
    row[forbidden_key] = "forbidden"

    with pytest.raises(PhaseAdapterError, match="keys mismatch"):
        _trajectory([row])


def test_duplicate_or_nonincreasing_steps_are_rejected() -> None:
    rows = [
        _row(0, train=0.5, test=0.0),
        _row(50, train=1.0, test=0.02),
        _row(50, train=1.0, test=0.03),
    ]

    with pytest.raises(PhaseAdapterError, match="strictly increasing"):
        _trajectory(rows)


def test_absolute_checkpoint_path_is_rejected() -> None:
    rows = _historical_rows()
    rows[0]["checkpoint_path"] = "/tmp/technical.pt"

    with pytest.raises(PhaseAdapterError, match="portable relative"):
        _trajectory(rows)


def test_tampered_trajectory_hash_is_rejected() -> None:
    sealed = _trajectory().sealed_mapping()
    sealed["points"][0]["test_accuracy"] = 0.1

    with pytest.raises(PhaseAdapterError, match="content hash mismatch"):
        validate_teacher_trajectory(sealed)


def test_missing_pre_is_explicitly_unavailable_and_transition_stays_unavailable() -> None:
    rows = [
        _row(0, train=0.5, test=0.20),
        _row(50, train=1.0, test=0.99),
        _row(100, train=1.0, test=0.99),
        _row(150, train=1.0, test=0.99),
        _row(200, train=1.0, test=0.99),
        _row(250, train=1.0, test=0.99),
    ]
    artifact = select_teacher_phases(_trajectory(rows))
    by_role = {item["role"]: item for item in artifact["phase_records"]}

    assert by_role["pre"]["availability_status"] == "unavailable"
    assert by_role["transition"]["availability_status"] == "unavailable"
    assert by_role["stable_post"]["availability_status"] == "selected"
    assert "training_step" not in by_role["pre"]


def test_missing_stable_is_explicitly_unavailable_without_replacement() -> None:
    rows = [
        _row(0, train=0.5, test=0.0),
        _row(50, train=1.0, test=0.02),
        _row(100, train=1.0, test=0.2),
        _row(150, train=1.0, test=0.5),
        _row(200, train=1.0, test=0.8),
    ]
    artifact = select_teacher_phases(_trajectory(rows))
    by_role = {item["role"]: item for item in artifact["phase_records"]}

    assert by_role["pre"]["availability_status"] == "selected"
    assert by_role["transition"]["availability_status"] == "unavailable"
    assert by_role["stable_post"]["availability_status"] == "unavailable"
    assert "training_step" not in by_role["stable_post"]


@dataclass(frozen=True)
class _InjectedTechnicalRule:
    rule_id: str = "technical-expanded-phase-rule"
    rule_version: str = "technical-only/v1"

    def select(self, trajectory):
        del trajectory
        return (
            PhaseDecision(
                role="pre",
                phase_label="technical-pre",
                availability_status="selected",
                selected_step=0,
            ),
            PhaseDecision(
                role="transition",
                phase_label="technical-transition",
                availability_status="selected",
                selected_step=150,
                transition_target=0.5,
                transition_absolute_distance=0.0,
            ),
            PhaseDecision(
                role="stable_post",
                phase_label="technical-stable",
                availability_status="selected",
                selected_step=450,
                stable_supporting_steps=(250, 300, 350, 400, 450),
            ),
        )


def test_expanded_rule_is_explicitly_injected_and_versioned() -> None:
    artifact = select_teacher_phases(
        _trajectory(),
        rule=_InjectedTechnicalRule(),
    )

    assert artifact["selection_rule"] == {
        "rule_id": "technical-expanded-phase-rule",
        "rule_version": "technical-only/v1",
    }
    assert [
        item["phase_label"]
        for item in artifact["phase_records"]
    ] == [
        "technical-pre",
        "technical-transition",
        "technical-stable",
    ]


@dataclass(frozen=True)
class _MasqueradingRule(_InjectedTechnicalRule):
    rule_id: str = HISTORICAL_RULE_ID


def test_injected_rule_cannot_masquerade_as_historical_rule() -> None:
    with pytest.raises(PhaseAdapterError, match="masquerade"):
        select_teacher_phases(
            _trajectory(),
            rule=_MasqueradingRule(),
        )


@dataclass(frozen=True)
class _UnknownStepRule(_InjectedTechnicalRule):
    def select(self, trajectory):
        del trajectory
        decisions = list(super().select(None))
        decisions[1] = PhaseDecision(
            role="transition",
            phase_label="technical-transition",
            availability_status="selected",
            selected_step=999999,
        )
        return tuple(decisions)


def test_injected_rule_cannot_select_absent_checkpoint() -> None:
    with pytest.raises(PhaseAdapterError, match="unknown trajectory step"):
        select_teacher_phases(
            _trajectory(),
            rule=_UnknownStepRule(),
        )


def test_phase_artifact_is_canonical_bound_and_tamper_evident() -> None:
    trajectory = _trajectory()
    artifact = select_teacher_phases(trajectory)

    validated = validate_phase_selection_artifact(
        artifact,
        trajectory=trajectory,
    )
    assert validated == artifact
    assert phase_artifact_sha256(artifact)

    tampered = copy.deepcopy(artifact)
    tampered["phase_records"][1]["test_accuracy"] = 0.51

    with pytest.raises(PhaseAdapterError, match="content hash mismatch"):
        validate_phase_selection_artifact(tampered)


def test_selected_phase_source_hash_is_bound_to_exact_trajectory_point() -> None:
    trajectory = _trajectory()
    artifact = select_teacher_phases(trajectory)

    selected = next(
        item
        for item in artifact["phase_records"]
        if item["role"] == "transition"
    )

    changed_rows = _historical_rows()
    changed_rows[3]["test_loss"] = 1.5
    changed_trajectory = _trajectory(changed_rows)

    # Selection step stays the same, but the immutable source point does not.
    assert (
        select_teacher_phases(changed_trajectory)["phase_records"][1][
            "training_step"
        ]
        == selected["training_step"]
    )
    assert (
        select_teacher_phases(changed_trajectory)["phase_records"][1][
            "source_point_sha256"
        ]
        != selected["source_point_sha256"]
    )


def test_teacher_seed_identity_is_preserved_per_trajectory() -> None:
    left = select_teacher_phases(_trajectory(seed=17))
    right = select_teacher_phases(_trajectory(seed=18))

    assert left["teacher_seed"] == 17
    assert right["teacher_seed"] == 18
    assert left["content_sha256"] != right["content_sha256"]


def test_phase_roles_have_one_canonical_order() -> None:
    artifact = select_teacher_phases(_trajectory())
    assert tuple(
        item["role"]
        for item in artifact["phase_records"]
    ) == PHASE_ROLES


def test_artifacts_never_gain_policy_authority() -> None:
    trajectory = _trajectory()
    sealed = trajectory.sealed_mapping()
    artifact = select_teacher_phases(trajectory)

    for record in (sealed, artifact):
        assert record["scientific_data"] is False
        assert record["production_eligible"] is False
        assert record["classification"] == "technical_fixture"

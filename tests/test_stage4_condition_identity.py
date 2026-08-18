from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    ConditionIdentityError,
    Stage3AvailabilityIndex,
    build_condition_id,
    parse_condition_id,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "followup/manifests/stage3_teacher_registry_v1.json"


@pytest.fixture(scope="module")
def stage3() -> Stage3AvailabilityIndex:
    registry = json.loads(REGISTRY.read_text())
    return Stage3AvailabilityIndex.from_registry(registry)


def test_stage3_index_preserves_15_13_2(stage3: Stage3AvailabilityIndex) -> None:
    assert len(stage3.cells) == 15
    assert sum(v == "selected" for v in stage3.cells.values()) == 13
    assert sum(v == "unavailable" for v in stage3.cells.values()) == 2
    assert stage3.availability(0, "pre-grokking") == "unavailable"
    assert stage3.availability(0, "50%") == "unavailable"
    assert stage3.availability(0, "stable post-grokking") == "selected"


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        (
            ConditionIdentity(
                teacher_seed=1,
                phase="stable post-grokking",
            ),
            "cfdid:v1:d2|teacher_seed=1|"
            "phase=stable%20post-grokking",
        ),
        (
            ConditionIdentity(
                teacher_seed=1,
                phase="stable post-grokking",
                distillation_condition="direct_teacher",
            ),
            "cfdid:v1:d3|teacher_seed=1|"
            "phase=stable%20post-grokking|"
            "distillation_condition=direct_teacher",
        ),
        (
            ConditionIdentity(
                teacher_seed=1,
                phase="stable post-grokking",
                distillation_condition="hard_target",
                student_initialization=0,
            ),
            "cfdid:v1:d4|teacher_seed=1|"
            "phase=stable%20post-grokking|"
            "distillation_condition=hard_target|"
            "student_initialization=0",
        ),
        (
            ConditionIdentity(
                teacher_seed=1,
                phase="stable post-grokking",
                distillation_condition="soft_target",
                student_initialization=2,
                discovery_method="synthetic-method/v1",
                fidelity_setting="synthetic-fidelity/v1",
                component_cap="synthetic-cap/v1",
                overlap_setting="synthetic-overlap/v1",
            ),
            "cfdid:v1:d8|teacher_seed=1|"
            "phase=stable%20post-grokking|"
            "distillation_condition=soft_target|"
            "student_initialization=2|"
            "discovery_method=synthetic-method%2Fv1|"
            "fidelity_setting=synthetic-fidelity%2Fv1|"
            "component_cap=synthetic-cap%2Fv1|"
            "overlap_setting=synthetic-overlap%2Fv1",
        ),
    ],
)
def test_build_and_parse_round_trip(
    stage3: Stage3AvailabilityIndex,
    identity: ConditionIdentity,
    expected: str,
) -> None:
    condition_id = build_condition_id(identity, stage3)
    assert condition_id == expected
    parsed = parse_condition_id(condition_id, stage3)
    assert parsed == identity
    assert build_condition_id(parsed, stage3) == condition_id


def test_unavailable_teacher_phase_prefix_valid_but_downstream_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    prefix = ConditionIdentity(
        teacher_seed=0,
        phase="pre-grokking",
    )
    assert (
        build_condition_id(prefix, stage3)
        == "cfdid:v1:d2|teacher_seed=0|phase=pre-grokking"
    )

    downstream = ConditionIdentity(
        teacher_seed=0,
        phase="pre-grokking",
        distillation_condition="direct_teacher",
    )
    with pytest.raises(
        ConditionIdentityError,
        match="unavailable Stage 3 cell",
    ):
        build_condition_id(downstream, stage3)


def test_direct_teacher_cannot_have_student_initialization(
    stage3: Stage3AvailabilityIndex,
) -> None:
    identity = ConditionIdentity(
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="direct_teacher",
        student_initialization=0,
    )
    with pytest.raises(
        ConditionIdentityError,
        match="direct_teacher cannot have student_initialization",
    ):
        build_condition_id(identity, stage3)


def test_parser_rejects_noncanonical_field_order(
    stage3: Stage3AvailabilityIndex,
) -> None:
    bad = (
        "cfdid:v1:d3|teacher_seed=1|"
        "distillation_condition=hard_target|"
        "phase=stable%20post-grokking"
    )
    with pytest.raises(
        ConditionIdentityError,
        match="canonical order",
    ):
        parse_condition_id(bad, stage3)

from __future__ import annotations

import json
import unicodedata
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
    return Stage3AvailabilityIndex.from_registry(
        json.loads(REGISTRY.read_text())
    )


def test_missing_field_rejected(stage3: Stage3AvailabilityIndex) -> None:
    bad = (
        "cfdid:v1:d3|teacher_seed=1|"
        "phase=stable%20post-grokking"
    )
    with pytest.raises(
        ConditionIdentityError,
        match=r"depth=3 requires 3 fields",
    ):
        parse_condition_id(bad, stage3)


def test_extra_field_rejected(stage3: Stage3AvailabilityIndex) -> None:
    bad = (
        "cfdid:v1:d3|teacher_seed=1|"
        "phase=stable%20post-grokking|"
        "distillation_condition=hard_target|"
        "student_initialization=0"
    )
    with pytest.raises(
        ConditionIdentityError,
        match=r"depth=3 requires 3 fields",
    ):
        parse_condition_id(bad, stage3)


@pytest.mark.parametrize(
    "phase",
    [
        "stable-post-grokking",
        "Stable post-grokking",
        "transition",
    ],
)
def test_invalid_phase_labels_rejected(
    stage3: Stage3AvailabilityIndex,
    phase: str,
) -> None:
    with pytest.raises(
        ConditionIdentityError,
        match="invalid phase",
    ):
        build_condition_id(
            ConditionIdentity(
                teacher_seed=1,
                phase=phase,
            ),
            stage3,
        )


@pytest.mark.parametrize(
    "condition",
    [
        "hard",
        "soft",
        "hard-target",
        "soft-target",
        "teacher",
        "DIRECT_TEACHER",
    ],
)
def test_invalid_or_alias_condition_labels_rejected(
    stage3: Stage3AvailabilityIndex,
    condition: str,
) -> None:
    with pytest.raises(
        ConditionIdentityError,
        match="invalid distillation_condition",
    ):
        build_condition_id(
            ConditionIdentity(
                teacher_seed=1,
                phase="stable post-grokking",
                distillation_condition=condition,
            ),
            stage3,
        )


def test_raw_delimiter_cannot_alias_encoded_value(
    stage3: Stage3AvailabilityIndex,
) -> None:
    canonical = (
        "cfdid:v1:d8|teacher_seed=1|"
        "phase=stable%20post-grokking|"
        "distillation_condition=hard_target|"
        "student_initialization=0|"
        "discovery_method=synthetic-method%2Fv1|"
        "fidelity_setting=synthetic-fidelity%2Fv1|"
        "component_cap=synthetic-cap%2Fv1|"
        "overlap_setting=synthetic-overlap%2Fv1"
    )
    bad = canonical.replace(
        "synthetic-method%2Fv1",
        "synthetic|method%2Fv1",
    )
    with pytest.raises(
        ConditionIdentityError,
        match=r"requires 8 fields",
    ):
        parse_condition_id(bad, stage3)


def test_raw_equals_cannot_alias_encoded_value(
    stage3: Stage3AvailabilityIndex,
) -> None:
    bad = (
        "cfdid:v1:d5|teacher_seed=1|"
        "phase=stable%20post-grokking|"
        "distillation_condition=hard_target|"
        "student_initialization=0|"
        "discovery_method=synthetic%3Dmethod%2Fv1"
    )
    with pytest.raises(
        ConditionIdentityError,
        match="version-reference grammar",
    ):
        parse_condition_id(bad, stage3)


def test_nfd_unicode_builder_input_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    # Deliberately decomposed Unicode; it is not a valid canonical phase
    # anyway, but the canonical serializer must not normalize silently.
    nfd = unicodedata.normalize("NFD", "é")
    assert unicodedata.normalize("NFC", nfd) != nfd

    with pytest.raises(
        ConditionIdentityError,
        match="invalid phase",
    ):
        build_condition_id(
            ConditionIdentity(
                teacher_seed=1,
                phase=nfd,
            ),
            stage3,
        )


def test_noncanonical_unicode_percent_encoding_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    # U+0065 + U+0301 is NFD "é". Parser must not silently NFC-normalize it.
    bad = (
        "cfdid:v1:d2|teacher_seed=1|"
        "phase=e%CC%81"
    )
    with pytest.raises(ConditionIdentityError):
        parse_condition_id(bad, stage3)


def test_lowercase_percent_escape_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    bad = (
        "cfdid:v1:d2|teacher_seed=1|"
        "phase=stable%20post%2dgrokking"
    )
    with pytest.raises(
        ConditionIdentityError,
        match="uppercase hexadecimal",
    ):
        parse_condition_id(bad, stage3)


def test_unnecessary_percent_escape_rejected_as_noncanonical(
    stage3: Stage3AvailabilityIndex,
) -> None:
    bad = (
        "cfdid:v1:d2|teacher_seed=1|"
        "phase=%73table%20post-grokking"
    )
    with pytest.raises(
        ConditionIdentityError,
        match="canonical percent-encoded form",
    ):
        parse_condition_id(bad, stage3)


def test_reordered_fields_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    bad = (
        "cfdid:v1:d3|"
        "phase=stable%20post-grokking|"
        "teacher_seed=1|"
        "distillation_condition=hard_target"
    )
    with pytest.raises(
        ConditionIdentityError,
        match="canonical order",
    ):
        parse_condition_id(bad, stage3)


def test_duplicate_field_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    bad = (
        "cfdid:v1:d3|teacher_seed=1|"
        "phase=stable%20post-grokking|"
        "phase=stable%20post-grokking"
    )
    with pytest.raises(
        ConditionIdentityError,
        match="canonical order",
    ):
        parse_condition_id(bad, stage3)


def test_hard_and_soft_never_share_condition_id(
    stage3: Stage3AvailabilityIndex,
) -> None:
    hard = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition="hard_target",
        ),
        stage3,
    )
    soft = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition="soft_target",
        ),
        stage3,
    )
    assert hard != soft


def test_direct_teacher_student_misuse_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    with pytest.raises(
        ConditionIdentityError,
        match="direct_teacher cannot have student_initialization",
    ):
        build_condition_id(
            ConditionIdentity(
                teacher_seed=1,
                phase="stable post-grokking",
                distillation_condition="direct_teacher",
                student_initialization=0,
            ),
            stage3,
        )


@pytest.mark.parametrize(
    "phase",
    ["pre-grokking", "50%"],
)
@pytest.mark.parametrize(
    "condition",
    ["direct_teacher", "hard_target", "soft_target"],
)
def test_unavailable_seed0_cells_cannot_spawn_conditions(
    stage3: Stage3AvailabilityIndex,
    phase: str,
    condition: str,
) -> None:
    with pytest.raises(
        ConditionIdentityError,
        match="unavailable Stage 3 cell",
    ):
        build_condition_id(
            ConditionIdentity(
                teacher_seed=0,
                phase=phase,
                distillation_condition=condition,
            ),
            stage3,
        )


def test_negative_student_initialization_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    with pytest.raises(
        ConditionIdentityError,
        match="student_initialization must be a non-negative integer",
    ):
        build_condition_id(
            ConditionIdentity(
                teacher_seed=1,
                phase="stable post-grokking",
                distillation_condition="hard_target",
                student_initialization=-1,
            ),
            stage3,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("discovery_method", "1"),
        ("discovery_method", "method"),
        ("fidelity_setting", "0.99"),
        ("fidelity_setting", "99%"),
        ("component_cap", "258"),
        ("component_cap", "0"),
        ("overlap_setting", "0.50"),
        ("overlap_setting", "50%"),
    ],
)
def test_unresolved_numeric_or_unversioned_settings_rejected(
    stage3: Stage3AvailabilityIndex,
    field: str,
    value: str,
) -> None:
    kwargs = {
        "teacher_seed": 1,
        "phase": "stable post-grokking",
        "distillation_condition": "hard_target",
        "student_initialization": 0,
        "discovery_method": "synthetic-method/v1",
        "fidelity_setting": "synthetic-fidelity/v1",
        "component_cap": "synthetic-cap/v1",
        "overlap_setting": "synthetic-overlap/v1",
    }
    kwargs[field] = value

    with pytest.raises(
        ConditionIdentityError,
        match="version-reference grammar",
    ):
        build_condition_id(
            ConditionIdentity(**kwargs),
            stage3,
        )


@pytest.mark.parametrize(
    "bad",
    [
        (
            "cfdid:v1:d2|teacher_seed=01|"
            "phase=stable%20post-grokking"
        ),
        (
            "cfdid:v1:d2|teacher_seed=+1|"
            "phase=stable%20post-grokking"
        ),
        (
            "cfdid:v1:d2|teacher_seed=-1|"
            "phase=stable%20post-grokking"
        ),
    ],
)
def test_noncanonical_integer_encodings_rejected(
    stage3: Stage3AvailabilityIndex,
    bad: str,
) -> None:
    with pytest.raises(
        ConditionIdentityError,
        match="canonical unsigned base-10 encoding",
    ):
        parse_condition_id(bad, stage3)


def test_noncontiguous_prefix_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    identity = ConditionIdentity(
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=None,
        discovery_method="synthetic-method/v1",
    )
    with pytest.raises(
        ConditionIdentityError,
        match="contiguous canonical prefix",
    ):
        build_condition_id(identity, stage3)


def test_roundtrip_mismatch_is_not_accepted(
    stage3: Stage3AvailabilityIndex,
) -> None:
    canonical = build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition="hard_target",
            student_initialization=0,
            discovery_method="synthetic-method/v1",
            fidelity_setting="synthetic-fidelity/v1",
            component_cap="synthetic-cap/v1",
            overlap_setting="synthetic-overlap/v1",
        ),
        stage3,
    )

    noncanonical = canonical.replace(
        "synthetic-method%2Fv1",
        "synthetic-method%2fv1",
    )

    assert noncanonical != canonical

    with pytest.raises(
        ConditionIdentityError,
        match="uppercase hexadecimal",
    ):
        parse_condition_id(noncanonical, stage3)

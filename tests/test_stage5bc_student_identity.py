from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import (
    Stage3AvailabilityIndex,
    parse_condition_id,
)
from circuit_families.stage4_seed_derivation import SeedEvidence
from circuit_families.stage5bc.student_identity import (
    StudentIdentityError,
    build_student_attempt_identity,
    build_student_condition_id,
    verify_student_attempt_identity,
)

REGISTRY = Path(
    "followup/manifests/stage3_teacher_registry_v1.json"
)


@pytest.fixture(scope="module")
def stage3() -> Stage3AvailabilityIndex:
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return Stage3AvailabilityIndex.from_registry(raw)


def _baseline(stage3: Stage3AvailabilityIndex):
    return build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=0,
        attempt_index=0,
        retry_index=0,
    )


def test_student_condition_is_exact_stage4_depth4_identity(
    stage3: Stage3AvailabilityIndex,
) -> None:
    condition_id = build_student_condition_id(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=7,
    )

    assert condition_id == (
        "cfdid:v1:d4|teacher_seed=1|"
        "phase=stable%20post-grokking|"
        "distillation_condition=hard_target|"
        "student_initialization=7"
    )

    parsed = parse_condition_id(condition_id, stage3)

    assert parsed.depth == 4
    assert parsed.teacher_seed == 1
    assert parsed.phase == "stable post-grokking"
    assert parsed.distillation_condition == "hard_target"
    assert parsed.student_initialization == 7


def test_attempt_retry_are_not_condition_id_coordinates(
    stage3: Stage3AvailabilityIndex,
) -> None:
    first = _baseline(stage3)
    second = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=0,
        attempt_index=4,
        retry_index=3,
    )

    assert first.condition_id == second.condition_id
    assert first.attempt_index != second.attempt_index
    assert first.retry_index != second.retry_index
    assert (
        first.training_seed.seed_material
        != second.training_seed.seed_material
    )


def test_training_and_tie_breaking_seed_evidence_is_stored(
    stage3: Stage3AvailabilityIndex,
) -> None:
    identity = _baseline(stage3)
    mapping = identity.to_mapping()

    assert mapping["attempt_index"] == 0
    assert mapping["retry_index"] == 0

    for field in ("training_seed", "tie_breaking_seed"):
        evidence = mapping[field]
        assert evidence["seed_derivation_version"] == "seed-derivation/v1"
        assert len(evidence["digest_sha256"]) == 64
        assert len(evidence["selected_bytes_hex"]) == 16
        assert isinstance(evidence["seed_value"], int)

    assert (
        mapping["training_seed"]["seed_material"]
        != mapping["tie_breaking_seed"]["seed_material"]
    )


def test_complete_attempt_identity_verifies(
    stage3: Stage3AvailabilityIndex,
) -> None:
    identity = _baseline(stage3)

    parsed = verify_student_attempt_identity(identity, stage3)

    assert parsed.depth == 4
    assert parsed.student_initialization == 0


def test_wrong_seed_evidence_is_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    identity = _baseline(stage3)

    bad_training = dataclasses.replace(
        identity.training_seed,
        seed_value=identity.training_seed.seed_value ^ 1,
    )
    bad_identity = dataclasses.replace(
        identity,
        training_seed=bad_training,
    )

    with pytest.raises(
        StudentIdentityError,
        match="seed evidence failed verification",
    ):
        verify_student_attempt_identity(
            bad_identity,
            stage3,
        )


@pytest.mark.parametrize(
    "change",
    [
        "teacher_seed",
        "phase",
        "distillation_condition",
        "student_initialization",
        "attempt_index",
        "retry_index",
    ],
)
def test_seed_material_is_sensitive_to_every_identity_coordinate(
    stage3: Stage3AvailabilityIndex,
    change: str,
) -> None:
    base_kwargs = {
        "stage3": stage3,
        "teacher_seed": 1,
        "phase": "stable post-grokking",
        "distillation_condition": "hard_target",
        "student_initialization": 0,
        "attempt_index": 0,
        "retry_index": 0,
    }

    changed_kwargs = dict(base_kwargs)

    replacements = {
        "teacher_seed": 2,
        "phase": "pre-grokking",
        "distillation_condition": "soft_target",
        "student_initialization": 1,
        "attempt_index": 1,
        "retry_index": 1,
    }
    changed_kwargs[change] = replacements[change]

    base = build_student_attempt_identity(**base_kwargs)
    changed = build_student_attempt_identity(**changed_kwargs)

    assert (
        base.training_seed.seed_material
        != changed.training_seed.seed_material
    )
    assert (
        base.training_seed.digest_sha256
        != changed.training_seed.digest_sha256
    )
    assert (
        base.tie_breaking_seed.seed_material
        != changed.tie_breaking_seed.seed_material
    )
    assert (
        base.tie_breaking_seed.digest_sha256
        != changed.tie_breaking_seed.digest_sha256
    )


def test_seed_purpose_is_explicit_and_sensitive(
    stage3: Stage3AvailabilityIndex,
) -> None:
    identity = _baseline(stage3)

    assert "purpose=training\n" in identity.training_seed.seed_material
    assert (
        "purpose=tie_breaking\n"
        in identity.tie_breaking_seed.seed_material
    )
    assert (
        identity.training_seed.digest_sha256
        != identity.tie_breaking_seed.digest_sha256
    )


def test_hard_soft_student_condition_ids_do_not_collide(
    stage3: Stage3AvailabilityIndex,
) -> None:
    hard = build_student_condition_id(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=0,
    )
    soft = build_student_condition_id(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="soft_target",
        student_initialization=0,
    )

    assert hard != soft


def test_unavailable_teacher_phase_cannot_form_student_identity(
    stage3: Stage3AvailabilityIndex,
) -> None:
    with pytest.raises(
        StudentIdentityError,
        match="unavailable Stage 3 cell",
    ):
        build_student_attempt_identity(
            stage3=stage3,
            teacher_seed=0,
            phase="pre-grokking",
            distillation_condition="hard_target",
            student_initialization=0,
            attempt_index=0,
            retry_index=0,
        )


def test_cross_process_seed_evidence_ignores_pythonhashseed() -> None:
    script = r'''
import json
from pathlib import Path
from circuit_families.stage4_condition_identity import Stage3AvailabilityIndex
from circuit_families.stage5bc.student_identity import build_student_attempt_identity

registry = json.loads(
    Path("followup/manifests/stage3_teacher_registry_v1.json")
    .read_text(encoding="utf-8")
)
stage3 = Stage3AvailabilityIndex.from_registry(registry)

identity = build_student_attempt_identity(
    stage3=stage3,
    teacher_seed=2,
    phase="50%",
    distillation_condition="soft_target",
    student_initialization=3,
    attempt_index=4,
    retry_index=5,
)

print(json.dumps(identity.to_mapping(), sort_keys=True))
'''

    outputs = []

    for hash_seed in ("1", "987654"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed

        result = subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=Path.cwd(),
            env=env,
            text=True,
        )
        outputs.append(result)

    assert outputs[0] == outputs[1]


def test_seed_objects_are_frozen_evidence_records(
    stage3: Stage3AvailabilityIndex,
) -> None:
    identity = _baseline(stage3)

    assert isinstance(identity.training_seed, SeedEvidence)
    assert isinstance(identity.tie_breaking_seed, SeedEvidence)

    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.attempt_index = 99

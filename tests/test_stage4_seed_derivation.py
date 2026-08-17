from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage4_seed_derivation import (
    SEED_MAX,
    SeedDerivationError,
    SeedInputs,
    derive_seed,
    parse_seed_material,
    verify_seed_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
SEED_SPEC = ROOT / "followup/configs/stage4_seed_derivation_spec_v1.json"
SEED_MODULE = ROOT / "src/circuit_families/stage4_seed_derivation.py"


@pytest.fixture(scope="module")
def stage3() -> Stage3AvailabilityIndex:
    return Stage3AvailabilityIndex.from_registry(
        json.loads(REGISTRY.read_text())
    )


def hard_student_id(
    stage3: Stage3AvailabilityIndex,
    *,
    teacher_seed: int = 1,
    phase: str = "stable post-grokking",
    condition: str = "hard_target",
    initialization: int = 0,
) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=teacher_seed,
            phase=phase,
            distillation_condition=condition,
            student_initialization=initialization,
        ),
        stage3,
    )


def discovery_id(
    stage3: Stage3AvailabilityIndex,
    *,
    teacher_seed: int = 1,
    phase: str = "stable post-grokking",
    condition: str = "hard_target",
    initialization: int = 0,
    method: str = "synthetic-method-a/v1",
    fidelity: str = "synthetic-fidelity-a/v1",
    cap: str = "synthetic-cap-a/v1",
    overlap: str = "synthetic-overlap-a/v1",
) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=teacher_seed,
            phase=phase,
            distillation_condition=condition,
            student_initialization=initialization,
            discovery_method=method,
            fidelity_setting=fidelity,
            component_cap=cap,
            overlap_setting=overlap,
        ),
        stage3,
    )


def test_all_part_j_known_vectors(
    stage3: Stage3AvailabilityIndex,
) -> None:
    spec = json.loads(SEED_SPEC.read_text())

    for name, expected in spec["known_test_vectors"].items():
        actual = derive_seed(
            SeedInputs(
                condition_id=expected["condition_id"],
                purpose=expected["purpose"],
                attempt_index=expected["attempt_index"],
                retry_index=expected["retry_index"],
            ),
            stage3,
        )

        assert actual.seed_derivation_version == "seed-derivation/v1", name
        assert actual.seed_material == expected["seed_material"], name
        assert actual.digest_sha256 == expected["digest_sha256"], name
        assert (
            actual.selected_bytes_hex
            == expected["selected_bytes_hex"]
        ), name
        assert actual.seed_value == expected["seed_value"], name

        recovered = verify_seed_evidence(actual, stage3)
        assert recovered.condition_id == expected["condition_id"]
        assert recovered.purpose == expected["purpose"]
        assert recovered.attempt_index == expected["attempt_index"]
        assert recovered.retry_index == expected["retry_index"]


def test_seed_range(stage3: Stage3AvailabilityIndex) -> None:
    evidence = derive_seed(
        SeedInputs(
            condition_id=hard_student_id(stage3),
            purpose="training",
            attempt_index=0,
            retry_index=0,
        ),
        stage3,
    )
    assert 0 <= evidence.seed_value <= SEED_MAX


def test_purpose_separation(stage3: Stage3AvailabilityIndex) -> None:
    condition_id = hard_student_id(stage3)

    training = derive_seed(
        SeedInputs(condition_id, "training", 0, 0),
        stage3,
    )
    tie = derive_seed(
        SeedInputs(condition_id, "tie_breaking", 0, 0),
        stage3,
    )

    assert training.seed_value != tie.seed_value
    assert training.seed_material != tie.seed_material


def test_attempt_and_retry_separation(
    stage3: Stage3AvailabilityIndex,
) -> None:
    condition_id = hard_student_id(stage3)

    base = derive_seed(
        SeedInputs(condition_id, "training", 0, 0),
        stage3,
    )
    changed_attempt = derive_seed(
        SeedInputs(condition_id, "training", 1, 0),
        stage3,
    )
    changed_retry = derive_seed(
        SeedInputs(condition_id, "training", 0, 1),
        stage3,
    )

    assert len(
        {
            base.seed_value,
            changed_attempt.seed_value,
            changed_retry.seed_value,
        }
    ) == 3


def test_every_identity_component_changes_discovery_seed(
    stage3: Stage3AvailabilityIndex,
) -> None:
    base_id = discovery_id(stage3)
    variants = {
        "teacher_seed": discovery_id(stage3, teacher_seed=2),
        "phase": discovery_id(stage3, phase="pre-grokking"),
        "distillation_condition": discovery_id(
            stage3,
            condition="soft_target",
        ),
        "student_initialization": discovery_id(
            stage3,
            initialization=1,
        ),
        "discovery_method": discovery_id(
            stage3,
            method="synthetic-method-b/v1",
        ),
        "fidelity_setting": discovery_id(
            stage3,
            fidelity="synthetic-fidelity-b/v1",
        ),
        "component_cap": discovery_id(
            stage3,
            cap="synthetic-cap-b/v1",
        ),
        "overlap_setting": discovery_id(
            stage3,
            overlap="synthetic-overlap-b/v1",
        ),
    }

    base_seed = derive_seed(
        SeedInputs(base_id, "discovery", 0, 0),
        stage3,
    ).seed_value

    variant_seeds = {}

    for dimension, condition_id in variants.items():
        assert condition_id != base_id, dimension
        seed = derive_seed(
            SeedInputs(condition_id, "discovery", 0, 0),
            stage3,
        ).seed_value
        assert seed != base_seed, dimension
        variant_seeds[dimension] = seed

    # Deterministic collision check for this prescribed test set.
    assert len(set(variant_seeds.values())) == len(variant_seeds)


@pytest.mark.parametrize(
    ("purpose", "depth_kind"),
    [
        ("training", "depth8"),
        ("tie_breaking", "depth8"),
        ("discovery", "depth4"),
    ],
)
def test_purpose_requires_exact_identity_depth(
    stage3: Stage3AvailabilityIndex,
    purpose: str,
    depth_kind: str,
) -> None:
    condition_id = (
        discovery_id(stage3)
        if depth_kind == "depth8"
        else hard_student_id(stage3)
    )

    expected = 4 if purpose in {"training", "tie_breaking"} else 8

    with pytest.raises(
        SeedDerivationError,
        match=rf"requires identity depth {expected}",
    ):
        derive_seed(
            SeedInputs(condition_id, purpose, 0, 0),
            stage3,
        )


def test_unknown_purpose_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    with pytest.raises(
        SeedDerivationError,
        match="unsupported seed purpose",
    ):
        derive_seed(
            SeedInputs(
                hard_student_id(stage3),
                "trainer",
                0,
                0,
            ),
            stage3,
        )


@pytest.mark.parametrize(
    ("attempt", "retry"),
    [
        (-1, 0),
        (0, -1),
        (True, 0),
        (0, False),
    ],
)
def test_invalid_attempt_retry_rejected(
    stage3: Stage3AvailabilityIndex,
    attempt,
    retry,
) -> None:
    with pytest.raises(
        SeedDerivationError,
        match="non-negative integer",
    ):
        derive_seed(
            SeedInputs(
                hard_student_id(stage3),
                "training",
                attempt,
                retry,
            ),
            stage3,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace(
            "namespace=circuit-families-distillation",
            "namespace=other",
        ),
        lambda text: text.replace(
            "seed_derivation_version=seed-derivation/v1",
            "seed_derivation_version=seed-derivation/v2",
        ),
        lambda text: text.replace(
            "purpose=training",
            "purpose=tie_breaking",
        ).replace(
            "tie_breaking\nattempt_index",
            "tie_breaking\nretry_index",
        ),
        lambda text: text.removesuffix("\n"),
        lambda text: text.replace(
            "attempt_index=0",
            "attempt_index=00",
        ),
    ],
)
def test_corrupted_seed_material_rejected(
    stage3: Stage3AvailabilityIndex,
    mutator,
) -> None:
    evidence = derive_seed(
        SeedInputs(
            hard_student_id(stage3),
            "training",
            0,
            0,
        ),
        stage3,
    )
    corrupted = mutator(evidence.seed_material)

    with pytest.raises(SeedDerivationError):
        parse_seed_material(corrupted, stage3)


def test_stored_seed_integer_mismatch_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    evidence = derive_seed(
        SeedInputs(
            hard_student_id(stage3),
            "training",
            0,
            0,
        ),
        stage3,
    )

    corrupted = replace(
        evidence,
        seed_value=(evidence.seed_value + 1) & SEED_MAX,
    )

    with pytest.raises(
        SeedDerivationError,
        match="stored seed_value does not match",
    ):
        verify_seed_evidence(corrupted, stage3)


def test_stored_digest_mismatch_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    evidence = derive_seed(
        SeedInputs(
            hard_student_id(stage3),
            "training",
            0,
            0,
        ),
        stage3,
    )

    replacement_digest = (
        ("0" if evidence.digest_sha256[0] != "0" else "1")
        + evidence.digest_sha256[1:]
    )

    corrupted = replace(
        evidence,
        digest_sha256=replacement_digest,
    )

    with pytest.raises(
        SeedDerivationError,
        match="stored digest_sha256 does not match",
    ):
        verify_seed_evidence(corrupted, stage3)


def test_stored_selected_bytes_mismatch_rejected(
    stage3: Stage3AvailabilityIndex,
) -> None:
    evidence = derive_seed(
        SeedInputs(
            hard_student_id(stage3),
            "training",
            0,
            0,
        ),
        stage3,
    )

    replacement = (
        ("0" if evidence.selected_bytes_hex[0] != "0" else "1")
        + evidence.selected_bytes_hex[1:]
    )

    corrupted = replace(
        evidence,
        selected_bytes_hex=replacement,
    )

    with pytest.raises(
        SeedDerivationError,
        match="stored selected_bytes_hex does not match",
    ):
        verify_seed_evidence(corrupted, stage3)


def test_cross_process_path_username_and_hashseed_stability(
    stage3: Stage3AvailabilityIndex,
    tmp_path: Path,
) -> None:
    condition_id = discovery_id(stage3)

    script = f"""
import json
from pathlib import Path
from circuit_families.stage4_condition_identity import Stage3AvailabilityIndex
from circuit_families.stage4_seed_derivation import SeedInputs, derive_seed

registry = json.loads(Path({str(REGISTRY)!r}).read_text())
stage3 = Stage3AvailabilityIndex.from_registry(registry)
evidence = derive_seed(
    SeedInputs(
        condition_id={condition_id!r},
        purpose="discovery",
        attempt_index=0,
        retry_index=0,
    ),
    stage3,
)
print(evidence.seed_value)
print(evidence.digest_sha256)
"""

    outputs = []

    for index, hashseed in enumerate(("1", "999999")):
        cwd = tmp_path / f"machine-{index}"
        cwd.mkdir()

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hashseed
        env["USER"] = f"collaborator{index}"
        env["LOGNAME"] = f"collaborator{index}"
        env["HOME"] = str(tmp_path / f"home-{index}")

        src = str(ROOT / "src")
        current_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            src
            if not current_pythonpath
            else src + os.pathsep + current_pythonpath
        )

        output = subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=cwd,
            env=env,
            text=True,
        ).strip()

        outputs.append(output)

    assert outputs[0] == outputs[1]


def test_no_executable_python_builtin_hash_call() -> None:
    tree = ast.parse(SEED_MODULE.read_text())

    calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hash"
        )
    ]

    assert calls == []

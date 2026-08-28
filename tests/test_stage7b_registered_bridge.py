from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from circuit_families.stage7b.registered_fixture import (
    RegisteredFixtureBindings,
    RegisteredFixtureError,
    canonical_modular_addition_domain,
    centred_logits,
    load_registered_fixture_request,
    validate_registered_fixture_identity,
)

REPO = Path(__file__).resolve().parents[1]
REQUEST_PATH = REPO / "followup/configs/stage7b/registered_fixture_request_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_request_load_remains_scientifically_closed() -> None:
    r = load_registered_fixture_request(REQUEST_PATH)
    assert r["technical_only"] is True
    assert r["scientific_data"] is False
    assert r["production_default"] is False
    assert r["ud_resolution"] is False
    assert r["stage8_execution"] is False
    assert r["endpoints"]["print_endpoint_values"] is False
    assert r["endpoints"]["interpret_endpoint_values"] is False


def test_canonical_domain_is_exact_12769_lexicographic_inputs() -> None:
    x = canonical_modular_addition_domain()
    assert x.shape == (12769, 3)
    assert x.dtype == np.int64
    assert tuple(x[0]) == (0, 0, 113)
    assert tuple(x[1]) == (0, 1, 113)
    assert tuple(x[112]) == (0, 112, 113)
    assert tuple(x[113]) == (1, 0, 113)
    assert tuple(x[-1]) == (112, 112, 113)

    pairs = x[:, :2]
    expected = np.array(
        [(a, b) for a in range(113) for b in range(113)],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(pairs, expected)


def test_centred_logits_remove_per_input_additive_gauge() -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(31, 113))
    shift = rng.normal(size=(31, 1))

    a = centred_logits(logits)
    b = centred_logits(logits + shift)

    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(a.mean(axis=1), 0.0, rtol=0.0, atol=1e-15)


def test_physical_identity_validation_uses_registry_manifest_and_checkpoint_hash(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    pred = tmp_path / "pred"
    registry_dir = repo / "followup/manifests"
    registry_dir.mkdir(parents=True)
    (pred / "manifests").mkdir(parents=True)
    ckpt_rel = (
        Path("checkpoints")
        / "stage18-main-training-s0-58b8c1235464"
        / "step_00005900.pt"
    )
    (pred / ckpt_rel).parent.mkdir(parents=True)

    manifest = pred / "manifests/training_stage18-main-training-s0-58b8c1235464.json"
    manifest.write_text(json.dumps({"seed": 0, "max_steps": 40000}) + "\n")
    ckpt = pred / ckpt_rel
    ckpt.write_bytes(b"temporary-stage7b-checkpoint")

    manifest_sha = _sha(manifest)
    ckpt_sha = _sha(ckpt)

    registry = {
        "records": [
            {
                "canonical_run_id": "stage18-main-training-s0-58b8c1235464",
                "teacher_seed": 0,
                "phase_label": "stable post-grokking",
                "training_step": 5900,
                "checkpoint_path": str(ckpt_rel),
                "checkpoint_sha256": ckpt_sha,
                "training_manifest_path": str(manifest.relative_to(pred)),
                "training_manifest_sha256": manifest_sha,
            }
        ]
    }
    (registry_dir / "stage3_teacher_registry_v1.json").write_text(
        json.dumps(registry) + "\n"
    )

    original = json.loads(REQUEST_PATH.read_text())
    request = json.loads(json.dumps(original))
    request["registered_teacher"]["checkpoint_sha256"] = ckpt_sha

    identity, checkpoint_path, manifest_path, record = (
        validate_registered_fixture_identity(
            repository_root=repo,
            predecessor_root=pred,
            request=request,
        )
    )

    assert identity.teacher_seed == 0
    assert checkpoint_path == ckpt
    assert manifest_path == manifest
    assert record["training_step"] == 5900


def test_wrong_checkpoint_hash_is_rejected_before_model_binding(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    pred = tmp_path / "pred"
    registry_dir = repo / "followup/manifests"
    registry_dir.mkdir(parents=True)
    (pred / "manifests").mkdir(parents=True)
    ckpt_rel = (
        Path("checkpoints")
        / "stage18-main-training-s0-58b8c1235464"
        / "step_00005900.pt"
    )
    (pred / ckpt_rel).parent.mkdir(parents=True)

    manifest = pred / "manifests/training_stage18-main-training-s0-58b8c1235464.json"
    manifest.write_text("{}\n")
    ckpt = pred / ckpt_rel
    ckpt.write_bytes(b"actual")
    actual_sha = _sha(ckpt)

    request = json.loads(REQUEST_PATH.read_text())
    request["registered_teacher"]["checkpoint_sha256"] = actual_sha

    registry = {
        "records": [
            {
                "canonical_run_id": "stage18-main-training-s0-58b8c1235464",
                "teacher_seed": 0,
                "phase_label": "stable post-grokking",
                "training_step": 5900,
                "checkpoint_path": str(ckpt_rel),
                "checkpoint_sha256": actual_sha,
                "training_manifest_path": str(manifest.relative_to(pred)),
                "training_manifest_sha256": _sha(manifest),
            }
        ]
    }
    (registry_dir / "stage3_teacher_registry_v1.json").write_text(
        json.dumps(registry) + "\n"
    )

    ckpt.write_bytes(b"tampered")

    with pytest.raises(RegisteredFixtureError, match="checkpoint hash mismatch"):
        validate_registered_fixture_identity(
            repository_root=repo,
            predecessor_root=pred,
            request=request,
        )


def test_binding_contract_has_exactly_two_adapter_names() -> None:
    def nope(**_: Any) -> Any:
        raise AssertionError("must not execute")

    bindings = RegisteredFixtureBindings(
        restore_model=nope,
        evaluate_teacher=nope,
        build_target_caches=nope,
        run_student_attempt=nope,
        assess_student_attempt=nope,
        run_discovery=nope,
        run_exact_endpoints=nope,
        build_excluded_outputs=nope,
        load_checkpoint_payload=nope,
        discovery_adapter_names=("greedy_deletion", "diversity_forced"),
    )
    assert bindings.discovery_adapter_names == (
        "greedy_deletion",
        "diversity_forced",
    )


def test_runner_is_cwd_independent_and_has_no_private_absolute_path() -> None:
    runner = REPO / "scripts/run_stage7b_registered_fixture.py"
    text = runner.read_text()
    assert "Path(__file__).resolve().parents[1]" in text
    assert "--predecessor-root" in text
    assert "--output-root" in text
    assert "/Users/" not in text
    assert "/private/" not in text


def test_bridge_defines_no_second_optimizer_or_discovery_algorithm() -> None:
    bridge = REPO / "src/circuit_families/stage7b/registered_fixture.py"
    text = bridge.read_text()

    assert "torch.optim." not in text
    assert "optimizer.step(" not in text
    assert "GreedyDeletionAdapter" in text
    assert "DiversityForcedAdapter" in text
    assert "Stage6AExactEvaluationBridge" in text
    assert "reduce_endpoint1" in text
    assert "recompute_endpoint2" in text
    assert "invoke_accepted" not in text


def test_part_c_source_does_not_embed_private_checkpoint_or_endpoint_values() -> None:
    for rel in (
        "src/circuit_families/stage7b/__init__.py",
        "src/circuit_families/stage7b/registered_fixture.py",
        "scripts/run_stage7b_registered_fixture.py",
    ):
        text = (REPO / rel).read_text()
        assert "/Users/" not in text
        assert "/private/" not in text
        assert "endpoint1_value" not in text
        assert "endpoint2_value" not in text


def test_production_student_path_uses_shared_lifecycle_and_registered_components() -> None:
    import inspect

    from circuit_families.stage7b import registered_fixture

    source = inspect.getsource(
        registered_fixture._build_accepted_bindings
    )

    assert "invoke_accepted(PreparedTrainer" not in source
    assert "TrainerLifecycle(" in source
    assert "build_student_attempt_identity(" in source
    assert "build_transformer(" in source
    assert "build_optimizer(" in source
    assert "HardLabelLossAdapter()" in source
    assert "GaugeInvariantSoftLossAdapter(" in source
    assert "TechnicalSoftTargetAdapter(" in source
    assert "outputs=outputs[:, -1, :]" in source
    assert (
        "technical_safety_step_limit=("
        in source
    )
    assert 'device_name != "mps"' in source
    assert "assess_hard_attempt(" in source
    assert "assess_soft_attempt(" in source
    assert "evaluate_hard_target_eligibility(" in source
    assert "evaluate_soft_target_eligibility(" in source
    assert "save_technical_resume_checkpoint(" in source
    assert "emit_technical_attempt_record(" in source

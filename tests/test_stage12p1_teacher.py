"""Focused/adversarial Stage 12-P1 teacher mechanics tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from circuit_families.stage12p1.tasks import (
    TASK_CONFIG_SCHEMA_VERSION,
    ModularAdditionImplementation,
    build_task_record,
)
from circuit_families.stage12p1.teacher import (
    TeacherProtocolError,
    TeacherTrainingAdapter,
    TeacherTrainingRequest,
    build_technical_tabular_teacher_data,
    emit_unavailable_teacher,
    validate_teacher_artifact,
)


def _task() -> dict:
    impl = ModularAdditionImplementation()
    return build_task_record(
        {
            "schema_version": TASK_CONFIG_SCHEMA_VERSION,
            "task_id": "technical-teacher-addition-m5",
            "implementation": impl.name,
            "implementation_version": impl.version,
            "modulus": 5,
            "input_domains": [list(range(5)), list(range(5))],
            "parameters": {},
            "split_identity": {
                "kind": "technical-teacher-split",
                "version": "v1",
                "train_indices": list(range(0, 20)),
                "test_indices": list(range(20, 25)),
            },
            "architecture_compatibility": {
                "input_arity": 2,
                "output_class_count": 5,
                "classification": "technical-fixture-only",
            },
            "scientific_data": False,
            "production_eligible": False,
        }
    )


def _request(
    tmp_path: Path,
    *,
    resume_id: str,
    stop_step: int = 4,
    checkpoint_interval: int = 1,
    checkpoint_retention: int = 2,
) -> TeacherTrainingRequest:
    return TeacherTrainingRequest(
        task_record=_task(),
        architecture_config={
            "kind": "tiny-linear-technical-fixture",
            "input_dim": 2,
            "output_dim": 5,
        },
        training_config={
            "loss": "cross_entropy",
            "classification": "technical-fixture-only",
        },
        optimizer_config={
            "kind": "sgd",
            "learning_rate": 0.05,
        },
        scheduler_config={
            "kind": "step_lr",
            "step_size": 1,
            "gamma": 0.95,
        },
        stopping_config={
            "technical_stop_step": stop_step,
        },
        backend_qualification={
            "backend_id": "cpu-technical-fixture",
            "device": "cpu",
            "qualified": True,
            "exact_resume_supported": True,
            "qualification_ref": "technical-test-only/v1",
        },
        model_seed_id="technical-model-seed/v1",
        model_seed=17,
        training_seed_id="technical-training-seed/v1",
        training_seed=23,
        max_technical_updates=6,
        checkpoint_interval=checkpoint_interval,
        checkpoint_retention=checkpoint_retention,
        output_root=tmp_path,
        resume_id=resume_id,
    )


def _model_constructor(*, task_record, architecture_config, seed, device):
    del task_record
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(int(architecture_config["input_dim"]), 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, int(architecture_config["output_dim"])),
    ).to(device)


def _loss_fn(*, outputs, targets, config):
    assert config["loss"] == "cross_entropy"
    return F.cross_entropy(outputs, targets)


def _optimizer_factory(*, model, config):
    assert config["kind"] == "sgd"
    return torch.optim.SGD(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )


def _scheduler_factory(*, optimizer, config):
    assert config["kind"] == "step_lr"
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["step_size"]),
        gamma=float(config["gamma"]),
    )


def _stop_rule(*, step, metrics, config):
    assert set(metrics) == {
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
    }
    return step >= int(config["technical_stop_step"])


def _adapter() -> TeacherTrainingAdapter:
    return TeacherTrainingAdapter(
        model_constructor=_model_constructor,
        loss_fn=_loss_fn,
        optimizer_factory=_optimizer_factory,
        scheduler_factory=_scheduler_factory,
        stop_rule=_stop_rule,
    )


def _artifact(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return validate_teacher_artifact(value)


def test_technical_tabular_data_is_bound_to_task_hashes() -> None:
    task = _task()
    data = build_technical_tabular_teacher_data(task)

    assert data.train_inputs.shape == (20, 2)
    assert data.train_targets.shape == (20,)
    assert data.test_inputs.shape == (5, 2)
    assert data.test_targets.shape == (5,)
    assert data.task_identity_sha256 == task["hashes"]["task_identity_sha256"]
    assert data.dataset_sha256 == task["hashes"]["dataset_sha256"]
    assert data.split_identity_sha256 == task["hashes"]["split_identity_sha256"]


def test_uninterrupted_teacher_run_seals_compact_technical_artifact(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_id="uninterrupted")
    result = _adapter().run(
        request=request,
        data=build_technical_tabular_teacher_data(request.task_record),
    )

    assert result.status == "completed"
    assert result.updates_completed == 4
    assert len(result.checkpoint_paths) <= request.checkpoint_retention

    artifact = _artifact(result.artifact_path)
    assert artifact["scientific_data"] is False
    assert artifact["production_eligible"] is False
    assert artifact["status"] == "completed"
    assert artifact["reason"] == "injected_stop_rule"
    assert artifact["updates_completed"] == 4
    assert [item["step"] for item in artifact["metric_snapshots"]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert artifact["state_hashes"] is not None
    assert artifact["environment"]["sha256"]
    assert artifact["request_hashes"]["task_identity_sha256"] == (
        request.task_record["hashes"]["task_identity_sha256"]
    )
    assert artifact["seed_identity"] == {
        "model_seed_id": "technical-model-seed/v1",
        "model_seed": 17,
        "training_seed_id": "technical-training-seed/v1",
        "training_seed": 23,
    }


def test_checkpoint_retention_is_bounded(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        resume_id="retention",
        stop_step=5,
        checkpoint_interval=1,
        checkpoint_retention=2,
    )
    result = _adapter().run(
        request=request,
        data=build_technical_tabular_teacher_data(request.task_record),
    )

    names = [path.name for path in result.checkpoint_paths]
    assert names == ["step_00000004.pt", "step_00000005.pt"]
    assert not list((tmp_path / "retention").rglob("*.tmp"))


def test_interrupted_resume_matches_uninterrupted_exact_state_and_metrics(
    tmp_path: Path,
) -> None:
    baseline_request = _request(
        tmp_path,
        resume_id="baseline",
        stop_step=4,
        checkpoint_interval=1,
        checkpoint_retention=4,
    )
    baseline_data = build_technical_tabular_teacher_data(
        baseline_request.task_record
    )
    baseline = _adapter().run(
        request=baseline_request,
        data=baseline_data,
    )
    baseline_artifact = _artifact(baseline.artifact_path)

    resumed_request = _request(
        tmp_path,
        resume_id="resumed",
        stop_step=4,
        checkpoint_interval=1,
        checkpoint_retention=4,
    )
    resumed_data = build_technical_tabular_teacher_data(
        resumed_request.task_record
    )

    interrupted = _adapter().run(
        request=resumed_request,
        data=resumed_data,
        interrupt_after_updates=2,
    )
    assert interrupted.status == "interrupted"
    assert interrupted.updates_completed == 2

    resumed = _adapter().run(
        request=resumed_request,
        data=resumed_data,
    )
    assert resumed.status == "completed"
    assert resumed.updates_completed == 4
    assert resumed.resumed_from_checkpoint_sha256 is not None

    resumed_artifact = _artifact(resumed.artifact_path)

    assert (
        baseline_artifact["state_hashes"]["model_state_sha256"]
        == resumed_artifact["state_hashes"]["model_state_sha256"]
    )
    assert (
        baseline_artifact["state_hashes"]["optimizer_state_sha256"]
        == resumed_artifact["state_hashes"]["optimizer_state_sha256"]
    )
    assert (
        baseline_artifact["state_hashes"]["scheduler_state_sha256"]
        == resumed_artifact["state_hashes"]["scheduler_state_sha256"]
    )
    assert (
        baseline_artifact["state_hashes"]["metrics_sha256"]
        == resumed_artifact["state_hashes"]["metrics_sha256"]
    )
    assert (
        baseline_artifact["metric_snapshots"]
        == resumed_artifact["metric_snapshots"]
    )


def test_completed_resume_identity_is_not_executed_twice(tmp_path: Path) -> None:
    calls = {"models": 0}

    def constructor(*, task_record, architecture_config, seed, device):
        calls["models"] += 1
        return _model_constructor(
            task_record=task_record,
            architecture_config=architecture_config,
            seed=seed,
            device=device,
        )

    adapter = TeacherTrainingAdapter(
        model_constructor=constructor,
        loss_fn=_loss_fn,
        optimizer_factory=_optimizer_factory,
        scheduler_factory=_scheduler_factory,
        stop_rule=_stop_rule,
    )
    request = _request(tmp_path, resume_id="dedupe")
    data = build_technical_tabular_teacher_data(request.task_record)

    first = adapter.run(request=request, data=data)
    second = adapter.run(request=request, data=data)

    assert first.status == "completed"
    assert second.status == "completed"
    assert second.reused_terminal_artifact is True
    assert first.artifact_path == second.artifact_path
    assert calls["models"] == 1


def test_completed_resume_id_rejects_changed_request_hashes(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_id="completed-stale")
    data = build_technical_tabular_teacher_data(request.task_record)

    first = _adapter().run(request=request, data=data)
    assert first.status == "completed"

    kwargs = dict(request.__dict__)
    kwargs["optimizer_config"] = {
        "kind": "sgd",
        "learning_rate": 0.051,
    }
    changed = TeacherTrainingRequest(**kwargs)

    with pytest.raises(
        TeacherProtocolError,
        match="terminal artifact request hashes are stale",
    ):
        _adapter().run(request=changed, data=data)



def test_numerical_failure_is_explicit_and_sealed(tmp_path: Path) -> None:
    def bad_loss(*, outputs, targets, config):
        del targets, config
        return outputs.sum() * torch.tensor(float("nan"))

    adapter = TeacherTrainingAdapter(
        model_constructor=_model_constructor,
        loss_fn=bad_loss,
        optimizer_factory=_optimizer_factory,
        scheduler_factory=_scheduler_factory,
        stop_rule=_stop_rule,
    )
    request = _request(tmp_path, resume_id="nonfinite")
    result = adapter.run(
        request=request,
        data=build_technical_tabular_teacher_data(request.task_record),
    )

    assert result.status == "numerical-failure"
    artifact = _artifact(result.artifact_path)
    assert artifact["status"] == "numerical-failure"
    assert artifact["scientific_data"] is False
    assert artifact["production_eligible"] is False


def test_general_execution_failure_is_explicit_and_sealed(tmp_path: Path) -> None:
    def broken_optimizer(*, model, config):
        del model, config
        raise RuntimeError("technical optimizer fixture failure")

    adapter = TeacherTrainingAdapter(
        model_constructor=_model_constructor,
        loss_fn=_loss_fn,
        optimizer_factory=broken_optimizer,
        scheduler_factory=_scheduler_factory,
        stop_rule=_stop_rule,
    )
    request = _request(tmp_path, resume_id="failed")
    result = adapter.run(
        request=request,
        data=build_technical_tabular_teacher_data(request.task_record),
    )

    assert result.status == "failed"
    artifact = _artifact(result.artifact_path)
    assert artifact["status"] == "failed"
    assert "RuntimeError" in artifact["reason"]


def test_unavailable_state_is_not_encoded_as_zero_or_replacement(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_id="unavailable")
    result = emit_unavailable_teacher(
        request,
        reason="technical fixture intentionally unavailable",
    )
    artifact = _artifact(result.artifact_path)

    assert result.status == "unavailable"
    assert artifact["status"] == "unavailable"
    assert artifact["updates_completed"] == 0
    assert artifact["metric_snapshots"] == []
    assert artifact["state_hashes"] is None
    assert artifact["reason"] == "technical fixture intentionally unavailable"


def test_unqualified_backend_is_rejected(tmp_path: Path) -> None:
    kwargs = dict(_request(tmp_path, resume_id="bad-backend").__dict__)
    kwargs["backend_qualification"] = {
        **kwargs["backend_qualification"],
        "qualified": False,
    }

    with pytest.raises(TeacherProtocolError, match="explicitly qualified"):
        TeacherTrainingRequest(**kwargs)


def test_data_identity_mismatch_is_rejected_before_training(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_id="wrong-data")
    data = build_technical_tabular_teacher_data(request.task_record)
    wrong = copy.copy(data)
    object.__setattr__(wrong, "dataset_sha256", "f" * 64)

    with pytest.raises(TeacherProtocolError, match="dataset hash mismatch"):
        _adapter().run(request=request, data=wrong)


def test_checkpoint_tampering_is_rejected_on_resume(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        resume_id="tamper",
        stop_step=4,
        checkpoint_retention=4,
    )
    data = build_technical_tabular_teacher_data(request.task_record)
    interrupted = _adapter().run(
        request=request,
        data=data,
        interrupt_after_updates=2,
    )
    latest = interrupted.checkpoint_paths[-1]

    raw = bytearray(latest.read_bytes())
    raw[len(raw) // 2] ^= 1
    latest.write_bytes(raw)

    with pytest.raises(
        TeacherProtocolError,
        match="physical hash mismatch",
    ):
        _adapter().run(request=request, data=data)


def test_request_change_rejects_stale_checkpoint(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        resume_id="stale",
        stop_step=4,
        checkpoint_retention=4,
    )
    data = build_technical_tabular_teacher_data(request.task_record)
    interrupted = _adapter().run(
        request=request,
        data=data,
        interrupt_after_updates=2,
    )
    assert interrupted.status == "interrupted"

    kwargs = dict(request.__dict__)
    kwargs["optimizer_config"] = {
        "kind": "sgd",
        "learning_rate": 0.051,
    }
    changed = TeacherTrainingRequest(**kwargs)

    with pytest.raises(
        TeacherProtocolError,
        match="request hashes are stale",
    ):
        _adapter().run(request=changed, data=data)


def test_artifact_content_hash_detects_tampering(tmp_path: Path) -> None:
    request = _request(tmp_path, resume_id="artifact-tamper")
    result = _adapter().run(
        request=request,
        data=build_technical_tabular_teacher_data(request.task_record),
    )
    artifact = json.loads(result.artifact_path.read_text())
    artifact["reason"] = "tampered"

    with pytest.raises(
        TeacherProtocolError,
        match="content hash mismatch",
    ):
        validate_teacher_artifact(artifact)


def test_same_seed_request_is_deterministic_across_independent_runs(
    tmp_path: Path,
) -> None:
    left_request = _request(tmp_path, resume_id="det-left")
    right_request = _request(tmp_path, resume_id="det-right")

    left = _adapter().run(
        request=left_request,
        data=build_technical_tabular_teacher_data(left_request.task_record),
    )
    right = _adapter().run(
        request=right_request,
        data=build_technical_tabular_teacher_data(right_request.task_record),
    )

    left_artifact = _artifact(left.artifact_path)
    right_artifact = _artifact(right.artifact_path)

    assert left_artifact["metric_snapshots"] == right_artifact["metric_snapshots"]
    assert (
        left_artifact["state_hashes"]["model_state_sha256"]
        == right_artifact["state_hashes"]["model_state_sha256"]
    )
    assert (
        left_artifact["state_hashes"]["optimizer_state_sha256"]
        == right_artifact["state_hashes"]["optimizer_state_sha256"]
    )

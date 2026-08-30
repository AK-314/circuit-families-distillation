"""Focused/adversarial compact-record and portable-CLI tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from circuit_families.stage12p1.phase import (
    build_teacher_trajectory,
    select_teacher_phases,
)
from circuit_families.stage12p1.records import (
    FoundationRecordError,
    build_foundation_record,
    foundation_record_sha256,
    teacher_attempt_identity,
    validate_foundation_record,
)
from circuit_families.stage12p1.tasks import (
    TASK_CONFIG_SCHEMA_VERSION,
    ModularAdditionImplementation,
    build_task_record,
)
from circuit_families.stage12p1.teacher import (
    TeacherTrainingAdapter,
    TeacherTrainingRequest,
    build_technical_tabular_teacher_data,
    emit_unavailable_teacher,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "validate_stage12p1_teacher_foundation.py"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _task() -> dict:
    implementation = ModularAdditionImplementation()
    return build_task_record(
        {
            "schema_version": TASK_CONFIG_SCHEMA_VERSION,
            "task_id": "technical-record-fixture",
            "implementation": implementation.name,
            "implementation_version": implementation.version,
            "modulus": 5,
            "input_domains": [
                list(range(5)),
                list(range(5)),
            ],
            "parameters": {},
            "split_identity": {
                "kind": "technical-record-split",
                "version": "v1",
                "train_indices": list(range(20)),
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
) -> TeacherTrainingRequest:
    return TeacherTrainingRequest(
        task_record=_task(),
        architecture_config={
            "kind": "technical-linear",
            "input_dim": 2,
            "output_dim": 5,
        },
        training_config={
            "loss": "cross_entropy",
        },
        optimizer_config={
            "kind": "sgd",
            "learning_rate": 0.01,
        },
        scheduler_config={
            "kind": "none",
        },
        stopping_config={
            "technical_stop_step": 0,
        },
        backend_qualification={
            "backend_id": "cpu-technical-record",
            "device": "cpu",
            "qualified": True,
            "exact_resume_supported": True,
            "qualification_ref": "technical-record-test/v1",
        },
        model_seed_id="technical-record-model-seed/v1",
        model_seed=31,
        training_seed_id="technical-record-training-seed/v1",
        training_seed=37,
        max_technical_updates=2,
        checkpoint_interval=1,
        checkpoint_retention=1,
        output_root=tmp_path,
        resume_id=resume_id,
    )


def _model(*, task_record, architecture_config, seed, device):
    del task_record
    torch.manual_seed(seed)
    return torch.nn.Linear(
        int(architecture_config["input_dim"]),
        int(architecture_config["output_dim"]),
    ).to(device)


def _loss(*, outputs, targets, config):
    assert config["loss"] == "cross_entropy"
    return F.cross_entropy(outputs, targets)


def _optimizer(*, model, config):
    return torch.optim.SGD(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )


def _scheduler(*, optimizer, config):
    del optimizer
    assert config["kind"] == "none"
    return None


def _stop(*, step, metrics, config):
    assert metrics
    return step >= int(config["technical_stop_step"])


def _adapter() -> TeacherTrainingAdapter:
    return TeacherTrainingAdapter(
        model_constructor=_model,
        loss_fn=_loss,
        optimizer_factory=_optimizer,
        scheduler_factory=_scheduler,
        stop_rule=_stop,
    )


def _phase(request, result):
    tests = {
        0: 0.00,
        50: 0.02,
        100: 0.15,
        150: 0.50,
        200: 0.90,
        250: 0.990,
        300: 0.995,
        350: 0.997,
        400: 0.999,
        450: 1.000,
    }
    rows = [
        {
            "training_step": step,
            "train_accuracy": 0.5 if step == 0 else 1.0,
            "test_accuracy": value,
            "train_loss": 0.1,
            "test_loss": 1.0,
            "checkpoint_path": (
                f"checkpoints/fixture/step_{step:08d}.pt"
            ),
            "checkpoint_sha256": _sha(f"checkpoint-{step}"),
        }
        for step, value in tests.items()
    ]

    trajectory = build_teacher_trajectory(
        teacher_seed_id="technical-record-teacher-seed/v1",
        teacher_seed=request.model_seed,
        task_identity_sha256=request.task_record["hashes"][
            "task_identity_sha256"
        ],
        teacher_artifact_sha256=result.artifact_sha256,
        records=rows,
    )
    return select_teacher_phases(trajectory)


def test_completed_record_contains_all_required_evidence(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_id="completed")
    result = _adapter().run(
        request=request,
        data=build_technical_tabular_teacher_data(
            request.task_record
        ),
    )

    record = build_foundation_record(
        request=request,
        result=result,
        phase_selection=_phase(request, result),
    )
    validated = validate_foundation_record(record)

    assert validated["scientific_data"] is False
    assert validated["production_eligible"] is False
    assert validated["task_identity"]["task_identity_sha256"]
    assert validated["attempt_identity"]["attempt_identity_sha256"]
    assert validated["sealed_teacher"]["status"] == "completed"
    assert validated["checkpoint_inventory"] == []
    assert validated["resume_lineage"] == []
    assert validated["phase_result"]["state"] == "available"
    assert validated["terminal_state"] == {
        "status": "completed",
        "reason": "injected_stop_rule_before_update",
        "terminal": True,
        "failure_or_unavailable": False,
    }
    assert foundation_record_sha256(validated)


def test_unavailable_record_preserves_unavailable_state(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_id="unavailable")
    result = emit_unavailable_teacher(
        request,
        reason="technical fixture unavailable",
    )

    record = build_foundation_record(
        request=request,
        result=result,
    )

    assert record["sealed_teacher"]["status"] == "unavailable"
    assert record["phase_result"] == {
        "state": "unavailable",
        "reason": "teacher_status:unavailable",
    }
    assert record["terminal_state"]["failure_or_unavailable"] is True


def test_failed_record_preserves_failure_without_phase_replacement(
    tmp_path: Path,
) -> None:
    def broken_model(**kwargs):
        del kwargs
        raise RuntimeError("technical failure fixture")

    adapter = TeacherTrainingAdapter(
        model_constructor=broken_model,
        loss_fn=_loss,
        optimizer_factory=_optimizer,
        scheduler_factory=_scheduler,
        stop_rule=_stop,
    )
    request = _request(tmp_path, resume_id="failed")
    result = adapter.run(
        request=request,
        data=build_technical_tabular_teacher_data(
            request.task_record
        ),
    )

    assert result.status == "failed"

    record = build_foundation_record(
        request=request,
        result=result,
    )

    assert record["terminal_state"]["status"] == "failed"
    assert record["terminal_state"]["failure_or_unavailable"] is True
    assert record["phase_result"]["state"] == "unavailable"


def test_completed_teacher_cannot_silently_omit_phase_result(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_id="missing-phase")
    result = _adapter().run(
        request=request,
        data=build_technical_tabular_teacher_data(
            request.task_record
        ),
    )

    with pytest.raises(
        FoundationRecordError,
        match="requires explicit phase-selection",
    ):
        build_foundation_record(
            request=request,
            result=result,
        )


def test_attempt_identity_changes_with_resume_coordinate(
    tmp_path: Path,
) -> None:
    left = teacher_attempt_identity(
        _request(tmp_path, resume_id="attempt-a")
    )
    right = teacher_attempt_identity(
        _request(tmp_path, resume_id="attempt-b")
    )

    assert (
        left["attempt_identity_sha256"]
        != right["attempt_identity_sha256"]
    )


def test_attempt_identity_tampering_is_detected(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_id="tamper")
    result = _adapter().run(
        request=request,
        data=build_technical_tabular_teacher_data(
            request.task_record
        ),
    )
    record = build_foundation_record(
        request=request,
        result=result,
        phase_selection=_phase(request, result),
    )

    record["attempt_identity"]["training_seed"] += 1

    with pytest.raises(
        FoundationRecordError,
        match="attempt identity hash mismatch",
    ):
        validate_foundation_record(record)


def test_absolute_artifact_path_is_rejected(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_id="path")
    result = _adapter().run(
        request=request,
        data=build_technical_tabular_teacher_data(
            request.task_record
        ),
    )
    record = build_foundation_record(
        request=request,
        result=result,
        phase_selection=_phase(request, result),
    )

    record["sealed_teacher"]["artifact_path"] = "/tmp/teacher.json"

    with pytest.raises(
        FoundationRecordError,
        match="portable relative",
    ):
        validate_foundation_record(record)


def test_phase_teacher_binding_tampering_is_detected(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_id="phase-binding")
    result = _adapter().run(
        request=request,
        data=build_technical_tabular_teacher_data(
            request.task_record
        ),
    )
    record = build_foundation_record(
        request=request,
        result=result,
        phase_selection=_phase(request, result),
    )

    record["phase_result"]["artifact"][
        "teacher_artifact_sha256"
    ] = "f" * 64

    with pytest.raises(ValueError):
        validate_foundation_record(record)


def test_record_content_hash_is_tamper_evident(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_id="record-hash")
    result = emit_unavailable_teacher(
        request,
        reason="technical fixture unavailable",
    )
    record = build_foundation_record(
        request=request,
        result=result,
    )

    tampered = copy.deepcopy(record)
    tampered["terminal_state"]["reason"] = "changed"

    with pytest.raises(
        FoundationRecordError,
        match="content hash mismatch",
    ):
        validate_foundation_record(tampered)


def test_validate_only_cli_is_cwd_and_pythonhashseed_stable(
    tmp_path: Path,
) -> None:
    outputs = []

    for hashseed in ("1", "999999"):
        cwd = tmp_path / f"cwd-{hashseed}"
        cwd.mkdir()

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hashseed

        completed = subprocess.run(
            [
                sys.executable,
                str(CLI),
                "--validate-only",
            ],
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip())

    assert outputs[0] == outputs[1]

    report = json.loads(outputs[0])
    assert report["stage12p1_validate_only"] == "PASS"
    assert report["scientific_data"] is False
    assert report["production_eligible"] is False
    assert report["production_teachers_trained"] is False

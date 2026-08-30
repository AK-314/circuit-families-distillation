from __future__ import annotations

import inspect
import runpy
from types import SimpleNamespace

import pytest

import circuit_families.stage12p2.checkpointing as checkpointing
from circuit_families.stage12p2 import (
    StudentCheckpointBindingError,
    load_student_resume_checkpoint,
    save_student_resume_checkpoint,
)

HELPERS = runpy.run_path("tests/test_stage12p2_training.py")
_stage3 = HELPERS["_stage3"]
_record = HELPERS["_record"]
_model_settings = HELPERS["_model_settings"]
_identity = HELPERS["_identity"]


def _identity_for(record):
    stage3 = _stage3()
    args = []
    kwargs = {}

    for parameter in inspect.signature(_identity).parameters.values():
        if parameter.name == "stage3":
            value = stage3
        elif parameter.name in {"record", "architecture", "architecture_record"}:
            value = record
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise AssertionError(
                f"unsupported required _identity helper argument: {parameter.name}"
            )

        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[parameter.name] = value

    return stage3, _identity(*args, **kwargs)


def _prepared_stub(record):
    return SimpleNamespace(settings=SimpleNamespace(model=_model_settings(record)))


def test_save_binds_exact_p2_identity_and_cache(monkeypatch, tmp_path) -> None:
    record = _record("alpha")
    stage3, identity = _identity_for(record)
    prepared = _prepared_stub(record)
    snapshot = object()
    sentinel = object()
    captured = {}

    def fake_save(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        checkpointing,
        "save_technical_resume_checkpoint",
        fake_save,
    )

    result = save_student_resume_checkpoint(
        tmp_path / "resume.pt",
        prepared=prepared,
        snapshot=snapshot,
        identity=identity,
        stage3=stage3,
    )

    assert result is sentinel
    assert captured["attempt_identity"] == identity.stage5_attempt
    assert captured["configuration_hashes"] == identity.checkpoint_configuration_hashes()
    assert captured["target_cache_manifest_sha256"] == identity.target_cache_manifest_sha256


def test_load_binds_exact_p2_identity_and_cache(monkeypatch, tmp_path) -> None:
    record = _record("alpha")
    stage3, identity = _identity_for(record)
    prepared = _prepared_stub(record)
    sentinel = object()
    captured = {}

    def fake_load(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        checkpointing,
        "load_technical_resume_checkpoint",
        fake_load,
    )

    result = load_student_resume_checkpoint(
        tmp_path / "resume.pt",
        prepared=prepared,
        expected_identity=identity,
        stage3=stage3,
        expected_file_sha256="f" * 64,
    )

    assert result is sentinel
    assert captured["expected_attempt_identity"] == identity.stage5_attempt
    assert captured["expected_configuration_hashes"] == identity.checkpoint_configuration_hashes()
    assert (
        captured["expected_target_cache_manifest_sha256"] == identity.target_cache_manifest_sha256
    )
    assert captured["expected_file_sha256"] == "f" * 64


def test_checkpoint_rejects_cross_architecture_prepared_trainer(
    monkeypatch,
    tmp_path,
) -> None:
    alpha = _record("alpha")
    beta = _record("beta")
    stage3, alpha_identity = _identity_for(alpha)
    prepared_beta = _prepared_stub(beta)

    monkeypatch.setattr(
        checkpointing,
        "save_technical_resume_checkpoint",
        lambda *args, **kwargs: pytest.fail("delegate must not be reached"),
    )

    with pytest.raises(
        StudentCheckpointBindingError,
        match="architecture",
    ):
        save_student_resume_checkpoint(
            tmp_path / "resume.pt",
            prepared=prepared_beta,
            snapshot=object(),
            identity=alpha_identity,
            stage3=stage3,
        )

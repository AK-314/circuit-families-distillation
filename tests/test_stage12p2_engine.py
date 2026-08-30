from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage5bc.student_identity import (
    build_student_attempt_identity,
)
from circuit_families.stage5bc.student_trainer import (
    HardTargetAdapter,
    OptimizerScheduleBundle,
    SoftTargetAdapter,
    TrainerLifecycle,
    TrainerSettingsBundle,
)
from circuit_families.stage5bc.target_cache import (
    TargetCacheBatch,
    build_target_cache,
    load_target_cache,
)
from circuit_families.stage6b import HardLabelLossAdapter
from circuit_families.stage12p2 import (
    ArchitectureModelConstructor,
    DuplicateStudentAttemptError,
    RollingCheckpointProfile,
    StudentAttemptEngineError,
    bind_student_training_identity,
    canonical_predecessor_record,
    default_technical_architecture_registry,
    record_unavailable_student_attempt,
    run_student_technical_attempt,
    technical_transformer_record,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE3_REGISTRY = ROOT / "followup/manifests/stage3_teacher_registry_v1.json"


def _stage3() -> Stage3AvailabilityIndex:
    return Stage3AvailabilityIndex.from_registry(
        json.loads(STAGE3_REGISTRY.read_text(encoding="utf-8"))
    )


def _records():
    canonical = canonical_predecessor_record()
    compatibility = canonical.compatibility

    technical_a = technical_transformer_record(
        family="technical",
        name="tiny-a",
        version="v1",
        n_layers=1,
        n_ctx=3,
        d_model=32,
        n_heads=4,
        d_head=8,
        d_mlp=64,
        d_vocab=114,
        d_vocab_out=113,
        compatibility=compatibility,
    )
    technical_b = technical_transformer_record(
        family="technical",
        name="tiny-b",
        version="v1",
        n_layers=2,
        n_ctx=3,
        d_model=48,
        n_heads=6,
        d_head=8,
        d_mlp=96,
        d_vocab=114,
        d_vocab_out=113,
        compatibility=compatibility,
    )
    return canonical, technical_a, technical_b


def _registry(records):
    registry = default_technical_architecture_registry()
    for record in records:
        registry.register(record)
    return registry


def _teacher_condition(
    stage3: Stage3AvailabilityIndex,
    condition: str,
) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition=condition,
        ),
        stage3,
    )


def _cache(
    tmp_path: Path,
    *,
    stage3: Stage3AvailabilityIndex,
    condition: str,
):
    root = tmp_path / f"cache-{condition}"
    input_ids = ("technical-input-0", "technical-input-1")
    logits = torch.zeros((2, 113), dtype=torch.float32)
    logits[0, 0] = 4.0
    logits[1, 1] = 4.0

    teacher_reference = {
        "record_type": "teacher_reference",
        "schema_version": "teacher_reference/v1",
        "condition_id": _teacher_condition(stage3, condition),
        "record_sha256": "5" * 64,
    }
    provenance_hashes = {
        "dataset_sha256": "6" * 64,
        "split_sha256": "7" * 64,
        "task_config_sha256": "8" * 64,
        "model_config_sha256": "9" * 64,
        "training_config_sha256": "a" * 64,
        "component_basis_sha256": "b" * 64,
    }

    build_target_cache(
        output_root=root,
        manifest_relative_path="manifest.json",
        payload_relative_path="payload.bin",
        completion_relative_path="completion.json",
        manifest_id=f"technical-stage12p2-{condition}-cache/v1",
        ordering_ref="technical-stage12p2-order/v1",
        expected_example_count=2,
        expected_class_count=113,
        teacher_reference=teacher_reference,
        provenance_hashes=provenance_hashes,
        batches=(
            TargetCacheBatch(
                input_ids=input_ids,
                raw_logits=logits,
            ),
        ),
        technical_fixture=True,
        stage4_record_serializable=False,
        expected_input_ids=input_ids,
    )

    return load_target_cache(
        output_root=root,
        manifest_relative_path="manifest.json",
        expected_input_ids=input_ids,
        expected_teacher_reference=teacher_reference,
        expected_provenance_hashes=provenance_hashes,
        expected_stage4_cache_kind=(
            "teacher_argmax" if condition == "hard_target" else "teacher_logits"
        ),
    )


def _identity(
    *,
    stage3,
    record,
    cache,
    condition,
    retry_index=0,
):
    stage5 = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition=condition,
        student_initialization=0,
        attempt_index=0,
        retry_index=retry_index,
    )
    return bind_student_training_identity(
        stage3=stage3,
        stage5_attempt=stage5,
        task_identity_sha256="1" * 64,
        target_cache_manifest=cache.manifest,
        architecture_record=record,
        model_seed_id="technical-model-seed/v1",
        model_seed=17,
        training_config_ref="technical-stage12p2-training/v1",
        training_config={
            "technical_fixture": True,
            "profile_injected": True,
        },
        backend_ref="technical-stage12p2-cpu/v1",
        backend_qualification={
            "backend": "cpu",
            "technical_fixture": True,
            "exact_resume_supported": True,
        },
    )


def _settings(record, *, hard: bool):
    return TrainerSettingsBundle(
        model={
            "architecture_ref": record.architecture_ref,
            "architecture_record_sha256": (record.to_mapping()["record_sha256"]),
        },
        loss=(
            {
                "loss_kind": "cross_entropy",
                "reduction": "mean",
            }
            if hard
            else {"technical_fixture": True}
        ),
        optimizer_schedule={
            "optimizer": "sgd",
            "technical_fixture": True,
        },
        stop={"technical_fixture": True},
    )


def _optimizer_factory(*, model, settings):
    return OptimizerScheduleBundle(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        scheduler=None,
    )


def _stop_after(updates: int):
    def stop_rule(*, progress, settings):
        return progress.updates_completed >= updates

    return stop_rule


def _soft_loss(*, outputs, targets, settings):
    student = outputs - outputs.mean(dim=-1, keepdim=True)
    teacher = targets.values - targets.values.mean(dim=-1, keepdim=True)
    return (student - teacher).square().mean()


def _lifecycle(
    *,
    registry,
    record,
    hard: bool,
    stop_after: int,
    numerical: bool = False,
):
    constructor = ArchitectureModelConstructor.from_record(
        registry=registry,
        record=record,
    )

    if hard:
        target_adapter = HardTargetAdapter()
        loss_adapter = HardLabelLossAdapter()
    else:
        target_adapter = SoftTargetAdapter()
        loss_adapter = _soft_loss

    if numerical:

        def nonfinite_loss(*, outputs, targets, settings):
            return outputs.sum() * torch.tensor(
                float("nan"),
                device=outputs.device,
            )

        loss_adapter = nonfinite_loss

    return TrainerLifecycle(
        model_constructor=constructor,
        target_adapter=target_adapter,
        loss_adapter=loss_adapter,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=_stop_after(stop_after),
        recorder=lambda event: None,
    )


def _inputs():
    return torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.int64,
    )


def _latest_resume(execution):
    candidates = [
        entry for entry in execution.checkpoints.entries if entry.role in {"rolling", "interrupted"}
    ]
    assert candidates
    return candidates[-1]


def test_shared_engine_runs_canonical_hard_and_technical_soft(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    canonical, technical_a, technical_b = _records()
    registry = _registry((canonical, technical_a, technical_b))

    hard_cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="hard_target",
    )
    soft_cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="soft_target",
    )

    hard_identity = _identity(
        stage3=stage3,
        record=canonical,
        cache=hard_cache,
        condition="hard_target",
    )
    soft_identity = _identity(
        stage3=stage3,
        record=technical_a,
        cache=soft_cache,
        condition="soft_target",
    )

    hard = run_student_technical_attempt(
        lifecycle=_lifecycle(
            registry=registry,
            record=canonical,
            hard=True,
            stop_after=1,
        ),
        stage3=stage3,
        identity=hard_identity,
        architecture_record=canonical,
        task_requirements=canonical.compatibility,
        cache=hard_cache,
        training_inputs=_inputs(),
        settings=_settings(canonical, hard=True),
        device="cpu",
        output_root=tmp_path / "hard-canonical",
        checkpoint_profile=RollingCheckpointProfile(
            interval_updates=1,
            retention_count=2,
        ),
        technical_safety_step_limit=3,
    )
    soft = run_student_technical_attempt(
        lifecycle=_lifecycle(
            registry=registry,
            record=technical_a,
            hard=False,
            stop_after=3,
        ),
        stage3=stage3,
        identity=soft_identity,
        architecture_record=technical_a,
        task_requirements=canonical.compatibility,
        cache=soft_cache,
        training_inputs=_inputs(),
        settings=_settings(technical_a, hard=False),
        device="cpu",
        output_root=tmp_path / "soft-technical-a",
        checkpoint_profile=RollingCheckpointProfile(
            interval_updates=1,
            retention_count=2,
        ),
        technical_safety_step_limit=5,
    )

    assert hard.status == "completed"
    assert soft.status == "completed"
    assert hard.architecture_ref != soft.architecture_ref
    assert hard.training_result.target_cache_kind == "teacher_argmax"
    assert soft.training_result.target_cache_kind == "teacher_logits"

    retained_rolling = [entry for entry in soft.checkpoints.entries if entry.role == "rolling"]
    assert len(retained_rolling) == 2
    assert all(Path(entry.path).exists() for entry in retained_rolling)
    assert any(entry.role == "terminal" for entry in soft.checkpoints.entries)


def test_distinct_second_technical_architecture_interrupts_and_resumes(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    canonical, technical_a, technical_b = _records()
    registry = _registry((canonical, technical_a, technical_b))
    cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="hard_target",
    )
    identity = _identity(
        stage3=stage3,
        record=technical_b,
        cache=cache,
        condition="hard_target",
    )
    lifecycle = _lifecycle(
        registry=registry,
        record=technical_b,
        hard=True,
        stop_after=2,
    )

    interrupted = run_student_technical_attempt(
        lifecycle=lifecycle,
        stage3=stage3,
        identity=identity,
        architecture_record=technical_b,
        task_requirements=canonical.compatibility,
        cache=cache,
        training_inputs=_inputs(),
        settings=_settings(technical_b, hard=True),
        device="cpu",
        output_root=tmp_path / "hard-technical-b",
        checkpoint_profile=RollingCheckpointProfile(
            interval_updates=1,
            retention_count=2,
        ),
        technical_safety_step_limit=4,
        interrupt_after_updates=1,
    )
    assert interrupted.status == "interrupted"
    assert interrupted.updates_completed == 1

    resume = _latest_resume(interrupted)
    completed = run_student_technical_attempt(
        lifecycle=lifecycle,
        stage3=stage3,
        identity=identity,
        architecture_record=technical_b,
        task_requirements=canonical.compatibility,
        cache=cache,
        training_inputs=_inputs(),
        settings=_settings(technical_b, hard=True),
        device="cpu",
        output_root=tmp_path / "hard-technical-b",
        checkpoint_profile=RollingCheckpointProfile(
            interval_updates=1,
            retention_count=2,
        ),
        technical_safety_step_limit=4,
        resume_checkpoint=resume.path,
        resume_checkpoint_sha256=resume.file_sha256,
    )

    assert completed.status == "completed"
    assert completed.updates_completed == 2
    assert any(entry.role == "resume_source" for entry in completed.checkpoints.entries)

    with pytest.raises(DuplicateStudentAttemptError):
        run_student_technical_attempt(
            lifecycle=lifecycle,
            stage3=stage3,
            identity=identity,
            architecture_record=technical_b,
            task_requirements=canonical.compatibility,
            cache=cache,
            training_inputs=_inputs(),
            settings=_settings(technical_b, hard=True),
            device="cpu",
            output_root=tmp_path / "hard-technical-b",
            checkpoint_profile=RollingCheckpointProfile(
                interval_updates=1,
                retention_count=2,
            ),
            technical_safety_step_limit=4,
        )


def test_explicit_failed_numerical_and_unavailable_statuses(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    canonical, technical_a, technical_b = _records()
    registry = _registry((canonical, technical_a, technical_b))
    cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="hard_target",
    )

    failed_identity = _identity(
        stage3=stage3,
        record=technical_a,
        cache=cache,
        condition="hard_target",
        retry_index=0,
    )
    failed = run_student_technical_attempt(
        lifecycle=_lifecycle(
            registry=registry,
            record=technical_a,
            hard=True,
            stop_after=99,
        ),
        stage3=stage3,
        identity=failed_identity,
        architecture_record=technical_a,
        task_requirements=canonical.compatibility,
        cache=cache,
        training_inputs=_inputs(),
        settings=_settings(technical_a, hard=True),
        device="cpu",
        output_root=tmp_path / "failed",
        checkpoint_profile=RollingCheckpointProfile(
            interval_updates=1,
            retention_count=1,
        ),
        technical_safety_step_limit=1,
    )
    assert failed.status == "failed"

    numerical_identity = _identity(
        stage3=stage3,
        record=technical_a,
        cache=cache,
        condition="hard_target",
        retry_index=1,
    )
    numerical = run_student_technical_attempt(
        lifecycle=_lifecycle(
            registry=registry,
            record=technical_a,
            hard=True,
            stop_after=99,
            numerical=True,
        ),
        stage3=stage3,
        identity=numerical_identity,
        architecture_record=technical_a,
        task_requirements=canonical.compatibility,
        cache=cache,
        training_inputs=_inputs(),
        settings=_settings(technical_a, hard=True),
        device="cpu",
        output_root=tmp_path / "numerical",
        checkpoint_profile=RollingCheckpointProfile(
            interval_updates=1,
            retention_count=1,
        ),
        technical_safety_step_limit=2,
    )
    assert numerical.status == "numerical-failure"

    unavailable_identity = _identity(
        stage3=stage3,
        record=technical_b,
        cache=cache,
        condition="hard_target",
        retry_index=2,
    )
    unavailable = record_unavailable_student_attempt(
        identity=unavailable_identity,
        output_root=tmp_path / "unavailable",
        reason="constructed_technical_unavailable_case",
    )
    assert unavailable.status == "unavailable"
    assert unavailable.updates_completed == 0


def test_cache_output_width_mismatch_rejects_before_training(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    canonical, technical_a, technical_b = _records()

    incompatible = technical_transformer_record(
        family="technical",
        name="wrong-output-width",
        version="v1",
        n_layers=1,
        n_ctx=3,
        d_model=32,
        n_heads=4,
        d_head=8,
        d_mlp=64,
        d_vocab=114,
        d_vocab_out=17,
        compatibility={
            **canonical.compatibility,
            "output_class_count": 17,
        },
    )
    registry = _registry((canonical, technical_a, technical_b, incompatible))
    cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="hard_target",
    )
    identity = _identity(
        stage3=stage3,
        record=incompatible,
        cache=cache,
        condition="hard_target",
    )

    with pytest.raises(
        StudentAttemptEngineError,
        match="target cache class count disagrees with architecture output width",
    ):
        run_student_technical_attempt(
            lifecycle=_lifecycle(
                registry=registry,
                record=incompatible,
                hard=True,
                stop_after=1,
            ),
            stage3=stage3,
            identity=identity,
            architecture_record=incompatible,
            task_requirements=incompatible.compatibility,
            cache=cache,
            training_inputs=_inputs(),
            settings=_settings(incompatible, hard=True),
            device="cpu",
            output_root=tmp_path / "mismatch",
            checkpoint_profile=RollingCheckpointProfile(
                interval_updates=1,
                retention_count=1,
            ),
            technical_safety_step_limit=2,
        )


def test_soft_technical_attempt_interrupts_resumes_and_seals(
    tmp_path: Path,
) -> None:
    stage3 = _stage3()
    canonical, technical_a, technical_b = _records()
    registry = _registry((canonical, technical_a, technical_b))
    cache = _cache(
        tmp_path,
        stage3=stage3,
        condition="soft_target",
    )
    identity = _identity(
        stage3=stage3,
        record=technical_b,
        cache=cache,
        condition="soft_target",
    )
    lifecycle = _lifecycle(
        registry=registry,
        record=technical_b,
        hard=False,
        stop_after=2,
    )
    output_root = tmp_path / "soft-technical-b-resume"

    interrupted = run_student_technical_attempt(
        lifecycle=lifecycle,
        stage3=stage3,
        identity=identity,
        architecture_record=technical_b,
        task_requirements=canonical.compatibility,
        cache=cache,
        training_inputs=_inputs(),
        settings=_settings(technical_b, hard=False),
        device="cpu",
        output_root=output_root,
        checkpoint_profile=RollingCheckpointProfile(
            interval_updates=1,
            retention_count=2,
        ),
        technical_safety_step_limit=4,
        interrupt_after_updates=1,
    )
    assert interrupted.status == "interrupted"

    resume = _latest_resume(interrupted)
    completed = run_student_technical_attempt(
        lifecycle=lifecycle,
        stage3=stage3,
        identity=identity,
        architecture_record=technical_b,
        task_requirements=canonical.compatibility,
        cache=cache,
        training_inputs=_inputs(),
        settings=_settings(technical_b, hard=False),
        device="cpu",
        output_root=output_root,
        checkpoint_profile=RollingCheckpointProfile(
            interval_updates=1,
            retention_count=2,
        ),
        technical_safety_step_limit=4,
        resume_checkpoint=resume.path,
        resume_checkpoint_sha256=resume.file_sha256,
    )

    assert completed.status == "completed"
    assert completed.updates_completed == 2
    terminal = [entry for entry in completed.checkpoints.entries if entry.role == "terminal"]
    assert len(terminal) == 1
    assert Path(terminal[0].path).exists()
    assert (output_root / "terminal_status.json").exists()

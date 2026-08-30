#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

STAGE3_REGISTRY = ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
import json

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
from circuit_families.stage6c import (
    TECHNICAL_POLICY_STATUS,
    TECHNICAL_SOFT_DISCREPANCY_METRIC,
    TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
    SoftRepresentationMetadata,
    TechnicalArgmaxRequirementMetadata,
    TechnicalSoftPolicy,
    TechnicalToleranceMetadata,
)
from circuit_families.stage6c.soft_target import CENTRING_REF
from circuit_families.stage12p2 import (
    ArchitectureModelConstructor,
    RollingCheckpointProfile,
    StudentSealingError,
    bind_student_training_identity,
    default_technical_architecture_registry,
    evaluate_hard_student_eligibility,
    evaluate_soft_student_eligibility,
    release_student_for_discovery,
    run_student_technical_attempt,
    seal_student_model,
)
from circuit_families.stage12p2.builders import (
    canonical_predecessor_record,
    technical_transformer_record,
)
from circuit_families.stage12p2.components import (
    transformer_component_inventory,
)


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


def _soft_policy(stage3):
    return TechnicalSoftPolicy(
        schema_version=TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
        policy_ref="technical-stage12p2-validation-soft-policy/v1",
        status=TECHNICAL_POLICY_STATUS,
        scientific_data=False,
        production_eligible=False,
        resolves_ud006=False,
        representation=SoftRepresentationMetadata(
            representation_ref="technical-stage12p2-validation-centred-logits/v1",
            cache_kind="teacher_logits",
            centering_ref=CENTRING_REF,
            teacher_condition_id=_teacher_condition(stage3, "soft_target"),
            ordering_ref="technical-stage12p2-order/v1",
            ordered_input_ids_sha256="d" * 64,
            temperature_candidate=None,
            normalization_candidate_ref=None,
        ),
        tolerance=TechnicalToleranceMetadata(
            metric_ref=TECHNICAL_SOFT_DISCREPANCY_METRIC,
            comparison="less_than_or_equal",
            candidate_value=1e-10,
            status=TECHNICAL_POLICY_STATUS,
        ),
        argmax_requirement=TechnicalArgmaxRequirementMetadata(
            requirement_ref="technical-stage12p2-validation-argmax/v1",
            candidate_required=False,
            status=TECHNICAL_POLICY_STATUS,
        ),
    )


def main() -> int:
    stage3 = _stage3()
    canonical, technical_a, technical_b = _records()
    records = (canonical, technical_a, technical_b)
    registry = default_technical_architecture_registry()
    for record in records:
        registry.register(record)

    for record in records:
        inventory = transformer_component_inventory(record)
        components = inventory.components
        print(
            "ARCH "
            f"ref={record.architecture_ref} "
            f"parameters={record.parameter_count} "
            f"components={len(components)}"
        )

    with tempfile.TemporaryDirectory(prefix="stage12p2-validation-") as raw_tmp:
        tmp = Path(raw_tmp)
        hard_cache = _cache(
            tmp,
            stage3=stage3,
            condition="hard_target",
        )
        soft_cache = _cache(
            tmp,
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
        resume_identity = _identity(
            stage3=stage3,
            record=technical_b,
            cache=hard_cache,
            condition="hard_target",
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
            output_root=tmp / "hard-canonical",
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
                stop_after=1,
            ),
            stage3=stage3,
            identity=soft_identity,
            architecture_record=technical_a,
            task_requirements=canonical.compatibility,
            cache=soft_cache,
            training_inputs=_inputs(),
            settings=_settings(technical_a, hard=False),
            device="cpu",
            output_root=tmp / "soft-technical-a",
            checkpoint_profile=RollingCheckpointProfile(
                interval_updates=1,
                retention_count=2,
            ),
            technical_safety_step_limit=3,
        )

        interrupted = run_student_technical_attempt(
            lifecycle=_lifecycle(
                registry=registry,
                record=technical_b,
                hard=True,
                stop_after=2,
            ),
            stage3=stage3,
            identity=resume_identity,
            architecture_record=technical_b,
            task_requirements=canonical.compatibility,
            cache=hard_cache,
            training_inputs=_inputs(),
            settings=_settings(technical_b, hard=True),
            device="cpu",
            output_root=tmp / "hard-technical-b",
            checkpoint_profile=RollingCheckpointProfile(
                interval_updates=1,
                retention_count=2,
            ),
            technical_safety_step_limit=4,
            interrupt_after_updates=1,
        )

        rolling = tuple(
            entry for entry in interrupted.checkpoints.entries if entry.role == "rolling"
        )[-1]

        resumed = run_student_technical_attempt(
            lifecycle=_lifecycle(
                registry=registry,
                record=technical_b,
                hard=True,
                stop_after=2,
            ),
            stage3=stage3,
            identity=resume_identity,
            architecture_record=technical_b,
            task_requirements=canonical.compatibility,
            cache=hard_cache,
            training_inputs=_inputs(),
            settings=_settings(technical_b, hard=True),
            device="cpu",
            output_root=tmp / "hard-technical-b",
            checkpoint_profile=RollingCheckpointProfile(
                interval_updates=1,
                retention_count=2,
            ),
            technical_safety_step_limit=4,
            resume_checkpoint=rolling.path,
            resume_checkpoint_sha256=rolling.file_sha256,
        )

        if (
            hard.status != "completed"
            or soft.status != "completed"
            or interrupted.status != "interrupted"
            or resumed.status != "completed"
        ):
            raise RuntimeError("technical lifecycle did not reach expected statuses")

        teacher_logits = torch.zeros((2, 113), dtype=torch.float32)
        teacher_logits[0, 0] = 4.0
        teacher_logits[1, 1] = 4.0
        hard_pass_logits = teacher_logits.clone()
        hard_fail_logits = teacher_logits.clone()
        hard_fail_logits[1, 1] = 0.0
        hard_fail_logits[1, 2] = 4.0

        hard_pass = evaluate_hard_student_eligibility(
            execution=hard,
            identity=hard_identity,
            teacher_decisions=(0, 1),
            student_dense_logits=hard_pass_logits,
            ordering_ref="technical-stage12p2-order/v1",
            ordered_input_ids_sha256="d" * 64,
            domain_complete=True,
        )
        hard_fail = evaluate_hard_student_eligibility(
            execution=hard,
            identity=hard_identity,
            teacher_decisions=(0, 1),
            student_dense_logits=hard_fail_logits,
            ordering_ref="technical-stage12p2-order/v1",
            ordered_input_ids_sha256="d" * 64,
            domain_complete=True,
        )

        soft_pass = evaluate_soft_student_eligibility(
            execution=soft,
            identity=soft_identity,
            teacher_logits=teacher_logits,
            student_dense_logits=(teacher_logits + torch.tensor([[7.0], [-3.0]])),
            policy=_soft_policy(stage3),
            ordering_ref="technical-stage12p2-order/v1",
            ordered_input_ids_sha256="d" * 64,
            domain_complete=True,
        )

        sealed = seal_student_model(
            execution=hard,
            identity=hard_identity,
            eligibility=hard_pass,
            dense_output_sha256=hard_pass.dense_output_sha256,
        )
        release = release_student_for_discovery(
            sealed=sealed,
            eligibility=hard_pass,
        )

        blocked = False
        try:
            seal_student_model(
                execution=hard,
                identity=hard_identity,
                eligibility=hard_fail,
                dense_output_sha256=hard_fail.dense_output_sha256,
            )
        except StudentSealingError:
            blocked = True
        if not blocked:
            raise RuntimeError("ineligible discovery boundary was not blocked")

        print(
            "LIFECYCLE "
            f"hard={hard.status} soft={soft.status} "
            f"interrupt={interrupted.status} resume={resumed.status}"
        )
        print(
            "ELIGIBILITY "
            f"hard_pass={hard_pass.status} "
            f"hard_fail={hard_fail.status} "
            f"soft_pass={soft_pass.status}"
        )
        print(f"DISCOVERY release={release.release_status} ineligible=BLOCKED")

    print("FIXTURE_SCOPE=SYNTHETIC_TECHNICAL")
    print("REPOSITORY_WRITES=NO")
    print("REAL_TRAINING=NO")
    print("REAL_TEACHER_CACHE=NO")
    print("PRODUCTION_ARCHITECTURE_SELECTED=NO")
    print("SCIENTIFIC_DATA=NO")
    print("STAGE12P2_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

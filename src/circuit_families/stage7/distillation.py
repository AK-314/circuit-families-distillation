"""Technical-only Stage 7A distillation integration.

This module wires the accepted Stage 5B/C target cache and shared trainer into
the accepted Stage 6B hard and Stage 6C soft eligibility/sealing lifecycles.

It defines no production hyperparameters, no replacement trainer, and no
scientific decision. Numeric fixture settings must be injected by the caller.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage5bc.attempt_records import (
    attempt_record_sha256,
    emit_technical_attempt_record,
    outcome_from_training_result,
)
from circuit_families.stage5bc.student_identity import (
    build_student_attempt_identity,
)
from circuit_families.stage5bc.student_trainer import (
    HardTargetAdapter,
    OptimizerScheduleBundle,
    TechnicalLoopSnapshot,
    TrainerLifecycle,
    TrainerSettingsBundle,
)
from circuit_families.stage5bc.target_cache import (
    FULL_DOMAIN_EXAMPLE_COUNT,
    TargetCacheBatch,
    build_target_cache,
    load_target_cache,
)
from circuit_families.stage5bc.technical_checkpoint import (
    save_technical_resume_checkpoint,
)
from circuit_families.stage6b import (
    CanonicalDecisionVector,
    HardAttemptLedger,
    HardLabelLossAdapter,
    assess_hard_attempt,
    canonical_decision_bytes,
    circuit_release_gate,
    evaluate_hard_target_eligibility,
    generate_hard_sealing_evidence,
)
from circuit_families.stage6c import (
    CENTRING_REF,
    SOFT_LOSS_KIND,
    TECHNICAL_POLICY_STATUS,
    TECHNICAL_SOFT_DISCREPANCY_METRIC,
    TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
    CanonicalSoftOutput,
    GaugeInvariantSoftLossAdapter,
    SoftAttemptLedger,
    SoftRepresentationMetadata,
    TechnicalArgmaxRequirementMetadata,
    TechnicalSoftPolicy,
    TechnicalSoftTargetAdapter,
    TechnicalToleranceMetadata,
    assess_soft_attempt,
    evaluate_soft_target_eligibility,
    generate_soft_sealing_evidence,
    soft_circuit_release_gate,
)

DISTILLATION_INTEGRATION_SCHEMA_VERSION: Final = (
    "stage7-technical-distillation-integration/v1"
)
SHARED_TRAINER_REFERENCE: Final = (
    "circuit_families.stage5bc.student_trainer.TrainerLifecycle"
)
TARGET_CACHE_BUILDER_REFERENCE: Final = (
    "circuit_families.stage5bc.target_cache.build_target_cache"
)
TARGET_CACHE_LOADER_REFERENCE: Final = (
    "circuit_families.stage5bc.target_cache.load_target_cache"
)
ORDERING_REF: Final = "technical-stage7-full-domain-order/v1"
ARCHITECTURE_REF: Final = "technical-stage7-replay-model/v1"
REPLICATION_REF: Final = "technical-stage7-replication-fixture/v1"
TRAINER_REF: Final = "technical-stage7-shared-trainer-fixture/v1"
HARD_ADAPTER_REF: Final = "technical-stage7-hard-adapter/v1"
SOFT_ADAPTER_REF: Final = "technical-stage7-soft-adapter/v1"


class Stage7DistillationError(ValueError):
    """Raised when the technical distillation integration violates its boundary."""


@dataclass(frozen=True)
class TechnicalDistillationFixtureConfig:
    """Explicit non-production settings for the tiny Stage 7A fixture."""

    hard_learning_rate: float
    soft_learning_rate: float
    technical_stop_step: int
    technical_safety_step_limit: int
    soft_tolerance: float
    scientific_data: bool = False
    production_eligible: bool = False
    production_default: bool = False
    resolves_decisions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "hard_learning_rate",
            "soft_learning_rate",
            "soft_tolerance",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise Stage7DistillationError(
                    f"{name} must be an explicitly injected finite number"
                )
            if not math.isfinite(float(value)):
                raise Stage7DistillationError(
                    f"{name} must be finite"
                )

        if self.hard_learning_rate <= 0:
            raise Stage7DistillationError(
                "hard_learning_rate must be positive"
            )

        if self.soft_learning_rate <= 0:
            raise Stage7DistillationError(
                "soft_learning_rate must be positive"
            )

        if self.soft_tolerance < 0:
            raise Stage7DistillationError(
                "soft_tolerance must be non-negative"
            )

        for name in (
            "technical_stop_step",
            "technical_safety_step_limit",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise Stage7DistillationError(
                    f"{name} must be a positive integer"
                )

        if self.technical_stop_step > self.technical_safety_step_limit:
            raise Stage7DistillationError(
                "successful fixture stop step must fit inside safety limit"
            )

        if self.scientific_data is not False:
            raise Stage7DistillationError(
                "technical fixture cannot contain scientific data"
            )

        if self.production_eligible is not False:
            raise Stage7DistillationError(
                "technical fixture cannot be production eligible"
            )

        if self.production_default is not False:
            raise Stage7DistillationError(
                "technical fixture cannot define a production default"
            )

        if self.resolves_decisions:
            raise Stage7DistillationError(
                "technical fixture cannot resolve unresolved decisions"
            )


class _ReplayModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float64)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.scale * inputs


def _model_constructor(
    *,
    seed: int,
    device: torch.device,
    settings: dict[str, Any],
) -> torch.nn.Module:
    if settings != {
        "profile_id": ARCHITECTURE_REF,
        "technical_fixture": True,
    }:
        raise Stage7DistillationError(
            "model settings must be explicitly technical"
        )

    torch.manual_seed(seed)
    model = _ReplayModel()
    model.eval()
    return model.to(device)


def _optimizer_factory(
    *,
    model: torch.nn.Module,
    settings: dict[str, Any],
) -> OptimizerScheduleBundle:
    if settings.get("technical_fixture") is not True:
        raise Stage7DistillationError(
            "optimizer settings must remain technical-only"
        )

    learning_rate = settings.get("learning_rate")

    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise Stage7DistillationError(
            "technical learning rate must be explicitly injected"
        )

    return OptimizerScheduleBundle(
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=float(learning_rate),
        ),
        scheduler=None,
    )


def _stop_rule(
    *,
    progress,
    settings: dict[str, Any],
) -> bool:
    if settings.get("technical_fixture") is not True:
        raise Stage7DistillationError(
            "stop settings must remain technical-only"
        )

    stop_step = settings.get("stop_step")

    if (
        isinstance(stop_step, bool)
        or not isinstance(stop_step, int)
        or stop_step <= 0
    ):
        raise Stage7DistillationError(
            "technical stop_step must be explicitly injected"
        )

    return progress.updates_completed >= stop_step


def _condition_id(
    stage3: Stage3AvailabilityIndex,
    *,
    teacher_seed: int,
    phase: str,
    condition: str,
    initialization: int | None = None,
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


def _file_sha256(source: Path) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _canonical_training_log_sha256(result) -> str:
    value = {
        "scientific_data": False,
        "production_eligible": False,
        "target_cache_kind": result.target_cache_kind,
        "terminal_reason": result.terminal_reason,
        "terminal_status": result.terminal_status,
        "updates_completed": result.updates_completed,
        "trajectory": [
            {
                "metrics": dict(progress.metrics),
                "step": progress.step,
                "updates_completed": progress.updates_completed,
            }
            for progress in result.trajectory
        ],
    }

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

    return hashlib.sha256(payload).hexdigest()


def _artifact(
    *,
    path: str,
    sha256: str,
    storage_class: str,
) -> dict[str, str]:
    return {
        "path": path,
        "sha256": sha256,
        "storage_class": storage_class,
    }


def _configuration_refs(condition: str) -> dict[str, str]:
    adapter_ref = (
        HARD_ADAPTER_REF
        if condition == "hard_target"
        else SOFT_ADAPTER_REF
    )

    return {
        "architecture_profile": ARCHITECTURE_REF,
        "trainer_profile": TRAINER_REF,
        "adapter_profile": adapter_ref,
    }


def _configuration_hashes(condition: str) -> dict[str, str]:
    return {
        key: hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()
        for key, value in _configuration_refs(condition).items()
    }


def _provenance_hashes() -> dict[str, str]:
    labels = (
        "dataset",
        "split",
        "task-config",
        "model-config",
        "training-config",
        "component-basis",
    )
    keys = (
        "dataset_sha256",
        "split_sha256",
        "task_config_sha256",
        "model_config_sha256",
        "training_config_sha256",
        "component_basis_sha256",
    )

    return {
        key: hashlib.sha256(
            f"technical-stage7:{label}".encode()
        ).hexdigest()
        for key, label in zip(keys, labels, strict=True)
    }


def _build_and_load_cache(
    *,
    output_root: Path,
    stage3: Stage3AvailabilityIndex,
    teacher_seed: int,
    phase: str,
    condition: str,
    input_ids: tuple[str, ...],
    raw_logits: torch.Tensor,
):
    cache_condition_id = _condition_id(
        stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        condition=condition,
    )

    teacher_reference = {
        "record_type": "teacher_reference",
        "schema_version": "teacher_reference/v1",
        "condition_id": cache_condition_id,
        "record_sha256": hashlib.sha256(
            (
                "technical-stage7-teacher:"
                f"{teacher_seed}:{phase}:{condition}"
            ).encode()
        ).hexdigest(),
    }

    provenance = _provenance_hashes()
    prefix = f"{condition}/cache"

    built = build_target_cache(
        output_root=output_root,
        manifest_relative_path=f"{prefix}/manifest.json",
        payload_relative_path=f"{prefix}/payload.bin",
        completion_relative_path=f"{prefix}/completion.json",
        manifest_id=f"technical-stage7-{condition}-cache/v1",
        ordering_ref=ORDERING_REF,
        expected_example_count=FULL_DOMAIN_EXAMPLE_COUNT,
        expected_class_count=2,
        teacher_reference=teacher_reference,
        provenance_hashes=provenance,
        batches=(
            TargetCacheBatch(
                input_ids=input_ids,
                raw_logits=raw_logits,
            ),
        ),
        technical_fixture=True,
        stage4_record_serializable=False,
        expected_input_ids=input_ids,
    )

    expected_kind = (
        "teacher_argmax"
        if condition == "hard_target"
        else "teacher_logits"
    )

    loaded = load_target_cache(
        output_root=output_root,
        manifest_relative_path=f"{prefix}/manifest.json",
        expected_input_ids=input_ids,
        expected_teacher_reference=teacher_reference,
        expected_provenance_hashes=provenance,
        expected_stage4_cache_kind=expected_kind,
    )

    return built, loaded


def _hard_settings(
    config: TechnicalDistillationFixtureConfig,
    *,
    stop_step: int,
) -> TrainerSettingsBundle:
    return TrainerSettingsBundle(
        model={
            "profile_id": ARCHITECTURE_REF,
            "technical_fixture": True,
        },
        loss={
            "loss_kind": "cross_entropy",
            "reduction": "mean",
        },
        optimizer_schedule={
            "technical_fixture": True,
            "learning_rate": config.hard_learning_rate,
        },
        stop={
            "technical_fixture": True,
            "stop_step": stop_step,
        },
    )


def _soft_settings(
    config: TechnicalDistillationFixtureConfig,
    *,
    policy: TechnicalSoftPolicy,
    stop_step: int,
) -> TrainerSettingsBundle:
    return TrainerSettingsBundle(
        model={
            "profile_id": ARCHITECTURE_REF,
            "technical_fixture": True,
        },
        loss={
            "loss_kind": SOFT_LOSS_KIND,
            "policy": policy,
            "reduction": "mean",
        },
        optimizer_schedule={
            "technical_fixture": True,
            "learning_rate": config.soft_learning_rate,
        },
        stop={
            "technical_fixture": True,
            "stop_step": stop_step,
        },
    )


def _soft_policy(
    *,
    stage3: Stage3AvailabilityIndex,
    teacher_seed: int,
    phase: str,
    ordered_input_ids_sha256: str,
    tolerance: float,
) -> TechnicalSoftPolicy:
    return TechnicalSoftPolicy(
        schema_version=TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
        policy_ref="technical-stage7-soft-policy/v1",
        status=TECHNICAL_POLICY_STATUS,
        scientific_data=False,
        production_eligible=False,
        resolves_ud006=False,
        representation=SoftRepresentationMetadata(
            representation_ref=(
                "technical-stage7-centred-logits/v1"
            ),
            cache_kind="teacher_logits",
            centering_ref=CENTRING_REF,
            teacher_condition_id=_condition_id(
                stage3,
                teacher_seed=teacher_seed,
                phase=phase,
                condition="soft_target",
            ),
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ordered_input_ids_sha256,
            temperature_candidate=None,
            normalization_candidate_ref=None,
        ),
        tolerance=TechnicalToleranceMetadata(
            metric_ref=TECHNICAL_SOFT_DISCREPANCY_METRIC,
            comparison="less_than_or_equal",
            candidate_value=float(tolerance),
            status=TECHNICAL_POLICY_STATUS,
        ),
        argmax_requirement=TechnicalArgmaxRequirementMetadata(
            requirement_ref=(
                "technical-stage7-soft-argmax-requirement/v1"
            ),
            candidate_required=True,
            status=TECHNICAL_POLICY_STATUS,
        ),
    )


def _run_attempt(
    *,
    output_root: Path,
    stage3: Stage3AvailabilityIndex,
    teacher_seed: int,
    phase: str,
    condition: str,
    initialization: int,
    loaded_cache,
    cache_manifest_sha256: str,
    lifecycle: TrainerLifecycle,
    settings: TrainerSettingsBundle,
    training_inputs: torch.Tensor,
    safety_step_limit: int,
):
    identity = build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        distillation_condition=condition,
        student_initialization=initialization,
        attempt_index=0,
        retry_index=0,
    )

    prepared = lifecycle.prepare(
        cache=loaded_cache,
        model_seed=identity.training_seed.seed_value,
        device="cpu",
        settings=settings,
    )

    result = lifecycle.run_technical(
        prepared=prepared,
        training_inputs=training_inputs,
        configuration_refs=_configuration_refs(condition),
        technical_safety_step_limit=safety_step_limit,
    )

    outcome_kind, failure_detail = outcome_from_training_result(
        result
    )

    checkpoint_artifact = None
    checkpoint_evidence = None

    if outcome_kind == "succeeded":
        checkpoint_path = (
            output_root
            / condition
            / "checkpoints"
            / f"initialization-{initialization}.pt"
        )

        checkpoint_evidence = save_technical_resume_checkpoint(
            checkpoint_path,
            prepared=prepared,
            snapshot=TechnicalLoopSnapshot(
                updates_completed=result.updates_completed,
                trajectory=result.trajectory,
                outer_training_mode=prepared.model.training,
            ),
            attempt_identity=identity,
            stage3=stage3,
            configuration_hashes=_configuration_hashes(
                condition
            ),
            target_cache_manifest_sha256=(
                cache_manifest_sha256
            ),
        )

        checkpoint_artifact = _artifact(
            path=(
                "technical/stage7/"
                f"{condition}/"
                f"initialization-{initialization}.pt"
            ),
            sha256=checkpoint_evidence.file_sha256,
            storage_class="external_checkpoint",
        )

    attempt_record = emit_technical_attempt_record(
        stage3=stage3,
        attempt_identity=identity,
        target_cache_reference={
            "record_type": "teacher_output_cache",
            "schema_version": "teacher_output_cache/v1",
            "condition_id": _condition_id(
                stage3,
                teacher_seed=teacher_seed,
                phase=phase,
                condition=condition,
            ),
            "record_sha256": cache_manifest_sha256,
        },
        outcome_kind=outcome_kind,
        student_architecture_ref=ARCHITECTURE_REF,
        replication_policy_ref=REPLICATION_REF,
        training_config_ref=TRAINER_REF,
        training_log=_artifact(
            path=(
                "technical/stage7/"
                f"{condition}/"
                f"initialization-{initialization}.json"
            ),
            sha256=_canonical_training_log_sha256(
                result
            ),
            storage_class="external_log",
        ),
        model_checkpoint=checkpoint_artifact,
        failure_detail=failure_detail,
    )

    sealed_attempt = copy.deepcopy(
        attempt_record
    )
    sealed_attempt["record_status"] = "sealed"

    outputs = None

    if outcome_kind == "succeeded":
        with torch.no_grad():
            outputs = prepared.model(
                training_inputs
            ).detach().clone()

    return {
        "identity": identity,
        "attempt_record": sealed_attempt,
        "result": result,
        "outputs": outputs,
        "checkpoint": checkpoint_evidence,
    }


def _validate_selected_teacher_cell(
    *,
    stage3: Stage3AvailabilityIndex,
    teacher_seed: int,
    phase: str,
) -> None:
    if stage3.availability(
        teacher_seed,
        phase,
    ) != "selected":
        raise Stage7DistillationError(
            "technical fixture requires public metadata for a selected "
            "Stage 3 teacher-phase cell"
        )


def run_technical_distillation_fixture(
    *,
    output_root: str | Path,
    stage3: Stage3AvailabilityIndex,
    teacher_seed: int,
    phase: str,
    config: TechnicalDistillationFixtureConfig,
) -> dict[str, Any]:
    """Exercise hard and soft technical distillation through accepted interfaces."""
    if not isinstance(
        stage3,
        Stage3AvailabilityIndex,
    ):
        raise Stage7DistillationError(
            "stage3 must be Stage3AvailabilityIndex"
        )

    if not isinstance(
        config,
        TechnicalDistillationFixtureConfig,
    ):
        raise Stage7DistillationError(
            "config must be TechnicalDistillationFixtureConfig"
        )

    _validate_selected_teacher_cell(
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
    )

    physical_root = Path(
        output_root
    ).expanduser().resolve(strict=False)

    physical_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if any(physical_root.iterdir()):
        raise Stage7DistillationError(
            "technical distillation fixture requires an empty isolated root"
        )

    decisions = (
        torch.arange(
            FULL_DOMAIN_EXAMPLE_COUNT,
            dtype=torch.int64,
        )
        % 2
    )

    one_hot = torch.nn.functional.one_hot(
        decisions,
        num_classes=2,
    ).to(torch.float64)

    raw_logits = (
        4.0 * one_hot
        - 2.0
    )

    input_ids = tuple(
        f"technical-stage7-input-{index:05d}"
        for index in range(
            FULL_DOMAIN_EXAMPLE_COUNT
        )
    )

    hard_built, hard_cache = (
        _build_and_load_cache(
            output_root=physical_root,
            stage3=stage3,
            teacher_seed=teacher_seed,
            phase=phase,
            condition="hard_target",
            input_ids=input_ids,
            raw_logits=raw_logits,
        )
    )

    soft_built, soft_cache = (
        _build_and_load_cache(
            output_root=physical_root,
            stage3=stage3,
            teacher_seed=teacher_seed,
            phase=phase,
            condition="soft_target",
            input_ids=input_ids,
            raw_logits=raw_logits,
        )
    )

    hard_cache_sha256 = _file_sha256(
        hard_built.manifest_path
    )
    soft_cache_sha256 = _file_sha256(
        soft_built.manifest_path
    )

    hard_manifest = (
        hard_cache.manifest.to_mapping()
    )
    soft_manifest = (
        soft_cache.manifest.to_mapping()
    )

    hard_order_sha256 = hard_manifest[
        "input_order"
    ]["ordered_input_ids_sha256"]

    soft_order_sha256 = soft_manifest[
        "input_order"
    ]["ordered_input_ids_sha256"]

    if hard_order_sha256 != soft_order_sha256:
        raise Stage7DistillationError(
            "hard/soft technical caches must share the same input order"
        )

    hard_events = []

    hard_lifecycle = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=HardTargetAdapter(),
        loss_adapter=HardLabelLossAdapter(),
        optimizer_schedule_factory=(
            _optimizer_factory
        ),
        stop_rule=_stop_rule,
        recorder=hard_events.append,
    )

    hard_inputs = one_hot.detach().clone()

    hard_success = _run_attempt(
        output_root=physical_root,
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        condition="hard_target",
        initialization=0,
        loaded_cache=hard_cache,
        cache_manifest_sha256=(
            hard_cache_sha256
        ),
        lifecycle=hard_lifecycle,
        settings=_hard_settings(
            config,
            stop_step=config.technical_stop_step,
        ),
        training_inputs=hard_inputs,
        safety_step_limit=(
            config.technical_safety_step_limit
        ),
    )

    hard_failure = _run_attempt(
        output_root=physical_root,
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        condition="hard_target",
        initialization=1,
        loaded_cache=hard_cache,
        cache_manifest_sha256=(
            hard_cache_sha256
        ),
        lifecycle=hard_lifecycle,
        settings=_hard_settings(
            config,
            stop_step=(
                config.technical_safety_step_limit
                + 1
            ),
        ),
        training_inputs=hard_inputs,
        safety_step_limit=(
            config.technical_safety_step_limit
        ),
    )

    if hard_success["outputs"] is None:
        raise Stage7DistillationError(
            "hard success path produced no dense output"
        )

    hard_teacher = CanonicalDecisionVector(
        role="direct_teacher",
        condition_id=_condition_id(
            stage3,
            teacher_seed=teacher_seed,
            phase=phase,
            condition="direct_teacher",
        ),
        ordering_ref=ORDERING_REF,
        ordered_input_ids_sha256=(
            hard_order_sha256
        ),
        decisions=tuple(
            hard_cache.argmax.tolist()
        ),
    )

    hard_student_values = tuple(
        hard_success["outputs"]
        .argmax(dim=-1)
        .to(torch.int64)
        .tolist()
    )

    hard_evaluation = (
        evaluate_hard_target_eligibility(
            teacher=hard_teacher,
            student=CanonicalDecisionVector(
                role="hard_target_student",
                condition_id=(
                    hard_success[
                        "attempt_record"
                    ]["condition_id"]
                ),
                ordering_ref=ORDERING_REF,
                ordered_input_ids_sha256=(
                    hard_order_sha256
                ),
                decisions=hard_student_values,
            ),
            stage3=stage3,
        )
    )

    hard_positive_assessment = (
        assess_hard_attempt(
            attempt_record=(
                hard_success[
                    "attempt_record"
                ]
            ),
            stage3=stage3,
            evaluation=hard_evaluation,
        )
    )

    hard_failed_assessment = (
        assess_hard_attempt(
            attempt_record=(
                hard_failure[
                    "attempt_record"
                ]
            ),
            stage3=stage3,
            evaluation=None,
        )
    )

    hard_ledger = HardAttemptLedger()
    hard_ledger.add(
        hard_positive_assessment
    )
    hard_ledger.add(
        hard_failed_assessment
    )

    hard_dense_sha256 = hashlib.sha256(
        canonical_decision_bytes(
            hard_student_values
        )
    ).hexdigest()

    hard_sealing = (
        generate_hard_sealing_evidence(
            assessment=(
                hard_positive_assessment
            ),
            stage3=stage3,
            checkpoint=(
                hard_positive_assessment
                .attempt_record[
                    "payload"
                ][
                    "model_checkpoint"
                ]
            ),
            dense_output=_artifact(
                path=(
                    "technical/stage7/"
                    "hard_target/"
                    "initialization-0-dense.bin"
                ),
                sha256=hard_dense_sha256,
                storage_class=(
                    "external_large_object"
                ),
            ),
            architecture_ref=(
                ARCHITECTURE_REF
            ),
        )
    )

    hard_positive_gate = (
        circuit_release_gate(
            assessment=(
                hard_positive_assessment
            ),
            sealing=hard_sealing,
        )
    )

    hard_failed_gate = (
        circuit_release_gate(
            assessment=(
                hard_failed_assessment
            ),
            sealing=None,
        )
    )

    policy = _soft_policy(
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        ordered_input_ids_sha256=(
            soft_order_sha256
        ),
        tolerance=config.soft_tolerance,
    )

    soft_events = []

    soft_lifecycle = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=(
            TechnicalSoftTargetAdapter(
                policy=policy,
                stage3=stage3,
            )
        ),
        loss_adapter=(
            GaugeInvariantSoftLossAdapter(
                policy=policy
            )
        ),
        optimizer_schedule_factory=(
            _optimizer_factory
        ),
        stop_rule=_stop_rule,
        recorder=soft_events.append,
    )

    soft_inputs = (
        soft_cache
        .centred_logits
        .detach()
        .clone()
    )

    soft_success = _run_attempt(
        output_root=physical_root,
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        condition="soft_target",
        initialization=0,
        loaded_cache=soft_cache,
        cache_manifest_sha256=(
            soft_cache_sha256
        ),
        lifecycle=soft_lifecycle,
        settings=_soft_settings(
            config,
            policy=policy,
            stop_step=config.technical_stop_step,
        ),
        training_inputs=soft_inputs,
        safety_step_limit=(
            config.technical_safety_step_limit
        ),
    )

    soft_failure = _run_attempt(
        output_root=physical_root,
        stage3=stage3,
        teacher_seed=teacher_seed,
        phase=phase,
        condition="soft_target",
        initialization=1,
        loaded_cache=soft_cache,
        cache_manifest_sha256=(
            soft_cache_sha256
        ),
        lifecycle=soft_lifecycle,
        settings=_soft_settings(
            config,
            policy=policy,
            stop_step=(
                config.technical_safety_step_limit
                + 1
            ),
        ),
        training_inputs=soft_inputs,
        safety_step_limit=(
            config.technical_safety_step_limit
        ),
    )

    if soft_success["outputs"] is None:
        raise Stage7DistillationError(
            "soft success path produced no dense output"
        )

    soft_teacher = CanonicalSoftOutput(
        role="soft_target_teacher",
        condition_id=_condition_id(
            stage3,
            teacher_seed=teacher_seed,
            phase=phase,
            condition="soft_target",
        ),
        ordering_ref=ORDERING_REF,
        ordered_input_ids_sha256=(
            soft_order_sha256
        ),
        logits=soft_cache.centred_logits,
        record_status="sealed",
    )

    soft_evaluation = (
        evaluate_soft_target_eligibility(
            teacher=soft_teacher,
            student=CanonicalSoftOutput(
                role="soft_target_student",
                condition_id=(
                    soft_success[
                        "attempt_record"
                    ]["condition_id"]
                ),
                ordering_ref=ORDERING_REF,
                ordered_input_ids_sha256=(
                    soft_order_sha256
                ),
                logits=(
                    soft_success[
                        "outputs"
                    ]
                ),
                record_status="sealed",
            ),
            policy=policy,
            stage3=stage3,
        )
    )

    soft_positive_assessment = (
        assess_soft_attempt(
            attempt_record=(
                soft_success[
                    "attempt_record"
                ]
            ),
            stage3=stage3,
            evaluation=soft_evaluation,
        )
    )

    soft_failed_assessment = (
        assess_soft_attempt(
            attempt_record=(
                soft_failure[
                    "attempt_record"
                ]
            ),
            stage3=stage3,
            evaluation=None,
        )
    )

    soft_ledger = SoftAttemptLedger()
    soft_ledger.add(
        soft_positive_assessment
    )
    soft_ledger.add(
        soft_failed_assessment
    )

    soft_sealing = (
        generate_soft_sealing_evidence(
            assessment=(
                soft_positive_assessment
            ),
            stage3=stage3,
            checkpoint=(
                soft_positive_assessment
                .attempt_record[
                    "payload"
                ][
                    "model_checkpoint"
                ]
            ),
            dense_output=_artifact(
                path=(
                    "technical/stage7/"
                    "soft_target/"
                    "initialization-0-centred.bin"
                ),
                sha256=(
                    soft_evaluation
                    .student_soft_output_sha256
                ),
                storage_class=(
                    "external_large_object"
                ),
            ),
            architecture_ref=(
                ARCHITECTURE_REF
            ),
        )
    )

    soft_positive_gate = (
        soft_circuit_release_gate(
            assessment=(
                soft_positive_assessment
            ),
            sealing=soft_sealing,
        )
    )

    soft_failed_gate = (
        soft_circuit_release_gate(
            assessment=(
                soft_failed_assessment
            ),
            sealing=None,
        )
    )

    if not hard_positive_gate.allowed:
        raise Stage7DistillationError(
            "eligible hard attempt was not released"
        )

    if hard_failed_gate.allowed:
        raise Stage7DistillationError(
            "failed hard attempt escaped passed-only gate"
        )

    if not soft_positive_gate.allowed:
        raise Stage7DistillationError(
            "eligible soft attempt was not released"
        )

    if soft_failed_gate.allowed:
        raise Stage7DistillationError(
            "failed soft attempt escaped passed-only gate"
        )

    if hard_ledger.attempt_count != 2:
        raise Stage7DistillationError(
            "hard attempt accounting lost an initialization"
        )

    if soft_ledger.attempt_count != 2:
        raise Stage7DistillationError(
            "soft attempt accounting lost an initialization"
        )

    if (
        hard_positive_assessment.classification
        != "eligible"
    ):
        raise Stage7DistillationError(
            "hard positive fixture did not pass full-domain eligibility"
        )

    if (
        hard_failed_assessment.classification
        != "training_failure"
    ):
        raise Stage7DistillationError(
            "hard failed attempt lost its failure classification"
        )

    if (
        soft_positive_assessment.status
        != "eligible"
    ):
        raise Stage7DistillationError(
            "soft positive fixture did not pass full-domain eligibility"
        )

    if "training_failure" not in (
        soft_failed_assessment.failure_kinds
    ):
        raise Stage7DistillationError(
            "soft failed attempt lost its failure classification"
        )

    hard_attempts = (
        hard_positive_assessment,
        hard_failed_assessment,
    )
    soft_attempts = (
        soft_positive_assessment,
        soft_failed_assessment,
    )

    return {
        "schema_version": (
            DISTILLATION_INTEGRATION_SCHEMA_VERSION
        ),
        "classification": (
            "synthetic_technical_only"
        ),
        "scientific_data": False,
        "production_eligible": False,
        "production_default": False,
        "resolves_decisions": [],
        "shared_trainer_reference": (
            SHARED_TRAINER_REFERENCE
        ),
        "target_cache_builder_reference": (
            TARGET_CACHE_BUILDER_REFERENCE
        ),
        "target_cache_loader_reference": (
            TARGET_CACHE_LOADER_REFERENCE
        ),
        "teacher_seed": teacher_seed,
        "phase": phase,
        "full_domain_example_count": (
            FULL_DOMAIN_EXAMPLE_COUNT
        ),
        "ordered_input_ids_sha256": (
            hard_order_sha256
        ),
        "direct_teacher_condition_id": (
            hard_teacher.condition_id
        ),
        "hard": {
            "target_cache_manifest_sha256": (
                hard_cache_sha256
            ),
            "target_cache_kind": (
                hard_lifecycle
                .target_cache_kind
            ),
            "attempt_count": (
                hard_ledger.attempt_count
            ),
            "attempted_initializations": [
                0,
                1,
            ],
            "attempt_record_sha256": [
                attempt_record_sha256(
                    item.attempt_record
                )
                for item in hard_attempts
            ],
            "classifications": [
                item.classification
                for item in hard_attempts
            ],
            "eligible_count": (
                hard_ledger
                .classification_count(
                    "eligible"
                )
            ),
            "training_failure_count": (
                hard_ledger
                .classification_count(
                    "training_failure"
                )
            ),
            "sealed_dense_model_sha256": (
                hard_sealing
                .stage4_record_sha256
            ),
            "eligible_release_allowed": (
                hard_positive_gate.allowed
            ),
            "failed_release_allowed": (
                hard_failed_gate.allowed
            ),
            "eligible_release_reason": (
                hard_positive_gate.reason
            ),
            "failed_release_reason": (
                hard_failed_gate.reason
            ),
        },
        "soft": {
            "target_cache_manifest_sha256": (
                soft_cache_sha256
            ),
            "target_cache_kind": (
                soft_lifecycle
                .target_cache_kind
            ),
            "attempt_count": (
                soft_ledger.attempt_count
            ),
            "attempted_initializations": [
                0,
                1,
            ],
            "attempt_record_sha256": [
                attempt_record_sha256(
                    item.attempt_record
                )
                for item in soft_attempts
            ],
            "statuses": [
                item.status
                for item in soft_attempts
            ],
            "failure_kinds": [
                list(item.failure_kinds)
                for item in soft_attempts
            ],
            "eligible_count": sum(
                item.status == "eligible"
                for item in soft_attempts
            ),
            "training_failure_count": (
                soft_ledger.failure_count(
                    "training_failure"
                )
            ),
            "sealed_dense_model_sha256": (
                soft_sealing
                .stage4_record_sha256
            ),
            "eligible_release_allowed": (
                soft_positive_gate.allowed
            ),
            "failed_release_allowed": (
                soft_failed_gate.allowed
            ),
            "eligible_release_reason": (
                soft_positive_gate.reason
            ),
            "failed_release_reason": (
                soft_failed_gate.reason
            ),
        },
        "hard_soft_condition_ids_distinct": (
            {
                item.attempt_record[
                    "condition_id"
                ]
                for item in hard_attempts
            }.isdisjoint(
                {
                    item.attempt_record[
                        "condition_id"
                    ]
                    for item in soft_attempts
                }
            )
        ),
        "failed_attempts_preserved": True,
        "passed_only_sealing": True,
        "real_optimization_configuration": False,
        "registered_fixture_execution": False,
    }

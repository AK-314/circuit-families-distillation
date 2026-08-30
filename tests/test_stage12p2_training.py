from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from circuit_families.stage4_condition_identity import Stage3AvailabilityIndex
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
from circuit_families.stage5bc.target_cache import TargetCacheManifest
from circuit_families.stage6b import HardLabelLossAdapter
from circuit_families.stage12p2 import (
    ArchitectureModelConstructor,
    ArchitectureRecord,
    ArchitectureRegistry,
    BuilderDescriptor,
    FinalPositionStudentModel,
    StudentTrainingContractError,
    bind_student_training_identity,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE3_REGISTRY = ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
CACHE_MANIFEST = ROOT / "tests/fixtures/stage5bc/technical_cache_manifest_v1.json"

BUILDER_REF = "technical-p2-training-test-builder/v1"
BUILDER_SHA256 = "a" * 64


class _TinySequenceModel(torch.nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, class_count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embedding(inputs)


class _TinyBuilder:
    descriptor = BuilderDescriptor(
        builder_ref=BUILDER_REF,
        implementation_sha256=BUILDER_SHA256,
    )

    def validate_record(self, record: ArchitectureRecord) -> None:
        if record.builder_ref != BUILDER_REF:
            raise ValueError("wrong builder")

    def build(
        self,
        *,
        record: ArchitectureRecord,
        seed: int,
        device: str | torch.device,
    ) -> torch.nn.Module:
        torch.manual_seed(seed)
        return _TinySequenceModel(class_count=int(record.dimensions["d_vocab_out"])).to(device)


def _record(name: str) -> ArchitectureRecord:
    return ArchitectureRecord(
        family="technical",
        name=name,
        version="v1",
        compatibility={"task_family": "technical"},
        dimensions={
            "n_layers": 1,
            "d_model": 4,
            "d_vocab_out": 3,
        },
        activation="relu",
        normalization=None,
        positional_embedding_type="standard",
        parameter_count=24,
        searchable_component_count=2,
        component_type_counts={
            "attention_head": 1,
            "mlp_neuron": 1,
        },
        initialization_ref="technical-p2-initialization/v1",
        builder_ref=BUILDER_REF,
        builder_sha256=BUILDER_SHA256,
        scientific_data=False,
        production_eligible=False,
    )


def _registry(*records: ArchitectureRecord) -> ArchitectureRegistry:
    registry = ArchitectureRegistry(
        builders={BUILDER_REF: _TinyBuilder()},
    )
    for record in records:
        registry.register(record)
    return registry


def _stage3() -> Stage3AvailabilityIndex:
    return Stage3AvailabilityIndex.from_registry(
        json.loads(STAGE3_REGISTRY.read_text(encoding="utf-8"))
    )


def _attempt(stage3: Stage3AvailabilityIndex):
    return build_student_attempt_identity(
        stage3=stage3,
        teacher_seed=1,
        phase="stable post-grokking",
        distillation_condition="hard_target",
        student_initialization=0,
        attempt_index=0,
        retry_index=0,
    )


def _identity(
    *,
    stage3: Stage3AvailabilityIndex,
    architecture: ArchitectureRecord,
):
    return bind_student_training_identity(
        stage3=stage3,
        stage5_attempt=_attempt(stage3),
        task_identity_sha256="1" * 64,
        target_cache_manifest=TargetCacheManifest.from_json_file(CACHE_MANIFEST),
        architecture_record=architecture,
        model_seed_id="technical-model-seed/v1",
        model_seed=17,
        training_config_ref="technical-p2-training/v1",
        training_config={
            "technical_fixture": True,
            "candidate_only": True,
        },
        backend_ref="technical-p2-backend/v1",
        backend_qualification={
            "backend_id": "cpu",
            "technical_fixture": True,
            "exact_resume_supported": True,
        },
    )


def _model_settings(record: ArchitectureRecord) -> dict[str, str]:
    return {
        "architecture_ref": record.architecture_ref,
        "architecture_record_sha256": record.to_mapping()["record_sha256"],
    }


def _optimizer_factory(*, model, settings):
    return OptimizerScheduleBundle(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        scheduler=None,
    )


class _Cache:
    def __init__(self) -> None:
        self.argmax = torch.tensor([0, 1], dtype=torch.int64)
        self.logits = torch.tensor(
            [
                [2.0, -1.0, -1.0],
                [-1.0, 2.0, -1.0],
            ],
            dtype=torch.float32,
        )

    def stage4_view(self, cache_kind: str) -> torch.Tensor:
        if cache_kind == "teacher_argmax":
            return self.argmax
        if cache_kind == "teacher_logits":
            return self.logits
        raise AssertionError(cache_kind)


def _soft_loss(*, outputs, targets, settings):
    student = outputs - outputs.mean(dim=-1, keepdim=True)
    teacher = targets.values - targets.values.mean(dim=-1, keepdim=True)
    return (student - teacher).square().mean()


def test_p2_identity_keeps_model_seed_distinct_from_stage5_training_seed() -> None:
    stage3 = _stage3()
    record = _record("alpha")
    identity = _identity(stage3=stage3, architecture=record)

    assert identity.model_seed == 17
    assert identity.model_seed_id == "technical-model-seed/v1"
    assert identity.model_seed != identity.stage5_attempt.training_seed.seed_value
    assert identity.student_initialization == 0
    assert identity.scientific_data is False
    assert identity.production_eligible is False
    assert identity.to_mapping()["condition"]["distillation_condition"] == "hard_target"


def test_architecture_changes_identity_and_checkpoint_binding() -> None:
    stage3 = _stage3()
    alpha = _identity(stage3=stage3, architecture=_record("alpha"))
    beta = _identity(stage3=stage3, architecture=_record("beta"))

    assert alpha.identity_sha256 != beta.identity_sha256
    assert alpha.checkpoint_configuration_hashes() != beta.checkpoint_configuration_hashes()
    assert (
        alpha.checkpoint_configuration_hashes()["architecture_record_sha256"]
        != beta.checkpoint_configuration_hashes()["architecture_record_sha256"]
    )


def test_identity_rejects_out_of_range_model_seed() -> None:
    stage3 = _stage3()
    identity = _identity(stage3=stage3, architecture=_record("alpha"))

    with pytest.raises(
        StudentTrainingContractError,
        match=r"model_seed must be an integer in \[0, 2\*\*32 - 1\]",
    ):
        replace(identity, model_seed=2**32)


def test_model_constructor_rejects_architecture_substitution_in_settings() -> None:
    alpha = _record("alpha")
    beta = _record("beta")
    registry = _registry(alpha, beta)
    constructor = ArchitectureModelConstructor.from_record(
        registry=registry,
        record=alpha,
    )

    with pytest.raises(
        StudentTrainingContractError,
        match="architecture_ref mismatch",
    ):
        constructor(
            seed=3,
            device=torch.device("cpu"),
            settings=_model_settings(beta),
        )


def test_model_constructor_rejects_stale_architecture_hash() -> None:
    record = _record("alpha")
    registry = _registry(record)
    constructor = ArchitectureModelConstructor.from_record(
        registry=registry,
        record=record,
    )
    settings = _model_settings(record)
    settings["architecture_record_sha256"] = "f" * 64

    with pytest.raises(
        StudentTrainingContractError,
        match="architecture record hash mismatch",
    ):
        constructor(
            seed=3,
            device=torch.device("cpu"),
            settings=settings,
        )


def test_final_position_adapter_returns_rank_two_dense_logits() -> None:
    model = FinalPositionStudentModel(_TinySequenceModel(class_count=3))
    inputs = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.int64,
    )

    sequence = model.base_model(inputs)
    expected = sequence[:, -1, :]
    actual = model(inputs)

    assert actual.ndim == 2
    assert actual.shape == (2, 3)
    assert torch.equal(actual, expected)


def test_hard_and_soft_paths_share_one_architecture_constructor() -> None:
    record = _record("alpha")
    registry = _registry(record)
    constructor = ArchitectureModelConstructor.from_record(
        registry=registry,
        record=record,
    )
    settings = _model_settings(record)
    cache = _Cache()

    hard = TrainerLifecycle(
        model_constructor=constructor,
        target_adapter=HardTargetAdapter(),
        loss_adapter=HardLabelLossAdapter(),
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=lambda *, progress, settings: False,
        recorder=lambda event: None,
    )
    soft = TrainerLifecycle(
        model_constructor=constructor,
        target_adapter=SoftTargetAdapter(),
        loss_adapter=_soft_loss,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=lambda *, progress, settings: False,
        recorder=lambda event: None,
    )

    hard_prepared = hard.prepare(
        cache=cache,
        model_seed=7,
        device="cpu",
        settings=TrainerSettingsBundle(
            model=settings,
            loss={
                "loss_kind": "cross_entropy",
                "reduction": "mean",
            },
            optimizer_schedule={"technical_fixture": True},
            stop={"technical_fixture": True},
        ),
    )
    soft_prepared = soft.prepare(
        cache=cache,
        model_seed=7,
        device="cpu",
        settings=TrainerSettingsBundle(
            model=settings,
            loss={"technical_fixture": True},
            optimizer_schedule={"technical_fixture": True},
            stop={"technical_fixture": True},
        ),
    )

    inputs = torch.tensor(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=torch.int64,
    )

    hard_outputs = hard_prepared.model(inputs)
    soft_outputs = soft_prepared.model(inputs)

    assert isinstance(hard_prepared.model, FinalPositionStudentModel)
    assert isinstance(soft_prepared.model, FinalPositionStudentModel)
    assert hard_outputs.shape == (2, 3)
    assert soft_outputs.shape == (2, 3)
    assert hard.target_cache_kind == "teacher_argmax"
    assert soft.target_cache_kind == "teacher_logits"
    assert torch.isfinite(
        hard.compute_loss(
            prepared=hard_prepared,
            outputs=hard_outputs,
        )
    )
    assert torch.isfinite(
        soft.compute_loss(
            prepared=soft_prepared,
            outputs=soft_outputs,
        )
    )

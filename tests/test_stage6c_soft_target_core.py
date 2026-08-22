from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from circuit_families.interpretability.centred_logit_fidelity import (
    centre_logits_across_classes,
)
from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage5bc.student_trainer import (
    OptimizerScheduleBundle,
    PreparedTargets,
    TrainerLifecycle,
    TrainerSettingsBundle,
)
from circuit_families.stage6c import (
    CENTRING_REF,
    SOFT_LOSS_KIND,
    TECHNICAL_POLICY_STATUS,
    TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
    GaugeInvariantSoftLossAdapter,
    SoftRepresentationMetadata,
    SoftTargetLossError,
    SoftTargetPolicyError,
    TechnicalArgmaxRequirementMetadata,
    TechnicalSoftPolicy,
    TechnicalSoftTargetAdapter,
    TechnicalToleranceMetadata,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
ORDERING_REF = "technical-stage6c-soft-order/v1"
ORDERING_SHA256 = "a" * 64


class _Manifest:
    def __init__(self, mapping) -> None:
        self._mapping = copy.deepcopy(mapping)

    def to_mapping(self):
        return copy.deepcopy(self._mapping)


class _SoftCache:
    def __init__(self, logits: torch.Tensor, manifest) -> None:
        self.logits = logits
        self.manifest = _Manifest(manifest)

    def stage4_view(self, cache_kind: str) -> torch.Tensor:
        if cache_kind != "teacher_logits":
            raise AssertionError(cache_kind)
        return self.logits


@pytest.fixture(scope="module")
def stage3() -> Stage3AvailabilityIndex:
    return Stage3AvailabilityIndex.from_registry(
        json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    )


def _condition_id(stage3: Stage3AvailabilityIndex, condition: str) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=1,
            phase="stable post-grokking",
            distillation_condition=condition,
        ),
        stage3,
    )


def _policy(
    stage3: Stage3AvailabilityIndex,
    *,
    condition: str = "soft_target",
    argmax_requirement: TechnicalArgmaxRequirementMetadata | None = None,
) -> TechnicalSoftPolicy:
    return TechnicalSoftPolicy(
        schema_version=TECHNICAL_SOFT_POLICY_SCHEMA_VERSION,
        policy_ref="technical-stage6c-soft-candidate/v1",
        status=TECHNICAL_POLICY_STATUS,
        scientific_data=False,
        production_eligible=False,
        resolves_ud006=False,
        representation=SoftRepresentationMetadata(
            representation_ref="technical-centred-teacher-logits/v1",
            cache_kind="teacher_logits",
            centering_ref=CENTRING_REF,
            teacher_condition_id=_condition_id(stage3, condition),
            ordering_ref=ORDERING_REF,
            ordered_input_ids_sha256=ORDERING_SHA256,
            temperature_candidate=None,
            normalization_candidate_ref=None,
        ),
        tolerance=TechnicalToleranceMetadata(
            metric_ref="technical-centred-logit-mse/v1",
            comparison="less_than_or_equal",
            candidate_value=0.25,
            status=TECHNICAL_POLICY_STATUS,
        ),
        argmax_requirement=argmax_requirement,
    )


def _manifest(policy: TechnicalSoftPolicy) -> dict[str, object]:
    return {
        "input_order": {
            "ordering_ref": policy.representation.ordering_ref,
            "ordered_input_ids_sha256": (
                policy.representation.ordered_input_ids_sha256
            ),
        },
        "teacher_reference": {
            "condition_id": policy.representation.teacher_condition_id,
        },
    }


def _model_constructor(*, seed: int, device: torch.device, settings):
    torch.manual_seed(seed)
    return torch.nn.Linear(2, 3).to(device)


def _optimizer_factory(*, model: torch.nn.Module, settings):
    return OptimizerScheduleBundle(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        scheduler=None,
    )


def _settings(policy: TechnicalSoftPolicy) -> TrainerSettingsBundle:
    return TrainerSettingsBundle(
        model={"technical_fixture": True},
        loss={
            "loss_kind": SOFT_LOSS_KIND,
            "policy": policy,
            "reduction": "mean",
        },
        optimizer_schedule={"technical_fixture": True},
        stop={"technical_fixture": True},
    )


def _lifecycle(
    policy: TechnicalSoftPolicy,
    stage3: Stage3AvailabilityIndex,
) -> TrainerLifecycle:
    return TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=TechnicalSoftTargetAdapter(policy=policy, stage3=stage3),
        loss_adapter=GaugeInvariantSoftLossAdapter(policy=policy),
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=lambda *, progress, settings: False,
        recorder=lambda event: None,
    )


def _teacher_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [2.0, -1.0, 0.5],
            [-0.5, 1.5, 0.25],
            [0.0, -2.0, 3.0],
        ],
        dtype=torch.float64,
    )


def _student_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [1.75, -0.75, 0.25],
            [-0.25, 1.25, 0.5],
            [0.25, -2.25, 3.0],
        ],
        dtype=torch.float64,
    )


def test_technical_policy_is_explicit_versioned_and_nonproduction(stage3) -> None:
    policy = _policy(stage3)
    first = policy.to_mapping()
    second = policy.to_mapping()

    assert first == second
    assert first["schema_version"] == TECHNICAL_SOFT_POLICY_SCHEMA_VERSION
    assert first["status"] == "technical_candidate_only"
    assert first["scientific_data"] is False
    assert first["production_eligible"] is False
    assert first["resolves_ud006"] is False
    assert first["argmax_requirement"] is None
    assert first["representation"]["temperature_candidate"] is None
    assert first["representation"]["normalization_candidate_ref"] is None
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second,
        sort_keys=True,
        allow_nan=False,
    )


def test_optional_argmax_metadata_remains_technical_only(stage3) -> None:
    candidate = TechnicalArgmaxRequirementMetadata(
        requirement_ref="technical-argmax-candidate/v1",
        candidate_required=True,
        status=TECHNICAL_POLICY_STATUS,
    )
    policy = _policy(stage3, argmax_requirement=candidate)

    assert policy.to_mapping()["argmax_requirement"] == {
        "candidate_required": True,
        "requirement_ref": "technical-argmax-candidate/v1",
        "status": "technical_candidate_only",
    }
    assert policy.resolves_ud006 is False
    assert policy.production_eligible is False


def test_valid_soft_loss_uses_accepted_shared_trainer(stage3) -> None:
    policy = _policy(stage3)
    teacher = _teacher_logits()
    student = _student_logits()
    lifecycle = _lifecycle(policy, stage3)
    prepared = lifecycle.prepare(
        cache=_SoftCache(teacher, _manifest(policy)),
        model_seed=7,
        device="cpu",
        settings=_settings(policy),
    )

    first = lifecycle.compute_loss(prepared=prepared, outputs=student)
    second = lifecycle.compute_loss(prepared=prepared, outputs=student.clone())
    expected = torch.mean(
        (
            centre_logits_across_classes(student)
            - centre_logits_across_classes(teacher)
        ).square()
    )

    assert lifecycle.target_cache_kind == "teacher_logits"
    assert prepared.targets.cache_kind == "teacher_logits"
    assert torch.equal(prepared.targets.values, centre_logits_across_classes(teacher))
    assert torch.equal(first, second)
    assert torch.allclose(first, expected, rtol=0.0, atol=1e-15)
    assert torch.isfinite(first)


def test_per_input_additive_gauge_shift_preserves_targets_loss_and_future_decision(
    stage3,
) -> None:
    policy = _policy(stage3)
    teacher = _teacher_logits()
    student = _student_logits()
    teacher_shift = torch.tensor([[9.0], [-3.5], [0.125]], dtype=torch.float64)
    student_shift = torch.tensor([[-4.0], [7.25], [2.0]], dtype=torch.float64)
    lifecycle = _lifecycle(policy, stage3)
    baseline = lifecycle.prepare(
        cache=_SoftCache(teacher, _manifest(policy)),
        model_seed=3,
        device="cpu",
        settings=_settings(policy),
    )
    shifted = lifecycle.prepare(
        cache=_SoftCache(teacher + teacher_shift, _manifest(policy)),
        model_seed=3,
        device="cpu",
        settings=_settings(policy),
    )

    baseline_loss = lifecycle.compute_loss(prepared=baseline, outputs=student)
    shifted_loss = lifecycle.compute_loss(
        prepared=shifted,
        outputs=student + student_shift,
    )
    candidate_tolerance = policy.tolerance.candidate_value

    assert torch.allclose(baseline.targets.values, shifted.targets.values)
    assert torch.allclose(baseline_loss, shifted_loss)
    assert bool(baseline_loss <= candidate_tolerance) == bool(
        shifted_loss <= candidate_tolerance
    )


@pytest.mark.parametrize(
    ("student", "message"),
    [
        (torch.zeros(3, dtype=torch.float32), "shape"),
        (torch.zeros((2, 3), dtype=torch.float32), "identical shapes"),
    ],
)
def test_wrong_student_shape_is_rejected(stage3, student, message: str) -> None:
    policy = _policy(stage3)
    lifecycle = _lifecycle(policy, stage3)
    prepared = lifecycle.prepare(
        cache=_SoftCache(_teacher_logits(), _manifest(policy)),
        model_seed=1,
        device="cpu",
        settings=_settings(policy),
    )

    with pytest.raises(SoftTargetLossError, match=message):
        lifecycle.compute_loss(prepared=prepared, outputs=student)


def test_wrong_ordering_or_identity_is_rejected(stage3) -> None:
    policy = _policy(stage3)
    adapter = TechnicalSoftTargetAdapter(policy=policy, stage3=stage3)
    wrong_order = _manifest(policy)
    wrong_order["input_order"]["ordered_input_ids_sha256"] = "b" * 64

    with pytest.raises(SoftTargetPolicyError, match="ordering mismatch"):
        adapter(_SoftCache(_teacher_logits(), wrong_order))

    hard_policy = _policy(stage3, condition="hard_target")
    with pytest.raises(SoftTargetPolicyError, match="soft_target teacher identity"):
        TechnicalSoftTargetAdapter(policy=hard_policy, stage3=stage3)


@pytest.mark.parametrize("source", ["student", "teacher"])
def test_nonfinite_logits_are_rejected(stage3, source: str) -> None:
    policy = _policy(stage3)
    teacher = _teacher_logits()
    student = _student_logits()
    if source == "teacher":
        teacher[0, 0] = float("inf")
        with pytest.raises(SoftTargetLossError, match="finite"):
            _lifecycle(policy, stage3).prepare(
                cache=_SoftCache(teacher, _manifest(policy)),
                model_seed=1,
                device="cpu",
                settings=_settings(policy),
            )
    else:
        student[0, 0] = float("nan")
        prepared = _lifecycle(policy, stage3).prepare(
            cache=_SoftCache(teacher, _manifest(policy)),
            model_seed=1,
            device="cpu",
            settings=_settings(policy),
        )
        with pytest.raises(SoftTargetLossError, match="finite"):
            GaugeInvariantSoftLossAdapter(policy=policy)(
                outputs=student,
                targets=prepared.targets,
                settings=_settings(policy).loss,
            )


def test_hard_target_cache_cannot_enter_soft_loss(stage3) -> None:
    policy = _policy(stage3)
    hard_targets = PreparedTargets(
        cache_kind="teacher_argmax",
        values=torch.tensor([0, 1, 2], dtype=torch.int64),
    )

    with pytest.raises(SoftTargetLossError, match="hard targets are forbidden"):
        GaugeInvariantSoftLossAdapter(policy=policy)(
            outputs=_student_logits(),
            targets=hard_targets,
            settings=_settings(policy).loss,
        )


def test_policy_rejects_production_or_ud_resolution_claims(stage3) -> None:
    policy = _policy(stage3)

    with pytest.raises(SoftTargetPolicyError, match="production_eligible=false"):
        replace(policy, production_eligible=True)
    with pytest.raises(SoftTargetPolicyError, match="must not resolve UD-006"):
        replace(policy, resolves_ud006=True)

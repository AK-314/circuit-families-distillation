from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from circuit_families.stage4_condition_identity import (
    ConditionIdentity,
    Stage3AvailabilityIndex,
    build_condition_id,
)
from circuit_families.stage5bc.student_trainer import (
    HardTargetAdapter,
    OptimizerScheduleBundle,
    PreparedTargets,
    SoftTargetAdapter,
    TrainerLifecycle,
    TrainerSettingsBundle,
)
from circuit_families.stage5bc.target_cache import FULL_DOMAIN_EXAMPLE_COUNT
from circuit_families.stage6b import (
    DECISION_VECTOR_HASH_VERSION,
    HARD_ELIGIBILITY_CRITERION,
    CanonicalDecisionVector,
    HardLabelLossAdapter,
    HardTargetEligibilityError,
    HardTargetLossError,
    evaluate_hard_target_eligibility,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "followup/manifests/stage3_teacher_registry_v1.json"
ORDERING_REF = "modular-addition-full-domain-order/v1"
ORDERING_SHA256 = "a" * 64


class _Cache:
    def stage4_view(self, cache_kind: str) -> torch.Tensor:
        if cache_kind == "teacher_argmax":
            return torch.tensor([0, 2, 1, 2], dtype=torch.int64)
        if cache_kind == "teacher_logits":
            return torch.zeros((4, 3), dtype=torch.float32)
        raise AssertionError(cache_kind)


@pytest.fixture(scope="module")
def stage3() -> Stage3AvailabilityIndex:
    return Stage3AvailabilityIndex.from_registry(
        json.loads(REGISTRY.read_text(encoding="utf-8"))
    )


@pytest.fixture(scope="module")
def decisions() -> tuple[int, ...]:
    return tuple(index % 113 for index in range(FULL_DOMAIN_EXAMPLE_COUNT))


def _condition_id(
    stage3: Stage3AvailabilityIndex,
    *,
    condition: str,
    seed: int = 1,
) -> str:
    return build_condition_id(
        ConditionIdentity(
            teacher_seed=seed,
            phase="pre-grokking",
            distillation_condition=condition,
            student_initialization=(0 if condition != "direct_teacher" else None),
        ),
        stage3,
    )


def _vector(
    stage3: Stage3AvailabilityIndex,
    decisions: tuple[int, ...],
    *,
    role: str,
    condition: str,
    seed: int = 1,
    ordering_sha256: str = ORDERING_SHA256,
) -> CanonicalDecisionVector:
    return CanonicalDecisionVector(
        role=role,
        condition_id=_condition_id(stage3, condition=condition, seed=seed),
        ordering_ref=ORDERING_REF,
        ordered_input_ids_sha256=ordering_sha256,
        decisions=decisions,
    )


def _model_constructor(*, seed: int, device: torch.device, settings) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Linear(2, 3).to(device)


def _optimizer_factory(*, model: torch.nn.Module, settings) -> OptimizerScheduleBundle:
    return OptimizerScheduleBundle(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
    )


def test_hard_loss_plugs_into_existing_shared_trainer_lifecycle() -> None:
    lifecycle = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=HardTargetAdapter(),
        loss_adapter=HardLabelLossAdapter(),
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=lambda *, progress, settings: False,
        recorder=lambda event: None,
    )
    prepared = lifecycle.prepare(
        cache=_Cache(),
        model_seed=7,
        device="cpu",
        settings=TrainerSettingsBundle(
            model={"technical_fixture": True},
            loss={"loss_kind": "cross_entropy", "reduction": "mean"},
            optimizer_schedule={"technical_fixture": True},
            stop={"technical_fixture": True},
        ),
    )
    outputs = torch.tensor(
        [[3.0, 0.0, -1.0], [0.0, -1.0, 2.0], [0.0, 2.0, -1.0], [0.0, -1.0, 2.0]],
        dtype=torch.float32,
    )

    loss = lifecycle.compute_loss(prepared=prepared, outputs=outputs)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert lifecycle.target_cache_kind == "teacher_argmax"


def test_hard_loss_has_no_implicit_settings_and_rejects_soft_targets() -> None:
    adapter = HardLabelLossAdapter()
    hard = PreparedTargets(
        cache_kind="teacher_argmax",
        values=torch.tensor([0, 1], dtype=torch.int64),
    )
    soft = SoftTargetAdapter()(_Cache())
    outputs = torch.zeros((2, 3), dtype=torch.float32)

    with pytest.raises(HardTargetLossError, match="require exactly"):
        adapter(outputs=outputs, targets=hard, settings={})
    with pytest.raises(HardTargetLossError, match="teacher_argmax"):
        adapter(
            outputs=torch.zeros((4, 3), dtype=torch.float32),
            targets=soft,
            settings={"loss_kind": "cross_entropy", "reduction": "mean"},
        )


def test_exact_full_domain_equality_is_eligible(stage3, decisions) -> None:
    teacher = _vector(
        stage3,
        decisions,
        role="direct_teacher",
        condition="direct_teacher",
    )
    student = _vector(
        stage3,
        decisions,
        role="hard_target_student",
        condition="hard_target",
    )

    first = evaluate_hard_target_eligibility(
        teacher=teacher,
        student=student,
        stage3=stage3,
    )
    second = evaluate_hard_target_eligibility(
        teacher=teacher,
        student=student,
        stage3=stage3,
    )

    assert first == second
    assert first.agreement_count == FULL_DOMAIN_EXAMPLE_COUNT
    assert first.total_count == FULL_DOMAIN_EXAMPLE_COUNT
    assert first.eligible is True
    assert first.teacher_decisions_sha256 == first.student_decisions_sha256
    assert first.criterion == HARD_ELIGIBILITY_CRITERION
    assert first.decision_hash_version == DECISION_VECTOR_HASH_VERSION
    assert first.scientific_data is False
    assert first.production_eligible is False
    assert first.to_mapping()["ordered_input_ids_sha256"] == ORDERING_SHA256


def test_one_changed_decision_is_ineligible_12768_of_12769(stage3, decisions) -> None:
    changed = list(decisions)
    changed[-1] = (changed[-1] + 1) % 113
    evidence = evaluate_hard_target_eligibility(
        teacher=_vector(
            stage3,
            decisions,
            role="direct_teacher",
            condition="direct_teacher",
        ),
        student=_vector(
            stage3,
            tuple(changed),
            role="hard_target_student",
            condition="hard_target",
        ),
        stage3=stage3,
    )

    assert evidence.agreement_count == FULL_DOMAIN_EXAMPLE_COUNT - 1
    assert evidence.total_count == FULL_DOMAIN_EXAMPLE_COUNT
    assert evidence.eligible is False
    assert evidence.teacher_decisions_sha256 != evidence.student_decisions_sha256


@pytest.mark.parametrize("role", ["direct_teacher", "hard_target_student"])
def test_wrong_length_fails(stage3, decisions, role: str) -> None:
    condition = "direct_teacher" if role == "direct_teacher" else "hard_target"
    short = _vector(
        stage3,
        decisions[:-1],
        role=role,
        condition=condition,
    )
    teacher = _vector(
        stage3,
        decisions,
        role="direct_teacher",
        condition="direct_teacher",
    )
    student = _vector(
        stage3,
        decisions,
        role="hard_target_student",
        condition="hard_target",
    )

    with pytest.raises(HardTargetEligibilityError, match="exactly 12769"):
        evaluate_hard_target_eligibility(
            teacher=short if role == "direct_teacher" else teacher,
            student=short if role == "hard_target_student" else student,
            stage3=stage3,
        )


def test_wrong_order_or_order_identity_fails(stage3, decisions) -> None:
    teacher = _vector(
        stage3,
        decisions,
        role="direct_teacher",
        condition="direct_teacher",
    )
    reordered = decisions[1:] + decisions[:1]
    reordered_evidence = evaluate_hard_target_eligibility(
        teacher=teacher,
        student=_vector(
            stage3,
            reordered,
            role="hard_target_student",
            condition="hard_target",
        ),
        stage3=stage3,
    )
    assert reordered_evidence.eligible is False
    assert reordered_evidence.agreement_count < FULL_DOMAIN_EXAMPLE_COUNT

    with pytest.raises(HardTargetEligibilityError, match="input-order identities"):
        evaluate_hard_target_eligibility(
            teacher=teacher,
            student=_vector(
                stage3,
                decisions,
                role="hard_target_student",
                condition="hard_target",
                ordering_sha256="b" * 64,
            ),
            stage3=stage3,
        )


def test_wrong_identity_and_hard_soft_confusion_fail(stage3, decisions) -> None:
    teacher = _vector(
        stage3,
        decisions,
        role="direct_teacher",
        condition="direct_teacher",
    )
    wrong_seed = _vector(
        stage3,
        decisions,
        role="hard_target_student",
        condition="hard_target",
        seed=2,
    )
    with pytest.raises(HardTargetEligibilityError, match="share teacher_seed and phase"):
        evaluate_hard_target_eligibility(
            teacher=teacher,
            student=wrong_seed,
            stage3=stage3,
        )

    soft_identity = replace(
        _vector(
            stage3,
            decisions,
            role="hard_target_student",
            condition="hard_target",
        ),
        condition_id=_condition_id(stage3, condition="soft_target"),
    )
    with pytest.raises(HardTargetEligibilityError, match="depth-4 hard_target"):
        evaluate_hard_target_eligibility(
            teacher=teacher,
            student=soft_identity,
            stage3=stage3,
        )

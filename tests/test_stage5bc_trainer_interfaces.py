from __future__ import annotations

import inspect

import pytest
import torch

from circuit_families.stage5bc.student_trainer import (
    HardTargetAdapter,
    OptimizerScheduleBundle,
    PreparedTargets,
    SoftTargetAdapter,
    TrainerEvent,
    TrainerInterfaceError,
    TrainerLifecycle,
    TrainerProgress,
    TrainerSettingsBundle,
)


class _TechnicalCache:
    def __init__(self) -> None:
        self.argmax = torch.tensor(
            [1, 0, 2, 1],
            dtype=torch.int64,
        )
        self.logits = torch.tensor(
            [
                [-1.0, 1.0, 0.0],
                [2.0, -1.0, -1.0],
                [-2.0, -1.0, 3.0],
                [-0.5, 1.0, -0.5],
            ],
            dtype=torch.float32,
        )

    def stage4_view(self, cache_kind: str) -> torch.Tensor:
        if cache_kind == "teacher_argmax":
            return self.argmax
        if cache_kind == "teacher_logits":
            return self.logits
        raise AssertionError(cache_kind)


def _settings() -> TrainerSettingsBundle:
    return TrainerSettingsBundle(
        model={
            "candidate_width": 2,
            "purpose": "technical_interface_test_only",
        },
        loss={
            "technical_scale": 1.0,
            "purpose": "interface_dispatch_test_only",
        },
        optimizer_schedule={
            "technical_learning_rate": 0.01,
            "scheduler": None,
            "purpose": "interface_dispatch_test_only",
        },
        stop={
            "technical_stop_step": 3,
            "purpose": "interface_dispatch_test_only",
        },
    )


def _model_constructor(
    *,
    seed: int,
    device: torch.device,
    settings,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    width = int(settings["candidate_width"])
    return torch.nn.Linear(width, 3).to(device)


def _optimizer_factory(
    *,
    model: torch.nn.Module,
    settings,
) -> OptimizerScheduleBundle:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(settings["technical_learning_rate"]),
    )
    return OptimizerScheduleBundle(
        optimizer=optimizer,
        scheduler=settings["scheduler"],
    )


def _technical_loss(
    *,
    outputs: torch.Tensor,
    targets: PreparedTargets,
    settings,
) -> torch.Tensor:
    assert targets.cache_kind in {
        "teacher_argmax",
        "teacher_logits",
    }
    return outputs.mean() * float(settings["technical_scale"])


def _technical_stop(
    *,
    progress: TrainerProgress,
    settings,
) -> bool:
    return progress.step >= int(settings["technical_stop_step"])


def test_lifecycle_requires_all_six_policy_components_explicitly() -> None:
    signature = inspect.signature(TrainerLifecycle)

    expected = {
        "model_constructor",
        "target_adapter",
        "loss_adapter",
        "optimizer_schedule_factory",
        "stop_rule",
        "recorder",
    }

    assert set(signature.parameters) == expected

    for parameter in signature.parameters.values():
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_hard_adapter_exposes_argmax_without_owning_a_loop() -> None:
    cache = _TechnicalCache()
    adapter = HardTargetAdapter()

    prepared = adapter(cache)

    assert prepared.cache_kind == "teacher_argmax"
    assert prepared.values.dtype == torch.int64
    assert torch.equal(prepared.values, cache.argmax)
    assert prepared.values.data_ptr() != cache.argmax.data_ptr()


def test_soft_adapter_exposes_centred_logits_without_loss_choice() -> None:
    cache = _TechnicalCache()
    adapter = SoftTargetAdapter()

    prepared = adapter(cache)

    assert prepared.cache_kind == "teacher_logits"
    assert prepared.values.is_floating_point()
    assert torch.equal(prepared.values, cache.logits)
    assert prepared.values.data_ptr() != cache.logits.data_ptr()


def test_hard_and_soft_use_the_same_lifecycle_class() -> None:
    events = []

    hard = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=HardTargetAdapter(),
        loss_adapter=_technical_loss,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=_technical_stop,
        recorder=events.append,
    )
    soft = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=SoftTargetAdapter(),
        loss_adapter=_technical_loss,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=_technical_stop,
        recorder=events.append,
    )

    assert type(hard) is TrainerLifecycle
    assert type(soft) is TrainerLifecycle
    assert hard.target_cache_kind == "teacher_argmax"
    assert soft.target_cache_kind == "teacher_logits"


@pytest.mark.parametrize(
    ("adapter", "expected_kind"),
    [
        (HardTargetAdapter(), "teacher_argmax"),
        (SoftTargetAdapter(), "teacher_logits"),
    ],
)
def test_shared_prepare_path_exercises_injected_candidates_only(
    adapter,
    expected_kind: str,
) -> None:
    events = []

    lifecycle = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=adapter,
        loss_adapter=_technical_loss,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=_technical_stop,
        recorder=events.append,
    )

    prepared = lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=7,
        device="cpu",
        settings=_settings(),
    )

    assert isinstance(prepared.model, torch.nn.Linear)
    assert prepared.targets.cache_kind == expected_kind
    assert isinstance(
        prepared.optimizer_schedule.optimizer,
        torch.optim.SGD,
    )
    assert prepared.optimizer_schedule.scheduler is None
    assert events == []


def test_loss_stop_and_recorder_dispatch_through_same_lifecycle() -> None:
    events = []

    lifecycle = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=HardTargetAdapter(),
        loss_adapter=_technical_loss,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=_technical_stop,
        recorder=events.append,
    )

    settings = _settings()
    prepared = lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=11,
        device="cpu",
        settings=settings,
    )

    outputs = torch.ones((4, 3), dtype=torch.float32)
    loss = lifecycle.compute_loss(
        prepared=prepared,
        outputs=outputs,
    )

    assert loss.ndim == 0
    assert float(loss) == pytest.approx(1.0)

    before = TrainerProgress(
        step=2,
        updates_completed=2,
        metrics={},
    )
    at_stop = TrainerProgress(
        step=3,
        updates_completed=3,
        metrics={},
    )

    assert lifecycle.should_stop(
        progress=before,
        settings=settings,
    ) is False
    assert lifecycle.should_stop(
        progress=at_stop,
        settings=settings,
    ) is True

    event = TrainerEvent(
        event_type="technical_interface_probe",
        step=0,
        payload={"scientific_data": False},
    )
    lifecycle.record(event)

    assert events == [event]


def test_target_adapter_cannot_supply_unknown_cache_kind() -> None:
    class BadAdapter:
        cache_kind = "invented_target"

        def __call__(self, cache):
            raise AssertionError("must not execute")

    with pytest.raises(
        TrainerInterfaceError,
        match="target_adapter.cache_kind",
    ):
        TrainerLifecycle(
            model_constructor=_model_constructor,
            target_adapter=BadAdapter(),
            loss_adapter=_technical_loss,
            optimizer_schedule_factory=_optimizer_factory,
            stop_rule=_technical_stop,
            recorder=lambda event: None,
        )


def test_optimizer_factory_must_return_explicit_bundle() -> None:
    def bad_factory(*, model, settings):
        return torch.optim.SGD(model.parameters(), lr=0.01)

    lifecycle = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=HardTargetAdapter(),
        loss_adapter=_technical_loss,
        optimizer_schedule_factory=bad_factory,
        stop_rule=_technical_stop,
        recorder=lambda event: None,
    )

    with pytest.raises(
        TrainerInterfaceError,
        match="OptimizerScheduleBundle",
    ):
        lifecycle.prepare(
            cache=_TechnicalCache(),
            model_seed=1,
            device="cpu",
            settings=_settings(),
        )


def test_loss_adapter_must_return_scalar_tensor() -> None:
    def bad_loss(*, outputs, targets, settings):
        return outputs

    lifecycle = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=HardTargetAdapter(),
        loss_adapter=bad_loss,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=_technical_stop,
        recorder=lambda event: None,
    )

    prepared = lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=1,
        device="cpu",
        settings=_settings(),
    )

    with pytest.raises(
        TrainerInterfaceError,
        match="scalar tensor",
    ):
        lifecycle.compute_loss(
            prepared=prepared,
            outputs=torch.ones((4, 3)),
        )


def test_no_training_update_occurs_during_prepare() -> None:
    lifecycle = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=HardTargetAdapter(),
        loss_adapter=_technical_loss,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=_technical_stop,
        recorder=lambda event: None,
    )

    prepared = lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=19,
        device="cpu",
        settings=_settings(),
    )

    before = [
        parameter.detach().clone()
        for parameter in prepared.model.parameters()
    ]

    after = [
        parameter.detach().clone()
        for parameter in prepared.model.parameters()
    ]

    assert all(
        torch.equal(left, right)
        for left, right in zip(before, after, strict=True)
    )

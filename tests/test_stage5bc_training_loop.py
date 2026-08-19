from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as functional

from circuit_families.stage5bc.student_trainer import (
    HardTargetAdapter,
    OptimizerScheduleBundle,
    PreparedTargets,
    SoftTargetAdapter,
    TechnicalTrainingResult,
    TrainerLifecycle,
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


def _inputs() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
        ],
        dtype=torch.float32,
    )


def _settings(
    *,
    stop_step: int = 2,
) -> TrainerSettingsBundle:
    return TrainerSettingsBundle(
        model={
            "candidate_input_width": 2,
            "candidate_output_width": 3,
            "purpose": "tiny_technical_fixture_only",
        },
        loss={
            "purpose": "explicit_technical_candidate_only",
        },
        optimizer_schedule={
            "candidate_learning_rate": 0.05,
            "purpose": "explicit_technical_candidate_only",
        },
        stop={
            "technical_stop_step": stop_step,
            "purpose": "explicit_technical_candidate_only",
        },
    )


def _configuration_refs() -> dict[str, str]:
    return {
        "architecture_profile": "technical-architecture-fixture/v1",
        "trainer_profile": "technical-trainer-fixture/v1",
        "adapter_profile": "technical-adapter-fixture/v1",
    }


def _model_constructor(
    *,
    seed: int,
    device: torch.device,
    settings,
) -> torch.nn.Module:
    torch.manual_seed(seed)

    model = torch.nn.Linear(
        int(settings["candidate_input_width"]),
        int(settings["candidate_output_width"]),
    )
    model.eval()
    return model.to(device)


def _optimizer_factory(
    *,
    model: torch.nn.Module,
    settings,
) -> OptimizerScheduleBundle:
    return OptimizerScheduleBundle(
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=float(settings["candidate_learning_rate"]),
        ),
        scheduler=None,
    )


def _hard_loss(
    *,
    outputs: torch.Tensor,
    targets: PreparedTargets,
    settings,
) -> torch.Tensor:
    assert targets.cache_kind == "teacher_argmax"
    return functional.cross_entropy(
        outputs,
        targets.values,
    )


def _soft_loss(
    *,
    outputs: torch.Tensor,
    targets: PreparedTargets,
    settings,
) -> torch.Tensor:
    assert targets.cache_kind == "teacher_logits"
    return functional.mse_loss(
        outputs,
        targets.values,
    )


def _stop_rule(
    *,
    progress,
    settings,
) -> bool:
    return progress.step >= int(settings["technical_stop_step"])


def _lifecycle(
    *,
    soft: bool,
    events: list,
) -> TrainerLifecycle:
    return TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=(
            SoftTargetAdapter()
            if soft
            else HardTargetAdapter()
        ),
        loss_adapter=_soft_loss if soft else _hard_loss,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=_stop_rule,
        recorder=events.append,
    )


@pytest.mark.parametrize("soft", [False, True])
def test_hard_and_soft_execute_through_one_common_loop(
    soft: bool,
) -> None:
    events = []
    lifecycle = _lifecycle(
        soft=soft,
        events=events,
    )

    prepared = lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=17,
        device="cpu",
        settings=_settings(stop_step=2),
    )

    assert prepared.model.training is False

    result = lifecycle.run_technical(
        prepared=prepared,
        training_inputs=_inputs(),
        configuration_refs=_configuration_refs(),
        technical_safety_step_limit=5,
    )

    assert isinstance(result, TechnicalTrainingResult)
    assert result.terminal_status == "stop_rule_met"
    assert result.terminal_reason == "injected_stop_rule"
    assert result.updates_completed == 2
    assert len(result.trajectory) == 2
    assert prepared.model.training is False
    assert result.model_training_mode_restored is True
    assert result.model_device == "cpu"

    expected_kind = (
        "teacher_logits"
        if soft
        else "teacher_argmax"
    )
    assert result.target_cache_kind == expected_kind

    assert [event.event_type for event in events] == [
        "technical_training_step",
        "technical_training_step",
        "technical_training_terminal",
    ]

    assert all(
        event.payload["scientific_data"] is False
        and event.payload["production_eligible"] is False
        for event in events
    )


def test_trajectory_and_final_parameters_are_deterministic() -> None:
    results = []
    states = []

    for _ in range(2):
        events = []
        lifecycle = _lifecycle(
            soft=False,
            events=events,
        )
        prepared = lifecycle.prepare(
            cache=_TechnicalCache(),
            model_seed=23,
            device="cpu",
            settings=_settings(stop_step=3),
        )

        result = lifecycle.run_technical(
            prepared=prepared,
            training_inputs=_inputs(),
            configuration_refs=_configuration_refs(),
            technical_safety_step_limit=5,
        )

        results.append(result)
        states.append(
            {
                name: tensor.detach().clone()
                for name, tensor in prepared.model.state_dict().items()
            }
        )

    assert results[0].trajectory == results[1].trajectory
    assert results[0].terminal_status == results[1].terminal_status

    assert states[0].keys() == states[1].keys()
    assert all(
        torch.equal(states[0][name], states[1][name])
        for name in states[0]
    )


def test_configuration_references_are_recorded_by_value() -> None:
    events = []
    lifecycle = _lifecycle(
        soft=False,
        events=events,
    )
    prepared = lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=31,
        device="cpu",
        settings=_settings(stop_step=1),
    )

    refs = _configuration_refs()

    result = lifecycle.run_technical(
        prepared=prepared,
        training_inputs=_inputs(),
        configuration_refs=refs,
        technical_safety_step_limit=3,
    )

    refs["trainer_profile"] = "mutated-after-run"

    assert result.configuration_refs["trainer_profile"] == (
        "technical-trainer-fixture/v1"
    )
    assert events[0].payload["configuration_refs"]["trainer_profile"] == (
        "technical-trainer-fixture/v1"
    )


def test_nonfinite_loss_returns_explicit_terminal_status() -> None:
    events = []

    def nonfinite_loss(
        *,
        outputs: torch.Tensor,
        targets: PreparedTargets,
        settings,
    ) -> torch.Tensor:
        return outputs.sum() * torch.tensor(
            float("nan"),
            device=outputs.device,
        )

    lifecycle = TrainerLifecycle(
        model_constructor=_model_constructor,
        target_adapter=HardTargetAdapter(),
        loss_adapter=nonfinite_loss,
        optimizer_schedule_factory=_optimizer_factory,
        stop_rule=_stop_rule,
        recorder=events.append,
    )

    prepared = lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=43,
        device="cpu",
        settings=_settings(stop_step=3),
    )

    before = copy.deepcopy(prepared.model.state_dict())

    result = lifecycle.run_technical(
        prepared=prepared,
        training_inputs=_inputs(),
        configuration_refs=_configuration_refs(),
        technical_safety_step_limit=5,
    )

    after = prepared.model.state_dict()

    assert result.terminal_status == "nonfinite_failure"
    assert result.terminal_reason == "nonfinite_loss"
    assert result.updates_completed == 0
    assert result.trajectory == ()

    assert all(
        torch.equal(before[name], after[name])
        for name in before
    )


def test_technical_safety_limit_is_explicit_terminal_status() -> None:
    events = []
    lifecycle = _lifecycle(
        soft=False,
        events=events,
    )

    prepared = lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=47,
        device="cpu",
        settings=_settings(stop_step=100),
    )

    result = lifecycle.run_technical(
        prepared=prepared,
        training_inputs=_inputs(),
        configuration_refs=_configuration_refs(),
        technical_safety_step_limit=2,
    )

    assert result.terminal_status == "technical_step_limit_exhausted"
    assert result.terminal_reason == (
        "mandatory_technical_safety_step_limit_reached"
    )
    assert result.updates_completed == 2


def test_stop_rule_can_terminate_before_first_update() -> None:
    events = []
    lifecycle = _lifecycle(
        soft=False,
        events=events,
    )

    prepared = lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=53,
        device="cpu",
        settings=_settings(stop_step=0),
    )

    before = copy.deepcopy(prepared.model.state_dict())

    result = lifecycle.run_technical(
        prepared=prepared,
        training_inputs=_inputs(),
        configuration_refs=_configuration_refs(),
        technical_safety_step_limit=3,
    )

    assert result.terminal_status == "stop_rule_met"
    assert result.terminal_reason == "stop_rule_before_first_update"
    assert result.updates_completed == 0

    after = prepared.model.state_dict()
    assert all(
        torch.equal(before[name], after[name])
        for name in before
    )


def test_model_device_and_prior_eval_mode_are_preserved() -> None:
    events = []
    lifecycle = _lifecycle(
        soft=False,
        events=events,
    )

    prepared = lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=59,
        device="cpu",
        settings=_settings(stop_step=1),
    )

    assert prepared.model.training is False

    before_devices = {
        name: str(parameter.device)
        for name, parameter in prepared.model.named_parameters()
    }

    result = lifecycle.run_technical(
        prepared=prepared,
        training_inputs=_inputs(),
        configuration_refs=_configuration_refs(),
        technical_safety_step_limit=3,
    )

    after_devices = {
        name: str(parameter.device)
        for name, parameter in prepared.model.named_parameters()
    }

    assert before_devices == after_devices
    assert prepared.model.training is False
    assert result.model_training_mode_restored is True


def test_loop_does_not_hide_target_count_mismatch() -> None:
    events = []
    lifecycle = _lifecycle(
        soft=False,
        events=events,
    )

    prepared = lifecycle.prepare(
        cache=_TechnicalCache(),
        model_seed=61,
        device="cpu",
        settings=_settings(stop_step=1),
    )

    with pytest.raises(
        ValueError,
        match="training input count must match prepared target count",
    ):
        lifecycle.run_technical(
            prepared=prepared,
            training_inputs=_inputs()[:3],
            configuration_refs=_configuration_refs(),
            technical_safety_step_limit=3,
        )

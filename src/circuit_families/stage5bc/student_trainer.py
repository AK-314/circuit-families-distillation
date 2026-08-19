"""Injected shared trainer lifecycle interfaces for Stage 5B.

Part I defines dependency-injection and target-adapter boundaries only.
The common training step/evaluation loop is implemented later in Part K.

No loss, temperature, optimizer, schedule, or stopping policy is selected
here as a scientific default.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from circuit_families.stage5bc.target_cache import (
    STAGE4_CACHE_KINDS,
    LoadedTargetCache,
)


class TrainerInterfaceError(ValueError):
    """Raised when an injected trainer component violates the interface."""


@dataclass(frozen=True)
class PreparedTargets:
    """Canonical target representation presented to the shared lifecycle."""

    cache_kind: str
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.cache_kind not in STAGE4_CACHE_KINDS:
            raise TrainerInterfaceError(
                f"unsupported target cache kind: {self.cache_kind!r}"
            )

        if not isinstance(self.values, torch.Tensor):
            raise TrainerInterfaceError(
                "prepared target values must be a torch.Tensor"
            )

        if self.cache_kind == "teacher_argmax":
            if self.values.ndim != 1:
                raise TrainerInterfaceError(
                    "teacher_argmax targets must be rank 1"
                )
            if self.values.dtype != torch.int64:
                raise TrainerInterfaceError(
                    "teacher_argmax targets must use int64"
                )

        if self.cache_kind == "teacher_logits":
            if self.values.ndim != 2:
                raise TrainerInterfaceError(
                    "teacher_logits targets must be rank 2"
                )
            if not self.values.is_floating_point():
                raise TrainerInterfaceError(
                    "teacher_logits targets must use a floating dtype"
                )


class HardTargetAdapter:
    """Expose cached teacher argmax labels through the shared target interface."""

    cache_kind = "teacher_argmax"

    def __call__(self, cache: LoadedTargetCache) -> PreparedTargets:
        values = cache.stage4_view(self.cache_kind).detach().clone()
        return PreparedTargets(
            cache_kind=self.cache_kind,
            values=values,
        )


class SoftTargetAdapter:
    """Expose cached centred teacher logits without choosing a soft loss."""

    cache_kind = "teacher_logits"

    def __call__(self, cache: LoadedTargetCache) -> PreparedTargets:
        values = cache.stage4_view(self.cache_kind).detach().clone()
        return PreparedTargets(
            cache_kind=self.cache_kind,
            values=values,
        )


class ModelConstructor(Protocol):
    """Injected technical model construction interface."""

    def __call__(
        self,
        *,
        seed: int,
        device: torch.device,
        settings: Mapping[str, Any],
    ) -> torch.nn.Module:
        ...


class TargetAdapter(Protocol):
    """Injected hard/soft target view with no training-loop ownership."""

    cache_kind: str

    def __call__(
        self,
        cache: LoadedTargetCache,
    ) -> PreparedTargets:
        ...


class LossAdapter(Protocol):
    """Injected candidate loss calculation; no default is supplied."""

    def __call__(
        self,
        *,
        outputs: torch.Tensor,
        targets: PreparedTargets,
        settings: Mapping[str, Any],
    ) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class OptimizerScheduleBundle:
    """Explicit optimizer plus optional explicitly supplied scheduler."""

    optimizer: torch.optim.Optimizer
    scheduler: object | None

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TrainerInterfaceError(
                "optimizer must be a torch.optim.Optimizer instance"
            )


class OptimizerScheduleFactory(Protocol):
    """Injected optimizer/schedule construction interface."""

    def __call__(
        self,
        *,
        model: torch.nn.Module,
        settings: Mapping[str, Any],
    ) -> OptimizerScheduleBundle:
        ...


@dataclass(frozen=True)
class TrainerProgress:
    """Minimal generic progress state consumed by an injected stop rule."""

    step: int
    updates_completed: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise TrainerInterfaceError("progress step must be an integer")
        if self.step < 0:
            raise TrainerInterfaceError(
                "progress step must be non-negative"
            )

        if (
            isinstance(self.updates_completed, bool)
            or not isinstance(self.updates_completed, int)
        ):
            raise TrainerInterfaceError(
                "updates_completed must be an integer"
            )
        if self.updates_completed < 0:
            raise TrainerInterfaceError(
                "updates_completed must be non-negative"
            )

        if not isinstance(self.metrics, Mapping):
            raise TrainerInterfaceError(
                "progress metrics must be a mapping"
            )


class StopRule(Protocol):
    """Injected candidate stopping rule; no default is supplied."""

    def __call__(
        self,
        *,
        progress: TrainerProgress,
        settings: Mapping[str, Any],
    ) -> bool:
        ...


@dataclass(frozen=True)
class TrainerEvent:
    """Generic event passed to the injected recorder."""

    event_type: str
    step: int
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, str) or not self.event_type:
            raise TrainerInterfaceError(
                "event_type must be a non-empty string"
            )
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise TrainerInterfaceError("event step must be an integer")
        if self.step < 0:
            raise TrainerInterfaceError(
                "event step must be non-negative"
            )
        if not isinstance(self.payload, Mapping):
            raise TrainerInterfaceError(
                "event payload must be a mapping"
            )


class Recorder(Protocol):
    """Injected lifecycle event sink."""

    def __call__(self, event: TrainerEvent) -> None:
        ...


@dataclass(frozen=True)
class TrainerSettingsBundle:
    """Explicit settings for every policy-bearing injected component."""

    model: Mapping[str, Any]
    loss: Mapping[str, Any]
    optimizer_schedule: Mapping[str, Any]
    stop: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field in (
            "model",
            "loss",
            "optimizer_schedule",
            "stop",
        ):
            value = getattr(self, field)
            if not isinstance(value, Mapping):
                raise TrainerInterfaceError(
                    f"{field} settings must be an explicit mapping"
                )
            object.__setattr__(
                self,
                field,
                copy.deepcopy(dict(value)),
            )


@dataclass(frozen=True)
class PreparedTrainer:
    """Objects prepared by the shared lifecycle before any training step."""

    model: torch.nn.Module
    targets: PreparedTargets
    optimizer_schedule: OptimizerScheduleBundle
    settings: TrainerSettingsBundle


TECHNICAL_TERMINAL_STATUSES = (
    "stop_rule_met",
    "nonfinite_failure",
    "technical_step_limit_exhausted",
)


@dataclass(frozen=True)
class TechnicalLoopSnapshot:
    """In-memory continuation counters for the one common technical loop."""

    updates_completed: int
    trajectory: tuple[TrainerProgress, ...]
    outer_training_mode: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.updates_completed, bool)
            or not isinstance(self.updates_completed, int)
            or self.updates_completed < 0
        ):
            raise TrainerInterfaceError(
                "snapshot updates_completed must be a non-negative integer"
            )

        if not isinstance(self.trajectory, tuple):
            raise TrainerInterfaceError(
                "snapshot trajectory must be a tuple"
            )

        if len(self.trajectory) != self.updates_completed:
            raise TrainerInterfaceError(
                "snapshot trajectory length must equal updates_completed"
            )

        for expected_step, progress in enumerate(
            self.trajectory,
            start=1,
        ):
            if not isinstance(progress, TrainerProgress):
                raise TrainerInterfaceError(
                    "snapshot trajectory entries must be TrainerProgress"
                )

            if (
                progress.step != expected_step
                or progress.updates_completed != expected_step
            ):
                raise TrainerInterfaceError(
                    "snapshot trajectory steps must be contiguous from 1"
                )

        if not isinstance(self.outer_training_mode, bool):
            raise TrainerInterfaceError(
                "snapshot outer_training_mode must be boolean"
            )


class TechnicalInterruption(RuntimeError):
    """Deliberate mechanics-only interruption carrying a resumable snapshot."""

    def __init__(self, snapshot: TechnicalLoopSnapshot) -> None:
        if not isinstance(snapshot, TechnicalLoopSnapshot):
            raise TrainerInterfaceError(
                "TechnicalInterruption requires TechnicalLoopSnapshot"
            )

        self.snapshot = snapshot
        super().__init__(
            "forced technical interruption after "
            f"{snapshot.updates_completed} updates"
        )


@dataclass(frozen=True)
class TechnicalTrainingResult:
    """Terminal mechanics record for one tiny technical training run."""

    terminal_status: str
    terminal_reason: str
    updates_completed: int
    trajectory: tuple[TrainerProgress, ...]
    configuration_refs: Mapping[str, str]
    target_cache_kind: str
    model_device: str
    model_training_mode_restored: bool

    def __post_init__(self) -> None:
        if self.terminal_status not in TECHNICAL_TERMINAL_STATUSES:
            raise TrainerInterfaceError(
                f"invalid technical terminal status: {self.terminal_status!r}"
            )

        if not isinstance(self.terminal_reason, str) or not self.terminal_reason:
            raise TrainerInterfaceError(
                "terminal_reason must be a non-empty string"
            )

        if (
            isinstance(self.updates_completed, bool)
            or not isinstance(self.updates_completed, int)
            or self.updates_completed < 0
        ):
            raise TrainerInterfaceError(
                "updates_completed must be a non-negative integer"
            )

        if not isinstance(self.trajectory, tuple):
            raise TrainerInterfaceError(
                "trajectory must be a tuple"
            )

        if any(
            not isinstance(item, TrainerProgress)
            for item in self.trajectory
        ):
            raise TrainerInterfaceError(
                "trajectory entries must be TrainerProgress"
            )

        if not isinstance(self.configuration_refs, Mapping):
            raise TrainerInterfaceError(
                "configuration_refs must be a mapping"
            )

        references = dict(self.configuration_refs)
        if not references:
            raise TrainerInterfaceError(
                "configuration_refs must not be empty"
            )

        for key, value in references.items():
            if not isinstance(key, str) or not key:
                raise TrainerInterfaceError(
                    "configuration_refs keys must be non-empty strings"
                )
            if not isinstance(value, str) or not value:
                raise TrainerInterfaceError(
                    "configuration_refs values must be non-empty strings"
                )

        object.__setattr__(
            self,
            "configuration_refs",
            copy.deepcopy(references),
        )

        if self.target_cache_kind not in STAGE4_CACHE_KINDS:
            raise TrainerInterfaceError(
                "target_cache_kind must be a Stage 4 cache kind"
            )

        if not isinstance(self.model_device, str) or not self.model_device:
            raise TrainerInterfaceError(
                "model_device must be a non-empty string"
            )

        if not isinstance(self.model_training_mode_restored, bool):
            raise TrainerInterfaceError(
                "model_training_mode_restored must be boolean"
            )


def _module_device(model: torch.nn.Module) -> torch.device:
    devices = {
        tensor.device
        for tensor in (
            *tuple(model.parameters()),
            *tuple(model.buffers()),
        )
    }

    if len(devices) > 1:
        raise TrainerInterfaceError(
            "technical trainer requires model parameters/buffers on one device"
        )

    if not devices:
        return torch.device("cpu")

    return next(iter(devices))


def _module_device_fingerprint(
    model: torch.nn.Module,
) -> tuple[tuple[str, str], ...]:
    items = []

    for name, parameter in model.named_parameters():
        items.append((f"parameter:{name}", str(parameter.device)))

    for name, buffer in model.named_buffers():
        items.append((f"buffer:{name}", str(buffer.device)))

    return tuple(items)


class TrainerLifecycle:
    """One lifecycle shared by hard-target and soft-target training.

    Part I intentionally stops at preparation and component dispatch.
    Part K adds the common technical step/evaluation loop to this lifecycle.
    """

    def __init__(
        self,
        *,
        model_constructor: ModelConstructor,
        target_adapter: TargetAdapter,
        loss_adapter: LossAdapter,
        optimizer_schedule_factory: OptimizerScheduleFactory,
        stop_rule: StopRule,
        recorder: Recorder,
    ) -> None:
        for name, value in (
            ("model_constructor", model_constructor),
            ("target_adapter", target_adapter),
            ("loss_adapter", loss_adapter),
            ("optimizer_schedule_factory", optimizer_schedule_factory),
            ("stop_rule", stop_rule),
            ("recorder", recorder),
        ):
            if value is None or not callable(value):
                raise TrainerInterfaceError(
                    f"{name} must be explicitly supplied and callable"
                )

        cache_kind = getattr(target_adapter, "cache_kind", None)
        if cache_kind not in STAGE4_CACHE_KINDS:
            raise TrainerInterfaceError(
                "target_adapter.cache_kind must be an allowed Stage 4 cache kind"
            )

        self.model_constructor = model_constructor
        self.target_adapter = target_adapter
        self.loss_adapter = loss_adapter
        self.optimizer_schedule_factory = optimizer_schedule_factory
        self.stop_rule = stop_rule
        self.recorder = recorder

    @property
    def target_cache_kind(self) -> str:
        """Return the selected adapter's cache view without owning a loop."""
        return str(self.target_adapter.cache_kind)

    def prepare(
        self,
        *,
        cache: LoadedTargetCache,
        model_seed: int,
        device: str | torch.device,
        settings: TrainerSettingsBundle,
    ) -> PreparedTrainer:
        """Prepare explicitly injected components; perform no training update."""
        if isinstance(model_seed, bool) or not isinstance(model_seed, int):
            raise TrainerInterfaceError(
                "model_seed must be an integer"
            )
        if model_seed < 0:
            raise TrainerInterfaceError(
                "model_seed must be non-negative"
            )
        if not isinstance(settings, TrainerSettingsBundle):
            raise TrainerInterfaceError(
                "settings must be a TrainerSettingsBundle"
            )

        selected_device = torch.device(device)

        model = self.model_constructor(
            seed=model_seed,
            device=selected_device,
            settings=settings.model,
        )
        if not isinstance(model, torch.nn.Module):
            raise TrainerInterfaceError(
                "model_constructor must return torch.nn.Module"
            )

        targets = self.target_adapter(cache)
        if not isinstance(targets, PreparedTargets):
            raise TrainerInterfaceError(
                "target_adapter must return PreparedTargets"
            )
        if targets.cache_kind != self.target_cache_kind:
            raise TrainerInterfaceError(
                "target_adapter returned inconsistent cache kind"
            )

        optimizer_schedule = self.optimizer_schedule_factory(
            model=model,
            settings=settings.optimizer_schedule,
        )
        if not isinstance(
            optimizer_schedule,
            OptimizerScheduleBundle,
        ):
            raise TrainerInterfaceError(
                "optimizer_schedule_factory must return "
                "OptimizerScheduleBundle"
            )

        return PreparedTrainer(
            model=model,
            targets=targets,
            optimizer_schedule=optimizer_schedule,
            settings=settings,
        )

    def compute_loss(
        self,
        *,
        prepared: PreparedTrainer,
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch to the injected loss adapter without choosing a loss."""
        if not isinstance(prepared, PreparedTrainer):
            raise TrainerInterfaceError(
                "prepared must be a PreparedTrainer"
            )
        if not isinstance(outputs, torch.Tensor):
            raise TrainerInterfaceError(
                "outputs must be a torch.Tensor"
            )

        loss = self.loss_adapter(
            outputs=outputs,
            targets=prepared.targets,
            settings=prepared.settings.loss,
        )

        if not isinstance(loss, torch.Tensor):
            raise TrainerInterfaceError(
                "loss_adapter must return a torch.Tensor"
            )
        if loss.ndim != 0:
            raise TrainerInterfaceError(
                "loss_adapter must return a scalar tensor"
            )

        return loss

    def should_stop(
        self,
        *,
        progress: TrainerProgress,
        settings: TrainerSettingsBundle,
    ) -> bool:
        """Dispatch to the injected stop rule."""
        if not isinstance(progress, TrainerProgress):
            raise TrainerInterfaceError(
                "progress must be TrainerProgress"
            )
        if not isinstance(settings, TrainerSettingsBundle):
            raise TrainerInterfaceError(
                "settings must be TrainerSettingsBundle"
            )

        decision = self.stop_rule(
            progress=progress,
            settings=settings.stop,
        )

        if not isinstance(decision, bool):
            raise TrainerInterfaceError(
                "stop_rule must return bool"
            )

        return decision

    def run_technical(
        self,
        *,
        prepared: PreparedTrainer,
        training_inputs: torch.Tensor,
        configuration_refs: Mapping[str, str],
        technical_safety_step_limit: int,
        resume_snapshot: TechnicalLoopSnapshot | None = None,
        snapshot_callback: Any | None = None,
        interrupt_after_updates: int | None = None,
    ) -> TechnicalTrainingResult:
        """Run or resume the one common mechanics-only training loop.

        The safety limit is a mandatory maximum *total* update count for a
        technical fixture. It is not a scientific stopping rule.

        ``snapshot_callback`` is invoked after each completed update. A forced
        interruption occurs only after that completed-state callback, ensuring
        a checkpoint can represent the exact continuation point.
        """
        if not isinstance(prepared, PreparedTrainer):
            raise TrainerInterfaceError(
                "prepared must be a PreparedTrainer"
            )

        if not isinstance(training_inputs, torch.Tensor):
            raise TrainerInterfaceError(
                "training_inputs must be a torch.Tensor"
            )

        if training_inputs.ndim < 1 or training_inputs.shape[0] <= 0:
            raise TrainerInterfaceError(
                "training_inputs must contain a non-empty example dimension"
            )

        if (
            isinstance(technical_safety_step_limit, bool)
            or not isinstance(technical_safety_step_limit, int)
            or technical_safety_step_limit <= 0
        ):
            raise TrainerInterfaceError(
                "technical_safety_step_limit must be a positive integer"
            )

        if not isinstance(configuration_refs, Mapping):
            raise TrainerInterfaceError(
                "configuration_refs must be an explicit mapping"
            )

        refs = copy.deepcopy(dict(configuration_refs))
        if not refs:
            raise TrainerInterfaceError(
                "configuration_refs must not be empty"
            )

        for key, value in refs.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or not value
            ):
                raise TrainerInterfaceError(
                    "configuration_refs must contain non-empty string pairs"
                )

        if snapshot_callback is not None and not callable(snapshot_callback):
            raise TrainerInterfaceError(
                "snapshot_callback must be callable when supplied"
            )

        if interrupt_after_updates is not None:
            if (
                isinstance(interrupt_after_updates, bool)
                or not isinstance(interrupt_after_updates, int)
                or interrupt_after_updates <= 0
            ):
                raise TrainerInterfaceError(
                    "interrupt_after_updates must be a positive integer"
                )

            if interrupt_after_updates > technical_safety_step_limit:
                raise TrainerInterfaceError(
                    "interrupt_after_updates cannot exceed technical safety limit"
                )

        model = prepared.model
        model_device = _module_device(model)

        if training_inputs.device != model_device:
            raise TrainerInterfaceError(
                "training_inputs must already be on the model device"
            )

        if prepared.targets.values.device != model_device:
            raise TrainerInterfaceError(
                "prepared targets must already be on the model device"
            )

        if training_inputs.is_floating_point() and not bool(
            torch.isfinite(training_inputs).all()
        ):
            raise TrainerInterfaceError(
                "training_inputs contain non-finite values"
            )

        if (
            prepared.targets.values.is_floating_point()
            and not bool(torch.isfinite(prepared.targets.values).all())
        ):
            raise TrainerInterfaceError(
                "prepared targets contain non-finite values"
            )

        if training_inputs.shape[0] != prepared.targets.values.shape[0]:
            raise TrainerInterfaceError(
                "training input count must match prepared target count"
            )

        optimizer = prepared.optimizer_schedule.optimizer
        scheduler = prepared.optimizer_schedule.scheduler

        if scheduler is not None:
            step_method = getattr(scheduler, "step", None)
            if step_method is None or not callable(step_method):
                raise TrainerInterfaceError(
                    "injected scheduler must expose callable step()"
                )

        if resume_snapshot is not None:
            if not isinstance(resume_snapshot, TechnicalLoopSnapshot):
                raise TrainerInterfaceError(
                    "resume_snapshot must be TechnicalLoopSnapshot"
                )

            if (
                resume_snapshot.updates_completed
                > technical_safety_step_limit
            ):
                raise TrainerInterfaceError(
                    "resume snapshot exceeds technical safety limit"
                )

            if (
                interrupt_after_updates is not None
                and interrupt_after_updates
                <= resume_snapshot.updates_completed
            ):
                raise TrainerInterfaceError(
                    "forced interruption point must be after resumed update count"
                )

            if model.training != resume_snapshot.outer_training_mode:
                raise TrainerInterfaceError(
                    "restored model outer training mode disagrees with snapshot"
                )

            original_training_mode = resume_snapshot.outer_training_mode
            trajectory = list(resume_snapshot.trajectory)
            updates_completed = resume_snapshot.updates_completed
        else:
            original_training_mode = model.training
            trajectory = []
            updates_completed = 0

        original_device_fingerprint = _module_device_fingerprint(model)

        terminal_status: str | None = None
        terminal_reason: str | None = None

        try:
            model.train(True)

            current_progress = TrainerProgress(
                step=updates_completed,
                updates_completed=updates_completed,
                metrics=(
                    trajectory[-1].metrics
                    if trajectory
                    else {}
                ),
            )

            if self.should_stop(
                progress=current_progress,
                settings=prepared.settings,
            ):
                terminal_status = "stop_rule_met"
                terminal_reason = (
                    "stop_rule_before_first_update"
                    if updates_completed == 0
                    else "stop_rule_before_next_resumed_update"
                )
            else:
                while updates_completed < technical_safety_step_limit:
                    optimizer.zero_grad(set_to_none=True)

                    outputs = model(training_inputs)

                    if not isinstance(outputs, torch.Tensor):
                        raise TrainerInterfaceError(
                            "technical model forward must return a torch.Tensor"
                        )

                    if not bool(torch.isfinite(outputs).all()):
                        terminal_status = "nonfinite_failure"
                        terminal_reason = "nonfinite_model_outputs"
                        break

                    loss = self.compute_loss(
                        prepared=prepared,
                        outputs=outputs,
                    )

                    if not bool(torch.isfinite(loss).all()):
                        terminal_status = "nonfinite_failure"
                        terminal_reason = "nonfinite_loss"
                        break

                    loss.backward()

                    gradient_is_finite = True

                    for parameter in model.parameters():
                        gradient = parameter.grad
                        if gradient is not None and not bool(
                            torch.isfinite(gradient).all()
                        ):
                            gradient_is_finite = False
                            break

                    if not gradient_is_finite:
                        terminal_status = "nonfinite_failure"
                        terminal_reason = "nonfinite_gradients"
                        break

                    optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

                    updates_completed += 1

                    progress = TrainerProgress(
                        step=updates_completed,
                        updates_completed=updates_completed,
                        metrics={
                            "loss": float(loss.detach().item()),
                        },
                    )
                    trajectory.append(progress)

                    self.record(
                        TrainerEvent(
                            event_type="technical_training_step",
                            step=updates_completed,
                            payload={
                                "scientific_data": False,
                                "production_eligible": False,
                                "target_cache_kind": self.target_cache_kind,
                                "configuration_refs": copy.deepcopy(refs),
                                "metrics": copy.deepcopy(
                                    dict(progress.metrics)
                                ),
                            },
                        )
                    )

                    snapshot = TechnicalLoopSnapshot(
                        updates_completed=updates_completed,
                        trajectory=tuple(trajectory),
                        outer_training_mode=original_training_mode,
                    )

                    if snapshot_callback is not None:
                        snapshot_callback(snapshot)

                    if interrupt_after_updates == updates_completed:
                        raise TechnicalInterruption(snapshot)

                    if self.should_stop(
                        progress=progress,
                        settings=prepared.settings,
                    ):
                        terminal_status = "stop_rule_met"
                        terminal_reason = "injected_stop_rule"
                        break

                if terminal_status is None:
                    terminal_status = "technical_step_limit_exhausted"
                    terminal_reason = (
                        "mandatory_technical_safety_step_limit_reached"
                    )
        finally:
            model.train(original_training_mode)

        restored_mode = model.training == original_training_mode
        final_device_fingerprint = _module_device_fingerprint(model)

        if final_device_fingerprint != original_device_fingerprint:
            raise TrainerInterfaceError(
                "technical loop changed model parameter/buffer device placement"
            )

        if not restored_mode:
            raise TrainerInterfaceError(
                "technical loop failed to restore model training/eval mode"
            )

        assert terminal_status is not None
        assert terminal_reason is not None

        result = TechnicalTrainingResult(
            terminal_status=terminal_status,
            terminal_reason=terminal_reason,
            updates_completed=updates_completed,
            trajectory=tuple(trajectory),
            configuration_refs=refs,
            target_cache_kind=self.target_cache_kind,
            model_device=str(model_device),
            model_training_mode_restored=restored_mode,
        )

        self.record(
            TrainerEvent(
                event_type="technical_training_terminal",
                step=updates_completed,
                payload={
                    "scientific_data": False,
                    "production_eligible": False,
                    "terminal_status": result.terminal_status,
                    "terminal_reason": result.terminal_reason,
                    "target_cache_kind": result.target_cache_kind,
                    "configuration_refs": copy.deepcopy(refs),
                },
            )
        )

        return result

    def record(self, event: TrainerEvent) -> None:
        """Dispatch an event to the injected recorder."""
        if not isinstance(event, TrainerEvent):
            raise TrainerInterfaceError(
                "event must be TrainerEvent"
            )
        self.recorder(event)

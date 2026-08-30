"""Method-native continuous optimization for Stage 12-R1 technical fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch

from circuit_families.stage6d.budgets import NativeBudgetLedger

from .gates import (
    GateConfig,
    GateRunIdentity,
    expected_l0_probability,
    gate_state_record,
    seeded_stochastic_gate_sample,
    validate_log_alpha,
)

CHECKPOINT_VERSION = "stage12r1-optimizer-checkpoint/v1"
OPTIMIZER_RECORD_VERSION = "stage12r1-native-optimizer/v1"

TerminalState = Literal[
    "completed",
    "exhausted",
    "interrupted",
    "numerical_failure",
]


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class OptimizerConfig:
    """Technical-only native optimizer configuration."""

    learning_rate: float
    max_steps: int
    sparsity_coefficient: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    learning_rate_decay: float = 1.0
    checkpoint_every: int = 1
    checkpoint_retention: int = 2
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        finite = (
            self.learning_rate,
            self.sparsity_coefficient,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.learning_rate_decay,
        )
        if not all(math.isfinite(x) for x in finite):
            raise ValueError("optimizer configuration values must be finite")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 0
        ):
            raise ValueError("max_steps must be a non-negative integer")
        if self.sparsity_coefficient < 0:
            raise ValueError("sparsity_coefficient must be non-negative")
        if not (0 <= self.beta1 < 1):
            raise ValueError("beta1 must lie in [0, 1)")
        if not (0 <= self.beta2 < 1):
            raise ValueError("beta2 must lie in [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0 < self.learning_rate_decay <= 1):
            raise ValueError("learning_rate_decay must lie in (0, 1]")
        if (
            isinstance(self.checkpoint_every, bool)
            or not isinstance(self.checkpoint_every, int)
            or self.checkpoint_every < 1
        ):
            raise ValueError("checkpoint_every must be a positive integer")
        if (
            isinstance(self.checkpoint_retention, bool)
            or not isinstance(self.checkpoint_retention, int)
            or self.checkpoint_retention < 1
        ):
            raise ValueError("checkpoint_retention must be a positive integer")
        if self.scientific_data:
            raise ValueError("optimizer config requires scientific_data=false")
        if self.production_eligible:
            raise ValueError(
                "optimizer config requires production_eligible=false"
            )

    def identity_sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class OptimizationTrajectoryPoint:
    step_index: int
    native_consumed: int
    fidelity_loss: float
    expected_l0_mean: float
    total_objective: float
    learning_rate: float
    deterministic_gate_mean: float

    def __post_init__(self) -> None:
        if self.step_index < 0 or self.native_consumed < 0:
            raise ValueError("trajectory indices must be non-negative")
        values = (
            self.fidelity_loss,
            self.expected_l0_mean,
            self.total_objective,
            self.learning_rate,
            self.deterministic_gate_mean,
        )
        if not all(math.isfinite(x) for x in values):
            raise ValueError("trajectory values must be finite")


@dataclass(frozen=True)
class NativeOptimizationResult:
    record_version: str
    terminal_state: TerminalState
    component_basis_identity: str
    component_count: int
    native_budget_unit: str
    native_budget_allowance: int
    native_budget_consumed: int
    next_step: int
    gate_state_sha256: str
    trajectory: tuple[OptimizationTrajectoryPoint, ...]
    latest_checkpoint: str | None
    failure_reason: str | None
    scientific_data: bool
    production_eligible: bool
    result_sha256: str


DenseMaskAdapter = Callable[[torch.Tensor, int], Any]
ObjectiveAdapter = Callable[[Any, int], torch.Tensor]
InterruptPredicate = Callable[[int], bool]


def _step_identity(
    identity: GateRunIdentity,
    step_index: int,
) -> GateRunIdentity:
    return replace(
        identity,
        stream_name=f"{identity.stream_name}/optimizer-step/{step_index}",
    )


def _checkpoint_payload(
    *,
    identity: GateRunIdentity,
    gate_config: GateConfig,
    optimizer_config: OptimizerConfig,
    component_basis_identity: str,
    component_count: int,
    next_step: int,
    log_alpha: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    trajectory: list[OptimizationTrajectoryPoint],
) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "run_identity_sha256": identity.material_sha256(),
        "gate_config_sha256": gate_config.identity_sha256(),
        "optimizer_config_sha256": optimizer_config.identity_sha256(),
        "component_basis_identity": component_basis_identity,
        "component_count": component_count,
        "next_step": next_step,
        "dtype": str(log_alpha.dtype),
        "log_alpha": [
            float(x) for x in log_alpha.detach().cpu().tolist()
        ],
        "exp_avg": [
            float(x) for x in exp_avg.detach().cpu().tolist()
        ],
        "exp_avg_sq": [
            float(x) for x in exp_avg_sq.detach().cpu().tolist()
        ],
        "trajectory": [asdict(point) for point in trajectory],
        "scientific_data": False,
        "production_eligible": False,
    }


def _write_atomic_checkpoint(
    directory: Path,
    payload: Mapping[str, Any],
    *,
    retention: int,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    next_step = int(payload["next_step"])
    destination = directory / f"optimizer-step-{next_step:08d}.json"

    envelope = {
        "payload": dict(payload),
        "payload_sha256": _sha256(payload),
    }
    encoded = _canonical_bytes(envelope)

    temporary = directory / f".{destination.name}.tmp-{os.getpid()}"
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)

    checkpoints = sorted(directory.glob("optimizer-step-*.json"))
    for stale in checkpoints[:-retention]:
        stale.unlink()

    return destination


def _load_checkpoint(
    path: Path,
    *,
    identity: GateRunIdentity,
    gate_config: GateConfig,
    optimizer_config: OptimizerConfig,
    component_basis_identity: str,
    component_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[OptimizationTrajectoryPoint],
]:
    raw = json.loads(path.read_text())
    if set(raw) != {"payload", "payload_sha256"}:
        raise ValueError("checkpoint envelope shape is invalid")

    payload = raw["payload"]
    if _sha256(payload) != raw["payload_sha256"]:
        raise ValueError("checkpoint payload hash mismatch")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("checkpoint version mismatch")
    if payload.get("run_identity_sha256") != identity.material_sha256():
        raise ValueError("checkpoint run identity mismatch")
    if payload.get("gate_config_sha256") != gate_config.identity_sha256():
        raise ValueError("checkpoint gate configuration mismatch")
    if (
        payload.get("optimizer_config_sha256")
        != optimizer_config.identity_sha256()
    ):
        raise ValueError("checkpoint optimizer configuration mismatch")
    if payload.get("component_basis_identity") != component_basis_identity:
        raise ValueError("checkpoint component basis identity mismatch")
    if payload.get("component_count") != component_count:
        raise ValueError("checkpoint component count mismatch")
    if payload.get("scientific_data") is not False:
        raise ValueError("checkpoint scientific_data boundary violated")
    if payload.get("production_eligible") is not False:
        raise ValueError("checkpoint production boundary violated")

    next_step = payload.get("next_step")
    if (
        isinstance(next_step, bool)
        or not isinstance(next_step, int)
        or not (0 <= next_step <= optimizer_config.max_steps)
    ):
        raise ValueError("checkpoint next_step is invalid")

    def tensor_field(name: str) -> torch.Tensor:
        values = payload.get(name)
        if not isinstance(values, list) or len(values) != component_count:
            raise ValueError(f"checkpoint {name} shape mismatch")
        tensor = torch.tensor(values, device=device, dtype=dtype)
        validate_log_alpha(tensor, component_count=component_count)
        return tensor

    log_alpha = tensor_field("log_alpha")
    exp_avg = tensor_field("exp_avg")
    exp_avg_sq = tensor_field("exp_avg_sq")

    trajectory_raw = payload.get("trajectory")
    if not isinstance(trajectory_raw, list):
        raise ValueError("checkpoint trajectory is invalid")
    trajectory = [
        OptimizationTrajectoryPoint(**record)
        for record in trajectory_raw
    ]
    if len(trajectory) != next_step:
        raise ValueError(
            "checkpoint trajectory length does not match next_step"
        )

    return next_step, log_alpha, exp_avg, exp_avg_sq, trajectory


def _result(
    *,
    terminal_state: TerminalState,
    component_basis_identity: str,
    component_count: int,
    native_budget_allowance: int,
    native_budget_consumed: int,
    next_step: int,
    log_alpha: torch.Tensor,
    gate_config: GateConfig,
    trajectory: list[OptimizationTrajectoryPoint],
    latest_checkpoint: Path | None,
    failure_reason: str | None,
) -> NativeOptimizationResult:
    gate_record = gate_state_record(
        log_alpha,
        gate_config,
        component_basis_identity=component_basis_identity,
        component_count=component_count,
    )

    payload = {
        "record_version": OPTIMIZER_RECORD_VERSION,
        "terminal_state": terminal_state,
        "component_basis_identity": component_basis_identity,
        "component_count": component_count,
        "native_budget_unit": "optimizer_step",
        "native_budget_allowance": native_budget_allowance,
        "native_budget_consumed": native_budget_consumed,
        "next_step": next_step,
        "gate_state_sha256": gate_record.state_sha256,
        "trajectory": [asdict(point) for point in trajectory],
        "latest_checkpoint": (
            None
            if latest_checkpoint is None
            else str(latest_checkpoint)
        ),
        "failure_reason": failure_reason,
        "scientific_data": False,
        "production_eligible": False,
    }

    return NativeOptimizationResult(
        **{
            **payload,
            "trajectory": tuple(trajectory),
        },
        result_sha256=_sha256(payload),
    )


def optimize_gates(
    *,
    initial_log_alpha: torch.Tensor,
    component_basis_identity: str,
    component_count: int,
    gate_config: GateConfig,
    run_identity: GateRunIdentity,
    optimizer_config: OptimizerConfig,
    native_budget_allowance: int,
    dense_mask_adapter: DenseMaskAdapter,
    objective_adapter: ObjectiveAdapter,
    checkpoint_directory: Path | None = None,
    resume_from: Path | None = None,
    interrupt_predicate: InterruptPredicate | None = None,
) -> NativeOptimizationResult:
    """Run bounded technical stochastic-gate optimization.

    Exact endpoint evaluation is deliberately absent from this function.
    """

    validate_log_alpha(
        initial_log_alpha,
        component_count=component_count,
    )
    if not component_basis_identity:
        raise ValueError("component_basis_identity must be non-empty")
    if (
        isinstance(native_budget_allowance, bool)
        or not isinstance(native_budget_allowance, int)
        or native_budget_allowance < 0
    ):
        raise ValueError(
            "native_budget_allowance must be non-negative"
        )

    device = initial_log_alpha.device
    dtype = initial_log_alpha.dtype

    if resume_from is None:
        next_step = 0
        log_alpha = initial_log_alpha.detach().clone()
        exp_avg = torch.zeros_like(log_alpha)
        exp_avg_sq = torch.zeros_like(log_alpha)
        trajectory: list[OptimizationTrajectoryPoint] = []
    else:
        (
            next_step,
            log_alpha,
            exp_avg,
            exp_avg_sq,
            trajectory,
        ) = _load_checkpoint(
            Path(resume_from),
            identity=run_identity,
            gate_config=gate_config,
            optimizer_config=optimizer_config,
            component_basis_identity=component_basis_identity,
            component_count=component_count,
            device=device,
            dtype=dtype,
        )

    log_alpha.requires_grad_(True)

    native = NativeBudgetLedger(
        unit="optimizer_step",
        allowance=native_budget_allowance,
    )
    if next_step:
        native.consume(
            next_step,
            detail={"source": "validated_resume_state"},
        )

    latest_checkpoint: Path | None = (
        Path(resume_from) if resume_from is not None else None
    )

    def finish(
        state: TerminalState,
        *,
        failure_reason: str | None = None,
    ) -> NativeOptimizationResult:
        return _result(
            terminal_state=state,
            component_basis_identity=component_basis_identity,
            component_count=component_count,
            native_budget_allowance=native_budget_allowance,
            native_budget_consumed=native.consumed,
            next_step=next_step,
            log_alpha=log_alpha,
            gate_config=gate_config,
            trajectory=trajectory,
            latest_checkpoint=latest_checkpoint,
            failure_reason=failure_reason,
        )

    if next_step >= optimizer_config.max_steps:
        native.terminate(
            detail={"reason": "configured_steps_complete"}
        )
        return finish("completed")

    if native.exhausted:
        native.terminate(
            detail={"reason": "native_budget_exhausted"}
        )
        return finish("exhausted")

    while next_step < optimizer_config.max_steps:
        if native.exhausted:
            native.terminate(
                detail={"reason": "native_budget_exhausted"}
            )
            return finish("exhausted")

        step_index = next_step

        sample = seeded_stochastic_gate_sample(
            log_alpha,
            gate_config,
            _step_identity(run_identity, step_index),
        )
        masked_output = dense_mask_adapter(sample, step_index)
        fidelity_loss = objective_adapter(masked_output, step_index)

        if (
            not isinstance(fidelity_loss, torch.Tensor)
            or fidelity_loss.numel() != 1
        ):
            raise TypeError(
                "objective_adapter must return one scalar tensor"
            )

        l0_mean = expected_l0_probability(
            log_alpha,
            gate_config,
        ).mean()
        total = (
            fidelity_loss
            + optimizer_config.sparsity_coefficient * l0_mean
        )

        if not torch.isfinite(total).item():
            native.fail(
                detail={"reason": "nonfinite_objective"}
            )
            return finish(
                "numerical_failure",
                failure_reason="nonfinite_objective",
            )

        total.backward()
        gradient = log_alpha.grad

        if (
            gradient is None
            or not torch.isfinite(gradient).all().item()
        ):
            native.fail(
                detail={"reason": "nonfinite_gradient"}
            )
            return finish(
                "numerical_failure",
                failure_reason="nonfinite_gradient",
            )

        native.consume(
            1,
            detail={
                "step_index": step_index,
                "unit": "optimizer_step",
            },
        )

        learning_rate = (
            optimizer_config.learning_rate
            * optimizer_config.learning_rate_decay**step_index
        )

        with torch.no_grad():
            exp_avg.mul_(optimizer_config.beta1).add_(
                gradient,
                alpha=1.0 - optimizer_config.beta1,
            )
            exp_avg_sq.mul_(
                optimizer_config.beta2
            ).addcmul_(
                gradient,
                gradient,
                value=1.0 - optimizer_config.beta2,
            )

            adam_step = step_index + 1
            bias1 = 1.0 - optimizer_config.beta1**adam_step
            bias2 = 1.0 - optimizer_config.beta2**adam_step

            denominator = exp_avg_sq.sqrt() / math.sqrt(bias2)
            denominator.add_(optimizer_config.epsilon)

            log_alpha.addcdiv_(
                exp_avg / bias1,
                denominator,
                value=-learning_rate,
            )

        log_alpha.grad = None
        next_step += 1

        with torch.no_grad():
            deterministic_mean = float(
                torch.clamp(
                    torch.sigmoid(log_alpha)
                    * (
                        gate_config.stretch_upper
                        - gate_config.stretch_lower
                    )
                    + gate_config.stretch_lower,
                    min=gate_config.clamp_min,
                    max=gate_config.clamp_max,
                ).mean().item()
            )

        trajectory.append(
            OptimizationTrajectoryPoint(
                step_index=step_index,
                native_consumed=native.consumed,
                fidelity_loss=float(
                    fidelity_loss.detach().item()
                ),
                expected_l0_mean=float(
                    l0_mean.detach().item()
                ),
                total_objective=float(
                    total.detach().item()
                ),
                learning_rate=learning_rate,
                deterministic_gate_mean=deterministic_mean,
            )
        )

        if (
            checkpoint_directory is not None
            and (
                next_step % optimizer_config.checkpoint_every == 0
                or next_step == optimizer_config.max_steps
            )
        ):
            latest_checkpoint = _write_atomic_checkpoint(
                Path(checkpoint_directory),
                _checkpoint_payload(
                    identity=run_identity,
                    gate_config=gate_config,
                    optimizer_config=optimizer_config,
                    component_basis_identity=component_basis_identity,
                    component_count=component_count,
                    next_step=next_step,
                    log_alpha=log_alpha,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    trajectory=trajectory,
                ),
                retention=optimizer_config.checkpoint_retention,
            )

        if (
            interrupt_predicate is not None
            and interrupt_predicate(next_step)
        ):
            native.terminate(
                detail={"reason": "technical_interruption"}
            )
            return finish("interrupted")

    native.terminate(
        detail={"reason": "configured_steps_complete"}
    )
    return finish("completed")

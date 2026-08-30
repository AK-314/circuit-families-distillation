"""Technical hard-concrete gate primitives for Stage 12-R1."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

GATE_RECORD_VERSION = "stage12r1-hard-concrete/v1"
RNG_DERIVATION_VERSION = "stage12r1-complete-identity-seed/v1"


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class GateConfig:
    """Technical-only hard-concrete configuration."""

    temperature: float
    stretch_lower: float
    stretch_upper: float
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    sample_epsilon: float = 1e-6
    scientific_data: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        values = (
            self.temperature,
            self.stretch_lower,
            self.stretch_upper,
            self.clamp_min,
            self.clamp_max,
            self.sample_epsilon,
        )
        if not all(math.isfinite(x) for x in values):
            raise ValueError("gate configuration values must be finite")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.stretch_lower >= 0:
            raise ValueError("stretch_lower must be negative")
        if self.stretch_upper <= 1:
            raise ValueError("stretch_upper must exceed one")
        if not (0 <= self.clamp_min < self.clamp_max <= 1):
            raise ValueError("clamp bounds must satisfy 0 <= min < max <= 1")
        if not (0 < self.sample_epsilon < 0.5):
            raise ValueError("sample_epsilon must lie in (0, 0.5)")
        if self.scientific_data:
            raise ValueError("technical gate config requires scientific_data=false")
        if self.production_eligible:
            raise ValueError(
                "technical gate config requires production_eligible=false"
            )

    def identity_sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class GateRunIdentity:
    """Complete RNG identity for one technical gate stream."""

    method_name: str
    method_version: str
    configuration_reference: str
    run_id: str
    condition_identity: str
    restart_index: int
    seed_value: int
    stream_name: str = "gate_training"

    def __post_init__(self) -> None:
        for value in (
            self.method_name,
            self.method_version,
            self.configuration_reference,
            self.run_id,
            self.condition_identity,
            self.stream_name,
        ):
            if not isinstance(value, str) or not value:
                raise ValueError("identity strings must be non-empty")
        if isinstance(self.restart_index, bool) or self.restart_index < 0:
            raise ValueError("restart_index must be a non-negative integer")
        if isinstance(self.seed_value, bool) or not isinstance(self.seed_value, int):
            raise TypeError("seed_value must be an integer")

    def material_sha256(self) -> str:
        return _sha256(
            {
                "rng_derivation_version": RNG_DERIVATION_VERSION,
                **asdict(self),
            }
        )

    def torch_seed(self) -> int:
        # torch.Generator.manual_seed accepts signed 64-bit-compatible values;
        # use 63 bits to keep the representation portable.
        return int(self.material_sha256()[:16], 16) & ((1 << 63) - 1)


@dataclass(frozen=True)
class GateStateRecord:
    version: str
    component_basis_identity: str
    component_count: int
    dtype: str
    log_alpha: tuple[float, ...]
    config_sha256: str
    state_sha256: str

    def __post_init__(self) -> None:
        if self.version != GATE_RECORD_VERSION:
            raise ValueError("unsupported gate-state record version")
        if not self.component_basis_identity:
            raise ValueError("component_basis_identity must be non-empty")
        if self.component_count < 1:
            raise ValueError("component_count must be positive")
        if len(self.log_alpha) != self.component_count:
            raise ValueError("gate-state length does not match component_count")
        if not all(math.isfinite(x) for x in self.log_alpha):
            raise ValueError("gate-state parameters must be finite")


def validate_log_alpha(
    log_alpha: torch.Tensor,
    *,
    component_count: int,
) -> torch.Tensor:
    if not isinstance(log_alpha, torch.Tensor):
        raise TypeError("log_alpha must be a torch.Tensor")
    if log_alpha.ndim != 1:
        raise ValueError("log_alpha must be one-dimensional")
    if component_count < 1:
        raise ValueError("component_count must be positive")
    if log_alpha.shape[0] != component_count:
        raise ValueError("log_alpha length does not match supplied component basis")
    if not log_alpha.dtype.is_floating_point:
        raise TypeError("log_alpha must use a floating dtype")
    if not torch.isfinite(log_alpha).all().item():
        raise ValueError("log_alpha must contain only finite values")
    return log_alpha


def expected_l0_probability(
    log_alpha: torch.Tensor,
    config: GateConfig,
) -> torch.Tensor:
    validate_log_alpha(log_alpha, component_count=log_alpha.shape[0])
    offset = config.temperature * math.log(
        -config.stretch_lower / config.stretch_upper
    )
    return torch.sigmoid(log_alpha - offset)


def deterministic_gate_values(
    log_alpha: torch.Tensor,
    config: GateConfig,
) -> torch.Tensor:
    validate_log_alpha(log_alpha, component_count=log_alpha.shape[0])
    stretched = (
        torch.sigmoid(log_alpha)
        * (config.stretch_upper - config.stretch_lower)
        + config.stretch_lower
    )
    return torch.clamp(
        stretched,
        min=config.clamp_min,
        max=config.clamp_max,
    )


def seeded_stochastic_gate_sample(
    log_alpha: torch.Tensor,
    config: GateConfig,
    identity: GateRunIdentity,
) -> torch.Tensor:
    validate_log_alpha(log_alpha, component_count=log_alpha.shape[0])

    # Generate the uniform stream on CPU from complete run identity, then move
    # it to the parameter device/dtype. This keeps CPU records reproducible and
    # remains usable with CUDA/MPS-resident differentiable tensors.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(identity.torch_seed())
    u = torch.rand(
        log_alpha.shape,
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    u = u.clamp(config.sample_epsilon, 1.0 - config.sample_epsilon)
    u = u.to(device=log_alpha.device, dtype=log_alpha.dtype)

    logistic_noise = torch.log(u) - torch.log1p(-u)
    relaxed = torch.sigmoid((log_alpha + logistic_noise) / config.temperature)
    stretched = (
        relaxed * (config.stretch_upper - config.stretch_lower)
        + config.stretch_lower
    )
    return torch.clamp(
        stretched,
        min=config.clamp_min,
        max=config.clamp_max,
    )


def deterministic_binary_mask(
    log_alpha: torch.Tensor,
    config: GateConfig,
    *,
    threshold: float,
    component_count: int,
) -> tuple[int, ...]:
    validate_log_alpha(log_alpha, component_count=component_count)
    if not math.isfinite(threshold) or not (0 <= threshold <= 1):
        raise ValueError("threshold must lie in [0, 1]")
    values = deterministic_gate_values(log_alpha, config)
    return tuple(int(x) for x in (values >= threshold).tolist())


def gate_state_record(
    log_alpha: torch.Tensor,
    config: GateConfig,
    *,
    component_basis_identity: str,
    component_count: int,
) -> GateStateRecord:
    validate_log_alpha(log_alpha, component_count=component_count)
    if not component_basis_identity:
        raise ValueError("component_basis_identity must be non-empty")

    values = tuple(float(x) for x in log_alpha.detach().cpu().tolist())
    payload = {
        "version": GATE_RECORD_VERSION,
        "component_basis_identity": component_basis_identity,
        "component_count": component_count,
        "dtype": str(log_alpha.dtype),
        "log_alpha": values,
        "config_sha256": config.identity_sha256(),
    }
    return GateStateRecord(
        **payload,
        state_sha256=_sha256(payload),
    )

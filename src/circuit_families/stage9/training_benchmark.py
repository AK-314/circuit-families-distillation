"""Technical-only backend benchmark for full-domain student updates."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from circuit_families.config import load_model_config, load_training_config
from circuit_families.data.modular_addition import generate_modular_addition_dataset
from circuit_families.models import build_transformer, parameter_count
from circuit_families.training import canonical_state_hash
from circuit_families.training.trainer import build_optimizer

BENCHMARK_SCHEMA_VERSION = "stage9-student-training-backend-benchmark/v1"
MODEL_SEED = 314159
FULL_DOMAIN_COUNT = 12_769
CLASS_COUNT = 113
UPDATE_COUNT_SCENARIOS = (1_000, 5_000, 10_000, 40_000)
ATTEMPT_COUNT_SCENARIOS = (90, 180)


class BenchmarkError(RuntimeError):
    """Raised when a requested backend benchmark cannot be trusted."""


@dataclass(frozen=True)
class TrialResult:
    device: str
    condition: str
    repeat_index: int
    update_seconds: float
    initial_state_sha256: str
    final_state_sha256: str
    dense_output_sha256: str
    dense_outputs: torch.Tensor
    state: dict[str, torch.Tensor]
    checkpoint_payload_bytes: int
    dense_output_bytes: int
    peak_rss_bytes: int


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = _canonical_json_bytes(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    )
    return hashlib.sha256(header + b"\n" + tensor.numpy().tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if platform.system() == "Darwin" else observed * 1024


def _state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone().contiguous()
        for name, value in model.named_parameters()
    }


def _checkpoint_payload_bytes(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    model_bytes = sum(
        value.numel() * value.element_size()
        for value in model.state_dict().values()
    )
    optimizer_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                optimizer_bytes += value.numel() * value.element_size()
    return model_bytes + optimizer_bytes


def _full_domain_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = generate_modular_addition_dataset(modulus=113, equals_token_id=113)
    inputs = torch.as_tensor(dataset["inputs"], dtype=torch.long, device=device)
    labels = torch.as_tensor(dataset["true_labels"], dtype=torch.long, device=device)
    return inputs, labels


def _dense_outputs_cpu(
    model: torch.nn.Module,
    inputs_cpu: torch.Tensor,
    *,
    batch_size: int = 256,
) -> torch.Tensor:
    model = model.to("cpu")
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, inputs_cpu.shape[0], batch_size):
            logits = model(inputs_cpu[start : start + batch_size])[:, -1, :]
            outputs.append(logits.detach().cpu())
    return torch.cat(outputs, dim=0).contiguous()


def _run_trial(
    *,
    repository_root: Path,
    device_name: str,
    condition: str,
    repeat_index: int,
) -> TrialResult:
    if condition not in {"hard_target", "soft_target"}:
        raise BenchmarkError(f"unsupported benchmark condition: {condition}")
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise BenchmarkError("MPS benchmark requested but MPS is unavailable")

    model_config = load_model_config(repository_root / "configs/model.yaml")
    training_config = load_training_config(repository_root / "configs/training.yaml")
    model = build_transformer(model_config, seed=MODEL_SEED, device=device)
    # build_transformer deliberately enables strict deterministic algorithms.
    # MPS lacks a deterministic implementation for one backward operation, so
    # the benchmark must restore the explicit Stage 7B warn-only policy after
    # model construction instead of silently inheriting strict mode.
    torch.use_deterministic_algorithms(True, warn_only=device.type == "mps")
    if parameter_count(model) != 227_313:
        raise BenchmarkError("benchmark architecture parameter count changed")
    optimizer = build_optimizer(model, training_config)
    inputs, labels = _full_domain_tensors(device)
    initial_hash = canonical_state_hash(model.state_dict())

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    logits = model(inputs)[:, -1, :]
    if condition == "hard_target":
        loss = functional.cross_entropy(logits, labels)
    else:
        target = functional.one_hot(labels, num_classes=CLASS_COUNT).to(logits.dtype)
        target = target - target.mean(dim=-1, keepdim=True)
        centred = logits - logits.mean(dim=-1, keepdim=True)
        loss = functional.mse_loss(centred, target)
    if not bool(torch.isfinite(loss)):
        raise BenchmarkError("benchmark loss became nonfinite")
    loss.backward()
    optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started

    for name, value in model.named_parameters():
        if not bool(torch.isfinite(value).all()):
            raise BenchmarkError(
                f"benchmark produced nonfinite model state: {device_name}/{condition}/{name}"
            )

    final_hash = canonical_state_hash(model.state_dict())
    checkpoint_bytes = _checkpoint_payload_bytes(model, optimizer)
    inputs_cpu, _ = _full_domain_tensors(torch.device("cpu"))
    dense = _dense_outputs_cpu(model, inputs_cpu)
    if not bool(torch.isfinite(dense).all()):
        raise BenchmarkError(
            f"benchmark produced nonfinite dense outputs: {device_name}/{condition}"
        )
    state = _state_copy(model)
    return TrialResult(
        device=device_name,
        condition=condition,
        repeat_index=repeat_index,
        update_seconds=elapsed,
        initial_state_sha256=initial_hash,
        final_state_sha256=final_hash,
        dense_output_sha256=_tensor_sha256(dense),
        dense_outputs=dense,
        state=state,
        checkpoint_payload_bytes=checkpoint_bytes,
        dense_output_bytes=dense.numel() * dense.element_size(),
        peak_rss_bytes=_peak_rss_bytes(),
    )


def _compare_pair(left: TrialResult, right: TrialResult) -> dict[str, Any]:
    if (left.device, left.condition) != (right.device, right.condition):
        raise BenchmarkError("trial comparison identities differ")
    if set(left.state) != set(right.state):
        raise BenchmarkError("trial state dictionaries differ")
    differing_elements = 0
    maximum_absolute_parameter_difference = 0.0
    squared_difference = 0.0
    squared_reference = 0.0
    differing_tensors: list[str] = []
    for name in sorted(left.state):
        a = left.state[name].to(torch.float64)
        b = right.state[name].to(torch.float64)
        unequal = a != b
        count = int(unequal.sum().item())
        if count:
            differing_tensors.append(name)
            differing_elements += count
        if a.numel():
            maximum_absolute_parameter_difference = max(
                maximum_absolute_parameter_difference,
                float(torch.max(torch.abs(a - b)).item()),
            )
        squared_difference += float(torch.sum((a - b).square()).item())
        squared_reference += float(torch.sum(a.square()).item())

    output_difference = left.dense_outputs.to(torch.float64) - right.dense_outputs.to(
        torch.float64
    )
    output_reference = left.dense_outputs.to(torch.float64)
    output_squared_difference = float(torch.sum(output_difference.square()).item())
    output_squared_reference = float(torch.sum(output_reference.square()).item())
    argmax_mismatches = int(
        torch.sum(
            torch.argmax(left.dense_outputs, dim=1)
            != torch.argmax(right.dense_outputs, dim=1)
        ).item()
    )
    return {
        "bitwise_state_equal": left.final_state_sha256 == right.final_state_sha256,
        "bitwise_dense_output_equal": (
            left.dense_output_sha256 == right.dense_output_sha256
        ),
        "differing_parameter_element_count": differing_elements,
        "differing_parameter_tensor_names": differing_tensors,
        "maximum_absolute_parameter_difference": (
            maximum_absolute_parameter_difference
        ),
        "relative_parameter_l2_difference": (
            math.sqrt(squared_difference / squared_reference)
            if squared_reference
            else 0.0
        ),
        "maximum_absolute_dense_output_difference": float(
            torch.max(torch.abs(output_difference)).item()
        ),
        "relative_dense_output_l2_difference": (
            math.sqrt(output_squared_difference / output_squared_reference)
            if output_squared_reference
            else 0.0
        ),
        "full_domain_argmax_mismatch_count": argmax_mismatches,
        "full_domain_count": FULL_DOMAIN_COUNT,
    }


def build_training_benchmark_report(
    trials: tuple[TrialResult, ...],
    *,
    source_bindings: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build technical benchmark evidence and linear planning scenarios."""
    grouped: dict[tuple[str, str], list[TrialResult]] = {}
    for trial in trials:
        grouped.setdefault((trial.device, trial.condition), []).append(trial)
    if not grouped:
        raise BenchmarkError("at least one benchmark trial is required")
    device_records = []
    per_device_steady_state: dict[str, float] = {}
    for (device, condition), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda item: item.repeat_index)
        if len(ordered) != 2:
            raise BenchmarkError("each backend/condition requires exactly two repeats")
        comparison = _compare_pair(ordered[0], ordered[1])
        median_seconds = statistics.median(item.update_seconds for item in ordered)
        steady_state_seconds = ordered[-1].update_seconds
        per_device_steady_state.setdefault(device, 0.0)
        per_device_steady_state[device] += steady_state_seconds / 2.0
        device_records.append(
            {
                "device": device,
                "condition": condition,
                "repeat_update_seconds": [item.update_seconds for item in ordered],
                "cold_start_update_seconds": ordered[0].update_seconds,
                "steady_state_update_seconds": steady_state_seconds,
                "median_update_seconds": median_seconds,
                "checkpoint_payload_bytes": max(
                    item.checkpoint_payload_bytes for item in ordered
                ),
                "dense_output_bytes": max(item.dense_output_bytes for item in ordered),
                "peak_rss_bytes": max(item.peak_rss_bytes for item in ordered),
                "comparison": comparison,
            }
        )

    backend_qualification = []
    for device in sorted(per_device_steady_state):
        records = [item for item in device_records if item["device"] == device]
        bitwise = all(
            item["comparison"]["bitwise_state_equal"]
            and item["comparison"]["bitwise_dense_output_equal"]
            for item in records
        )
        semantic = all(
            item["comparison"]["full_domain_argmax_mismatch_count"] == 0
            for item in records
        )
        if device == "cpu" and bitwise:
            disposition = "verified_deterministic_reference_candidate"
        elif device == "mps":
            disposition = "development_only_not_definitive"
        else:
            disposition = "requires_additional_qualification"
        backend_qualification.append(
            {
                "device": device,
                "bitwise_reproducible": bitwise,
                "semantic_argmax_reproducible": semantic,
                "disposition": disposition,
            }
        )

    projections = []
    for device, seconds_per_update in sorted(per_device_steady_state.items()):
        for updates in UPDATE_COUNT_SCENARIOS:
            for attempts in ATTEMPT_COUNT_SCENARIOS:
                projections.append(
                    {
                        "device": device,
                        "updates_per_attempt": updates,
                        "attempt_count": attempts,
                        "linear_wall_clock_hours_one_worker": (
                            seconds_per_update * updates * attempts / 3600.0
                        ),
                        "seconds_per_update_basis": (
                            "mean_of_hard_and_soft_second_repeat_steady_state"
                        ),
                        "interpretation": "upper_level_planning_scenario_not_frozen",
                    }
                )

    maximum_checkpoint = max(item["checkpoint_payload_bytes"] for item in device_records)
    maximum_dense = max(item["dense_output_bytes"] for item in device_records)
    storage = [
        {
            "attempt_count": attempts,
            "checkpoint_plus_dense_output_bytes": attempts
            * (maximum_checkpoint + maximum_dense),
            "excludes_intermediate_checkpoints": True,
        }
        for attempts in ATTEMPT_COUNT_SCENARIOS
    ]
    intermediate_checkpoint_storage = [
        {
            "updates_per_attempt": updates,
            "attempt_count": attempts,
            "checkpoint_interval_updates": 50,
            "checkpoint_count_per_attempt_including_initial_and_final": (
                updates // 50 + 1
            ),
            "checkpoint_bytes": (
                attempts * (updates // 50 + 1) * maximum_checkpoint
            ),
            "interpretation": "legacy_interval_scenario_not_frozen",
        }
        for updates in UPDATE_COUNT_SCENARIOS
        for attempts in ATTEMPT_COUNT_SCENARIOS
    ]
    report: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "classification": "technical_benchmark_only",
        "scientific_data": False,
        "production_eligible": False,
        "resolves_decisions": [],
        "architecture_parameter_count": 227_313,
        "full_domain_count": FULL_DOMAIN_COUNT,
        "conditions": ["hard_target", "soft_target"],
        "repeats_per_backend_condition": 2,
        "native_updates_per_trial": 1,
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "source_bindings": dict(sorted((source_bindings or {}).items())),
        "measurements": device_records,
        "backend_qualification": backend_qualification,
        "unmeasured_backends": [
            {
                "device": "cuda",
                "disposition": "must_pass_same_exact_and_semantic_qualification_before_use",
            }
        ],
        "runtime_projections": projections,
        "storage_projections": storage,
        "intermediate_checkpoint_storage_projections": (
            intermediate_checkpoint_storage
        ),
        "decision_boundary": {
            "optimizer_schedule_stopping_attempt_cap_frozen": False,
            "cpu_reference_candidate_only": True,
            "mps_definitive_training_authorized": False,
            "scientific_red_team_started": False,
        },
        "stage9_complete": True,
        "stage10_started": False,
    }
    report["report_sha256"] = hashlib.sha256(_canonical_json_bytes(report)).hexdigest()
    return report


def run_training_benchmark(
    *,
    repository_root: str | Path,
    devices: tuple[str, ...],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    trials = tuple(
        _run_trial(
            repository_root=root,
            device_name=device,
            condition=condition,
            repeat_index=repeat_index,
        )
        for device in devices
        for condition in ("hard_target", "soft_target")
        for repeat_index in range(2)
    )
    return build_training_benchmark_report(
        trials,
        source_bindings={
            "benchmark_module_sha256": _file_sha256(Path(__file__)),
            "model_config_sha256": _file_sha256(root / "configs/model.yaml"),
            "training_config_sha256": _file_sha256(root / "configs/training.yaml"),
        },
    )


def validate_training_benchmark_report(record: dict[str, Any]) -> None:
    """Validate a committed machine-specific Stage 9 benchmark report."""
    if record.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkError("unsupported Stage 9 benchmark report schema")
    if record.get("scientific_data") is not False:
        raise BenchmarkError("Stage 9 report cannot contain scientific data")
    if record.get("production_eligible") is not False:
        raise BenchmarkError("Stage 9 report cannot grant production eligibility")
    if record.get("resolves_decisions") != []:
        raise BenchmarkError("Stage 9 report cannot resolve decisions")
    if record.get("stage9_complete") is not True:
        raise BenchmarkError("Stage 9 report is not complete")
    if record.get("stage10_started") is not False:
        raise BenchmarkError("Stage 9 report must predate Stage 10")
    supplied_hash = record.get("report_sha256")
    payload = dict(record)
    payload.pop("report_sha256", None)
    expected_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if supplied_hash != expected_hash:
        raise BenchmarkError("Stage 9 benchmark report hash mismatch")
    measurements = record.get("measurements")
    if not isinstance(measurements, list):
        raise BenchmarkError("Stage 9 measurements must be a list")
    identities = {
        (item.get("device"), item.get("condition"))
        for item in measurements
        if isinstance(item, dict)
    }
    if identities != {
        ("cpu", "hard_target"),
        ("cpu", "soft_target"),
        ("mps", "hard_target"),
        ("mps", "soft_target"),
    }:
        raise BenchmarkError("Stage 9 backend/condition coverage is incomplete")
    qualifications = {
        item["device"]: item
        for item in record.get("backend_qualification", [])
        if isinstance(item, dict) and isinstance(item.get("device"), str)
    }
    if not qualifications.get("cpu", {}).get("bitwise_reproducible"):
        raise BenchmarkError("CPU did not qualify as deterministic reference")
    if qualifications.get("mps", {}).get("disposition") != (
        "development_only_not_definitive"
    ):
        raise BenchmarkError("MPS must remain development-only")

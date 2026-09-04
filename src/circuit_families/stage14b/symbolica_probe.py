"""Synthetic, non-scientific qualification probe for a Symbolica practice node."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from circuit_families.config import load_model_config
from circuit_families.data.modular_addition import generate_modular_addition_dataset
from circuit_families.interpretability.centred_logit_fidelity import (
    CentredLogitPredictiveAccumulator,
    centre_logits_across_classes,
)
from circuit_families.interpretability.component_ablation import masked_model_logits
from circuit_families.interpretability.masks import ComponentMask
from circuit_families.models import build_transformer, parameter_count
from circuit_families.stage9 import run_training_benchmark

from .records import Stage14BError, canonical_json_bytes, canonical_sha256, with_boundary
from .resources import inventory_resource_pool, qualify_backend

PROBE_SCHEMA_VERSION = "stage14-symbolica-practice-node-probe/v1"
FULL_DOMAIN_COUNT = 12_769
CLASS_COUNT = 113
COMPONENT_COUNT = 516
MEASURED_REPEATS = 3
WARMUP_REPEATS = 1
TARGET_CUDA_DEVICES = 16
TARGET_CPU_CORES = 64
TOTAL_WINDOW_HOURS = 96
AUDIT_RESERVE_HOURS = 12


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if platform.system() == "Darwin" else observed * 1024


def _available_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def _cuda_devices() -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
        for index in range(torch.cuda.device_count())
        for properties in (torch.cuda.get_device_properties(index),)
    ]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _torch_thread_limit(threads: int):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _timed_repeats(operation: Callable[[], float], *, device: torch.device) -> dict[str, Any]:
    for _ in range(WARMUP_REPEATS):
        value = operation()
        if not isinstance(value, float) or not torch.isfinite(torch.tensor(value)):
            raise Stage14BError("warmup produced a non-finite scalar")
    elapsed: list[float] = []
    values: list[float] = []
    for _ in range(MEASURED_REPEATS):
        _synchronize(device)
        started = time.perf_counter()
        value = operation()
        _synchronize(device)
        elapsed.append(time.perf_counter() - started)
        values.append(value)
    if not all(seconds > 0 for seconds in elapsed):
        raise Stage14BError("probe recorded a non-positive duration")
    return {
        "discarded_warmups": WARMUP_REPEATS,
        "measured_repeats": MEASURED_REPEATS,
        "elapsed_seconds": elapsed,
        "median_seconds": statistics.median(elapsed),
        "minimum_seconds": min(elapsed),
        "maximum_seconds": max(elapsed),
        "finite_scalar_outputs": values,
    }


def _full_domain(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = generate_modular_addition_dataset(modulus=113, equals_token_id=113)
    inputs = torch.as_tensor(dataset["inputs"], dtype=torch.long, device=device)
    labels = torch.as_tensor(dataset["true_labels"], dtype=torch.long, device=device)
    if inputs.shape[0] != FULL_DOMAIN_COUNT:
        raise Stage14BError("canonical synthetic domain size changed")
    return inputs, labels


def _training_measurements(repository_root: Path) -> dict[str, Any]:
    report = run_training_benchmark(
        repository_root=repository_root,
        devices=("cpu", "cuda"),
    )
    wanted = {
        ("cpu", "hard_target"),
        ("cpu", "soft_target"),
        ("cuda", "hard_target"),
        ("cuda", "soft_target"),
    }
    measurements = [
        item
        for item in report["measurements"]
        if (item["device"], item["condition"]) in wanted
    ]
    if {(item["device"], item["condition"]) for item in measurements} != wanted:
        raise Stage14BError("training benchmark omitted a required device/condition")
    qualifications = {
        item["device"]: item for item in report["backend_qualification"]
    }
    if not qualifications["cpu"]["bitwise_reproducible"]:
        raise Stage14BError("CPU training reference was not bitwise reproducible")
    if not qualifications["cuda"]["semantic_argmax_reproducible"]:
        raise Stage14BError("CUDA training repeats were not semantically reproducible")
    return {
        "source_schema_version": report["schema_version"],
        "architecture_parameter_count": report["architecture_parameter_count"],
        "full_domain_count": report["full_domain_count"],
        "native_updates_per_trial": report["native_updates_per_trial"],
        "measurements": measurements,
        "backend_qualification": report["backend_qualification"],
        "source_report_sha256": report["report_sha256"],
        "interpretation": (
            "actual one-update full-domain timings; short-run evidence, not a "
            "long-run efficiency guarantee"
        ),
    }


def _model_forward_cross_backend(repository_root: Path) -> dict[str, Any]:
    config = load_model_config(repository_root / "configs/model.yaml")
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    cpu_model = build_transformer(config, seed=223607, device=cpu)
    cuda_model = build_transformer(config, seed=223607, device=cuda)
    cuda_model.load_state_dict(cpu_model.state_dict())
    cpu_model.eval()
    cuda_model.eval()
    cpu_inputs, _ = _full_domain(cpu)
    batch_size = 256
    cpu_parts = []
    cuda_parts = []
    with torch.inference_mode():
        for start in range(0, FULL_DOMAIN_COUNT, batch_size):
            stop = min(start + batch_size, FULL_DOMAIN_COUNT)
            cpu_logits = cpu_model(cpu_inputs[start:stop])[:, -1, :]
            cuda_logits = cuda_model(cpu_inputs[start:stop].to(cuda))[:, -1, :]
            cpu_parts.append(centre_logits_across_classes(cpu_logits).cpu())
            cuda_parts.append(centre_logits_across_classes(cuda_logits).cpu())
    reference = torch.cat(cpu_parts, dim=0).to(torch.float64)
    observed = torch.cat(cuda_parts, dim=0).to(torch.float64)
    difference = observed - reference
    maximum_absolute = float(torch.max(torch.abs(difference)).item())
    reference_norm = float(torch.linalg.vector_norm(reference).item())
    relative_l2 = float(torch.linalg.vector_norm(difference).item()) / max(
        reference_norm, 1e-12
    )
    mismatches = int(
        torch.sum(torch.argmax(reference, dim=1) != torch.argmax(observed, dim=1)).item()
    )
    absolute_tolerance = 1e-6
    relative_tolerance = 1e-5
    within_tolerance = (
        maximum_absolute <= absolute_tolerance or relative_l2 <= relative_tolerance
    )
    if not within_tolerance or mismatches:
        raise Stage14BError(
            "full-domain CPU/CUDA model-forward semantic qualification failed: "
            f"max_abs={maximum_absolute}, relative_l2={relative_l2}, "
            f"argmax_mismatches={mismatches}"
        )
    return {
        "full_domain_count": FULL_DOMAIN_COUNT,
        "class_count": CLASS_COUNT,
        "batch_size": batch_size,
        "same_cpu_initialized_state_loaded_on_cuda": True,
        "maximum_absolute_centred_logit_difference": maximum_absolute,
        "relative_centred_logit_l2_difference": relative_l2,
        "full_domain_argmax_mismatch_count": mismatches,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "within_tolerance": within_tolerance,
        "semantic_status": "PASS",
    }


def _continuous_masked_logits(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    gate_values: torch.Tensor,
) -> torch.Tensor:
    head_values = gate_values[:4].view(1, 1, 4, 1)
    neuron_values = gate_values[4:].view(1, 1, 512)

    def head_hook(activation: torch.Tensor, _hook: object) -> torch.Tensor:
        return activation * head_values

    def neuron_hook(activation: torch.Tensor, _hook: object) -> torch.Tensor:
        return activation * neuron_values

    with model.hooks(
        fwd_hooks=[
            ("blocks.0.attn.hook_z", head_hook),
            ("blocks.0.mlp.hook_post", neuron_hook),
        ]
    ):
        return model(inputs)[:, -1, :]


def _model_in_loop_hard_concrete(repository_root: Path) -> dict[str, Any]:
    device = torch.device("cuda")
    model = build_transformer(
        load_model_config(repository_root / "configs/model.yaml"),
        seed=141421,
        device=device,
    )
    if parameter_count(model) != 227_313:
        raise Stage14BError("canonical architecture parameter count changed")
    model.eval()
    model.requires_grad_(False)
    inputs, _ = _full_domain(device)
    with torch.inference_mode():
        reference = centre_logits_across_classes(model(inputs)[:, -1, :]).detach()
    log_alpha = torch.zeros(COMPONENT_COUNT, device=device, requires_grad=True)
    optimizer = torch.optim.Adam((log_alpha,), lr=0.01)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    def operation() -> float:
        optimizer.zero_grad(set_to_none=True)
        gates = torch.sigmoid(log_alpha)
        logits = _continuous_masked_logits(model, inputs, gates)
        centred = centre_logits_across_classes(logits)
        fidelity_loss = functional.mse_loss(centred, reference)
        objective = fidelity_loss + 0.001 * gates.mean()
        if not bool(torch.isfinite(objective)):
            raise Stage14BError("model-in-loop objective became non-finite")
        objective.backward()
        if log_alpha.grad is None or not bool(torch.isfinite(log_alpha.grad).all()):
            raise Stage14BError("model-in-loop gate gradient is missing or non-finite")
        optimizer.step()
        return float(objective.detach().cpu())

    timing = _timed_repeats(operation, device=device)
    timing.update(
        {
            "component_count": COMPONENT_COUNT,
            "full_domain_count": FULL_DOMAIN_COUNT,
            "model_parameter_count": parameter_count(model),
            "precision": "float32",
            "objective": "centred-logit-MSE-plus-gate-mean-technical-proxy",
            "steps_per_production_restart": 5000,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "interpretation": (
                "actual full-domain differentiable model-in-loop gate steps; "
                "three-step timing is extrapolative and must retain margin"
            ),
        }
    )
    return timing


def _exact_mask_evaluation_single_thread(repository_root: Path) -> dict[str, Any]:
    device = torch.device("cpu")
    model = build_transformer(
        load_model_config(repository_root / "configs/model.yaml"),
        seed=173205,
        device=device,
    )
    model.eval()
    inputs, _ = _full_domain(device)
    batch_size = 256
    reference_parts = []
    with torch.inference_mode():
        for start in range(0, FULL_DOMAIN_COUNT, batch_size):
            logits = model(inputs[start : start + batch_size])[:, -1, :]
            reference_parts.append(centre_logits_across_classes(logits))
    reference = torch.cat(reference_parts, dim=0).contiguous()
    mask = ComponentMask(
        attention_head_mask=(1, 1, 1, 1),
        mlp_neuron_mask=tuple(0 if index % 4 == 0 else 1 for index in range(512)),
    )

    def operation() -> float:
        accumulator = CentredLogitPredictiveAccumulator(
            expected_example_count=FULL_DOMAIN_COUNT,
            class_count=CLASS_COUNT,
        )
        for start in range(0, FULL_DOMAIN_COUNT, batch_size):
            stop = min(start + batch_size, FULL_DOMAIN_COUNT)
            logits = masked_model_logits(model, inputs[start:stop], mask)[:, -1, :]
            accumulator.update(
                reference[start:stop],
                centre_logits_across_classes(logits),
                start_index=start,
            )
        return float(accumulator.finalize())

    timing = _timed_repeats(operation, device=device)
    if max(timing["finite_scalar_outputs"]) - min(timing["finite_scalar_outputs"]) > 1e-12:
        raise Stage14BError("CPU exact-evaluation result was not repeatable")
    timing.update(
        {
            "full_domain_count": FULL_DOMAIN_COUNT,
            "inference_batch_size": batch_size,
            "retained_component_count": mask.retained_component_count,
            "all_campaign_exact_evaluations": 615_424,
            "execution_target": "standalone CPU workers",
            "torch_cpu_threads": 1,
            "interpretation": (
                "actual full-domain exact masked evaluations on one CPU core; "
                "campaign concurrency still requires a multi-worker scaling check"
            ),
        }
    )
    return timing


def _exact_mask_evaluation(repository_root: Path) -> dict[str, Any]:
    with _torch_thread_limit(1):
        return _exact_mask_evaluation_single_thread(repository_root)


def _nvidia_smi_summary() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            query,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error_type": type(exc).__name__, "rows": []}
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "available": completed.returncode == 0,
        "return_code": completed.returncode,
        "rows": rows,
        "stderr_excerpt": completed.stderr.strip()[:500],
    }


def _planning_projection(
    training: Mapping[str, Any],
    hard_concrete: Mapping[str, Any],
    exact: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    cuda_training = [
        item
        for item in training["measurements"]
        if item["device"] == "cuda"
    ]
    seconds_per_update = statistics.median(
        item["steady_state_update_seconds"] for item in cuda_training
    )
    hard_seconds = float(hard_concrete["median_seconds"])
    exact_seconds = float(exact["median_seconds"])
    scope = json.loads(
        (
            repository_root
            / "followup/manifests/stage13_scope_resource_projection_v3.json"
        ).read_text(encoding="utf-8")
    )["scopes"]["protected_core"]
    central = next(
        item for item in scope["scenarios"] if item["scenario"] == "central"
    )["compute"]
    # Frozen central workload counts from Stage 13 v3. These calculations are
    # intentionally transparent and do not themselves authorize execution.
    central_training_updates = int(central["training_canonical_equivalent_updates"])
    central_hard_steps = int(central["hard_concrete_model_in_loop_steps"])
    exact_evaluations = 615_424
    raw_gpu_device_hours = (
        central_training_updates * seconds_per_update
        + central_hard_steps * hard_seconds
    ) / 3600.0
    raw_cpu_core_hours = exact_evaluations * exact_seconds / 3600.0
    gpu_efficiency = 0.70
    cpu_efficiency = 0.70
    gpu_capacity_wall = raw_gpu_device_hours / (TARGET_CUDA_DEVICES * gpu_efficiency)
    cpu_capacity_wall = raw_cpu_core_hours / (TARGET_CPU_CORES * cpu_efficiency)
    return {
        "scope": "protected-core central-count diagnostic projection",
        "target_cuda_devices": TARGET_CUDA_DEVICES,
        "target_cpu_cores": TARGET_CPU_CORES,
        "total_window_hours": TOTAL_WINDOW_HOURS,
        "audit_reserve_hours": AUDIT_RESERVE_HOURS,
        "science_window_hours": TOTAL_WINDOW_HOURS - AUDIT_RESERVE_HOURS,
        "assumed_gpu_scheduling_efficiency": gpu_efficiency,
        "assumed_cpu_scheduling_efficiency": cpu_efficiency,
        "measured_seconds_per_training_update": seconds_per_update,
        "measured_seconds_per_model_in_loop_gate_step": hard_seconds,
        "measured_seconds_per_exact_evaluation_one_cpu_core": exact_seconds,
        "central_training_update_count": central_training_updates,
        "central_hard_concrete_step_count": central_hard_steps,
        "exact_evaluation_count": exact_evaluations,
        "raw_gpu_device_hours": raw_gpu_device_hours,
        "raw_exact_cpu_core_hours": raw_cpu_core_hours,
        "gpu_capacity_wall_hours": gpu_capacity_wall,
        "exact_cpu_capacity_wall_hours": cpu_capacity_wall,
        "additive_capacity_wall_hours_before_other_work": gpu_capacity_wall
        + cpu_capacity_wall,
        "fits_science_window_before_other_work": (
            gpu_capacity_wall + cpu_capacity_wall
            <= TOTAL_WINDOW_HOURS - AUDIT_RESERVE_HOURS
        ),
        "not_included": [
            "queue latency and scheduler launch overhead",
            "teacher/student early stopping variation",
            "Fourier trials, calibration, packing, merge and export",
            "filesystem contention and multi-worker scaling",
            "alternate-architecture throughput differences",
        ],
        "decision": "diagnostic_only_not_launch_authorization",
    }


def validate_probe_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise Stage14BError("unsupported Symbolica probe schema")
    for field in (
        "scientific_data",
        "production_eligible",
        "definitive_execution_started",
        "stage15_started",
        "registered_or_private_artifacts_accessed",
    ):
        if report.get(field) is not False:
            raise Stage14BError(f"probe report violates {field}=false")
    if report.get("probe_status") != "PASS":
        raise Stage14BError("probe report is not complete")
    payload = dict(report)
    supplied = payload.pop("report_sha256", None)
    if supplied != canonical_sha256(payload):
        raise Stage14BError("Symbolica probe report hash mismatch")


def run_symbolica_probe(repository_root: Path, output_root: Path) -> dict[str, Any]:
    """Run the bounded synthetic practice-node probe and write its report."""
    root = repository_root.resolve()
    output = output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise Stage14BError("probe output directory must be empty")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise Stage14BError("Symbolica probe requires at least one CUDA GPU")

    inventory = inventory_resource_pool(
        provider="symbolica",
        site="practice-node",
        pool_reference="symbolica-practice-node-observation/v1",
        permitted_roots=(output,),
        production_pool=False,
    )
    correctness = {
        backend: qualify_backend(
            inventory,
            backend=backend,
            output_root=output / f"correctness-{backend}",
            absolute_tolerance=1e-6,
            relative_tolerance=1e-5,
            memory_ceiling_bytes=max(4 * 1024**3, _peak_rss_bytes() * 4),
            disk_ceiling_bytes=1024**2,
            time_ceiling_seconds=120,
        )
        for backend in ("cpu", "cuda")
    }
    model_forward = _model_forward_cross_backend(root)
    training = _training_measurements(root)
    hard_concrete = _model_in_loop_hard_concrete(root)
    exact = _exact_mask_evaluation(root)
    projection = _planning_projection(
        training,
        hard_concrete,
        exact,
        repository_root=root,
    )
    report = with_boundary(
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "purpose": "synthetic practice-node qualification and planning evidence",
            "probe_status": "PASS",
            "source_commit": _git_head(root),
            "source_bindings": {
                "model_config_sha256": _file_sha256(root / "configs/model.yaml"),
                "training_config_sha256": _file_sha256(root / "configs/training.yaml"),
                "stage13_projection_sha256": _file_sha256(
                    root / "followup/manifests/stage13_scope_resource_projection_v3.json"
                ),
            },
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_devices": _cuda_devices(),
                "logical_cpu_count": os.cpu_count(),
                "available_cpu_affinity_count": _available_cpu_count(),
                "torch_cpu_threads": torch.get_num_threads(),
                "peak_process_rss_bytes": _peak_rss_bytes(),
                "nvidia_smi": _nvidia_smi_summary(),
            },
            "inventory": inventory,
            "correctness_microqualification": correctness,
            "full_domain_model_forward_qualification": model_forward,
            "representative_training": training,
            "representative_model_in_loop_hard_concrete": hard_concrete,
            "representative_exact_mask_evaluation": exact,
            "diagnostic_projection": projection,
            "limitations": [
                "one practice node is not a two-node scaling measurement",
                "three timed repeats do not establish long-run throughput",
                "the exact-evaluation test does not establish 64-worker efficiency",
                "no queue, preemption, shared-filesystem or interconnect stress test",
                "no registered teacher or student checkpoint was read",
                "this probe cannot authorize Stage 15",
            ],
            "registered_or_private_artifacts_accessed": False,
            "stage15_started": False,
        }
    )
    report["report_sha256"] = canonical_sha256(report)
    validate_probe_report(report)
    (output / "symbolica-probe-report.json").write_bytes(canonical_json_bytes(report))
    return report

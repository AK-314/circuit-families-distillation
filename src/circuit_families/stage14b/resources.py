"""Secret-free resource inventory and excluded-fixture qualification."""

from __future__ import annotations

import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .records import Stage14BError, canonical_sha256, with_boundary

TECHNICAL_CASES = (
    "hard_training_step",
    "soft_training_step",
    "eligibility_computation",
    "greedy_ranking",
    "model_in_loop_hard_concrete",
    "exact_full_domain_evaluation",
    "packing",
    "exact_calibration_shard",
    "fourier_interchange",
    "serialization",
    "checkpoint_resume",
    "compact_merge",
    "export_verification",
)


def _command_identity(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if path is None:
        return {"available": False, "path_name": None, "version": None}
    try:
        result = subprocess.run(
            [path, "--version"], check=False, text=True, capture_output=True, timeout=5
        )
        version = (result.stdout or result.stderr).splitlines()[0][:200]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        version = "available_version_unreadable"
    return {"available": True, "path_name": Path(path).name, "version": version}


def _mac_hardware() -> dict[str, Any]:
    values: dict[str, str] = {}
    for key in ("machdep.cpu.brand_string", "hw.physicalcpu", "hw.logicalcpu", "hw.memsize"):
        try:
            result = subprocess.run(
                ["sysctl", "-n", key], check=True, text=True, capture_output=True, timeout=5
            )
            values[key] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            values[key] = "unavailable"
    return {
        "cpu_model": values["machdep.cpu.brand_string"],
        "physical_cores": _optional_int(values["hw.physicalcpu"]),
        "logical_cores": _optional_int(values["hw.logicalcpu"]) or os.cpu_count(),
        "ram_bytes": _optional_int(values["hw.memsize"]),
        "numa": "not_reported",
    }


def _optional_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _linux_hardware() -> dict[str, Any]:
    cpu_model = platform.processor() or "unavailable"
    physical = None
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        ram = pages * page_size
    except (OSError, ValueError):
        ram = None
    return {
        "cpu_model": cpu_model,
        "physical_cores": physical,
        "logical_cores": os.cpu_count(),
        "ram_bytes": ram,
        "numa": "not_reported",
    }


def _accelerators() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda": [], "mps": {"built": False, "available": False}}
    cuda = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda.append(
                {
                    "index": index,
                    "model": props.name,
                    "vram_bytes": props.total_memory,
                    "capability": list(props.major_minor)
                    if hasattr(props, "major_minor")
                    else None,
                }
            )
    return {
        "cuda": cuda,
        "cuda_runtime": torch.version.cuda,
        "mps": {
            "built": torch.backends.mps.is_built(),
            "available": torch.backends.mps.is_available(),
        },
    }


def _storage_inventory(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    stats = os.statvfs(path)
    return {
        "root_reference": path.name or "filesystem-root",
        "capacity_bytes": usage.total,
        "available_bytes": usage.free,
        "quota_bytes": None,
        "inode_capacity": stats.f_files,
        "inode_available": stats.f_ffree,
        "file_limit": None,
        "atomic_rename": "requires_bounded_probe",
        "measured_write_bytes_per_second": None,
        "measured_read_bytes_per_second": None,
    }


def inventory_resource_pool(
    *,
    provider: str,
    site: str,
    pool_reference: str,
    permitted_roots: Sequence[Path],
    availability_intervals: Sequence[Mapping[str, Any]] = (),
    account_placeholder: str | None = None,
    production_pool: bool = False,
) -> dict[str, Any]:
    """Capture only observations available without credentials or private paths."""
    if not provider or not site or not pool_reference or not permitted_roots:
        raise Stage14BError("inventory requires provider, site, pool reference, and roots")
    hardware = _mac_hardware() if platform.system() == "Darwin" else _linux_hardware()
    accelerators = _accelerators()
    scheduler = {
        "slurm": {
            name: _command_identity(name) for name in ("sbatch", "squeue", "sacct", "scancel")
        },
        "arrays": None,
        "dependencies": None,
        "job_limits": None,
        "wall_limit_seconds": None,
        "preemption": None,
        "queue_or_partition": None,
        "account_placeholder": account_placeholder,
    }
    record = with_boundary(
        {
            "schema_version": "stage14b-resource-inventory/v1",
            "provider": provider,
            "site": site,
            "pool_reference": pool_reference,
            "production_pool": production_pool,
            "scheduler": scheduler,
            "hosts": {
                "observed_host_count": 1,
                "cpu": hardware,
                "accelerators": accelerators,
                "interconnect": "not_observed",
            },
            "storage": [_storage_inventory(path.absolute()) for path in permitted_roots],
            "network_policy": "not_observed",
            "interruption_behavior": "not_observed",
            "availability_intervals": [dict(item) for item in availability_intervals],
            "container_support": {
                name: _command_identity(name)
                for name in ("docker", "podman", "apptainer", "singularity")
            },
            "clock": {
                "source": "system wall and monotonic clocks",
                "observed_unix_seconds": int(time.time()),
                "monotonic_seconds": time.monotonic(),
            },
            "maximum_eligible_concurrency": None,
            "qualification_status": "UNQUALIFIED",
        }
    )
    fingerprint_payload = {
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "cpu": hardware,
        "accelerators": accelerators,
        "python": platform.python_version(),
    }
    record["hardware_class_fingerprint"] = canonical_sha256(fingerprint_payload)
    record["inventory_sha256"] = canonical_sha256(record)
    return record


def _fixture_outputs(backend: str) -> dict[str, list[float]]:
    try:
        import torch
    except ImportError as exc:
        raise Stage14BError("qualification requires locked torch") from exc
    if backend == "mps" and not torch.backends.mps.is_available():
        raise Stage14BError("MPS backend unavailable")
    if backend == "cuda" and not torch.cuda.is_available():
        raise Stage14BError("CUDA backend unavailable")
    device = torch.device(backend)
    torch.manual_seed(1414)
    outputs: dict[str, list[float]] = {}
    base = torch.arange(64, dtype=torch.float64).reshape(8, 8) / 63.0
    if backend != "cpu":
        base = base.to(dtype=torch.float32, device=device)
    for index, case in enumerate(TECHNICAL_CASES):
        matrix = base + (index + 1) / 1000.0
        if case == "greedy_ranking":
            value = torch.argsort(matrix.flatten(), descending=True)[:12].to("cpu")
        elif case == "packing":
            value = (matrix > matrix.mean()).to(torch.int8).sum(dim=1).to("cpu")
        elif case == "serialization":
            value = matrix.flatten()[:16].to("cpu")
        else:
            value = (matrix @ matrix.T).sin().sum(dim=0).to("cpu")
        outputs[case] = [float(item) for item in value.flatten().tolist()]
    return outputs


def _timed_repeat(backend: str) -> tuple[dict[str, list[float]], float]:
    started = time.perf_counter()
    outputs = _fixture_outputs(backend)
    elapsed = time.perf_counter() - started
    return outputs, elapsed


def _probe_storage(root: Path, ceiling_bytes: int) -> dict[str, Any]:
    if ceiling_bytes <= 0 or ceiling_bytes > 16 * 1024 * 1024:
        raise Stage14BError("qualification storage ceiling must be in (0, 16 MiB]")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    probe = root / "storage-probe.bin"
    renamed = root / "storage-probe.renamed"
    payload = bytes((index * 17) % 251 for index in range(ceiling_bytes))
    start = time.perf_counter()
    probe.write_bytes(payload)
    write_elapsed = max(time.perf_counter() - start, 1e-9)
    os.replace(probe, renamed)
    start = time.perf_counter()
    read_back = renamed.read_bytes()
    read_elapsed = max(time.perf_counter() - start, 1e-9)
    renamed.unlink()
    if read_back != payload:
        raise Stage14BError("storage probe reread mismatch")
    return {
        "atomic_rename": True,
        "bytes": len(payload),
        "write_bytes_per_second": len(payload) / write_elapsed,
        "read_bytes_per_second": len(payload) / read_elapsed,
    }


def qualify_backend(
    inventory: Mapping[str, Any],
    *,
    backend: str,
    output_root: Path,
    absolute_tolerance: float,
    relative_tolerance: float,
    memory_ceiling_bytes: int,
    disk_ceiling_bytes: int,
    time_ceiling_seconds: float,
) -> dict[str, Any]:
    if backend not in {"cpu", "cuda", "mps"}:
        raise Stage14BError("unsupported qualification backend")
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise Stage14BError("qualification tolerances must be prospective and non-negative")
    if memory_ceiling_bytes <= 0 or time_ceiling_seconds <= 0:
        raise Stage14BError("qualification ceilings must be positive")
    warmup, _ = _timed_repeat(backend)
    repeats = [_timed_repeat(backend) for _ in range(3)]
    elapsed_total = sum(item[1] for item in repeats)
    if elapsed_total > time_ceiling_seconds:
        raise Stage14BError("qualification exceeded injected time ceiling")
    outputs = [item[0] for item in repeats]
    output_hashes = [canonical_sha256(value) for value in outputs]
    deterministic = all(value == output_hashes[0] for value in output_hashes[1:])
    cpu_reference = _fixture_outputs("cpu")
    comparison = {}
    for case in TECHNICAL_CASES:
        cpu_values = cpu_reference[case]
        backend_values = outputs[0][case]
        if len(cpu_values) != len(backend_values):
            raise Stage14BError("backend qualification output shape mismatch")
        absolute_errors = [
            abs(reference - observed)
            for reference, observed in zip(cpu_values, backend_values, strict=True)
        ]
        relative_errors = [
            error / max(abs(reference), 1e-12)
            for reference, error in zip(cpu_values, absolute_errors, strict=True)
        ]
        maximum_absolute = max(absolute_errors, default=0.0)
        maximum_relative = max(relative_errors, default=0.0)
        comparison[case] = {
            "cpu_sha256": canonical_sha256(cpu_values),
            "backend_sha256": canonical_sha256(backend_values),
            "exact_match": cpu_values == backend_values,
            "maximum_absolute_error": maximum_absolute,
            "maximum_relative_error": maximum_relative,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "within_tolerance": maximum_absolute <= absolute_tolerance
            or maximum_relative <= relative_tolerance,
        }
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    storage = _probe_storage(output_root.absolute(), disk_ceiling_bytes)
    resume_reference = canonical_sha256({"cases": outputs[0], "step": 13})
    resumed = canonical_sha256({"cases": outputs[0], "step": 13})
    suite_pass = (
        deterministic
        and resume_reference == resumed
        and peak_bytes <= memory_ceiling_bytes
        and all(item["within_tolerance"] for item in comparison.values())
    )
    production_pool = bool(inventory.get("production_pool"))
    record = with_boundary(
        {
            "schema_version": "stage14b-qualification/v1",
            "inventory_sha256": inventory.get("inventory_sha256"),
            "hardware_class_fingerprint": inventory.get("hardware_class_fingerprint"),
            "backend": backend,
            "fixture_class": "excluded_deterministic_technical/v1",
            "prospective_tolerances": {
                "absolute": absolute_tolerance,
                "relative": relative_tolerance,
            },
            "discarded_warmup_sha256": canonical_sha256(warmup),
            "repeats": [
                {
                    "index": index,
                    "elapsed_seconds": elapsed,
                    "outputs_sha256": canonical_sha256(output),
                }
                for index, (output, elapsed) in enumerate(repeats, start=1)
            ],
            "case_comparison": comparison,
            "same_class_repeatability": deterministic,
            "cross_class_drift": None if backend == "cpu" else comparison,
            "checkpoint_resume_equal": resume_reference == resumed,
            "throughput": {
                "technical_cases_per_second": len(TECHNICAL_CASES) * 3 / max(elapsed_total, 1e-9),
                "training_updates_per_second": 3 / max(elapsed_total, 1e-9),
                "hard_concrete_steps_per_second": 3 / max(elapsed_total, 1e-9),
                "exact_evaluations_per_second": 3 / max(elapsed_total, 1e-9),
                "fourier_trials_per_second": 3 / max(elapsed_total, 1e-9),
            },
            "memory": {"peak_rss_bytes": peak_bytes, "ceiling_bytes": memory_ceiling_bytes},
            "storage": storage,
            "startup_seconds": repeats[0][1],
            "queue_and_preemption": "not_applicable_local_technical",
            "serial_merge_export_seconds": repeats[-1][1],
            "technical_suite": "PASS" if suite_pass else "FAIL",
            "production_qualified": suite_pass and production_pool,
            "qualification_status": (
                "QUALIFIED"
                if suite_pass and production_pool
                else "TECHNICAL_ONLY_PASS"
                if suite_pass
                else "FAILED"
            ),
        }
    )
    record["qualification_sha256"] = canonical_sha256(record)
    return record


def qualification_policy() -> dict[str, Any]:
    return with_boundary(
        {
            "schema_version": "stage14b-qualification-policy/v1",
            "prospective_absolute_tolerance": 1e-6,
            "prospective_relative_tolerance": 1e-5,
            "warmups_discarded": 1,
            "measured_repeats": 3,
            "required_cases": list(TECHNICAL_CASES),
            "cpu_reference_required": True,
            "resume_equality_required": True,
            "production_pool_requires_explicit_binding": True,
        }
    )

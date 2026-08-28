"""Registered-teacher Stage 10 resource benchmark without endpoint reporting."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from circuit_families.interpretability.centred_logit_fidelity import (
    CentredLogitPredictiveAccumulator,
    centre_logits_across_classes,
)
from circuit_families.interpretability.component_ablation import masked_model_logits
from circuit_families.interpretability.masks import ComponentMask
from circuit_families.interpretability.sparse_search import rank_retained_components
from circuit_families.stage6d import load_technical_profiles
from circuit_families.stage7b.registered_fixture import (
    canonical_modular_addition_domain,
    load_registered_fixture_request,
    production_bindings,
    validate_registered_fixture_identity,
)

DISCOVERY_BENCHMARK_SCHEMA_VERSION = "stage10-discovery-compute-benchmark/v1"
PRIMARY_MODEL_COUNT = 105


class DiscoveryBenchmarkError(RuntimeError):
    """Raised when discovery timing or determinism evidence is incomplete."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if platform.system() == "Darwin" else observed * 1024


def _timed_exact_evaluation(
    *,
    model: torch.nn.Module,
    domain: torch.Tensor,
    reference: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    accumulator = CentredLogitPredictiveAccumulator(
        expected_example_count=domain.shape[0],
        class_count=reference.shape[1],
    )
    mask = ComponentMask.all_retained()
    started = time.perf_counter()
    for start in range(0, domain.shape[0], batch_size):
        stop = min(start + batch_size, domain.shape[0])
        with torch.inference_mode():
            logits = masked_model_logits(model, domain[start:stop], mask)[:, -1, :]
        accumulator.update(
            reference[start:stop],
            centre_logits_across_classes(logits),
            start_index=start,
        )
    fidelity = accumulator.finalize()
    elapsed = time.perf_counter() - started
    if not math.isfinite(fidelity):
        raise DiscoveryBenchmarkError("exact mask evaluation became nonfinite")
    return {
        "elapsed_seconds": elapsed,
        "finite": True,
        "result_sha256": _hash({"fidelity": fidelity}),
    }


def _timed_ranking(
    *,
    model: torch.nn.Module,
    domain: torch.Tensor,
    pseudo_targets: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    ranking = rank_retained_components(
        model,
        domain,
        pseudo_targets,
        ComponentMask.all_retained(),
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - started
    identities = [
        item.component_identifier for item in ranking.ranked_components
    ]
    if len(identities) != 516 or len(set(identities)) != 516:
        raise DiscoveryBenchmarkError("ranking did not cover the 516-component basis")
    return {
        "elapsed_seconds": elapsed,
        "component_count": len(identities),
        "ordering_sha256": _hash(identities),
    }


def build_discovery_benchmark_report(
    *,
    exact_evaluation_trials: list[dict[str, Any]],
    ranking_trials: list[dict[str, Any]],
    method_trials: list[dict[str, Any]],
    source_bindings: dict[str, str],
    peak_rss_bytes: int,
    hardware: dict[str, str],
) -> dict[str, Any]:
    """Build validated Stage 10 mechanics and planning projections."""
    if len(exact_evaluation_trials) != 2 or len(ranking_trials) != 2:
        raise DiscoveryBenchmarkError("exact evaluation and ranking require two trials")
    if exact_evaluation_trials[0]["result_sha256"] != exact_evaluation_trials[1][
        "result_sha256"
    ]:
        raise DiscoveryBenchmarkError("exact evaluation was not deterministic")
    if ranking_trials[0]["ordering_sha256"] != ranking_trials[1][
        "ordering_sha256"
    ]:
        raise DiscoveryBenchmarkError("ranking was not deterministic")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trial in method_trials:
        grouped.setdefault(str(trial["method_name"]), []).append(trial)
    if set(grouped) != {"greedy_deletion", "diversity_forced"}:
        raise DiscoveryBenchmarkError("both accepted discovery methods are required")

    method_records = []
    for method_name in sorted(grouped):
        trials = sorted(grouped[method_name], key=lambda item: item["repeat_index"])
        if len(trials) != 2:
            raise DiscoveryBenchmarkError("each discovery method requires two trials")
        deterministic = trials[0]["evidence_sha256"] == trials[1]["evidence_sha256"]
        if not deterministic:
            raise DiscoveryBenchmarkError(
                f"discovery evidence was not deterministic: {method_name}"
            )
        method_records.append(
            {
                "method_name": method_name,
                "native_budget_unit": trials[0]["native_budget_unit"],
                "native_budget_allowance": trials[0]["native_budget_allowance"],
                "exact_evaluation_allowance": trials[0][
                    "exact_evaluation_allowance"
                ],
                "repeat_elapsed_seconds": [item["elapsed_seconds"] for item in trials],
                "cold_start_elapsed_seconds": trials[0]["elapsed_seconds"],
                "steady_state_elapsed_seconds": trials[1]["elapsed_seconds"],
                "evidence_sha256": trials[0]["evidence_sha256"],
                "stopping_status": trials[0]["stopping_status"],
                "native_budget_consumed": trials[0]["native_budget_consumed"],
                "exact_budget_consumed": trials[0]["exact_budget_consumed"],
                "proposal_count": trials[0]["proposal_count"],
                "deterministic_reproduction": True,
            }
        )

    exact_steady = exact_evaluation_trials[1]["elapsed_seconds"]
    ranking_steady = ranking_trials[1]["elapsed_seconds"]
    method_projections = [
        {
            "method_name": item["method_name"],
            "model_count": PRIMARY_MODEL_COUNT,
            "one_worker_hours_at_technical_fixture_budget": (
                item["steady_state_elapsed_seconds"] * PRIMARY_MODEL_COUNT / 3600.0
            ),
            "budget_status": "technical_fixture_only_not_frozen",
        }
        for item in method_records
    ]
    report: dict[str, Any] = {
        "schema_version": DISCOVERY_BENCHMARK_SCHEMA_VERSION,
        "classification": "registered_teacher_technical_benchmark_only",
        "scientific_data": False,
        "production_eligible": False,
        "endpoint_values_recorded": False,
        "resolves_decisions": [],
        "registered_teacher_count": 1,
        "primary_model_count_projection": PRIMARY_MODEL_COUNT,
        "hardware": dict(sorted(hardware.items())),
        "source_bindings": dict(sorted(source_bindings.items())),
        "exact_evaluation": {
            "repeat_elapsed_seconds": [
                item["elapsed_seconds"] for item in exact_evaluation_trials
            ],
            "steady_state_seconds": exact_steady,
            "deterministic_result_hash": exact_evaluation_trials[0]["result_sha256"],
            "full_domain_count": 12_769,
            "inference_batch_size": 256,
        },
        "ranking_pass": {
            "repeat_elapsed_seconds": [item["elapsed_seconds"] for item in ranking_trials],
            "steady_state_seconds": ranking_steady,
            "deterministic_ordering_hash": ranking_trials[0]["ordering_sha256"],
            "component_count": 516,
        },
        "methods": method_records,
        "method_projections": method_projections,
        "peak_rss_bytes": peak_rss_bytes,
        "safe_concurrency": {
            "measured_process_peak_rss_bytes": peak_rss_bytes,
            "default_workers_on_measured_mac": 1,
            "increase_only_after_memory_monitoring": True,
        },
        "ledger_reuse": {
            "fidelity_threshold_reducer_sensitivity_without_new_search": True,
            "component_cap_sensitivity_without_new_search": True,
            "overlap_sensitivity_without_new_search": True,
            "packing_recomputation_without_new_search": True,
            "discovery_trajectory_threshold_change_may_require_new_search": True,
            "reuse_limited_to_masks_already_present_in_sealed_ledger": True,
        },
        "budget_interpretation": {
            "method_native_units_resource_equivalent": False,
            "common_exact_allowance_required": True,
            "raw_cross_method_packing_resource_matched": False,
            "final_budget_frozen": False,
        },
        "stage10_complete": True,
        "stage11_started": False,
        "scientific_red_team_started": False,
    }
    report["report_sha256"] = _hash(report)
    return report


def run_discovery_benchmark(
    *,
    repository_root: str | Path,
    predecessor_root: str | Path,
) -> dict[str, Any]:
    """Physically benchmark accepted discovery mechanics on one teacher."""
    root = Path(repository_root).resolve()
    predecessor = Path(predecessor_root).resolve()
    request_path = root / "followup/configs/stage7b/registered_fixture_request_v1.json"
    request = load_registered_fixture_request(request_path)
    _, checkpoint_path, _, _ = validate_registered_fixture_identity(
        repository_root=root,
        predecessor_root=predecessor,
        request=request,
    )
    bindings = production_bindings(
        repository_root=root,
        predecessor_root=predecessor,
        request=request,
    )
    payload = bindings.load_checkpoint_payload(checkpoint_path)
    model = bindings.restore_model(
        checkpoint_path=checkpoint_path,
        checkpoint_payload=payload,
        device="cpu",
    )
    domain = torch.as_tensor(
        canonical_modular_addition_domain(), dtype=torch.long, device="cpu"
    )
    batches = []
    with torch.inference_mode():
        for start in range(0, domain.shape[0], 256):
            batches.append(model(domain[start : start + 256])[:, -1, :].detach())
    full_logits = torch.cat(batches, dim=0)
    reference = centre_logits_across_classes(full_logits)
    pseudo_targets = torch.argmax(full_logits, dim=-1)

    exact_trials = [
        _timed_exact_evaluation(
            model=model,
            domain=domain,
            reference=reference,
            batch_size=256,
        )
        for _ in range(2)
    ]
    ranking_trials = [
        _timed_ranking(
            model=model,
            domain=domain,
            pseudo_targets=pseudo_targets,
            batch_size=256,
        )
        for _ in range(2)
    ]

    profiles = {
        profile.method_name: profile
        for profile in load_technical_profiles(
            root / "followup/configs/stage6d/technical_discovery_profiles_v1.json"
        )
    }
    method_trials = []
    for adapter_name in bindings.discovery_adapter_names:
        profile = profiles[adapter_name]
        for repeat_index in range(2):
            started = time.perf_counter()
            result = bindings.run_discovery(
                adapter_name=adapter_name,
                subject_kind="teacher",
                subject=model,
                request=request,
                output_root=Path("/private/tmp/stage10-no-output"),
            )
            elapsed = time.perf_counter() - started
            adapter_result = result["adapter_result"]
            evidence = asdict(adapter_result)
            method_trials.append(
                {
                    "method_name": adapter_name,
                    "repeat_index": repeat_index,
                    "elapsed_seconds": elapsed,
                    "native_budget_unit": profile.native_budget_unit,
                    "native_budget_allowance": profile.native_budget_allowance,
                    "exact_evaluation_allowance": profile.exact_evaluation_allowance,
                    "stopping_status": adapter_result.stopping_status,
                    "native_budget_consumed": adapter_result.native_budget_consumed,
                    "exact_budget_consumed": (
                        adapter_result.exact_evaluation_consumed
                    ),
                    "proposal_count": adapter_result.proposal_count,
                    "evidence_sha256": _hash(evidence),
                }
            )

    return build_discovery_benchmark_report(
        exact_evaluation_trials=exact_trials,
        ranking_trials=ranking_trials,
        method_trials=method_trials,
        source_bindings={
            "benchmark_module_sha256": _file_sha256(Path(__file__)),
            "registered_request_sha256": _file_sha256(request_path),
            "discovery_profiles_sha256": _file_sha256(
                root / "followup/configs/stage6d/technical_discovery_profiles_v1.json"
            ),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
        },
        peak_rss_bytes=_peak_rss_bytes(),
        hardware={
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
    )


def validate_discovery_benchmark_report(record: dict[str, Any]) -> None:
    """Validate a committed Stage 10 report and its scientific firewall."""
    if record.get("schema_version") != DISCOVERY_BENCHMARK_SCHEMA_VERSION:
        raise DiscoveryBenchmarkError("unsupported Stage 10 report schema")
    for field in ("scientific_data", "production_eligible", "endpoint_values_recorded"):
        if record.get(field) is not False:
            raise DiscoveryBenchmarkError(f"Stage 10 requires {field}=false")
    if record.get("resolves_decisions") != []:
        raise DiscoveryBenchmarkError("Stage 10 cannot resolve decisions")
    if record.get("stage10_complete") is not True:
        raise DiscoveryBenchmarkError("Stage 10 report is incomplete")
    if record.get("stage11_started") is not False:
        raise DiscoveryBenchmarkError("Stage 10 report crossed the Stage 11 boundary")
    if record.get("scientific_red_team_started") is not False:
        raise DiscoveryBenchmarkError("Stage 10 report crossed the red-team boundary")
    payload = dict(record)
    supplied = payload.pop("report_sha256", None)
    if supplied != _hash(payload):
        raise DiscoveryBenchmarkError("Stage 10 report hash mismatch")
    methods = record.get("methods")
    if not isinstance(methods, list) or {
        item.get("method_name") for item in methods if isinstance(item, dict)
    } != {"greedy_deletion", "diversity_forced"}:
        raise DiscoveryBenchmarkError("Stage 10 method coverage is incomplete")
    if not all(item.get("deterministic_reproduction") is True for item in methods):
        raise DiscoveryBenchmarkError("Stage 10 method reproduction failed")

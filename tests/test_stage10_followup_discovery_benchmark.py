from __future__ import annotations

import hashlib
import json
from pathlib import Path

from circuit_families.stage10 import (
    build_discovery_benchmark_report,
    validate_discovery_benchmark_report,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "followup/benchmarks/stage10_discovery_compute_benchmark_m5max_v1.json"
)


def _method(name: str, repeat: int) -> dict[str, object]:
    unit = (
        "ranked_component_proposals"
        if name == "greedy_deletion"
        else "restart_ranked_proposals"
    )
    return {
        "method_name": name,
        "repeat_index": repeat,
        "elapsed_seconds": 2.0 + repeat,
        "native_budget_unit": unit,
        "native_budget_allowance": 5,
        "exact_evaluation_allowance": 3,
        "stopping_status": "completed",
        "native_budget_consumed": 2,
        "exact_budget_consumed": 3,
        "proposal_count": 2,
        "evidence_sha256": ("a" if name == "greedy_deletion" else "b") * 64,
    }


def test_report_preserves_method_specific_budgets_and_firewall() -> None:
    report = build_discovery_benchmark_report(
        exact_evaluation_trials=[
            {"elapsed_seconds": 1.0, "result_sha256": "c" * 64},
            {"elapsed_seconds": 0.5, "result_sha256": "c" * 64},
        ],
        ranking_trials=[
            {"elapsed_seconds": 2.0, "ordering_sha256": "d" * 64},
            {"elapsed_seconds": 1.0, "ordering_sha256": "d" * 64},
        ],
        method_trials=[
            _method(name, repeat)
            for name in ("greedy_deletion", "diversity_forced")
            for repeat in range(2)
        ],
        source_bindings={"fixture": "e" * 64},
        peak_rss_bytes=1000,
        hardware={"device": "cpu"},
    )
    assert report["stage10_complete"] is True
    assert report["stage11_started"] is False
    assert report["scientific_red_team_started"] is False
    assert report["scientific_data"] is False
    assert report["endpoint_values_recorded"] is False
    assert report["budget_interpretation"] == {
        "method_native_units_resource_equivalent": False,
        "common_exact_allowance_required": True,
        "raw_cross_method_packing_resource_matched": False,
        "final_budget_frozen": False,
    }
    assert {item["native_budget_unit"] for item in report["methods"]} == {
        "ranked_component_proposals",
        "restart_ranked_proposals",
    }


def test_ledger_reuse_boundary_is_explicit() -> None:
    report = build_discovery_benchmark_report(
        exact_evaluation_trials=[
            {"elapsed_seconds": 1.0, "result_sha256": "c" * 64},
            {"elapsed_seconds": 0.5, "result_sha256": "c" * 64},
        ],
        ranking_trials=[
            {"elapsed_seconds": 2.0, "ordering_sha256": "d" * 64},
            {"elapsed_seconds": 1.0, "ordering_sha256": "d" * 64},
        ],
        method_trials=[
            _method(name, repeat)
            for name in ("greedy_deletion", "diversity_forced")
            for repeat in range(2)
        ],
        source_bindings={},
        peak_rss_bytes=1000,
        hardware={},
    )
    assert report["ledger_reuse"] == {
        "fidelity_threshold_reducer_sensitivity_without_new_search": True,
        "component_cap_sensitivity_without_new_search": True,
        "overlap_sensitivity_without_new_search": True,
        "packing_recomputation_without_new_search": True,
        "discovery_trajectory_threshold_change_may_require_new_search": True,
        "reuse_limited_to_masks_already_present_in_sealed_ledger": True,
    }


def test_committed_registered_teacher_report_is_bound_and_endpoint_blind() -> None:
    record = json.loads(REPORT.read_text(encoding="utf-8"))
    validate_discovery_benchmark_report(record)
    module_path = ROOT / "src/circuit_families/stage10/discovery_benchmark.py"
    assert record["source_bindings"]["benchmark_module_sha256"] == (
        hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    assert record["endpoint_values_recorded"] is False
    assert record["safe_concurrency"]["default_workers_on_measured_mac"] == 1

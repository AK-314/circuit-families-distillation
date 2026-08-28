from __future__ import annotations

import json
from pathlib import Path

import torch

from circuit_families.stage9.training_benchmark import (
    TrialResult,
    build_training_benchmark_report,
    validate_training_benchmark_report,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "followup/benchmarks/stage9_training_backend_benchmark_m5max_v1.json"
)


def _trial(*, device: str, condition: str, repeat: int, delta: float = 0.0):
    state = {"weight": torch.tensor([1.0 + delta], dtype=torch.float32)}
    outputs = torch.tensor([[1.0 + delta, 0.0]], dtype=torch.float32)
    return TrialResult(
        device=device,
        condition=condition,
        repeat_index=repeat,
        update_seconds=2.0 + repeat,
        initial_state_sha256="a" * 64,
        final_state_sha256=("b" if delta == 0.0 else "c") * 64,
        dense_output_sha256=("d" if delta == 0.0 else "e") * 64,
        dense_outputs=outputs,
        state=state,
        checkpoint_payload_bytes=100,
        dense_output_bytes=50,
        peak_rss_bytes=1000,
    )


def test_cpu_exact_pair_is_reference_candidate() -> None:
    trials = tuple(
        _trial(device="cpu", condition=condition, repeat=repeat)
        for condition in ("hard_target", "soft_target")
        for repeat in range(2)
    )
    report = build_training_benchmark_report(trials)
    qualification = report["backend_qualification"][0]
    assert qualification == {
        "device": "cpu",
        "bitwise_reproducible": True,
        "semantic_argmax_reproducible": True,
        "disposition": "verified_deterministic_reference_candidate",
    }
    assert report["scientific_data"] is False
    assert report["production_eligible"] is False
    assert report["decision_boundary"]["scientific_red_team_started"] is False


def test_mps_near_equal_pair_is_development_only() -> None:
    trials = []
    for condition in ("hard_target", "soft_target"):
        trials.append(_trial(device="mps", condition=condition, repeat=0))
        trials.append(
            _trial(device="mps", condition=condition, repeat=1, delta=1e-7)
        )
    report = build_training_benchmark_report(tuple(trials))
    qualification = report["backend_qualification"][0]
    assert qualification["bitwise_reproducible"] is False
    assert qualification["semantic_argmax_reproducible"] is True
    assert qualification["disposition"] == "development_only_not_definitive"


def test_storage_projection_is_bounded_and_explicit() -> None:
    trials = tuple(
        _trial(device="cpu", condition=condition, repeat=repeat)
        for condition in ("hard_target", "soft_target")
        for repeat in range(2)
    )
    report = build_training_benchmark_report(trials)
    assert report["storage_projections"] == [
        {
            "attempt_count": 90,
            "checkpoint_plus_dense_output_bytes": 13_500,
            "excludes_intermediate_checkpoints": True,
        },
        {
            "attempt_count": 180,
            "checkpoint_plus_dense_output_bytes": 27_000,
            "excludes_intermediate_checkpoints": True,
        },
    ]
    assert all(
        item["interpretation"] == "upper_level_planning_scenario_not_frozen"
        for item in report["runtime_projections"]
    )


def test_committed_m5max_report_is_complete_and_hash_bound() -> None:
    record = json.loads(REPORT.read_text(encoding="utf-8"))
    validate_training_benchmark_report(record)
    assert record["backend_qualification"] == [
        {
            "bitwise_reproducible": True,
            "device": "cpu",
            "disposition": "verified_deterministic_reference_candidate",
            "semantic_argmax_reproducible": True,
        },
        {
            "bitwise_reproducible": False,
            "device": "mps",
            "disposition": "development_only_not_definitive",
            "semantic_argmax_reproducible": True,
        },
    ]

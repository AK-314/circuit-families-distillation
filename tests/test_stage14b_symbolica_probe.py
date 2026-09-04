from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from circuit_families.stage14b import symbolica_probe
from circuit_families.stage14b.records import Stage14BError, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _training_fixture(seconds: float = 0.02) -> dict[str, object]:
    return {
        "measurements": [
            {
                "device": "cuda",
                "condition": condition,
                "steady_state_update_seconds": seconds,
            }
            for condition in ("hard_target", "soft_target")
        ]
    }


def test_projection_uses_frozen_stage13_v3_counts() -> None:
    report = symbolica_probe._planning_projection(
        _training_fixture(),
        {"median_seconds": 0.03},
        {"median_seconds": 0.04},
        repository_root=ROOT,
    )
    assert report["central_training_update_count"] == 15_013_440
    assert report["central_hard_concrete_step_count"] == 15_300_000
    assert report["exact_evaluation_count"] == 615_424
    expected_gpu_hours = (15_013_440 * 0.02 + 15_300_000 * 0.03) / 3600
    assert report["raw_gpu_device_hours"] == pytest.approx(expected_gpu_hours)
    assert report["decision"] == "diagnostic_only_not_launch_authorization"


def test_probe_report_validation_rejects_boundary_and_hash_changes() -> None:
    report = {
        "schema_version": symbolica_probe.PROBE_SCHEMA_VERSION,
        "probe_status": "PASS",
        "scientific_data": False,
        "production_eligible": False,
        "definitive_execution_started": False,
        "stage15_started": False,
        "registered_or_private_artifacts_accessed": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    symbolica_probe.validate_probe_report(report)

    corrupted = dict(report)
    corrupted["stage15_started"] = True
    with pytest.raises(Stage14BError, match="stage15_started=false"):
        symbolica_probe.validate_probe_report(corrupted)

    corrupted = dict(report)
    corrupted["probe_status"] = "FAIL"
    with pytest.raises(Stage14BError, match="not complete"):
        symbolica_probe.validate_probe_report(corrupted)

    corrupted = dict(report)
    corrupted["purpose"] = "post-hash mutation"
    with pytest.raises(Stage14BError, match="hash mismatch"):
        symbolica_probe.validate_probe_report(corrupted)


def test_probe_requires_cuda_before_running_measurements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(symbolica_probe.torch.cuda, "is_available", lambda: False)
    with pytest.raises(Stage14BError, match="requires at least one CUDA GPU"):
        symbolica_probe.run_symbolica_probe(ROOT, tmp_path / "output")


def test_probe_composes_and_authenticates_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(symbolica_probe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(symbolica_probe.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        symbolica_probe,
        "inventory_resource_pool",
        lambda **_kwargs: {"inventory_sha256": "inventory", "production_pool": False},
    )
    monkeypatch.setattr(
        symbolica_probe,
        "qualify_backend",
        lambda _inventory, **kwargs: {
            "backend": kwargs["backend"],
            "technical_suite": "PASS",
            "production_qualified": False,
        },
    )
    measurements = [
        {
            "device": "cuda",
            "condition": condition,
            "steady_state_update_seconds": 0.02,
        }
        for condition in ("hard_target", "soft_target")
    ]
    monkeypatch.setattr(
        symbolica_probe,
        "_training_measurements",
        lambda _root: {"measurements": measurements},
    )
    monkeypatch.setattr(
        symbolica_probe,
        "_model_forward_cross_backend",
        lambda _root: {"semantic_status": "PASS"},
    )
    monkeypatch.setattr(
        symbolica_probe,
        "_model_in_loop_hard_concrete",
        lambda _root: {"median_seconds": 0.03},
    )
    monkeypatch.setattr(
        symbolica_probe,
        "_exact_mask_evaluation",
        lambda _root: {"median_seconds": 0.04},
    )
    monkeypatch.setattr(symbolica_probe, "_nvidia_smi_summary", lambda: {"rows": []})
    monkeypatch.setattr(symbolica_probe, "_cuda_devices", lambda: [])
    monkeypatch.setattr(symbolica_probe, "_git_head", lambda _root: "abc123")

    report = symbolica_probe.run_symbolica_probe(ROOT, tmp_path / "output")
    symbolica_probe.validate_probe_report(report)
    assert report["source_commit"] == "abc123"
    assert report["scientific_data"] is False
    assert report["stage15_started"] is False
    assert (tmp_path / "output/symbolica-probe-report.json").is_file()


def test_shell_wrapper_is_valid_bash() -> None:
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/run_stage14_symbolica_probe.sh")],
        check=True,
    )


def test_torch_thread_limit_restores_prior_setting() -> None:
    before = symbolica_probe.torch.get_num_threads()
    with symbolica_probe._torch_thread_limit(1):
        assert symbolica_probe.torch.get_num_threads() == 1
    assert symbolica_probe.torch.get_num_threads() == before


def test_cli_marks_all_scientific_boundaries() -> None:
    source = (ROOT / "scripts/run_stage14_symbolica_probe.py").read_text(
        encoding="utf-8"
    )
    for expected in (
        "scientific_data=false",
        "production_eligible=false",
        "definitive_execution_started=false",
        "stage15_started=false",
    ):
        assert expected in source

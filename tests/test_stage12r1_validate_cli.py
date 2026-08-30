from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts.validate_stage12r1_independent_discovery import (
    run_validation,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_stage12r1_independent_discovery.py"
PYTHON = ROOT / ".venv/bin/python"


def _run(cwd: Path, hash_seed: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    return subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--validate-only",
        ],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_validate_report_has_explicit_technical_boundaries() -> None:
    report = run_validation()

    assert report["classification"] == "synthetic_technical_only"
    assert report["scientific_data"] is False
    assert report["production_eligible"] is False
    assert report["registered_model_access"] is False
    assert report["native_budget_unit"] == "optimizer_step"
    assert report["resume_matched"] is True
    assert report["endpoint1_retained_proportion"] == pytest.approx(6 / 516)
    assert report["stage6e_qualified_count"] >= 2
    assert report["stage6e_packing_lower_bound"] >= 1


def test_validate_cli_runs_from_repository_root() -> None:
    result = _run(ROOT, "1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "STAGE12R1_VALIDATE=PASS" in result.stdout
    assert "scientific_data=NO" in result.stdout
    assert "production_eligible=NO" in result.stdout


def test_validate_cli_runs_from_unrelated_directory(tmp_path) -> None:
    result = _run(tmp_path, "2")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "STAGE12R1_VALIDATE=PASS" in result.stdout


def test_hash_seed_does_not_change_validation_record(tmp_path) -> None:
    first = _run(tmp_path, "11")
    second = _run(tmp_path, "97")

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr

    first_hash = next(
        line
        for line in first.stdout.splitlines()
        if line.startswith("report_sha256=")
    )
    second_hash = next(
        line
        for line in second.stdout.splitlines()
        if line.startswith("report_sha256=")
    )

    assert first_hash == second_hash

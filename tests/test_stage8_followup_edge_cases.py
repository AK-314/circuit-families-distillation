from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from circuit_families.stage8 import EdgeCaseMatrixError, run_edge_case_matrix

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "followup/configs/stage8/technical_edge_case_matrix_v1.json"
SCRIPT = ROOT / "scripts/validate_stage8_edge_cases.py"


def test_complete_prescribed_matrix_passes() -> None:
    result = run_edge_case_matrix(repository_root=ROOT)
    assert result["case_count"] == 14
    assert result["passed_count"] == 14
    assert result["failed_count"] == 0
    assert result["stage8_complete"] is True
    assert result["stage9_started"] is False
    assert result["scientific_data"] is False
    assert result["production_eligible"] is False
    assert len(result["record_sha256"]) == 64


def test_matrix_rejects_changed_expectation(tmp_path: Path) -> None:
    record = json.loads(MATRIX.read_text(encoding="utf-8"))
    record["cases"][0]["expected"] = "wrong"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(EdgeCaseMatrixError, match="cases failed"):
        run_edge_case_matrix(repository_root=ROOT, matrix_path=path)


def test_matrix_rejects_scientific_or_production_authority(tmp_path: Path) -> None:
    for field in ("scientific_data", "production_eligible"):
        record = json.loads(MATRIX.read_text(encoding="utf-8"))
        record[field] = True
        path = tmp_path / f"{field}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(EdgeCaseMatrixError):
            run_edge_case_matrix(repository_root=ROOT, matrix_path=path)


def test_cli_is_cwd_independent_and_hashseed_deterministic(tmp_path: Path) -> None:
    outputs = []
    for seed in ("1", "987654321"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repository-root",
                str(ROOT),
            ],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])["stage8_complete"] is True

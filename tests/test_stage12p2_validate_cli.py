from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/validate_stage12p2_multi_architecture.py"
PYTHON = Path(sys.executable)


def _run(cwd: Path, *, seed: str = "1") -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    return subprocess.run(
        [str(PYTHON), str(VALIDATOR)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_contract(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("ARCH ref=") == 3
    assert (
        "LIFECYCLE hard=completed soft=completed interrupt=interrupted resume=completed"
    ) in result.stdout
    assert "ELIGIBILITY hard_pass=passed hard_fail=ineligible soft_pass=passed" in result.stdout
    assert "DISCOVERY release=released ineligible=BLOCKED" in result.stdout
    assert "FIXTURE_SCOPE=SYNTHETIC_TECHNICAL" in result.stdout
    assert "REPOSITORY_WRITES=NO" in result.stdout
    assert "REAL_TRAINING=NO" in result.stdout
    assert "REAL_TEACHER_CACHE=NO" in result.stdout
    assert "PRODUCTION_ARCHITECTURE_SELECTED=NO" in result.stdout
    assert "SCIENTIFIC_DATA=NO" in result.stdout
    assert result.stdout.rstrip().endswith("STAGE12P2_VALIDATION=PASS")


def test_validator_from_repository_root() -> None:
    _assert_contract(_run(ROOT))


def test_validator_from_unrelated_cwd() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _assert_contract(_run(Path(tmp)))


def test_validator_output_is_hashseed_deterministic() -> None:
    first = _run(ROOT, seed="1")
    second = _run(ROOT, seed="913")
    _assert_contract(first)
    _assert_contract(second)
    assert first.stdout == second.stdout

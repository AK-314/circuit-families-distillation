from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/validate_stage6b_hard_eligibility.py"


def _run_validator(cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("STAGE6B_REEXECUTED", None)
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_required_diagnostics(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr
    assert "POSITIVE agreement_count=12769 total_count=12769" in result.stdout
    assert "eligibility_status=ELIGIBLE failure_kind=NONE" in result.stdout
    assert "sealing_status=SEALED_HASH_CONSISTENT downstream_gate=ALLOWED" in (
        result.stdout
    )
    assert "NEGATIVE agreement_count=12768 total_count=12769" in result.stdout
    assert "failure_kind=subperfect_agreement" in result.stdout
    assert "sealing_status=UNSEALED downstream_gate=BLOCKED" in result.stdout
    assert "negative_attempt_retained=YES" in result.stdout
    assert result.stdout.count("teacher_decision_hash=") == 2
    assert result.stdout.count("student_decision_hash=") == 2
    assert "VALIDATION_WRITES=NO" in result.stdout
    assert result.stdout.rstrip().endswith("STAGE6B_VALIDATION=PASS")


def test_validate_only_cli_from_repository_root() -> None:
    _assert_required_diagnostics(_run_validator(ROOT))


def test_validate_only_cli_is_cwd_independent_and_writes_nothing(
    tmp_path: Path,
) -> None:
    before = tuple(tmp_path.iterdir())

    result = _run_validator(tmp_path)

    _assert_required_diagnostics(result)
    assert tuple(tmp_path.iterdir()) == before

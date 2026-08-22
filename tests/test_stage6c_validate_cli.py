from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/validate_stage6c_soft_eligibility.py"
HASH_RE = re.compile(
    r"^(POSITIVE|NEGATIVE).*teacher_soft_hash=([0-9a-f]{64}) "
    r"student_soft_hash=([0-9a-f]{64})",
    re.MULTILINE,
)


def _run_validator(cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("STAGE6C_REEXECUTED", None)
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
    assert "POSITIVE domain_count=12769/12769" in result.stdout
    assert "argmax_agreement=12769/12769 discrepancy=0 tolerance=0" in result.stdout
    assert "eligibility_status=ELIGIBLE failure_kinds=NONE" in result.stdout
    assert "sealing_status=SEALED_HASH_CONSISTENT downstream_gate=ALLOWED" in (
        result.stdout
    )
    assert "NEGATIVE domain_count=12769/12769" in result.stdout
    assert "argmax_agreement=12768/12769" in result.stdout
    assert "failure_kinds=tolerance_failure,argmax_rule_failure" in result.stdout
    assert "sealing_status=UNSEALED downstream_gate=BLOCKED" in result.stdout
    assert "negative_attempt_retained=YES" in result.stdout
    assert "FIXTURE_SCOPE=SYNTHETIC_TECHNICAL_IN_MEMORY" in result.stdout
    assert "VALIDATION_WRITES=NO" in result.stdout
    assert "REAL_TRAINING=NO" in result.stdout
    assert "REAL_TEACHER_CACHE=NO" in result.stdout
    assert "SCIENTIFIC_DATA=NO" in result.stdout
    assert result.stdout.rstrip().endswith("STAGE6C_VALIDATION=PASS")


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def test_validate_only_cli_from_repository_root() -> None:
    result = _run_validator(ROOT)

    _assert_required_diagnostics(result)


def test_validate_only_cli_is_cwd_independent_and_writes_nothing(
    tmp_path: Path,
) -> None:
    before = tuple(tmp_path.rglob("*"))

    result = _run_validator(tmp_path)

    _assert_required_diagnostics(result)
    assert tuple(tmp_path.rglob("*")) == before


def test_validate_only_output_and_hashes_are_deterministic() -> None:
    before = _git_status()

    first = _run_validator(ROOT)
    second = _run_validator(ROOT)

    _assert_required_diagnostics(first)
    _assert_required_diagnostics(second)
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == ""
    assert _git_status() == before

    hashes = HASH_RE.findall(first.stdout)
    assert [label for label, _, _ in hashes] == ["POSITIVE", "NEGATIVE"]
    positive_teacher, positive_student = hashes[0][1:]
    negative_teacher, negative_student = hashes[1][1:]
    assert positive_teacher == positive_student
    assert negative_teacher == positive_teacher
    assert negative_student != negative_teacher


def test_cli_fixture_is_isolated_from_training_and_teacher_cache_surfaces() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")

    assert "TrainerLifecycle" not in source
    assert "TechnicalSoftTargetAdapter" not in source
    assert "REAL_TRAINING=NO" in source
    assert "REAL_TEACHER_CACHE=NO" in source
    assert "torch.tensor" in source
    assert "torch.load" not in source
    assert "open(" not in source

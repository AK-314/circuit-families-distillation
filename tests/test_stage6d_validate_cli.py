from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "validate_stage6d_discovery_adapters.py"


def test_cli_is_cwd_independent_read_only_and_reports_required_fields(tmp_path):
    before = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    after = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert before == after

    output = completed.stdout
    assert output.count("method=") == 2
    assert "native_unit=" in output
    assert "native_usage=" in output
    assert "exact_usage=" in output
    assert "proposals=" in output
    assert "restarts=" in output
    assert "termination=" in output
    assert "exact_ledger_sha256=" in output
    assert "exact_ledger_counts=" in output
    assert "resource_warning=" in output
    assert "STAGE6D_VALIDATE=PASS" in output
    assert "scientific_data=NO" in output

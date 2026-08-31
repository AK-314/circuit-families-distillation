"""Validate-only CLI tests for Phase I E2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/run_phase1_e2_random_mask_fidelity.py").resolve()


def test_validate_only_rejects_missing_frozen_registry(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repository-root",
            str(tmp_path),
            "--source-root",
            str(tmp_path),
            "--validate-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "phase1_e2_random_mask_fidelity_null.json" in result.stderr

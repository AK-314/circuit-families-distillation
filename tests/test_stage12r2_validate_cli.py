from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_stage12r2_basis_sensitivity.py"


def run_validator(
    *,
    cwd: Path,
    hash_seed: str,
) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--validate-only",
            "--json",
        ],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_validate_cli_is_portable_and_hash_seed_deterministic(
    tmp_path: Path,
) -> None:
    root_seed_1 = run_validator(cwd=REPO_ROOT, hash_seed="1")
    root_seed_2 = run_validator(cwd=REPO_ROOT, hash_seed="901")
    unrelated = run_validator(cwd=tmp_path, hash_seed="901")

    assert root_seed_1["report_hash"] == root_seed_2["report_hash"]
    assert root_seed_2["report_hash"] == unrelated["report_hash"]
    assert root_seed_1["classification"] == "synthetic_technical_only"
    assert root_seed_1["scientific_data"] is False
    assert root_seed_1["production_eligible"] is False
    assert root_seed_1["rd004_resolved"] is False
    assert root_seed_1["registered_model_access"] is False

    assert root_seed_1["canonical"]["round_trip_mask_exact"] is True
    assert root_seed_1["attention_refinement"]["all_on_identity"] is True
    assert root_seed_1["partitions"]["distinct"] is True
    assert root_seed_1["rotations"]["nontrivial_all_on_identity"] is True
    assert root_seed_1["invalid_cross_basis_comparison_rejected"] is True

    evidence = root_seed_1["exact_ledger_consumption"]
    assert evidence["exact_fidelity_preserved"] == -0.125
    assert evidence["state"] == "evaluated"

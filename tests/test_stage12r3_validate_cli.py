from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from circuit_families.stage12r2.contracts import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_stage12r3_packing_calibration.py"


def _run(
    *,
    cwd: Path,
    hash_seed: str,
    json_mode: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    env.pop("PYTHONPATH", None)

    command = [
        sys.executable,
        str(SCRIPT),
        "--validate-only",
    ]
    if json_mode:
        command.append("--json")

    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _report(
    *,
    cwd: Path,
    hash_seed: str,
) -> dict[str, object]:
    completed = _run(cwd=cwd, hash_seed=hash_seed)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_validate_cli_is_portable_from_unrelated_cwd(tmp_path: Path) -> None:
    root = _report(cwd=ROOT, hash_seed="1")
    unrelated = _report(cwd=tmp_path, hash_seed="1")

    assert root == unrelated


def test_validate_cli_hash_is_pythonhashseed_independent(
    tmp_path: Path,
) -> None:
    first = _report(cwd=tmp_path, hash_seed="1")
    second = _report(cwd=tmp_path, hash_seed="987654")

    assert first["report_hash"] == second["report_hash"]
    assert first == second


def test_report_hash_is_canonical_content_hash() -> None:
    report = _report(cwd=ROOT, hash_seed="7")
    claimed = report.pop("report_hash")

    assert claimed == canonical_sha256(report)


def test_report_contains_four_distinct_layers() -> None:
    report = _report(cwd=ROOT, hash_seed="1")
    layers = report["layers"]

    assert set(layers) == {
        "combinatorial_floor",
        "ordinary_restart_baseline",
        "local_exact_perturbation",
        "tractable_feasible_region",
    }
    identities = [layer["profile_identity"] for layer in layers.values()]
    assert len(set(identities)) == 4


def test_report_preserves_required_outcome_classes_without_interpretation() -> None:
    report = _report(cwd=ROOT, hash_seed="1")
    outcomes = report["outcome_preservation"]

    assert set(outcomes) == {
        "null",
        "negative",
        "zero",
        "failed",
        "unavailable",
        "censored",
    }
    assert outcomes["null"]["value"] is None
    assert outcomes["negative"]["local_nonqualified_neighbor_count"] > 0
    assert outcomes["zero"]["failed_search_packing_lower_bound"] == 0
    assert outcomes["failed"]["search_procedure_failed"] is True
    assert outcomes["unavailable"]["status"] == "unavailable"
    assert outcomes["censored"]["exact_censored_count"] > 0


def test_report_preserves_claim_boundaries() -> None:
    report = _report(cwd=ROOT, hash_seed="1")
    boundaries = report["claim_boundaries"]

    assert boundaries["ordinary_restart_independent_method_claim"] is False
    assert boundaries["local_inherited_fidelity"] is False
    assert boundaries["local_surrogate_fidelity"] is False
    assert boundaries["tractable_main_scale_transfer"] is False
    assert boundaries["mechanism_count_claim"] is False
    assert boundaries["production_packing_policy_selected"] is False
    assert boundaries["rd_006_open"] is True
    assert boundaries["rd_008_open"] is True
    assert boundaries["rd_009_open"] is True


def test_validate_cli_is_technical_only_and_creates_no_outputs() -> None:
    report = _report(cwd=ROOT, hash_seed="1")
    execution = report["execution"]

    assert execution == {
        "validate_only": True,
        "outputs_created": False,
        "registered_data_loaded": False,
        "registered_model_execution": False,
        "scientific_execution": False,
        "scientific_data": False,
        "production_eligible": False,
    }


def test_layer_specific_calibration_contracts_are_exercised() -> None:
    report = _report(cwd=ROOT, hash_seed="1")
    layers = report["layers"]

    combinatorial = layers["combinatorial_floor"]
    assert combinatorial["fidelity_claim"] is False
    assert combinatorial["exact_evaluation_count"] == 0
    assert combinatorial["endpoint2_claim"] is False

    ordinary = layers["ordinary_restart_baseline"]
    assert ordinary["duplicate_recovery_present"] is True
    assert ordinary["discovery_relationship"] == (
        "same_discovery_family_ordinary_restart"
    )

    local = layers["local_exact_perturbation"]
    assert local["qualified_proposal_count"] > 0
    assert local["nonqualified_proposal_count"] > 0
    assert local["inherited_fidelity_used"] is False
    assert local["surrogate_fidelity_used"] is False

    tractable = layers["tractable_feasible_region"]
    assert tractable["certificate_exactness"] == "exact"
    assert tractable["certificate_exhaustive"] is True
    assert tractable["search_missed_feasible_count"] > 0
    assert tractable["packing_gap"] > 0


def test_human_cli_reports_pass_and_hash(tmp_path: Path) -> None:
    completed = _run(
        cwd=tmp_path,
        hash_seed="37",
        json_mode=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "classification=synthetic_technical_only" in completed.stdout
    assert "registered_data_loaded=false" in completed.stdout
    assert "scientific_execution=false" in completed.stdout
    assert "report_hash=" in completed.stdout
    assert "STAGE12R3_VALIDATE=PASS" in completed.stdout


def test_validate_only_flag_is_required(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--validate-only" in completed.stderr


@pytest.mark.parametrize("hash_seed", ["0", "1", "123", "99991"])
def test_json_report_contains_no_invocation_cwd(
    tmp_path: Path,
    hash_seed: str,
) -> None:
    report = _report(cwd=tmp_path, hash_seed=hash_seed)
    encoded = json.dumps(report, sort_keys=True)

    assert str(ROOT) not in encoded
    assert str(tmp_path) not in encoded

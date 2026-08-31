"""CLI tests for Phase I E1 validate-only behavior."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/run_phase1_e1_jaccard_null.py").resolve()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_complete_fixture(root: Path) -> Path:
    circuits = root / "circuits.csv"
    pairs = root / "pairs.csv"
    manifest = root / "manifest.json"
    common = {
        "model_seed": 0,
        "checkpoint_step": 9050,
        "cell_id": "fixture-cell",
        "fidelity_numerator": 99,
        "fidelity_denominator": 100,
        "distinctness_numerator": 1,
        "distinctness_denominator": 2,
    }
    _write_csv(
        circuits,
        [
            {
                **common,
                "circuit_id": circuit_id,
                "mask_sha256": digit * 64,
                "retained_heads": 4,
                "retained_neurons": 2,
                "retained_components": 6,
                "threshold_pass": "True",
            }
            for circuit_id, digit in (("C1", "1"), ("C2", "2"))
        ],
    )
    _write_csv(
        pairs,
        [
            {
                **common,
                "circuit_i": "C1",
                "circuit_j": "C2",
                "intersection_count": 5,
                "union_count": 7,
                "jaccard_numerator": 5,
                "jaccard_denominator": 7,
            }
        ],
    )
    manifest.write_text(
        json.dumps(
            {
                "outputs": {
                    "circuits.csv": _sha(circuits),
                    "pairs.csv": _sha(pairs),
                }
            }
        ),
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "experiment_type": "phase1_e1_size_matched_jaccard_null",
        "analysis_id": "fixture",
        "source": {
            "manifest": {"path": "manifest.json", "sha256": _sha(manifest)},
            "circuits_table": {"path": "circuits.csv", "sha256": _sha(circuits)},
            "pairwise_overlap_table": {"path": "pairs.csv", "sha256": _sha(pairs)},
        },
        "component_universe": {
            "attention_head_count": 4,
            "mlp_neuron_count": 512,
            "total_count": 516,
            "attention_head_identifiers": ["H0", "H1", "H2", "H3"],
            "mlp_neuron_identifier_template": "N0-N511",
        },
        "comparison_set": {
            "scope": "within_family_pairs_only",
            "checkpoint_step": 9050,
            "model_seeds": [0],
            "fidelity_numerator": 99,
            "fidelity_denominator": 100,
            "distinctness_numerator": 1,
            "distinctness_denominator": 2,
            "cell_ids": ["fixture-cell"],
            "expected_family_sizes": {"0": 2},
            "expected_circuit_count": 2,
            "expected_pair_count": 1,
        },
        "null_models": ["size_matched", "basis_stratified"],
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_validate_only_succeeds_without_writing_analysis_outputs(tmp_path: Path) -> None:
    config = _write_complete_fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repository-root",
            str(tmp_path),
            "--source-root",
            str(tmp_path),
            "--config",
            str(config),
            "--validate-only",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPYCACHEPREFIX": "/tmp/cfd-e1-test-pycache"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "validation_passed"
    assert not (tmp_path / "pairwise_jaccard_null.csv").exists()


def test_cli_requires_output_unless_validate_only() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            "missing-source-root",
            "--validate-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Required E1 manifest is absent" in result.stderr
    assert not Path("results/phase1_e1_jaccard_null").exists()

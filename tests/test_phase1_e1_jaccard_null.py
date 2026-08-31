"""Focused tests for Phase I E1 size-matched Jaccard nulls."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from circuit_families.analysis.phase1_e1_jaccard_null import (
    CircuitRecord,
    E1ValidationError,
    ObservedPair,
    analyse_pair,
    convolve_intersections,
    derive_null_seed,
    hypergeometric_pmf,
    jaccard_from_intersection,
    validate_inputs,
)


def circuit(identifier: str, *, heads: int, neurons: int) -> CircuitRecord:
    return CircuitRecord(
        model_seed=0,
        checkpoint_step=9050,
        cell_id="fixture-cell",
        circuit_id=identifier,
        mask_sha256=("a" if identifier == "C1" else "b") * 64,
        retained_heads=heads,
        retained_neurons=neurons,
        retained_components=heads + neurons,
    )


def pair() -> ObservedPair:
    left = circuit("C1", heads=4, neurons=2)
    right = circuit("C2", heads=4, neurons=3)
    return ObservedPair(
        pair_id="s0:fixture-cell:C1--C2",
        model_seed=0,
        checkpoint_step=9050,
        cell_id="fixture-cell",
        left=left,
        right=right,
        intersection=5,
        union=8,
    )


def test_hypergeometric_exact_combinatorics() -> None:
    distribution = hypergeometric_pmf(5, 2, 2)
    assert [point.intersection for point in distribution] == [0, 1, 2]
    assert [point.probability for point in distribution] == pytest.approx([3 / 10, 6 / 10, 1 / 10])
    assert math.fsum(point.probability for point in distribution) == pytest.approx(1.0)


def test_hypergeometric_boundary_and_empty_jaccard() -> None:
    assert hypergeometric_pmf(4, 4, 2)[0].intersection == 2
    assert hypergeometric_pmf(4, 0, 0)[0].probability == 1.0
    assert jaccard_from_intersection(0, 0, 0) == 1.0
    with pytest.raises(E1ValidationError, match="cannot exceed"):
        hypergeometric_pmf(4, 5, 0)


def test_basis_stratified_distribution_is_exact_convolution() -> None:
    heads = hypergeometric_pmf(2, 1, 1)
    neurons = hypergeometric_pmf(3, 1, 1)
    combined = convolve_intersections(heads, neurons)
    assert [point.intersection for point in combined] == [0, 1, 2]
    assert [point.probability for point in combined] == pytest.approx([1 / 3, 1 / 2, 1 / 6])


def test_seed_and_sampling_are_deterministic_and_pair_specific() -> None:
    observed = pair()
    first = analyse_pair(
        observed,
        "basis_stratified",
        analysis_id="fixture",
        draw_count=1000,
        confidence_level=0.95,
    )
    second = analyse_pair(
        observed,
        "basis_stratified",
        analysis_id="fixture",
        draw_count=1000,
        confidence_level=0.95,
    )
    assert first == second
    assert derive_null_seed("fixture", observed.pair_id, "size_matched") != derive_null_seed(
        "fixture", observed.pair_id, "basis_stratified"
    )
    changed = replace(observed, pair_id=observed.pair_id + "-changed")
    assert derive_null_seed("fixture", changed.pair_id, "size_matched") != derive_null_seed(
        "fixture", observed.pair_id, "size_matched"
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_inputs_rejects_incomplete_pair_set(tmp_path: Path) -> None:
    circuits_path = tmp_path / "circuits.csv"
    pairs_path = tmp_path / "pairs.csv"
    manifest_path = tmp_path / "manifest.json"
    circuit_rows = []
    for index in range(1, 4):
        circuit_rows.append(
            {
                "model_seed": 0,
                "checkpoint_step": 9050,
                "cell_id": "fixture-cell",
                "fidelity_numerator": 99,
                "fidelity_denominator": 100,
                "distinctness_numerator": 1,
                "distinctness_denominator": 2,
                "circuit_id": f"C{index}",
                "mask_sha256": str(index) * 64,
                "retained_heads": 4,
                "retained_neurons": 2,
                "retained_components": 6,
                "threshold_pass": "True",
            }
        )
    _write_csv(circuits_path, circuit_rows)
    _write_csv(
        pairs_path,
        [
            {
                "model_seed": 0,
                "checkpoint_step": 9050,
                "cell_id": "fixture-cell",
                "fidelity_numerator": 99,
                "fidelity_denominator": 100,
                "distinctness_numerator": 1,
                "distinctness_denominator": 2,
                "circuit_i": "C1",
                "circuit_j": "C2",
                "intersection_count": 5,
                "union_count": 7,
                "jaccard_numerator": 5,
                "jaccard_denominator": 7,
            }
        ],
    )
    manifest_path.write_text("{}\n", encoding="utf-8")
    source = {
        "manifest": {"path": "manifest.json", "sha256": _sha(manifest_path)},
        "circuits_table": {"path": "circuits.csv", "sha256": _sha(circuits_path)},
        "pairwise_overlap_table": {"path": "pairs.csv", "sha256": _sha(pairs_path)},
    }
    manifest_path.write_text(
        json.dumps({"outputs": {record["path"]: record["sha256"] for record in source.values()}}),
        encoding="utf-8",
    )
    source["manifest"]["sha256"] = _sha(manifest_path)
    config = {
        "schema_version": 1,
        "experiment_type": "phase1_e1_size_matched_jaccard_null",
        "analysis_id": "fixture",
        "source": source,
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
            "expected_family_sizes": {"0": 3},
            "expected_circuit_count": 3,
            "expected_pair_count": 3,
        },
        "null_models": ["size_matched", "basis_stratified"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(E1ValidationError, match="pair count"):
        validate_inputs(config_path, source_root=tmp_path)

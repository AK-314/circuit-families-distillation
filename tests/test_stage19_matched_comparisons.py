from __future__ import annotations

from decimal import Decimal

from circuit_families.analysis.stage19_matched_comparisons import (
    CHECKPOINT_STEPS,
    Circuit,
    build_stage19_tables,
    optimal_matching,
)


def _circuit(label: str, value: str, size: int = 10) -> Circuit:
    return Circuit(
        cell_id="left" if label.startswith("L") else "right",
        circuit_id=label,
        mask_sha256=label.lower().ljust(64, "0"),
        model_seed=0,
        checkpoint_index=1,
        checkpoint_step=200,
        retained_components=size,
        fidelity=Decimal(value),
    )


def test_matching_maximises_cardinality_before_minimising_cost() -> None:
    left = (_circuit("L1", "0.000"), _circuit("L2", "0.020"))
    right = (_circuit("R1", "0.009"), _circuit("R2", "0.011"))
    result = optimal_matching(
        left,
        right,
        value=lambda circuit: circuit.fidelity,
        tolerance=Decimal("0.01"),
    )
    assert [(match.left.circuit_id, match.right.circuit_id) for match in result.matches] == [
        ("L1", "R1"),
        ("L2", "R2"),
    ]
    assert result.total_absolute_difference == Decimal("0.018")
    assert result.unmatched_left == ()
    assert result.unmatched_right == ()


def test_matching_reports_unmatched_circuits() -> None:
    result = optimal_matching(
        (_circuit("L1", "0.90"),),
        (_circuit("R1", "0.95"),),
        value=lambda circuit: circuit.fidelity,
        tolerance=Decimal("0.01"),
    )
    assert result.matches == ()
    assert [circuit.circuit_id for circuit in result.unmatched_left] == ["L1"]
    assert [circuit.circuit_id for circuit in result.unmatched_right] == ["R1"]


def _family(seed: int, index: int, step: int, size: int) -> dict[str, object]:
    return {
        "global_cell_index": seed * len(CHECKPOINT_STEPS) + index,
        "cell_id": f"s{seed}-step{step}-f99of100-d1of2",
        "model_seed": seed,
        "checkpoint_index": index,
        "checkpoint_step": step,
        "fidelity_numerator": 99,
        "fidelity_denominator": 100,
        "displayed_fidelity": "0.990",
        "distinctness_numerator": 1,
        "distinctness_denominator": 2,
        "displayed_jaccard_cutoff": "0.50",
        "primary_cell": True,
        "family_size": size,
        "right_censored": False,
        "status": "complete" if size else "sparsity_failure",
        "stopping_reason": "family_complete" if size else "requested_member_1_sparsity_failure",
    }


def _circuit_row(seed: int, step: int, label: str, fidelity: str, size: int) -> dict[str, object]:
    return {
        "cell_id": f"s{seed}-step{step}-f99of100-d1of2",
        "circuit_id": label,
        "mask_sha256": (label + str(step)).lower().ljust(64, "0"),
        "model_seed": seed,
        "checkpoint_index": CHECKPOINT_STEPS.index(step) + 1,
        "checkpoint_step": step,
        "retained_components": size,
        "fidelity": fidelity,
    }


def test_stage19_tables_preserve_empty_cells_and_build_both_matches() -> None:
    families = [
        _family(seed, index, step, int(seed == 0 and step in (200, 3400)))
        for seed in range(5)
        for index, step in enumerate(CHECKPOINT_STEPS, start=1)
    ]
    circuits = [
        _circuit_row(0, 200, "C1", "0.991", 10),
        _circuit_row(0, 3400, "C1", "0.995", 13),
    ]
    tables = build_stage19_tables(
        stage18_run_id="stage18-test",
        circuit_rows=circuits,
        family_rows=families,
        overlap_rows=(),
    )
    assert len(tables.matched_fidelity_summary) == 105
    assert len(tables.matched_sparsity_summary) == 105
    fidelity = next(
        row
        for row in tables.matched_fidelity_summary
        if row["model_seed"] == 0
        and row["left_checkpoint_step"] == 200
        and row["right_checkpoint_step"] == 3400
    )
    assert fidelity["comparison_status"] == "matched"
    assert fidelity["matched_count"] == 1
    sparsity = next(
        row
        for row in tables.matched_sparsity_summary
        if row["model_seed"] == 0
        and row["left_checkpoint_step"] == 200
        and row["right_checkpoint_step"] == 3400
    )
    assert sparsity["comparison_status"] == "matched"
    assert sparsity["fidelity_change_right_minus_left"] == Decimal("0.004")
    assert len(tables.empty_cells) == 33
    assert all(row["handling"] == "reported_as_empty_not_imputed" for row in tables.empty_cells)
    assert len(tables.pareto_frontiers) == 2

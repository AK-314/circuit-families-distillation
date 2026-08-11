from __future__ import annotations

from circuit_families.analysis.stage22_freeze import (
    prediction_quantities,
    prediction_table_block,
    resolution_rows,
)


def _protocol() -> str:
    rows = [
        "| Quantity | H1 prediction | H2 prediction | Mixed or unresolved interpretation |",
        "| ----- | ----- | ----- | ----- |",
        *[
            f"| {quantity} | h1 | h2 | mixed |"
            for quantity in (
                "Recovered structural family size",
                "Transfer-distinct group count",
                "Circuit size",
                "Pairwise structural overlap",
                "Cross-subset transfer",
                "Timing of change",
                "Matched-fidelity result",
                "Matched-sparsity result",
                "Empty-family transition",
            )
        ],
    ]
    return (
        "before\n## **6\\. Frozen prediction table**\n"
        + "\n".join(rows)
        + ("\n### **Control interpretation table**\nafter")
    )


def test_prediction_table_extraction_and_resolution_are_exact() -> None:
    protocol = _protocol()
    assert len(prediction_table_block(protocol).splitlines()) == 11
    assert len(prediction_quantities(protocol)) == 9
    rows = resolution_rows(protocol)
    assert len(rows) == 9
    assert rows[-1]["resolution_category"] == "Supported"
    assert all(row["provisional_pending_stage18_reproduction"] for row in rows)

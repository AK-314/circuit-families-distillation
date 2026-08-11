from __future__ import annotations

import numpy as np

from circuit_families.analysis.stage21_figures import numeric_matrix, principal_figures_caption


def test_numeric_matrix_preserves_missing_values_as_nan() -> None:
    rows = (
        {"seed": 0, "step": 10, "value": "1.5"},
        {"seed": 1, "step": 20, "value": "2.5"},
        {"seed": 1, "step": 10, "value": ""},
    )
    matrix = numeric_matrix(
        rows,
        row_values=(0, 1),
        column_values=(10, 20),
        row_key="seed",
        column_key="step",
        value_key="value",
    )
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 1.5
    assert np.isnan(matrix[0, 1])
    assert np.isnan(matrix[1, 0])
    assert matrix[1, 1] == 2.5


def test_principal_figure_captions_cover_every_figure_and_aggregation_unit() -> None:
    caption = principal_figures_caption()
    assert all(f"Figure {index}." in caption for index in range(1, 6))
    assert "independently trained model seed" in caption
    assert "unweighted mean" in caption
    assert "not treated as independent replications" in caption
    assert "not imputed" in caption
    assert "logged every 50 training steps" in caption
    assert "structural distance is one minus overlap" in caption
    assert "maximum absolute fidelity difference" in caption
    assert "no-generalisation control was unavailable" in caption
    assert "stage 18 reproduction comparison was pending" in caption.lower()

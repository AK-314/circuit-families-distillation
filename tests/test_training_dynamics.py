"""Tests for checkpoint-based training-dynamics diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from circuit_families.analysis.training_dynamics import (
    model_parameter_group_norms,
    optimizer_moment_norms,
    parameter_group,
    plot_loss_norm_diagnostics,
    read_diagnostic_csv,
    write_diagnostic_caption,
    write_diagnostic_csv,
)


def test_parameter_groups_are_disjoint_and_complete() -> None:
    assert parameter_group("embed.W_E") == "embedding"
    assert parameter_group("pos_embed.W_pos") == "positional_embedding"
    assert parameter_group("blocks.0.attn.W_Q") == "attention"
    assert parameter_group("blocks.0.mlp.W_in") == "mlp"
    assert parameter_group("unembed.W_U") == "unembedding"

    with pytest.raises(
        ValueError,
        match="Unclassified trainable parameter",
    ):
        parameter_group("unknown.weight")


def test_parameter_group_norms_are_calculated_without_double_counting() -> None:
    state = {
        "embed.W_E": torch.tensor([3.0, 4.0]),
        "pos_embed.W_pos": torch.tensor([12.0]),
        "blocks.0.attn.W_Q": torch.tensor([5.0]),
        "blocks.0.mlp.W_in": torch.tensor([8.0, 15.0]),
        "unembed.W_U": torch.tensor([7.0, 24.0]),
        "blocks.0.attn.mask": torch.empty((0, 0)),
        "blocks.0.attn.IGNORE": torch.tensor(0.0),
    }

    norms = model_parameter_group_norms(state)

    assert norms["embedding_parameter_norm"] == 5.0
    assert norms["positional_embedding_norm"] == 12.0
    assert norms["attention_parameter_norm"] == 5.0
    assert norms["mlp_parameter_norm"] == 17.0
    assert norms["unembedding_parameter_norm"] == 25.0

    expected_total = math.sqrt(
        5.0**2
        + 12.0**2
        + 5.0**2
        + 17.0**2
        + 25.0**2
    )

    assert norms["total_parameter_norm"] == expected_total


def test_optimizer_moment_norms_are_extracted_correctly() -> None:
    optimizer_state = {
        "state": {
            0: {
                "exp_avg": torch.tensor([3.0, 4.0]),
                "exp_avg_sq": torch.tensor([5.0, 12.0]),
                "step": torch.tensor(1.0),
            },
            1: {
                "exp_avg": torch.tensor([12.0]),
                "exp_avg_sq": torch.tensor([84.0]),
                "step": torch.tensor(1.0),
            },
        },
        "param_groups": [],
    }

    first, second = optimizer_moment_norms(optimizer_state)

    assert first == 13.0
    assert second == 85.0


def test_empty_optimizer_state_has_zero_moment_norms() -> None:
    first, second = optimizer_moment_norms(
        {
            "state": {},
            "param_groups": [],
        }
    )

    assert first == 0.0
    assert second == 0.0


def _diagnostic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for step in (0, 50, 100, 150, 200):
        rows.append(
            {
                "training_step": step,
                "train_loss": 1.0 / (step + 1),
                "test_loss": 2.0 / (step + 1),
                "train_accuracy": min(step / 100, 1.0),
                "test_accuracy": min(step / 200, 1.0),
                "total_parameter_norm": 10.0 + step / 100,
                "embedding_parameter_norm": 2.0,
                "positional_embedding_norm": 1.0,
                "attention_parameter_norm": 5.0,
                "mlp_parameter_norm": 7.0,
                "unembedding_parameter_norm": 3.0 + step / 200,
                "optimizer_first_moment_norm": step / 1000,
                "optimizer_second_moment_norm": step / 10000,
                "checkpoint_path": (
                    f"checkpoints/test-run/step_{step:08d}.pt"
                ),
                "checkpoint_sha256": "a" * 64,
            }
        )

    return rows


def test_diagnostic_table_generation_is_deterministic(
    tmp_path: Path,
) -> None:
    rows = _diagnostic_rows()

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    write_diagnostic_csv(rows, first)
    write_diagnostic_csv(rows, second)

    assert first.read_bytes() == second.read_bytes()
    assert read_diagnostic_csv(first) == read_diagnostic_csv(second)


def test_diagnostic_figure_regenerates_from_source_table(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    write_diagnostic_csv(_diagnostic_rows(), source)
    rows = read_diagnostic_csv(source)

    first_png = tmp_path / "first.png"
    first_pdf = tmp_path / "first.pdf"
    second_png = tmp_path / "second.png"
    second_pdf = tmp_path / "second.pdf"

    plot_loss_norm_diagnostics(
        rows,
        run_id="test-run",
        stable_post_step=200,
        png_path=first_png,
        pdf_path=first_pdf,
    )
    plot_loss_norm_diagnostics(
        rows,
        run_id="test-run",
        stable_post_step=200,
        png_path=second_png,
        pdf_path=second_pdf,
    )

    assert first_png.read_bytes() == second_png.read_bytes()
    assert first_pdf.read_bytes() == second_pdf.read_bytes()

    first_caption = tmp_path / "first.txt"
    second_caption = tmp_path / "second.txt"

    write_diagnostic_caption(
        run_id="test-run",
        checkpoint_count=len(rows),
        stable_post_step=200,
        output_path=first_caption,
    )
    write_diagnostic_caption(
        run_id="test-run",
        checkpoint_count=len(rows),
        stable_post_step=200,
        output_path=second_caption,
    )

    assert first_caption.read_bytes() == second_caption.read_bytes()
